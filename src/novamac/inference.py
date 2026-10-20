"""
The Novamac class: one transient scene (one band), its data, the model
built from it, the annealed HMC sampler that draws from the posterior, and
the diagnostics and plots that read the sampler's products.

    nm = Novamac(data=..., weight=..., mask=..., is_template=..., crpix=..., crtan=...,
                 crval=..., CD=..., psf_n=..., psf_Rd=..., psf_q=..., psf_PA=...,
                 zeropoint=27.5, ps_center=[[x, y]], prior="path/to/prior/final",
                 out_dir="fits/01234567", name="01234567_g", num_chains=4)
    nm.sample()
    nm.plot_all()

Everything the inference needs is given at construction; the data-specific
preparation (cutting cutouts, reading WCS and PSF calibrations, units) stays
outside the package. Optional inputs are estimated from the data when not
given: the initial scene (a drizzled stack of the template visits, see
init_scene.py), the per-image sky (border median) and the initial
point-source fluxes (aperture photometry of the data minus the initial
scene). Every sampler setting is a class attribute with a documented
default; pass any of them as keywords to override per instance, or set
them on the class to change the default for every instance.

Schedule (see hmc.py for the method)
------------------------------------
  warmup   warmup_steps at t_start       HMC only; finds the step size
  anneal   anneal_steps, t: t_start -> 0  per step: Heun predictor of the
                                          scene's reverse diffusion (prior score
                                          only), then one HMC corrector
                                          transition at the new t (likelihood)
  hold     hold_steps at t = 0            HMC only at the true posterior; the
                                          step sizes adapt for hold_adapt_chunks
                                          more chunks, then every kernel is fixed
                                          and the saved states are the draws

Mass matrix: fixed in shape, inv_mass(t) = inv_mass0 + mass_sigma_fraction *
sigma(t)^2 on the scene, with inv_mass0 a Gauss-Newton estimate of the t = 0
posterior variance made once at setup. Nothing about it is learned from the
chain. The step size is adapted per chain every chunk_steps (x0.7 / x1.2,
faster during warmup) and is the one thing to watch: a collapsing step size
means some direction is badly preconditioned or the target is misbehaving.

Only the scene is sampled from the start; every other block is frozen at its
initial estimate until the diffusion time in release_t (the transient flux
can be released once the data constrain it, see release_latest_t).

Products: see products.py. The plots and diagnostics work off those files,
so a run can be sampled in one job and plotted in another by constructing
the same Novamac again.
"""

import json
import traceback
from pathlib import Path

import caskade as ck
import jax
import jax.numpy as jnp
import numpy as np

from ._backend import DTYPE
from .build_likelihood import build_likelihood
from .build_prior import build_prior
from .hmc import (
    SCORE_FIELDS,
    adapt_step_size,
    block_indices,
    hutchinson_jj_diag,
    make_block_curvature_fn,
    make_broadening_fn,
    make_chunk_runner,
    make_linear_gibbs_fn,
    make_log_parts,
    make_log_variance_fn,
    make_nuisance_log_prior,
    make_score_fn,
    make_sigma_fn,
    make_target,
    release_inverse_mass,
)
from .host_prior import Prior, load_prior
from .init_scene import (
    aperture_fluxes,
    border_sky,
    cd_arcsec,
    data_pixel_scale,
    prepare_initial_scene,
)
from .psf import fit_moffat_psfs
from .products import RunFiles
from .sampler import (
    build_schedule,
    calibrate_prior_offset,
    find_initial_step_size,
    flat_data_weight,
    linear_coordinates,
    parameter_blocks,
    save_chain_states,
    setup_inverse_mass,
    start_time,
    write_trace,
)
from .utils import Timer, enable_console_logging, log, save_json

NATIVE_PRIOR_PIXEL_ARCSEC = (
    0.2  # the Rubin pixel the host priors are trained on, unless the prior's config says otherwise
)


class Novamac:
    """Transient scene inference for one band. See the module docstring."""

    # Sampler settings (class-level defaults; any of them can be passed as a
    # keyword to the constructor)
    #####################################################################
    seed = 42
    num_chains = 4
    num_integration_steps = 8  # leapfrog steps per HMC transition on the scene

    # Prior noise level sigma(t) at which the schedule starts (the warmup
    # target). None means t = 1, i.e. sigma_max. Starting from full prior noise
    # does not work with a log-flux scene: once sigma_t >~ 2 the chain holds
    # pixels many e-folds from the data and the likelihood penalty on them grows
    # doubly exponentially as t decreases, the step size collapses and the chain
    # freezes. sigma ~ 1 (one e-fold of pixel noise) is broad but recoverable.
    sigma_start = 1.0
    warmup_steps = 256  # at t = t_start, HMC only
    anneal_steps = 4096  # t from t_start down to 0, predictor + corrector
    hold_steps = 512  # at t = 0: HMC at the true posterior; a drift across the saved states means the chain is still settling
    # The step sizes keep adapting for this many chunks of the hold (the cut at
    # the hold start is a guess) and are then frozen, chain restarts stop, and
    # the states saved from then on (trace "draws_from_step") are the posterior
    # draws: a state-dependent adaptation or restart during collection would
    # break the fixed-kernel stationarity the draws rely on.
    hold_adapt_chunks = 4
    # The anneal acceptance is inflated by the predictor (each corrector starts
    # from a slightly displaced state, so its proposals tend to go uphill), which
    # lets the step size grow past what plain HMC at t = 0 can use. The step size
    # is cut by this factor once when the hold starts, instead of waiting for the
    # adaptation to walk it down at step_down per chunk.
    hold_step_factor = 0.8
    # Likewise when a block is released: the step size the scene-only anneal has
    # grown to is far too large for the new, stiffer coordinates, and waiting for
    # the adaptation means chunks at zero acceptance. The (non-scene) step size
    # is cut by this factor per block released.
    release_step_factor = 0.9
    chunk_steps = 32  # step size adaptation period; also the jitted unit of work
    save_every = 128  # model state written to disk every save_every steps (multiple of chunk_steps)

    # Predictor on/off (corrector-only annealing when False: HMC alone tracks the
    # moving target), and its noise: fraction of the reverse-SDE noise (1 =
    # reverse SDE, 0 = probability-flow ODE); the drift is scaled accordingly.
    predictor = True
    predictor_noise = 0.0

    # Mass matrix: inv_mass(t) = inv_mass0 + mass_sigma_fraction * sigma(t)^2 on
    # the scene. 0.5 is the width the product of two factors each convolved to
    # width sigma_t has in the delta-function limit; the diagonal likelihood
    # broadening makes the actual target somewhat narrower, which only costs a
    # smaller step size while sigma_t is large.
    mass_sigma_fraction = 0.5
    gn_stride = 8  # scene probe stride for the setup Gauss-Newton estimate (stride^2 probes)

    # Diffusion time at which each parameter block is released (frozen at its
    # initial value while t is above this). The scene is always free. While the
    # scene is noisy the broadened likelihood says nothing about these blocks, so
    # sampling them early only lets them wander under their priors; the log flux
    # in particular would then face an exp-shaped cliff when the likelihood
    # sharpens. None = never frozen (free from the start).
    # The scene scale starts at the unit conversion (scene_scale0) and is
    # sampled with the other non-scene parameters under its soft log-uniform
    # prior: the model renders scale * I and the prior sees I, so a free scale
    # lets a global brightness offset between the data and the prior's
    # expectation be absorbed by one coordinate instead of by every scene
    # pixel. freeze_scene_scale keeps it at the unit conversion. It is always
    # frozen in the prior-only and frozen test modes, where nothing constrains it.
    freeze_scene_scale = True
    release_t = {
        "scene_scale": 0.4,  # the scene's overall scale factor: free from the start (if not frozen)
        "sky": 0.7,
        "alignment": 0.7,  # per-image crtan
        "ps_flux": 0.3,
        "ps_center": 0.3,
        "psf": 0.3,
    }
    # Blocks listed here are not released at release_t unconditionally: from
    # release_t on, their curvature is probed every release_probe_every chunks and
    # they are released once the median curvature-based variance (over the
    # coordinates the likelihood constrains) has dropped below release_mass_ratio
    # times the prior-width variance, i.e. once the data constrain the block
    # clearly better than the prior does. At release_latest_t they are released anyway.
    release_latest_t = {}
    release_mass_ratio = 0.25
    release_probe_every = 8  # chunks between curvature probes (each probe costs a few HVPs)
    # Once per chunk the sky levels and point source fluxes are redrawn from their
    # exact conditional given everything else (hmc.make_linear_gibbs_fn): the
    # model is linear in them, so this is a weighted least-squares solve per
    # image and an exact Metropolis-Hastings move. Only coordinates that have been
    # released take part. The proposal's mean and covariance (the likelihood's
    # conditional of the physical values, before the prior and the positivity
    # constraint) are stored per chunk.
    linear_gibbs = True
    # Blocked HMC: each step is predictor -> HMC on the scene -> HMC on the
    # released non-scene blocks (alignment, PSF, centres, fluxes, sky, scale)
    # with the scene held fixed, with its own step size, leapfrog count and
    # acceptance (hmc.make_chunk_runner, log_prior_nuisance). One shared step
    # size starved those blocks: the scene's coupled curvature pins it near
    # 1e-3, and with the Gauss-Newton mass every coordinate then moves ~eps * L
    # of its posterior width per transition.
    nuisance_hmc = True
    nuisance_integration_steps = 10
    # A chain whose step size has collapsed (an unlucky start, or a stiff spot
    # the adaptation walked down into) is dragged by the unadjusted predictor
    # while its corrector cannot correct, and it does not recover on its own.
    # A chain whose scene step size has been below restart_step_ratio x the
    # chain median for restart_chunks consecutive chunks is restarted from the
    # state and step sizes of the healthiest chain (the highest log density
    # among the others). Checked after every chunk from the second chunk on.
    # Two more stall signatures, same restart, same persistence:
    #   dead:    scene acceptance below restart_dead_acceptance (the step is far
    #            too large; the few moves that do get accepted are wild jumps)
    #   outlier: log density further from the median of the other chains than
    #            restart_outlier_factor x their spread (max - min, at least
    #            restart_outlier_min), in either direction: a chain in a basin
    #            of its own (needs three or more chains)
    # A restarted chain shares its past with its source, so the chains are no
    # longer independent; prefer that to a dead chain.
    restart_collapsed_chains = True
    restart_step_ratio = 0.1
    restart_chunks = 2
    restart_dead_acceptance = 0.05
    restart_outlier_factor = 5.0
    restart_outlier_min = 2000.0
    # The host prior's per-pixel energy level is subtracted inside its sum
    # (model.meta.prior_energy_offset, a function of sigma; see build_prior):
    # the raw total is ~1e7 at small sigma, where float32 resolves only ~1, and a
    # Metropolis test needs ~0.01. The level is calibrated once at setup on this
    # object's initial scene noised to prior_offset_sigmas values of sigma and
    # interpolated in log sigma. Besides making the test work, the logged log
    # prior then reads as "distance from a typical noised scene at this sigma".
    prior_energy_offset = True
    prior_offset_sigmas = 64
    # Evaluate the likelihood on the Tweedie estimate of the clean scene,
    # x + sigma_t^2 * prior score, instead of on the noisy x_t (hmc.Target). On the
    # noisy scene the likelihood squeezes the sigma_t noise out wherever the data
    # constrain it and leaves it in the PSF margin (the "box"); on the denoised
    # scene it is insensitive to the noise and the annealed states stay noised
    # versions of the t = 0 posterior. The broadening then uses the denoiser's
    # residual variance sigma^2 s^2 / (sigma^2 + s^2), s the prior's data scale.
    # Off at t = 0, so the hold samples the plain posterior. The denoiser's
    # Jacobian is taken as the identity in the force unless tweedie_exact_jacobian
    # (one extra Hessian-vector product of the prior per evaluation); either way
    # the Metropolis test is exact for the target.
    tweedie = True
    tweedie_exact_jacobian = False
    # Render the scene at x - v_t/2 inside the likelihood, v_t the residual
    # log-variance the broadening uses (hmc.Target): the data constrain the local
    # average of exp(x), which is exp(mu + v/2) for a field with log-mean mu and
    # scatter v, so without this the log-level is pushed down by v/2 wherever
    # the data have a say and not in the PSF margin (the dark box). Zero at t = 0.
    mean_correction = True
    # When the likelihood is switched on, like a parameter block's release time:
    #   None      from the first step (warmup included)
    #   "anneal"  off during the warmup, on from the first annealing step
    #   a float   on once t <= that value
    # Until then the target is the prior alone, no block is released and the
    # Gibbs refresh is skipped. When the likelihood comes on the step size is cut
    # by likelihood_step_factor and the adaptation uses the warmup's faster rates
    # for rewarm_chunks chunks. On real data a prior-only warmup collapsed the
    # chains: the prior carries the scene away from the data and the likelihood
    # is then too stiff. Keep None unless the start is known to be far from the prior.
    likelihood_t = None
    likelihood_step_factor = 0.5
    rewarm_chunks = 4  # also used after every block release, for the non-scene step

    initial_step_size = 0.1  # starting point of the step size search
    initial_step_clip = (
        2.0  # per-chain initial step sizes are clipped to within this factor of the chain median
    )
    target_acceptance = 0.6
    acceptance_tol = 0.05
    step_down, step_up = 0.7, 1.2  # during annealing / hold
    # Faster during warmup: the initial search happens far from the typical set
    # where gradients are large, so it starts tiny and must catch up quickly
    step_down_warmup, step_up_warmup = 0.5, 1.5

    hutchinson_probes = 256  # for diag(J J^T); relative error ~ sqrt(2 / probes)
    hutchinson_batch = 32

    # Test modes.
    # likelihood=False: the likelihood is replaced by 0, every non-scene block
    # stays frozen, and a reference draw from the prior's probability-flow
    # sampler is stored in the trace for comparison; the run then produces a
    # draw from the prior alone, started like a real run (t_start from
    # sigma_start, the initial scene plus noise) so that it isolates the prior
    # and the initialisation. prior_only_from_noise instead starts at t = 1
    # from a draw of the prior's t = 1 marginal, which tests the prior alone.
    likelihood = True
    pflow_reference_steps = 256
    prior_only_from_noise = False
    # freeze_non_scene: keep the likelihood but never release any non-scene
    # block, so only the scene pixels are sampled against prior and
    # likelihood; fluxes and the other parameters stay at their initial values.
    # Separates "the scene sampling is wrong" from "the block releases break it".
    freeze_non_scene = False
    # Posterior runs start the scene at the initial scene plus Gaussian noise of
    # sigma(t_start) in log I, i.e. a draw of the prior's t_start marginal
    # conditioned on that scene, rather than at the clean scene itself (which is
    # far from the warmup target). The Jacobian diagonal and the Gauss-Newton
    # mass are still evaluated at the clean scene.
    start_noise = True
    # |log I| beyond this overflows exp() in float32 (the limit is ~88). The
    # target is truncated to this box on the scene and the predictor is clipped
    # to it, which matters only near t = 1 in test mode (sigma_max ~ 27 puts the
    # box at ~3 sigma); posterior runs never come near it.
    max_abs_log_scene = 80.0
    stuck_acceptance = 0.1  # per-chunk acceptance below this is reported
    min_step_size = 1e-8  # step size below this is reported as collapsed

    # Initial scene (init_scene.py). Pixels below init_threshold_sigma x the
    # stack's own noise (measured on it, scaled per pixel by the coverage) are
    # blended toward zero flux, which the prior's transform puts on its sky
    # pedestal; None disables. init_sky_border: per-image residual sky
    # subtracted before stacking (and the default initial sky): median of this
    # many border pixels (0: none). The stack uses the templates if there are
    # at least init_min_templates, otherwise every image. init_subsample:
    # subpixels per scene pixel into which each data pixel is cut.
    init_threshold_sigma = 3.0
    init_sky_border = 4
    init_min_templates = 4
    init_subsample = 5
    # Aperture for the initial point-source fluxes when none are given:
    # radius in units of the Moffat FWHM (the enclosed fraction is corrected).
    init_aperture_fwhm = 1.5

    # Prior configuration (build_prior). prior_scene_factor: the host prior is
    # shown the scene this many times brighter than the likelihood renders it
    # (log I + log(factor) goes to the prior); 1 switches it off. prior_widths:
    # the Gaussian prior widths of the non-scene parameters; "wcs" None means
    # one data pixel.
    prior_scene_factor = 1.0
    prior_widths = {
        "wcs": None,  # arcsec; None = one data pixel
        "ps_position": 0.5,  # arcsec
        "logflux": 2.0,  # natural log of the point source flux
        "psf_n": 0.5,
        "psf_Rd": 0.2,  # relative
        "psf_q": 0.1,
        "psf_PA": 0.2,  # rad
        "scene_scale": 3.0,  # natural log; None for an improper flat prior
    }

    def __init__(
        self,
        data,
        *,
        is_template,
        crpix,
        crtan,
        crval,
        CD,
        zeropoint,
        ps_center,
        prior,
        psf_n=None,
        psf_Rd=None,
        psf_q=None,
        psf_PA=None,
        psf_images=None,
        weight=None,
        variance=None,
        mask=None,
        psf_shape=None,
        psf_upsample=1,
        ps_flux=None,
        sky=None,
        init_image=None,
        prior_pixel_scale=None,
        scene_pixel_scale=None,
        band="",
        out_dir=".",
        name="novamac",
        target_names=None,
        mjd=None,
        reference_flux=None,
        verbose=True,
        **settings,
    ):
        """
        Shapes: N images of H x W pixels, P point sources.

        Data (required):
          data (N, H, W)            cutouts, one per epoch, in the data's flux units
          weight or variance (N, H, W)  inverse variance / variance per pixel (one of them)
          mask (N, H, W)            True for pixels to ignore (default: none)
          is_template (N,)          True for epochs without the transient
          crpix (N, 2), crtan (N, 2), crval (2,), CD (N, 2, 2)
                                    WCS per image: reference pixel (col, row), its
                                    tangent-plane position (arcsec), the tangent
                                    point (world coords) and the pixel matrix
                                    (arcsec or degrees per pixel)
          psf_n, psf_Rd, psf_q, psf_PA (N,)  Moffat PSF per image (Rd in data pixels, PA in rad), or
          psf_images (N, h, w)      PSF images instead, fitted with the Moffat model (novamac.psf)
          psf_shape, psf_upsample   footprint of the rendered PSF in data pixels (default: the PSF
                                    images' size, else 15 x 15) and the oversampling of the PSF
                                    (images and model alike)
          zeropoint                 AB zero point of the data: 1 data unit = 10^((8.90 - zp) / 2.5) Jy
          ps_center (P, 2)          initial tangent-plane position of the point source(s), arcsec
          prior                     host prior checkpoint directory, or a loaded host_prior.Prior

        Optional initial values (estimated from the data when None):
          ps_flux (P, N)            initial point-source flux per image (aperture photometry)
          sky (N,)                  initial residual sky per image, flux per arcsec^2 (border median)
          init_image (Hs, Ws)       initial scene on the scene grid, data units per arcsec^2
                                    (drizzled stack of the templates)

        Geometry: the scene grid is the prior's training grid (the prior is
        size-locked), at scene_pixel_scale (default: the prior's pixel,
        prior_pixel_scale x the prior's pooling factor; prior_pixel_scale
        defaults to the prior config's pixel scale or 0.2 arcsec). The scene is
        centred on ps_center[0].

        Products and plots go to out_dir with the prefix `name`. target_names,
        mjd and reference_flux (e.g. a catalogue light curve, (N,)) are used
        for labels and plots only. verbose logs progress to stdout. Any other
        keyword must be a sampler setting (see the class attributes).
        """
        if verbose:
            enable_console_logging()
        unknown = set(settings) - set(self.setting_names())
        if unknown:
            raise TypeError(f"unknown settings {sorted(unknown)}; see Novamac.setting_names()")
        for k in ("release_t", "release_latest_t", "prior_widths"):  # never share a mutable default
            setattr(self, k, dict(getattr(self, k)))
        for k, v in settings.items():
            setattr(
                self, k, dict(getattr(self, k), **v) if isinstance(getattr(self, k), dict) else v
            )

        # Data
        self.data = np.asarray(data, dtype=float)
        if self.data.ndim != 3:
            raise ValueError(f"data must be (N, H, W), got {self.data.shape}")
        if (weight is None) == (variance is None):
            raise ValueError("give exactly one of weight and variance")
        if weight is None:
            variance = np.asarray(variance, dtype=float)
            weight = np.where(
                np.isfinite(variance) & (variance > 0),
                1.0 / np.where(variance > 0, variance, 1.0),
                0.0,
            )
        self.weight = np.asarray(weight, dtype=float)
        self.mask = (
            np.zeros(self.data.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        )
        self.is_template = np.asarray(is_template, dtype=bool)
        self.crpix = np.asarray(crpix, dtype=float)
        self.crtan = np.asarray(crtan, dtype=float)
        self.crval = np.asarray(crval, dtype=float)
        self.CD = np.asarray(CD, dtype=float)
        self.psf_upsample = int(psf_upsample)
        given = [a is not None for a in (psf_n, psf_Rd, psf_q, psf_PA)]
        if psf_images is not None:
            if any(given):
                raise ValueError(
                    "give either psf_images or the Moffat parameters psf_n, psf_Rd, psf_q, psf_PA, not both"
                )
            psf_images = [np.asarray(im, dtype=float) for im in psf_images]
            if len(psf_images) != self.data.shape[0]:
                raise ValueError("psf_images must have one image per epoch")
            fitted = fit_moffat_psfs(psf_images, upsample=self.psf_upsample)
            psf_n, psf_Rd, psf_q, psf_PA = (
                fitted[k] for k in ("psf_n", "psf_Rd", "psf_q", "psf_PA")
            )
            if psf_shape is None:
                h, w = psf_images[0].shape
                psf_shape = (h // self.psf_upsample, w // self.psf_upsample)
        elif not all(given):
            raise ValueError(
                "give the Moffat PSF parameters psf_n, psf_Rd, psf_q, psf_PA (all four) or psf_images"
            )
        self.psf_n, self.psf_Rd, self.psf_q, self.psf_PA = (
            np.asarray(a, dtype=float) for a in (psf_n, psf_Rd, psf_q, psf_PA)
        )
        self.psf_shape = (15, 15) if psf_shape is None else tuple(int(s) for s in psf_shape)
        self.zeropoint = float(zeropoint)
        self.ps_center = np.asarray(ps_center, dtype=float)
        if self.ps_center.shape == (2,):
            self.ps_center = self.ps_center[None, :]
        self.ps_flux = None if ps_flux is None else np.asarray(ps_flux, dtype=float)
        self.sky = None if sky is None else np.asarray(sky, dtype=float)
        self.init_image = None if init_image is None else np.asarray(init_image, dtype=float)
        self.band = str(band)
        self.name = str(name)
        self.files = RunFiles(out_dir, self.name)
        self.mjd = None if mjd is None else np.asarray(mjd, dtype=float)
        self.reference_flux = (
            None if reference_flux is None else np.asarray(reference_flux, dtype=float)
        )
        N = self.n_images
        self.target_names = (
            list(target_names)
            if target_names is not None
            else [f"target_t{int(t)}_{i}" for i, t in enumerate(self.is_template)]
        )
        for nm_, shape in (("weight", self.data.shape), ("mask", self.data.shape), ("is_template", (N,)), ("crpix", (N, 2)),
                           ("crtan", (N, 2)), ("crval", (2,)), ("CD", (N, 2, 2)), ("psf_n", (N,)), ("psf_Rd", (N,)),
                           ("psf_q", (N,)), ("psf_PA", (N,))):  # fmt: skip
            if getattr(self, nm_).shape != shape:
                raise ValueError(f"{nm_} must have shape {shape}, got {getattr(self, nm_).shape}")
        if self.ps_flux is not None and self.ps_flux.shape != (self.n_sources, N):
            raise ValueError(
                f"ps_flux must have shape {(self.n_sources, N)}, got {self.ps_flux.shape}"
            )
        if len(self.target_names) != N:
            raise ValueError("target_names must have one entry per image")
        unmasked = ~self.mask
        if not np.all(np.isfinite(self.data[unmasked])):
            raise ValueError("non-finite data in unmasked pixels")
        if not np.all(np.isfinite(self.weight[unmasked])) or np.any(self.weight[unmasked] < 0):
            raise ValueError("non-finite or negative weight in unmasked pixels")

        # Prior and units
        self.prior = prior if isinstance(prior, Prior) else load_prior(prior)
        cfg = self.prior.config
        if prior_pixel_scale is None:
            prior_pixel_scale = cfg.get(
                "pixel_scale", cfg.get("patch", {}).get("pixel_scale", NATIVE_PRIOR_PIXEL_ARCSEC)
            )
        self.prior_pixel_scale = float(prior_pixel_scale)  # arcsec per native training pixel
        self.scene_pixel_scale = float(
            self.prior_pixel_scale * self.prior.pool_factor
            if scene_pixel_scale is None
            else scene_pixel_scale
        )
        self.scene_shape = (int(self.prior.size), int(self.prior.size))
        self.data_pixel_scale = data_pixel_scale(self.CD)
        # The model renders scale * I with I in the prior's units (nJy per native
        # prior pixel; average pooling keeps the units) and scale * I in data
        # units per arcsec^2, so the scale is a fixed unit conversion,
        #     scale = (1 / prior pixel area) * (data units per nJy).
        self.njy_per_count = 10 ** ((8.90 - self.zeropoint) / 2.5) * 1e9
        self.scene_scale0 = 1.0 / (self.prior_pixel_scale**2 * self.njy_per_count)
        if self.init_image is not None and self.init_image.shape != self.scene_shape:
            raise ValueError(
                f"init_image must be on the scene grid {self.scene_shape} (the prior's), got {self.init_image.shape}"
            )

        self.likelihood_args = None  # built by prepare()
        self.init_info = None
        self.model = None  # built by build_model()
        self.trace = None  # the last sample() call's trace

    # Settings
    #####################################################################
    @classmethod
    def setting_names(cls):
        """Names of every sampler setting (the class attributes)."""
        return sorted(
            k for k, v in vars(Novamac).items()
            if not k.startswith("_") and not callable(v) and not isinstance(v, (property, classmethod, staticmethod))
        )  # fmt: skip

    def settings(self):
        """The settings in force for this instance."""
        return {
            k: (dict(v) if isinstance(v, dict) else v)
            for k, v in ((k, getattr(self, k)) for k in self.setting_names())
        }

    @property
    def n_images(self):
        return self.data.shape[0]

    @property
    def n_sources(self):
        return self.ps_center.shape[0]

    # Preparation
    #####################################################################
    def prepare(self):
        """Initial sky, initial scene and the likelihood arguments (the
        initial fluxes are estimated in build_model, since they need the
        rendered scene). Idempotent."""
        if self.likelihood_args is not None:
            return self
        border = (
            border_sky(self.data, self.mask, self.init_sky_border) if self.init_sky_border else None
        )
        if self.sky is None:
            # the model's sky is a surface brightness (flux per arcsec^2)
            area = np.abs(np.linalg.det(cd_arcsec(self.CD)))
            self.sky = np.zeros(self.n_images) if border is None else border / area
        scene, info = prepare_initial_scene(
            self.data, self.weight, self.mask, self.is_template, self.crpix, self.crtan, self.CD, self.scene_shape,
            self.scene_pixel_scale, self.ps_center[0], scene_scale=self.scene_scale0, softening=self.prior.transform.softening,
            prior_scene_factor=self.prior_scene_factor, init_image=self.init_image, sky=border,
            threshold_sigma=self.init_threshold_sigma, min_templates=self.init_min_templates, subsample=self.init_subsample,
        )  # fmt: skip
        self.init_info = info
        self.likelihood_args = {
            "data": self.data, "var": None, "weight": self.weight, "mask": self.mask, "band": self.band,
            "is_template": self.is_template, "crpix": self.crpix, "crtan": self.crtan, "crval": self.crval, "CD": self.CD,
            "psf_n": self.psf_n, "psf_Rd": self.psf_Rd, "psf_q": self.psf_q, "psf_PA": self.psf_PA,
            "psf_upsample": self.psf_upsample, "Hp": self.psf_shape[0], "Wp": self.psf_shape[1],
            "ps_center0": self.ps_center, "ps_flux0": None if self.ps_flux is None else np.clip(np.abs(self.ps_flux), 1e-30, None),
            "sky_I0": self.sky, "pix_scale": self.scene_pixel_scale, "pix_init": scene, "scene_scale0": self.scene_scale0,
            "zp": self.zeropoint, "target_names": self.target_names,
        }  # fmt: skip
        return self

    def build_model(self):
        """The AstroPhot model (likelihood + prior), built once and cached.
        Every plot and diagnostic uses the same build, so the parameter
        layout matches the chain files."""
        if self.model is not None:
            return self.model
        self.prepare()
        la = dict(self.likelihood_args)
        estimate = la["ps_flux0"] is None
        if estimate:
            la["ps_flux0"] = np.ones((self.n_sources, self.n_images))
        model = build_likelihood(**la)
        if estimate:
            self.ps_flux = self._estimate_fluxes(model)
            self.likelihood_args["ps_flux0"] = self.ps_flux
            for p in range(self.n_sources):
                model.models[f"ps_batch_model_{p}"].model.flux.allflux.value = self.ps_flux[p]
            model.meta.ps_flux0 = self.ps_flux
        widths = dict(self.prior_widths)
        if widths.get("wcs") is None:
            widths["wcs"] = self.data_pixel_scale
        self.model = build_prior(
            model, self.prior, prior_scene_factor=self.prior_scene_factor,
            **{f"{k}_prior_width": v for k, v in widths.items()},
        )  # fmt: skip
        return self.model

    def _estimate_fluxes(self, model):
        """Aperture photometry of the data minus the rendered initial scene
        and sky, per source and image; epochs without a detection start at
        their aperture noise (positive, and the centre of the log-flux prior).
        The scene rendered here is the raw stack (before the soft threshold
        and the prior's transform, which both move faint pixels), so the
        residual is unbiased where the host is faint."""
        pix = model.models.pixelated_batch_model.model
        prepared = np.array(pix.I.npvalue)
        raw = np.asarray(self.init_info["raw"], dtype=float) / self.scene_scale0
        pix.I.value = np.where(raw > 0, raw, 1e-6 * max(float(raw.max()), 1e-30))
        for p in range(self.n_sources):
            model.models[f"ps_batch_model_{p}"].model.flux.allflux.value = np.full(
                self.n_images, 1e-6
            )
        rendered = np.asarray(model().flatten("data")).reshape(self.data.shape)
        pix.I.value = prepared
        residual = self.data - rendered
        fluxes = np.ones((self.n_sources, self.n_images))
        for p in range(self.n_sources):
            flux, err = aperture_fluxes(residual, self.weight, self.mask, self.crpix, self.crtan, self.CD, self.ps_center[p],
                                        self.psf_n, self.psf_Rd, self.init_aperture_fwhm)  # fmt: skip
            est = np.maximum(flux, err)
            est = np.where(np.isfinite(est) & (est > 0), est, 1.0)
            fluxes[p] = np.where(self.is_template, 1.0, est)
            sn = ~self.is_template
            log(
                f"initial flux of source {p} from aperture photometry ({self.init_aperture_fwhm:g} FWHM): "
                f"{np.array2string(flux[sn], precision=3)} +- {np.array2string(err[sn], precision=3)} on the {int(sn.sum())} transient epochs"
            )
        return fluxes

    def geometry_report(self):
        """Lines describing the pixel geometry: data cutouts, PSF footprint,
        scene grid and the prior's grid, plus the largest cutout the scene can
        model without PSF spill from outside its footprint."""
        self.prepare()
        N, H, W = self.data.shape
        Hp, Wp, up = self.psf_shape[0], self.psf_shape[1], self.psf_upsample
        Hpix, Wpix = self.scene_shape
        pix_scale = self.scene_pixel_scale
        ratio = self.data_pixel_scale / pix_scale  # scene pixels per data pixel (the upsampling)
        footprint = Hpix / ratio  # scene footprint in data pixels
        prior = self.prior
        psf_px = max(Hp, Wp)
        max_cutout = footprint - psf_px
        needed_scene = (H + psf_px) * ratio
        prior_pix = self.prior_pixel_scale * prior.pool_factor
        return [
            f"geometry: {N} cutouts of {H}x{W} data px at {self.data_pixel_scale:.4f} arcsec/px",
            f"  PSF: {Hp}x{Wp} data px, rendered at {up}x ({Hp * up}x{Wp * up} px)",
            f"  scene: {Hpix}x{Wpix} scene px at {pix_scale:.4f} arcsec/px = {ratio:.2f}x upsampling, "
            f"footprint {footprint:.1f} data px ({Hpix * pix_scale:.1f} arcsec)"
            f"; scale = {self.scene_scale0:.4g} (ZP {self.zeropoint:g}: {self.njy_per_count:.4g} nJy/count, "
            f"prior pixel {self.prior_pixel_scale**2:g} arcsec^2)"
            + (f"; the prior sees the scene x{self.prior_scene_factor:g} brighter (prior_scene_factor)" if self.prior_scene_factor != 1 else ""),
            f"  prior: trained on a {prior.size}x{prior.size} grid (pool {prior.pool_factor}), sigma in "
            f"[{prior.sigma_min:.3g}, {prior.sigma_max:.3g}]; one prior pixel = {prior_pix:.3f} arcsec vs the "
            f"{pix_scale:.4f} arcsec scene pixel (ratio {prior_pix / pix_scale:.2f}"
            + (": the prior sees structures that many times larger than it was trained on)" if abs(prior_pix / pix_scale - 1) > 1e-3 else ")"),
            f"  coverage: the convolution needs scene beyond the data by half the PSF on every side. "
            f"This cutout + PSF = {H + psf_px} data px = {needed_scene:.1f} scene px (scene has {Hpix}); "
            f"the largest cutout this scene models fully is footprint - PSF = {max_cutout:.1f} data px"
            + (f" and the {H}-px cutout exceeds it by {H - max_cutout:.1f} px (outer ring sees PSF spill from outside the scene)"
               if H > max_cutout else f", so the {H}-px cutout fits"),
        ]  # fmt: skip

    # Schedule helpers
    #####################################################################
    def _likelihood_on(self, step, t):
        """Whether the likelihood is part of the target for the chunk starting at (step, t)."""
        if not self.likelihood:
            return False
        if self.likelihood_t is None:
            return True
        if self.likelihood_t == "anneal":
            return step >= self.warmup_steps
        return t <= float(self.likelihood_t)

    def phase_of(self, step):
        if step <= self.warmup_steps:
            return "warmup"
        if step <= self.warmup_steps + self.anneal_steps:
            return "anneal"
        return "hold"

    # Sampling
    #####################################################################
    def sample(self, seed=None):
        """Run the annealed sampler and write the products (products.py).
        Returns the trace dict (also kept as self.trace)."""
        if self.save_every % self.chunk_steps != 0:
            raise ValueError("save_every must be a multiple of chunk_steps")
        C = self.num_chains
        timings = {}
        rng = jax.random.key(self.seed if seed is None else seed)
        files = self.files
        files.mkdir()

        with Timer("build model", timings):
            model = self.build_model()
        likelihood_args = self.likelihood_args
        log(
            f"{self.name}: {self.n_images} cutouts ({int(self.is_template.sum())} templates), {self.n_sources} point source(s)"
        )
        for line in self.geometry_report():
            log(line)

        # Parameter layout: group 0 is sampled (unconstrained inside ValidContext),
        # group 1 is the template indicator, dynamic for vmapping but never sampled.
        with ck.ValidContext(model):
            x0, group1 = model.get_values()
        x0 = jnp.asarray(x0, dtype=DTYPE)
        group1 = jnp.asarray(group1, dtype=DTYPE)
        as_dev = lambda a: jnp.asarray(np.asarray(a), dtype=DTYPE)  # noqa: E731
        dim = x0.shape[0]
        log_lik, log_prior, images = make_log_parts(model, group1)
        log_prior_nuisance = make_nuisance_log_prior(model, group1) if self.nuisance_hmc else None
        if self.nuisance_hmc and log_prior_nuisance is None:
            log(
                "WARNING: the model has no log_prior_nuisance; non-scene blocks move in the scene HMC with the shared step size"
            )
        no_likelihood = not self.likelihood
        if no_likelihood:
            log(
                "TEST MODE: likelihood switched off, sampling the prior alone; non-scene blocks stay frozen"
                + (
                    "; starting from the t=1 marginal"
                    if self.prior_only_from_noise
                    else "; starting like a real run (initial scene + noise at t_start)"
                )
            )
            log_lik = jax.jit(lambda x, extra_var: jnp.zeros((), dtype=x.dtype))
        sde = model.meta.host_prior.sde
        sigma_fn = make_sigma_fn(sde)
        tweedie = self.tweedie and not no_likelihood
        data_std = self.prior.config["sde"].get("data_std")
        if tweedie and data_std is None:
            raise ValueError("tweedie needs the prior's data_std (sde.data_std in its config.json)")
        broaden_fn = make_broadening_fn(
            sde, float(data_std) if tweedie else None
        )  # extra variance = broaden_fn(t) * diag(JJ^T)
        log_var_fn = (
            make_log_variance_fn(sde, float(data_std) if tweedie else None)
            if (self.mean_correction and not no_likelihood)
            else None
        )
        from_noise = no_likelihood and self.prior_only_from_noise
        t_start = 1.0 if from_noise else start_time(sde, self.sigma_start)
        log(
            f"schedule starts at t={t_start:.3f} (sigma={float(sigma_fn(t_start)):.3g}; sigma range [{sde.sigma_min:.3g}, {sde.sigma_max:.3g}])"
        )

        pix = model.models.pixelated_batch_model.model
        scene_idx = block_indices(model, [pix.I])
        scene_dev_idx = jnp.asarray(scene_idx)
        prior_offset_table = None
        if self.prior_energy_offset:
            rng, key_off = jax.random.split(rng)
            with Timer("prior energy offset calibration", timings):
                scene_log = np.asarray(x0)[scene_idx].reshape(
                    pix.I.shape
                )  # log I of the initial scene
                prior_offset_table = calibrate_prior_offset(
                    model, scene_log, key_off, self.prior_offset_sigmas
                )
            sg, val = prior_offset_table
            log(
                f"prior energy offset per pixel: {val[0]:.4g} at sigma={sg[0]:.3g} ... {val[-1]:.4g} at sigma={sg[-1]:.3g} "
                f"(x{len(scene_idx)} pixels; subtracted inside the prior's sum so the logged log prior is relative to a typical noised scene)"
            )
        scene_mask = np.zeros(dim)
        scene_mask[scene_idx] = 1.0
        target = make_target(
            log_lik,
            log_prior,
            sigma_fn,
            as_dev(scene_mask),
            tweedie,
            self.tweedie_exact_jacobian,
            log_var_fn,
        )
        if log_var_fn is not None:
            log(
                f"mean correction: the likelihood renders the scene at x - v_t/2 (v_{{t_start}}/2 = {0.5 * float(log_var_fn(t_start)):.3g} in log I)"
            )
        # the prior-only target used while the likelihood is off (no likelihood cost at all)
        zero_lik = jax.jit(lambda x, extra_var: jnp.zeros((), dtype=x.dtype))
        target_off = make_target(zero_lik, log_prior, sigma_fn, as_dev(scene_mask), False, False)
        if not no_likelihood and self.likelihood_t is not None:
            log(
                f"likelihood switched on {'from the first annealing step' if self.likelihood_t == 'anneal' else f'once t <= {float(self.likelihood_t):g}'}; "
                f"prior alone before that (step size x{self.likelihood_step_factor:g} at the switch, then {self.rewarm_chunks} chunks of warmup-rate adaptation)"
            )
        if tweedie:
            log(
                f"Tweedie: likelihood evaluated at x + sigma_t^2 * prior score on the scene for t > 0 (plain at t = 0); "
                f"broadening uses sigma^2 s^2 / (sigma^2 + s^2) with s = {float(data_std):.3g}"
                + (
                    "; exact denoiser Jacobian in the force"
                    if self.tweedie_exact_jacobian
                    else "; identity Jacobian in the force"
                )
            )
        pflow_reference = None
        for line in self.init_info["report"]:
            log(line)
        if from_noise:
            # start the scene from the t = 1 marginal: x ~ N(data_mean, sigma_max^2)
            # in the prior's coordinate, which for the sampler (log I) is the same
            # thing since I = exp(x) is the prior's to_flux
            rng, key_start = jax.random.split(rng)
            prior = model.meta.host_prior
            noise = prior.sde.data_mean + prior.sde.sigma_max * jax.random.normal(
                key_start, (len(scene_idx),), DTYPE
            )
            noise = noise - float(
                np.log(getattr(model.meta, "prior_scene_factor", 1.0))
            )  # prior units -> scene units
            n_clipped = int(jnp.sum(jnp.abs(noise) > self.max_abs_log_scene))
            noise = jnp.clip(noise, -self.max_abs_log_scene, self.max_abs_log_scene)
            x0 = x0.at[jnp.asarray(scene_idx)].set(noise)
            if n_clipped:
                log(
                    f"  {n_clipped} start pixels clipped to |log I| <= {self.max_abs_log_scene:g} (float32 exp range)"
                )
            log(f"scene started from the t=1 marginal (sigma_max={prior.sde.sigma_max:.3g})")
        if no_likelihood:
            rng, key_ref = jax.random.split(rng)
            prior = model.meta.host_prior
            with Timer("pflow reference sample (includes compile)", timings):
                ref = prior.sample(key_ref, n=C, n_steps=self.pflow_reference_steps)
                # (chains, H, W) in the scene's units (the prior's flux divided by the factor it is shown)
                pflow_reference = np.asarray(prior.to_flux(ref))[:, 0] / float(
                    getattr(model.meta, "prior_scene_factor", 1.0)
                )
            log(f"pflow reference drawn with {self.pflow_reference_steps} Heun steps")
        blocks = parameter_blocks(model, self.release_t, self.release_latest_t)
        # per-coordinate release time handed to the runner: +inf = free, -inf =
        # frozen; a block's entries are set when it is actually released
        t_release = np.full(dim, np.inf)
        if self.freeze_non_scene and not no_likelihood:
            log("FROZEN MODE: likelihood kept, non-scene blocks never released (scene pixels only)")
        for b in blocks:
            if (no_likelihood or self.freeze_non_scene) and b.name != "scene_scale":
                b.freeze()  # never released
            if b.name == "scene_scale" and (
                self.freeze_scene_scale or no_likelihood or self.freeze_non_scene
            ):
                b.freeze()  # fixed unit conversion (nothing constrains it without the likelihood)
            t_release[b.idx] = np.inf if b.t_earliest >= t_start else -np.inf
        log(f"{dim} parameters: {len(scene_idx)} scene pixels")
        for b in blocks:
            when = f"t <= {b.t_earliest:.3g} (sigma {float(sigma_fn(min(b.t_earliest, 1.0))):.3g})"
            if b.t_latest < b.t_earliest:
                when += f" if the curvature criterion holds, by t <= {b.t_latest:.3g} regardless"
            log(f"  {b.name:11s} {len(b.idx):5d} coords, released at {when}")

        # Jacobian diagonal for the likelihood broadening, once, at the initial scene
        # ----------------------------------------------------------------
        rng, key = jax.random.split(rng)
        if no_likelihood:
            jj = jnp.zeros(int(np.prod(self.data.shape)), dtype=DTYPE)
        else:
            with Timer("Jacobian diagonal (Hutchinson, includes compile)", timings):
                jj = hutchinson_jj_diag(
                    images, x0, scene_idx, key, self.hutchinson_probes, self.hutchinson_batch
                )
                jax.block_until_ready(jj)
        jj_np = np.asarray(jj)
        if not np.all(np.isfinite(jj_np)):
            raise RuntimeError(f"{self.name}: non-finite Jacobian diagonal")
        data_var = 1.0 / np.where(self.weight > 0, self.weight, np.inf).ravel()
        ratio_start = float(broaden_fn(t_start)) * jj_np / data_var
        log(
            f"diag(JJ^T): median {np.median(jj_np):.3g}, max {np.max(jj_np):.3g}; "
            f"extra/data variance at the start: median {np.median(ratio_start):.3g}, max {np.max(ratio_start):.3g}"
        )

        # Mass matrix, once: Gauss-Newton t = 0 variance estimate
        # ----------------------------------------------------------------
        with Timer("Gauss-Newton mass matrix (includes compile)", timings):
            inv_mass0, reference_var, flat, gn_curv = setup_inverse_mass(
                model,
                images,
                x0,
                scene_idx,
                blocks,
                self.weight,
                self.mask,
                self.gn_stride,
                likelihood=not no_likelihood,
            )
        gibbs_fn, lin = None, None
        if self.linear_gibbs and not no_likelihood:
            lin = linear_coordinates(model, flat)
            data_flat, weight_flat = flat_data_weight(model)
            gibbs_fn = make_linear_gibbs_fn(
                images,
                as_dev(data_flat),
                as_dev(weight_flat),
                broaden_fn,
                lin,
                denoise=target.denoise,
            )
            n_sky, n_flux = int(np.sum(lin["source"] < 0)), int(np.sum(lin["source"] >= 0))
            log(
                f"linear Gibbs refresh once per chunk on {n_sky} sky levels + {n_flux} point source fluxes (released coordinates only)"
            )
        if np.any(flat):
            # no likelihood and no prior curvature (template-image fluxes): frozen for good
            t_release[flat] = -np.inf
            log(
                f"{int(np.sum(flat))} coordinates have neither likelihood nor prior curvature and stay frozen"
            )
        log(
            f"inv_mass0 (t=0 variance estimate): scene median {np.median(inv_mass0[scene_idx]):.3g} "
            f"(prior sigma_min^2 = {float(sde.sigma_min) ** 2:.3g}); mass at the start adds "
            f"{self.mass_sigma_fraction} * sigma^2 = {self.mass_sigma_fraction * float(sigma_fn(t_start)) ** 2:.3g} on the scene"
        )
        for b in blocks:
            log(
                f"  {b.name:11s} median inv_mass0 {np.median(inv_mass0[b.idx]):.3g}, prior-width variance {np.median(reference_var[b.idx]):.3g}"
            )

        # Sanity check the starting point at both ends of the schedule
        # ----------------------------------------------------------------
        with Timer("log density at x0 (includes compile)", timings):
            checks = {}
            for t in (t_start, 0.0):
                ll = float(log_lik(x0, broaden_fn(t) * jj))
                lp = float(log_prior(x0, t))
                checks[t] = (ll, lp)
                log(f"  t={t:.3g}: log likelihood {ll:.6g}, log prior {lp:.6g}")
        if not all(np.isfinite(v) for pair in checks.values() for v in pair):
            raise RuntimeError(f"{self.name}: non-finite log density at the starting point")

        # Prepare outputs (clean start, stale chain files removed)
        # ----------------------------------------------------------------
        chain_paths = [files.chain(c) for c in range(C)]
        for p in chain_paths:
            p.unlink(missing_ok=True)
        tpath = files.trace

        ts = build_schedule(t_start, self.warmup_steps, self.anneal_steps, self.hold_steps)
        ts_prev = np.concatenate([[ts[0]], ts[:-1]])  # t of the step before each step
        total = len(ts)
        chunk_bounds = [
            (s, min(s + self.chunk_steps, total)) for s in range(0, total, self.chunk_steps)
        ]
        n_chunks = len(chunk_bounds)
        # from this step on every kernel is fixed (no adaptation, no restarts): the draws
        hold_start = self.warmup_steps + self.anneal_steps
        draws_from = (
            min(total, hold_start + self.hold_adapt_chunks * self.chunk_steps)
            if self.hold_steps > 0
            else total
        )
        trace = {
            # settings
            "name": self.name,
            "settings": json.dumps(self.settings(), default=str),
            "num_chains": C,
            "warmup_steps": self.warmup_steps,
            "anneal_steps": self.anneal_steps,
            "hold_steps": self.hold_steps,
            "chunk_steps": self.chunk_steps,
            "t_start": t_start,
            "mass_sigma_fraction": self.mass_sigma_fraction,
            "predictor_noise": self.predictor_noise,
            "predictor": self.predictor,
            "tweedie": tweedie,
            **({"prior_offset_sigma": prior_offset_table[0], "prior_offset_value": prior_offset_table[1]} if prior_offset_table is not None else {}),
            "mean_correction": log_var_fn is not None,
            "likelihood_t": str(self.likelihood_t),
            "likelihood_on_step": 0 if self._likelihood_on(0, t_start) else -1,  # set when the switch happens
            "draws_from_step": draws_from,
            "score_fields": np.array(SCORE_FIELDS),
            "scene_idx": scene_idx,
            "scene_shape": np.array(pix.I.shape),
            **{f"block_idx_{b.name}": b.idx for b in blocks},
            "inv_mass0": inv_mass0.astype(np.float32),
            **({"pflow_reference": pflow_reference.astype(np.float32)} if pflow_reference is not None else {}),
            "gauss_newton_curvature": gn_curv.astype(np.float32),
            "jj_diag": jj_np.astype(np.float32),
            # per step
            "t": ts,
            "logdensity": np.full((total, C), np.nan),
            "loglik": np.full((total, C), np.nan),
            "logprior": np.full((total, C), np.nan),
            "accept_prob": np.full((total, C), np.nan),
            "log_alpha": np.full((total, C), np.nan),  # log Metropolis ratio (minus the energy error)
            "divergent": np.zeros((total, C), dtype=bool),
            "predictor_rms": np.full((total, C), np.nan),  # rms scene displacement by the predictor
            # non-scene HMC sub-step (nan while it does not exist)
            "accept_prob_nuis": np.full((total, C), np.nan),
            "log_alpha_nuis": np.full((total, C), np.nan),
            "divergent_nuis": np.zeros((total, C), dtype=bool),
            "steps_done": 0,
            # per chunk (values in force *during* that chunk; scores at its end)
            "chunk_step": np.array([e for _, e in chunk_bounds]),
            "chunk_t": ts[[e - 1 for _, e in chunk_bounds]],
            "step_size": np.full((n_chunks, C), np.nan),
            "step_size_nuis": np.full((n_chunks, C), np.nan),
            "scores": np.full((n_chunks, C, len(SCORE_FIELDS)), np.nan),
            "chunk_logdensity": np.full((n_chunks, C), np.nan),  # of the state at the end of the chunk, after the Gibbs refresh
            # linear Gibbs refresh at the end of each chunk: acceptance, and the
            # proposal's mean / covariance of the physical sky + flux values (the
            # likelihood's conditional, before the prior and the positivity constraint)
            **({
                "lin_idx": lin["idx"], "lin_image": lin["image"], "lin_source": lin["source"], "lin_is_log": lin["is_log"],
                "gibbs_accept": np.full((n_chunks, C), np.nan),
                "cond_mean": np.full((n_chunks, C, len(lin["idx"])), np.nan, dtype=np.float32),
                "cond_cov": np.full((n_chunks, C, len(lin["idx"]), len(lin["idx"])), np.nan, dtype=np.float32),
            } if lin is not None else {}),
            # which step / t each saved chain state corresponds to
            "saved_step": [],
            "saved_t": [],
            # block releases and curvature probes
            "block_names": np.array([b.name for b in blocks]),
            "block_t_earliest": np.array([b.t_earliest for b in blocks]),
            "block_t_latest": np.array([b.t_latest for b in blocks]),
            "block_release_step": np.full(len(blocks), -1),
            "block_release_t": np.full(len(blocks), np.nan),
            "probe_block": [],
            "probe_step": [],
            "probe_ratio": [],
            # chain restarts: (step, chain restarted, chain it was copied from, reason)
            "restart_step": [],
            "restart_chain": [],
            "restart_source": [],
            "restart_reason": [],
        }  # fmt: skip
        self.trace = trace

        # Sampler state
        # ----------------------------------------------------------------
        positions = jnp.tile(x0, (C, 1))
        if self.start_noise and not from_noise:
            # noise the clean initial scene up to the warmup target's level, independently per chain
            rng, key_start = jax.random.split(rng)
            s0 = float(sigma_fn(t_start))
            noised = positions[:, scene_dev_idx] + s0 * jax.random.normal(
                key_start, (C, len(scene_idx)), DTYPE
            )
            positions = positions.at[:, scene_dev_idx].set(
                jnp.clip(noised, -self.max_abs_log_scene, self.max_abs_log_scene)
            )
            ll = float(log_lik(positions[0], broaden_fn(t_start) * jj))
            lp = float(log_prior(positions[0], t_start))
            log(
                f"scene start = initial scene + N(0, {s0:.3g}^2) in log I per chain; chain 0 at t={t_start:.3g}: log likelihood {ll:.6g}, log prior {lp:.6g}"
            )
            if not (np.isfinite(ll) and np.isfinite(lp)):
                raise RuntimeError(
                    f"{self.name}: non-finite log density at the noised starting point"
                )
        jj_dev = as_dev(jj)
        inv_mass0_dev = as_dev(inv_mass0)
        scene_mask_dev = as_dev(scene_mask)
        inv_mass_at = (
            lambda t: inv_mass0 + self.mass_sigma_fraction * float(sigma_fn(t)) ** 2 * scene_mask
        )  # noqa: E731
        runners = {}
        make_runner = lambda n, on, nuis: make_chunk_runner(  # noqa: E731
            target if on else target_off, broaden_fn, sigma_fn, self.num_integration_steps, n, self.mass_sigma_fraction,
            self.predictor_noise, max_abs_scene=self.max_abs_log_scene, predictor=self.predictor,
            nuisance_log_prior=log_prior_nuisance if nuis else None, nuisance_steps=self.nuisance_integration_steps,
        )  # fmt: skip
        lik_on = self._likelihood_on(0, t_start)
        nuis_on = False  # the non-scene sub-step exists once a non-scene block has been released
        step_size_nuis = np.full(C, self.initial_step_size)
        rewarm_until = (
            -1
        )  # chunk index until which the adaptation uses the warmup rates (set at the likelihood switch)
        rewarm_nuis_until = -1  # same for the non-scene step (set when it first exists)
        stall_chunks = {
            k: np.zeros(C, dtype=int) for k in ("collapsed", "dead", "outlier")
        }  # consecutive chunks flagged, per chain
        score_fns = {
            True: make_score_fn(target, broaden_fn, scene_idx),
            False: make_score_fn(target_off, broaden_fn, scene_idx),
        }
        curvature_fns = {
            b.name: make_block_curvature_fn(target, broaden_fn, b.groups) for b in blocks
        }
        # blocks that are free from the start count as released at step 0: they
        # go to the non-scene HMC sub-step from the beginning
        for bi, b in enumerate(blocks):
            if b.t_earliest >= t_start:
                b.released_chunk = -1
                trace["block_release_step"][bi] = 0
                trace["block_release_t"][bi] = t_start
                log(
                    f"{b.name} is free from the start (t_earliest {b.t_earliest:.3g} >= t_start {t_start:.3g})"
                )
                if log_prior_nuisance is not None and lik_on:
                    nuis_on = True

        with Timer("initial step size search (includes compile)", timings):
            runners[(1, lik_on, nuis_on)] = make_runner(1, lik_on, nuis_on)
            rng, step_size, step_size_nuis = find_initial_step_size(
                runners[(1, lik_on, nuis_on)],
                rng,
                positions,
                inv_mass0,
                t_start,
                jj_dev,
                t_release,
                scene_mask,
                as_dev,
                initial_step_size=self.initial_step_size,
                tune_nuis=nuis_on,
            )
        if C > 1:
            # the chains start statistically alike, so one chain's search landing
            # far from the others is a fluke of the single-step test (a first move
            # from a bad spot is uphill and accepted at any step); the first chunk
            # at a step too large by 10x makes one wild accepted jump that the
            # chain never recovers from. Clip to a factor initial_step_clip of the median.
            for arr in (step_size, step_size_nuis) if nuis_on else (step_size,):
                med = float(np.median(arr))
                np.clip(arr, med / self.initial_step_clip, med * self.initial_step_clip, out=arr)
        log(
            f"initial step sizes {np.array2string(step_size, precision=3)}"
            + (f", non-scene {np.array2string(step_size_nuis, precision=3)}" if nuis_on else "")
        )

        # Tweedie estimate of the scene at every saved state, for the plots: the
        # score is one prior gradient per chain, and saving it here is far cheaper
        # than rebuilding the model to recompute it later.
        tweedie_fn = jax.jit(jax.vmap(
            lambda x, t: jnp.exp(x[scene_dev_idx] + sigma_fn(t) ** 2 * target.prior_grad(x, t)[scene_dev_idx]).reshape(pix.I.shape),
            in_axes=(0, None),
        ))  # fmt: skip
        scene_hat = []

        def save_tweedie(t):
            scene_hat.append(np.asarray(tweedie_fn(positions, float(t)), dtype=np.float32))
            np.savez(files.tweedie, scene_hat=np.stack(scene_hat), saved_step=np.array(trace["saved_step"]),
                     saved_t=np.array(trace["saved_t"]))  # fmt: skip

        save_chain_states(model, positions, group1, chain_paths)
        trace["saved_step"].append(0)
        trace["saved_t"].append(ts[0])
        write_trace(tpath, trace)
        save_tweedie(ts[0])
        log(
            f"schedule: {self.warmup_steps} warmup + {self.anneal_steps} anneal + {self.hold_steps} hold = {total} steps in {n_chunks} chunks"
        )

        # Chunk loop
        # ----------------------------------------------------------------
        for ci, (s0, s1) in enumerate(chunk_bounds):
            n = s1 - s0
            on_now = self._likelihood_on(s0, float(ts[s0]))
            if on_now and not lik_on:
                trace["likelihood_on_step"] = s0
                step_size = step_size * self.likelihood_step_factor
                rewarm_until = ci + self.rewarm_chunks
                log(
                    f"likelihood on at step {s0} (t={ts[s0]:.3f}); step size x{self.likelihood_step_factor:g} -> "
                    f"{np.array2string(np.asarray(step_size), precision=3)}, warmup-rate adaptation for {self.rewarm_chunks} chunks"
                )
                ev_on = broaden_fn(float(ts[s0])) * jj_dev
                ll_on = np.array(
                    [
                        float(log_lik(target.denoise(positions[c], float(ts[s0])), ev_on))
                        for c in range(C)
                    ]
                )
                log(
                    f"  log likelihood of the prior-settled states: {np.array2string(ll_on, precision=4)}"
                )
                if not np.all(np.isfinite(ll_on)):
                    log(
                        "  WARNING: non-finite likelihood at the switch; the prior-only phase carried the scene out of the "
                        "likelihood's range (float32 exp overflow) and the chain cannot recover from here"
                    )
            lik_on = on_now
            if s0 == self.warmup_steps + self.anneal_steps and self.hold_steps > 0:
                step_size = step_size * self.hold_step_factor
                step_size_nuis = step_size_nuis * self.hold_step_factor
                log(
                    f"hold starts: step size x{self.hold_step_factor:g} -> {np.array2string(np.asarray(step_size), precision=3)}"
                    + (
                        f", non-scene {np.array2string(np.asarray(step_size_nuis), precision=3)}"
                        if nuis_on
                        else ""
                    )
                )

            # Block releases. Once the schedule reaches a block's earliest time its
            # curvature is probed (every release_probe_every chunks); it is
            # released when the curvature-based variance is below
            # release_mass_ratio x the prior-width variance, or when the latest
            # time is reached. The mass itself is fixed (inv_mass0).
            t_chunk = float(ts[s1 - 1])
            for bi, b in enumerate(blocks):
                if b.released_chunk is not None or t_chunk > b.t_earliest or not lik_on:
                    continue  # (nothing is released while the likelihood is off)
                forced = t_chunk <= b.t_latest
                due = (
                    b.last_probe_chunk is None
                    or ci - b.last_probe_chunk >= self.release_probe_every
                )
                if not (forced or due):
                    continue
                with Timer(f"release curvature {b.name}", timings, quiet=True) as tc:
                    curvature = np.mean(
                        [
                            np.asarray(curvature_fns[b.name](positions[c], float(ts[s0]), jj_dev))
                            for c in range(C)
                        ],
                        axis=0,
                    )
                b.last_probe_chunk = ci
                var = release_inverse_mass(curvature, reference_var[b.idx])
                constrained = curvature > 0
                ratio = (
                    float(np.median(var[constrained]) / np.median(reference_var[b.idx]))
                    if constrained.any()
                    else np.inf
                )
                trace["probe_block"].append(bi)
                trace["probe_step"].append(s0)
                trace["probe_ratio"].append(ratio)
                summary = (
                    f"curvature variance median {np.median(var[constrained]) if constrained.any() else np.nan:.3g} vs "
                    f"prior-width {np.median(reference_var[b.idx]):.3g} (ratio {ratio:.3g}), "
                    f"{int(np.sum(~constrained))} coords unconstrained, {tc.elapsed:.1f} s"
                )
                if forced or ratio < self.release_mass_ratio:
                    b.released_chunk = ci
                    trace["block_release_step"][bi] = s0
                    trace["block_release_t"][bi] = ts[s0]
                    t_release[b.idx[~flat[b.idx]]] = ts[
                        s0
                    ]  # active from this chunk on (template fluxes stay frozen)
                    why = (
                        "latest time reached"
                        if forced and ratio >= self.release_mass_ratio
                        else "criterion met"
                    )
                    if log_prior_nuisance is not None:
                        rewarm_nuis_until = (
                            ci + self.rewarm_chunks
                        )  # re-tune the non-scene step quickly after every release
                        nuis_on = True
                        step_size_nuis = step_size_nuis * self.release_step_factor
                        cut = f"non-scene step size x{self.release_step_factor:g} -> {np.array2string(np.asarray(step_size_nuis), precision=3)}"
                    else:
                        step_size = step_size * self.release_step_factor
                        cut = f"step size x{self.release_step_factor:g} -> {np.array2string(np.asarray(step_size), precision=3)}"
                    log(f"released {b.name} at step {s0} (t={ts[s0]:.3f}, {why}): {summary}; {cut}")
                else:
                    log(f"probed {b.name} at step {s0} (t={ts[s0]:.3f}), stays frozen: {summary}")

            if (n, lik_on, nuis_on) not in runners:
                runners[(n, lik_on, nuis_on)] = make_runner(n, lik_on, nuis_on)
            rng, key = jax.random.split(rng)
            with Timer("sampling", timings, quiet=True) as tm:
                positions, ctrace = runners[(n, lik_on, nuis_on)](
                    jax.random.split(key, C), positions, as_dev(step_size), as_dev(step_size_nuis), inv_mass0_dev,
                    as_dev(ts[s0:s1]), as_dev(ts_prev[s0:s1]), jj_dev, as_dev(t_release), scene_mask_dev,
                )  # fmt: skip
                jax.block_until_ready(positions)
            for k, v in ctrace.items():
                trace[k][s0:s1] = np.asarray(v).T
            trace["steps_done"] = s1

            # Exact refresh of the released sky levels and point source fluxes
            gibbs_acc = None
            if gibbs_fn is not None and lik_on:
                active = (t_chunk <= t_release[lin["idx"]]).astype(np.float64)
                if active.any():
                    rng, key = jax.random.split(rng)
                    with Timer("linear gibbs", timings, quiet=True):
                        positions, g_acc, c_mean, c_cov = gibbs_fn(
                            jax.random.split(key, C), positions, t_chunk, jj_dev, as_dev(active)
                        )
                        jax.block_until_ready(positions)
                    gibbs_acc = np.asarray(g_acc)
                    trace["gibbs_accept"][ci] = gibbs_acc
                    trace["cond_mean"][ci] = np.asarray(c_mean)
                    trace["cond_cov"][ci] = np.asarray(c_cov)

            # Diagnostics at the end of the chunk, with the parameters used in it;
            # the log density is that of the state the chunk ends on (after the
            # Gibbs refresh), which is what the restart check below compares
            with Timer("scores", timings, quiet=True):
                scores, logp_post = score_fns[lik_on](
                    positions, t_chunk, jj_dev, as_dev(inv_mass_at(t_chunk))
                )
                scores, logp_end = np.asarray(scores), np.asarray(logp_post)
            trace["scores"][ci] = scores
            trace["chunk_logdensity"][ci] = logp_end
            trace["step_size"][ci] = step_size

            # Step size adaptation for the next chunk (until the draws start)
            acceptance = np.asarray(ctrace["accept_prob"]).mean(axis=1)
            acceptance_nuis = (
                np.nanmean(np.asarray(ctrace["accept_prob_nuis"]), axis=1) if nuis_on else None
            )
            adapting = s1 < draws_from
            if adapting:
                warming = s1 <= self.warmup_steps or ci < rewarm_until
                step_size = adapt_step_size(
                    step_size, acceptance, self.target_acceptance, self.acceptance_tol,
                    self.step_down_warmup if warming else self.step_down, self.step_up_warmup if warming else self.step_up,
                )  # fmt: skip
                if nuis_on:
                    warming_nuis = s1 <= self.warmup_steps or ci < rewarm_nuis_until
                    step_size_nuis = adapt_step_size(
                        step_size_nuis, acceptance_nuis, self.target_acceptance, self.acceptance_tol,
                        self.step_down_warmup if warming_nuis else self.step_down, self.step_up_warmup if warming_nuis else self.step_up,
                    )  # fmt: skip
            trace["step_size_nuis"][ci] = step_size_nuis if nuis_on else np.nan
            if s1 == draws_from:
                log(
                    f"step sizes frozen at step {s1}: {np.array2string(np.asarray(step_size), precision=3)}"
                    + (
                        f", non-scene {np.array2string(np.asarray(step_size_nuis), precision=3)}"
                        if nuis_on
                        else ""
                    )
                    + "; no more adaptation or restarts, the states saved from here on are the posterior draws"
                )

            # Stalled chains restart from the healthiest chain (until the draws start)
            if self.restart_collapsed_chains and C > 1 and ci >= 1 and adapting:
                flags = {
                    "collapsed": step_size < self.restart_step_ratio * float(np.median(step_size)),
                    "dead": acceptance < self.restart_dead_acceptance,
                    "outlier": np.zeros(C, dtype=bool),
                }
                if C > 2:
                    for c in range(C):
                        others = np.delete(logp_end, c)
                        spread = max(
                            float(np.max(others) - np.min(others)), self.restart_outlier_min
                        )
                        flags["outlier"][c] = (
                            abs(logp_end[c] - float(np.median(others)))
                            > self.restart_outlier_factor * spread
                        )
                for nm_, flag in flags.items():
                    stall_chunks[nm_] = np.where(flag, stall_chunks[nm_] + 1, 0)
                stalled = np.zeros(C, dtype=bool)
                reasons = {}
                for nm_, count in stall_chunks.items():
                    hit = count >= self.restart_chunks
                    stalled |= hit
                    for c in np.where(hit)[0]:
                        reasons.setdefault(int(c), []).append(nm_)
                if stalled.any() and not stalled.all():
                    best = int(np.argmax(np.where(stalled, -np.inf, logp_end)))
                    bad = np.where(stalled)[0]
                    positions = positions.at[jnp.asarray(bad)].set(positions[best])
                    step_size = np.where(stalled, step_size[best], step_size)
                    step_size_nuis = np.where(stalled, step_size_nuis[best], step_size_nuis)
                    for count in stall_chunks.values():
                        count[stalled] = 0
                    for c in bad:
                        trace["restart_step"].append(s1)
                        trace["restart_chain"].append(int(c))
                        trace["restart_source"].append(best)
                        trace["restart_reason"].append(",".join(reasons[int(c)]))
                    why = "; ".join(f"chain {int(c)}: {'+'.join(reasons[int(c)])}" for c in bad)
                    log(
                        f"stalled for {self.restart_chunks} chunks ({why}): restarted from chain {best}'s state and step sizes "
                        f"at step {s1} [{self.phase_of(s1)}]"
                    )

            # Save model states
            if s1 % self.save_every == 0 or s1 == total:
                with Timer("save state", timings, quiet=True):
                    save_chain_states(model, positions, group1, chain_paths)
                    trace["saved_step"].append(s1)
                    trace["saved_t"].append(ts[s1 - 1])
                    write_trace(tpath, trace)
                    save_tweedie(ts[s1 - 1])

            # Report
            lik_over_prior = (
                scores[:, SCORE_FIELDS.index("lik_mnorm_scene")]
                / scores[:, SCORE_FIELDS.index("prior_mnorm_scene")]
            )
            n_div = int(np.asarray(ctrace["divergent"]).sum())
            pred = np.asarray(ctrace["predictor_rms"]).mean(axis=1)
            fmt = lambda a: np.array2string(np.asarray(a), precision=3)  # noqa: E731
            log(
                f"step {s1}/{total} t={t_chunk:.3f} [{self.phase_of(s1)}] {tm.elapsed:.1f} s | "
                f"acc {fmt(acceptance)} | eps {fmt(step_size)} | pred rms {fmt(pred)} | "
                f"logp {fmt(np.asarray(ctrace['logdensity'])[:, -1])} | "
                f"div {n_div} | scene score lik/prior {fmt(lik_over_prior)}"
                + (
                    f" | nuis acc {fmt(acceptance_nuis)} eps {fmt(step_size_nuis)}"
                    if nuis_on
                    else ""
                )
                + (f" | gibbs acc {fmt(gibbs_acc)}" if gibbs_acc is not None else "")
            )
            if not np.all(np.isfinite(np.asarray(ctrace["logdensity"]))):
                log("  WARNING: non-finite log density in this chunk")
            for c in np.where(acceptance < self.stuck_acceptance)[0]:
                log(f"  WARNING: chain {c} looks stuck (acceptance {acceptance[c]:.3f})")
            for c in np.where(step_size < self.min_step_size)[0]:
                log(f"  WARNING: chain {c} step size collapsed to {step_size[c]:.3g}")

        save_json(files.timings, timings)
        log(f"finished {self.name}; total {sum(timings.values()) / 60:.1f} min")
        return trace

    # Products, plots and diagnostics
    #####################################################################
    def load_trace(self):
        """The trace of the run on disk (products.py)."""
        return self.files.load_trace()

    def plot_all(self, raise_errors=False):
        """Every figure (plots.py) from the products on disk."""
        from . import plots

        return plots.plot_all(self, raise_errors=raise_errors)

    def plot_progress(self):
        from . import plots

        return plots.plot_progress(self)

    def plot_scenes(self, max_panels=16):
        from . import plots

        return plots.plot_scenes(self, max_panels=max_panels)

    def plot_tweedie(self, max_panels=16):
        from . import plots

        return plots.plot_tweedie(self, max_panels=max_panels)

    def plot_lightcurve(self):
        from . import plots

        return plots.plot_lightcurve(self)

    def plot_flux_traces(self):
        from . import plots

        return plots.plot_flux_traces(self)

    def plot_parameter_traces(self):
        from . import plots

        return plots.plot_parameter_traces(self)

    def plot_fit_state(self, chain=0, index=-1):
        from . import plots

        return plots.plot_fit_state(self, chain=chain, index=index)

    def describe_init(self):
        """Where the initial scene sits relative to the prior (diagnostics.py);
        returns the report lines (also logged)."""
        from . import diagnostics

        return diagnostics.describe_init(self)

    def diagnose_stall(self, n_last=4):
        """Post-mortem of a run whose chain stalled (diagnostics.py); returns
        the report lines (also logged)."""
        from . import diagnostics

        return diagnostics.diagnose_stall(self, n_last=n_last)

    def run(self, plots=True):
        """sample() then plot_all(); plotting errors are logged, not raised."""
        trace = self.sample()
        if plots:
            try:
                self.plot_all()
            except Exception:  # noqa: BLE001
                log(f"plotting failed:\n{traceback.format_exc()}")
        return trace

    def __repr__(self):
        return f"Novamac({self.name!r}: {self.n_images} images, {self.n_sources} source(s), out_dir={str(self.files.out_dir)!r})"


__all__ = ["Novamac", "Path"]
