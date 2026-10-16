# construct the prior model
import types

import astrophot as ap
import jax.numpy as jnp

from .host_prior import load_prior


def build_prior(
    model: ap.Model,
    host_prior_loc: str,
    wcs_prior_width: float = 0.3,
    ps_position_prior_width: float = 0.5,
    logflux_prior_width: float = 2.0,
    psf_n_prior_width: float = 0.5,
    psf_Rd_prior_width: float = 0.2,
    psf_q_prior_width: float = 0.1,
    psf_PA_prior_width: float = 0.2,
    scene_scale_prior_width: float = 3.0,
    prior_scene_factor: float = 1.0,
):
    """
    Taking a model generated with `build_likelihood_model` this will construct a
    prior function and add it to the model. This will return an unnormalized log
    prior density.

    Args:
      - model: The full constructed AstroPhot model
      - host_prior_loc: The location of the host prior, a path to the checkpoint directory (str)
      - prior_scene_factor: the host prior is evaluated at log(I) + log(prior_scene_factor),
        i.e. it is shown a scene this many times brighter than the one the likelihood
        renders (an experiment: brighter training objects come with flatter, simpler
        surroundings). 1 means the prior sees the scene as it is (float)
      - wcs_prior_width: The width of the Gaussian prior on the WCS parameters
        in tangent plane coordinates, in arcseconds (float)
      - ps_position_prior_width: The width of the Gaussian prior on the PS positions
        in tangent plane coordinates, in arcseconds (float)
      - logflux_prior_width: The width of the Gaussian prior on the log of the pixel fluxes (float)
      - scene_scale_prior_width: The width (natural log) of the Gaussian prior on log of the
        scene scale factor, centred on its initial value; broad (3 = a factor of 20 at one
        sigma) so it acts as a soft log-uniform prior. None for no prior (improper flat in log).
    """
    # Collect prior configuration
    P = model.meta.shapes["P"]
    crtan0 = jnp.array(model.meta.crtan0)
    ps_center0 = jnp.array(model.meta.ps_center0)
    log_flux0 = jnp.log(jnp.array(model.meta.ps_flux0))
    psf_n0 = jnp.array(model.meta.psf_n0)
    psf_Rd0 = jnp.array(model.meta.psf_Rd0)
    psf_q0 = jnp.array(model.meta.psf_q0)
    psf_PA0 = jnp.array(model.meta.psf_PA0)
    not_template = jnp.array(~model.meta.is_template)
    host_prior = load_prior(host_prior_loc)
    # Keep the prior object and the widths reachable from the model, e.g. for
    # a sampler that needs the sigma(t) schedule or wants to seed a mass
    # matrix from the prior widths.
    model.meta.host_prior = host_prior
    model.meta.prior_scene_factor = float(prior_scene_factor)
    log_prior_scene_factor = float(jnp.log(prior_scene_factor))
    model.meta.prior_widths = {
        "wcs": wcs_prior_width,
        "ps_position": ps_position_prior_width,
        "logflux": logflux_prior_width,
        "psf_n": psf_n_prior_width,
        "psf_Rd": psf_Rd_prior_width,
        "psf_q": psf_q_prior_width,
        "psf_PA": psf_PA_prior_width,
        "scene_scale": scene_scale_prior_width,
    }
    log_scene_scale0 = float(jnp.log(model.meta.scene_scale0))

    def _nuisance_log_prior(self):
        """Every prior term except the host prior on the scene pixels (all
        Gaussian, no network); see log_prior_nuisance."""
        lp = 0.0

        # Prior on image positions (crtan, for image alignment)
        lp = lp - 0.5 * ap.backend.sum(
            ((self.models.sky_batch_model.target.crtan - crtan0) / wcs_prior_width) ** 2
        )

        for p in range(P):
            # Prior on point source positions
            lp = lp - 0.5 * ap.backend.sum(
                (
                    (self.models[f"ps_batch_model_{p}"].model.center.value - ps_center0[p])
                    / ps_position_prior_width
                )
                ** 2
            )

            # Prior on point source log fluxes. Template images carry no point
            # source (flux = allflux * 0), so take the log of `allflux` (always
            # > 0) and mask with not_template. Using `flux` here would give
            # log(0) * 0 = nan for every template image.
            log_allflux = jnp.log(self.models[f"ps_batch_model_{p}"].model.flux.allflux.value)
            lp = lp - 0.5 * ap.backend.sum(
                ((log_allflux - log_flux0[p]) / logflux_prior_width) ** 2 * not_template
            )

        # PSF prior (the PSF params are per image, shape (N,), so sum over images)
        psf = self.models.pixelated_batch_model.model.psf
        lp = lp - 0.5 * ap.backend.sum((psf.n.value - psf_n0) ** 2) / psf_n_prior_width**2
        lp = lp - 0.5 * ap.backend.sum((psf.Rd.value / psf_Rd0 - 1) ** 2) / psf_Rd_prior_width**2
        lp = lp - 0.5 * ap.backend.sum((psf.q.value - psf_q0) ** 2) / psf_q_prior_width**2
        lp = lp - 0.5 * ap.backend.sum((psf.PA.value - psf_PA0) ** 2) / psf_PA_prior_width**2

        # Soft log-uniform prior on the scene scale factor
        if scene_scale_prior_width is not None:
            log_scale = jnp.log(self.models.pixelated_batch_model.model.scale.value)
            lp = lp - 0.5 * ((log_scale - log_scene_scale0) / scene_scale_prior_width) ** 2

        return lp

    @ap.param.forward
    def log_prior_nuisance(self):
        """
        The prior on everything except the scene pixels: image alignment,
        point source positions and log fluxes, PSF shape, scene scale. All
        Gaussian and independent of t, and cheap (no network), so a sampler
        can update these blocks with the scene held fixed without evaluating
        the host prior.
        """
        return _nuisance_log_prior(self)

    @ap.param.forward
    def log_prior(self, t=0.0):
        """
        Unnormalized log prior. `t` in [0, 1] is the diffusion time of the
        host-galaxy prior: the pixel prior is evaluated at noise level
        sigma(t) of the VE schedule, so t=1 is the broadest (nearly Gaussian)
        prior and t=0 the sharpest one the model was trained on. The Gaussian
        priors on the other parameters do not depend on t.

        If `self.meta.prior_energy_offset` is set to a function of sigma, its
        value is subtracted from every scene pixel's energy before the sum
        (host_prior.log_prob's `pixel_offset`): a sampler sets it to the
        typical per-pixel energy at each sigma so the float32 total stays
        small enough to resolve Metropolis differences. A function of sigma
        alone leaves every ratio at fixed t unchanged.
        """
        lp = _nuisance_log_prior(self)
        # Prior on pixel fluxes (in the prior's own units: the scene scale absorbs the rest)
        # the prior sees the scene prior_scene_factor times brighter than the likelihood does
        log_pix = jnp.log(self.models.pixelated_batch_model.model.I.value) + log_prior_scene_factor
        sigma = host_prior.sde.sigma(t)
        offset_fn = getattr(self.meta, "prior_energy_offset", None)
        pixel_offset = offset_fn(sigma) if offset_fn is not None else 0.0
        lp = lp + host_prior.log_prob(log_pix, sigma, pixel_offset)
        return lp

    model.log_prior_nuisance = types.MethodType(log_prior_nuisance, model)
    model.log_prior = types.MethodType(log_prior, model)

    return model
