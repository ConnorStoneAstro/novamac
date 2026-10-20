"""
Pieces of the annealed sampler that the Novamac class assembles: the
schedule, the parameter blocks and their release rules, the prior energy
offset, the linear-coordinate table for the Gibbs refresh, the setup mass
matrix, the initial step size search and the chain-file writers. The method
itself is described in hmc.py; the loop that runs it is Novamac.sample.
"""

import os

import caskade as ck
import jax
import jax.numpy as jnp
import numpy as np

from ._backend import DTYPE
from .hmc import block_indices, flat_index, make_gauss_newton_fn, strided_pixel_groups
from .utils import log


# Schedule
#####################################################################
def start_time(sde, sigma_start):
    """t at which sigma(t) = sigma_start on the prior's VE schedule (1 if sigma_start is None)."""
    if sigma_start is None:
        return 1.0
    t = float(np.log(sigma_start / sde.sigma_min) / np.log(sde.sigma_max / sde.sigma_min))
    if not 0.0 < t <= 1.0:
        raise ValueError(f"sigma_start={sigma_start} is outside the prior's sigma range [{sde.sigma_min}, {sde.sigma_max}]")
    return t


def build_schedule(t_start, warmup_steps, anneal_steps, hold_steps):
    """Diffusion time t of every step: warmup at t_start, linear t_start -> 0, hold at 0."""
    anneal = t_start * (1.0 - np.arange(1, anneal_steps + 1) / anneal_steps)
    return np.concatenate([np.full(warmup_steps, t_start), anneal, np.zeros(hold_steps)])


# Parameter blocks
#####################################################################
class Block:
    """A frozen-then-released set of coordinates. `groups` partitions `idx`
    into sets that share no data pixels / are decoupled in the Hessian (one
    probe each, see hmc.make_gauss_newton_fn and make_block_curvature_fn).
    Released at t <= t_earliest if the curvature criterion is met (probed
    periodically), at t <= t_latest regardless; t_latest == t_earliest means
    an unconditional release."""

    def __init__(self, name, groups, t_earliest, t_latest=None):
        self.name = name
        self.groups = [np.asarray(g, dtype=int) for g in groups]
        self.idx = np.concatenate(self.groups)
        self.t_earliest = np.inf if t_earliest is None else float(t_earliest)
        self.t_latest = self.t_earliest if t_latest is None else float(t_latest)
        assert self.t_latest <= self.t_earliest, f"{name}: latest release must not precede earliest"
        self.released_chunk = None  # chunk index at which the block was released
        self.last_probe_chunk = None

    def freeze(self):
        self.t_earliest = self.t_latest = -np.inf


BLOCK_NAMES = ("scene_scale", "sky", "alignment", "ps_flux", "ps_center", "psf")


def parameter_blocks(model, release_t, release_latest_t):
    """The non-scene blocks with their decoupled groups: per-image parameters
    decouple across images (the likelihood is a sum over images), so one
    group per parameter type / component."""
    pix = model.models.pixelated_batch_model.model
    psf, sky = pix.psf, model.models.sky_batch_model.model
    crtan = [t.crtan for t in model.target.images]
    ps = [model.models[f"ps_batch_model_{p}"].model for p in range(model.meta.shapes["P"])]
    groups = {
        "scene_scale": [block_indices(model, [pix.scale])],
        "sky": [block_indices(model, [sky.I0])],
        "alignment": [block_indices(model, crtan, element=(k,)) for k in range(2)],
        "ps_flux": [block_indices(model, [m.flux.allflux]) for m in ps],
        "ps_center": [[i] for m in ps for i in block_indices(model, [m.center])],
        "psf": [block_indices(model, [p]) for p in (psf.n, psf.Rd, psf.q, psf.PA)],
    }
    unknown = set(release_t) - set(groups)
    if unknown:
        raise ValueError(f"release_t names unknown blocks {sorted(unknown)}; known: {BLOCK_NAMES}")
    blocks = [Block(name, g, release_t.get(name), release_latest_t.get(name)) for name, g in groups.items()]
    return [b for b in blocks if len(b.idx)]


# Prior energy offset
#####################################################################
def calibrate_prior_offset(model, scene_log, rng_key, n_sigmas=64):
    """
    Install model.meta.prior_energy_offset: the mean per-pixel energy of the
    initial scene noised to sigma, E(scene_log + sigma z, sigma) / N, on a
    log-spaced grid of `n_sigmas` sigma over the prior's range, interpolated
    in log sigma. Must run before log_prior is first traced. Returns
    (sigmas, values).
    """
    prior = model.meta.host_prior
    sde = prior.sde
    sigmas = np.exp(np.linspace(np.log(sde.sigma_min), np.log(sde.sigma_max), n_sigmas))
    # in the prior's coordinate: the prior is shown the scene prior_scene_factor times brighter
    scene = jnp.asarray(scene_log, DTYPE) + float(np.log(getattr(model.meta, "prior_scene_factor", 1.0)))
    n_pix = scene.size
    z = jax.random.normal(rng_key, scene.shape, DTYPE)
    values = np.array([float(prior.energy(scene + float(sg) * z, float(sg))) / n_pix for sg in sigmas])
    log_sig = jnp.asarray(np.log(sigmas), DTYPE)
    if np.all(values > 0):
        # the "dae" level is ~ (s^2 + sigma^2) / sigma^2, a power law at small
        # sigma: linear interpolation in log-log is nearly exact, whereas linear
        # in the value overshoots between nodes by a few percent of ~600 per
        # pixel, which showed as a sawtooth of ~1e5 in the logged log prior
        log_vals = jnp.asarray(np.log(values), DTYPE)
        model.meta.prior_energy_offset = lambda sigma: jnp.exp(jnp.interp(jnp.log(sigma), log_sig, log_vals))
    else:
        vals = jnp.asarray(values, DTYPE)
        model.meta.prior_energy_offset = lambda sigma: jnp.interp(jnp.log(sigma), log_sig, vals)
    return sigmas, values


# Data layout
#####################################################################
def flat_data_weight(model):
    """Data and inverse variance (0 on masked / bad pixels) flattened in exactly
    the order log_likelihood / model_image_flat use."""
    target = model.target[model.window]
    data = np.asarray(target.flatten("data"), dtype=np.float64)
    variance = np.asarray(target.flatten("variance"), dtype=np.float64)
    mask = np.asarray(target.flatten("mask"), dtype=bool)
    weight = np.where(mask | ~np.isfinite(variance) | (variance <= 0), 0.0, 1.0 / np.where(variance > 0, variance, 1.0))
    return data, weight


def linear_coordinates(model, frozen):
    """
    Table of the coordinates that enter the model image linearly in their
    physical value (see hmc.make_linear_gibbs_fn): every image's sky level
    (group 0, identity transform, no prior) and every non-template image's
    point source flux for each source p (group p + 1, log transform, the
    Gaussian prior of build_prior on log flux). `frozen` marks coordinates
    that never move (template fluxes); they are left out.
    """
    entries = flat_index(model)
    sky = model.models.sky_batch_model.model
    P = model.meta.shapes["P"]
    N = len(model.target.images)
    H, W = np.asarray(model.target.images[0].data).shape[-2:]
    log_flux0 = np.log(np.asarray(model.meta.ps_flux0, dtype=np.float64))
    w = float(model.meta.prior_widths["logflux"])
    params = {id(sky.I0): (0, False, None)}
    for p in range(P):
        ps = model.models[f"ps_batch_model_{p}"].model
        params[id(ps.flux.allflux)] = (p + 1, True, p)
    idx, group, image, is_log, mean, prec, source = [], [], [], [], [], [], []
    for i, (param, ix) in enumerate(entries):
        if id(param) not in params or frozen[i]:
            continue
        g, lg, src = params[id(param)]
        n = int(ix[0]) if len(ix) else 0
        idx.append(i)
        group.append(g)
        image.append(n)
        is_log.append(lg)
        source.append(-1 if src is None else src)
        if lg:
            mean.append(float(np.broadcast_to(log_flux0[src], (N,))[n]) if log_flux0.ndim else float(log_flux0))
            prec.append(1.0 / w**2)
        else:
            mean.append(0.0)
            prec.append(0.0)
    return {
        "idx": np.array(idx, dtype=int), "group": np.array(group, dtype=int), "image": np.array(image, dtype=int),
        "is_log": np.array(is_log, dtype=bool), "prior_mean": np.array(mean), "prior_prec": np.array(prec),
        "source": np.array(source, dtype=int), "pix_image": np.repeat(np.arange(N), int(H) * int(W)), "n_images": N,
    }  # fmt: skip


# Mass matrix
#####################################################################
def prior_precision(model, weight, mask):
    """
    Per-coordinate precision of the Gaussian priors in unconstrained space
    (0 where there is no prior), and a mask of the template-image flux
    coordinates, which have neither prior nor likelihood and are frozen for
    good. Also returns the prior-width variances (1 / precision, with the
    sky's data-only guess 1 / sum(weight)) used as the reference in the
    release criterion.
    """
    widths = model.meta.prior_widths
    sigma_min = float(model.meta.host_prior.sde.sigma_min)
    pix = model.models.pixelated_batch_model.model
    psf, sky = pix.psf, model.models.sky_batch_model.model
    is_template = np.asarray(model.meta.is_template, dtype=bool)
    unmasked_weight = np.sum(np.asarray(weight) * ~np.asarray(mask, dtype=bool), axis=(1, 2))

    # (prior precision, reference variance) per Param; callables take the element index
    scale_w = widths.get("scene_scale")
    rules = {
        id(pix.I): (1.0 / sigma_min**2, sigma_min**2),  # the prior at sigma_min is at most this curved
        id(pix.scale): (0.0 if scale_w is None else 1.0 / scale_w**2, 1.0 if scale_w is None else scale_w**2),
        id(psf.n): (1.0 / widths["psf_n"] ** 2, widths["psf_n"] ** 2),
        id(psf.Rd): (1.0 / widths["psf_Rd"] ** 2, widths["psf_Rd"] ** 2),  # log-transformed: relative width
        id(psf.q): (1.0 / widths["psf_q"] ** 2, widths["psf_q"] ** 2),
        id(psf.PA): (1.0 / widths["psf_PA"] ** 2, widths["psf_PA"] ** 2),
        id(sky.I0): lambda idx: (0.0, 1.0 / unmasked_weight[idx[0] if idx else 0]),
    }
    for target in model.target.images:
        rules[id(target.crtan)] = (1.0 / widths["wcs"] ** 2, widths["wcs"] ** 2)
    template_flux_params = set()
    for p in range(model.meta.shapes["P"]):
        ps = model.models[f"ps_batch_model_{p}"].model
        rules[id(ps.center)] = (1.0 / widths["ps_position"] ** 2, widths["ps_position"] ** 2)
        rules[id(ps.flux.allflux)] = lambda idx: (
            (0.0, widths["logflux"] ** 2) if is_template[idx[0] if idx else 0]
            else (1.0 / widths["logflux"] ** 2, widths["logflux"] ** 2)
        )  # fmt: skip
        template_flux_params.add(id(ps.flux.allflux))

    entries = flat_index(model)
    precision = np.zeros(len(entries))
    reference = np.ones(len(entries))
    template_flux = np.zeros(len(entries), dtype=bool)
    unknown = set()
    for i, (param, idx) in enumerate(entries):
        rule = rules.get(id(param))
        if rule is None:
            unknown.add(param.name)
            continue
        precision[i], reference[i] = rule(idx) if callable(rule) else rule
        if id(param) in template_flux_params and is_template[idx[0] if idx else 0]:
            template_flux[i] = True
    if unknown:
        log(f"WARNING: no prior rule for {sorted(unknown)}, treated as unconstrained")
    return precision, reference, template_flux


def setup_inverse_mass(model, images, x0, scene_idx, blocks, weight, mask, gn_stride, likelihood=True):
    """
    inv_mass0: Gauss-Newton estimate of the t = 0 posterior variance per
    coordinate, 1 / (diag(J^T W J) + prior precision), evaluated once at the
    initial state. The scene is probed with a stride pattern, the blocks with
    their groups. Coordinates with neither likelihood nor prior curvature
    (template-image fluxes) are reported and frozen by the caller.
    Returns (inv_mass0, reference variance, flat mask, curvature).
    """
    pix = model.models.pixelated_batch_model.model
    groups = strided_pixel_groups(scene_idx, pix.I.shape, gn_stride) + [g for b in blocks for g in b.groups]
    idx = np.concatenate(groups)
    curvature = np.zeros(x0.shape[0])
    if likelihood:  # no likelihood, no likelihood curvature (and no inf * 0 from an overflowing start)
        _, w = flat_data_weight(model)  # weights in exactly the order log_likelihood / images flatten the data
        gauss_newton = make_gauss_newton_fn(images, groups)
        curvature[idx] = np.nan_to_num(np.asarray(gauss_newton(x0, jnp.asarray(w, DTYPE)), dtype=np.float64))
    precision, reference, template_flux = prior_precision(model, weight, mask)
    total = np.maximum(curvature, 0.0) + precision
    flat = total <= 0
    inv_mass0 = np.where(flat, 1.0, 1.0 / np.where(flat, 1.0, total))
    return inv_mass0, reference, flat | template_flux, curvature


# Initial step size
#####################################################################
def find_initial_step_size(runner_1step, rng, positions, inv_mass0, t, jj, t_release, scene_mask, as_dev,
                           initial_step_size=0.1, tune_nuis=False, max_rounds=30):
    """
    Stan-style search: double (halve) the step size while single HMC steps
    are accepted with probability above (below) 0.5, stop at the first
    crossing. Per chain. With tune_nuis the non-scene sub-step's step size
    is searched at the same time from its own acceptance. Returns
    (rng, step_size, step_size_nuis).
    """
    num_chains = positions.shape[0]
    keys = ("accept_prob", "accept_prob_nuis") if tune_nuis else ("accept_prob",)
    step = {k: np.full(num_chains, initial_step_size) for k in keys}
    direction = {k: np.zeros(num_chains) for k in keys}
    done = {k: np.zeros(num_chains, dtype=bool) for k in keys}
    ts = as_dev(np.array([t]))
    nuis = lambda: step.get("accept_prob_nuis", step["accept_prob"])  # noqa: E731
    for _ in range(max_rounds):
        rng, key = jax.random.split(rng)
        _, ctrace = runner_1step(
            jax.random.split(key, num_chains), positions, as_dev(step["accept_prob"]), as_dev(nuis()), as_dev(inv_mass0),
            ts, ts, jj, as_dev(t_release), as_dev(scene_mask),
        )  # fmt: skip
        for k in keys:
            acc = np.asarray(ctrace[k])[:, 0]
            new_direction = np.where(acc > 0.5, 1.0, -1.0)
            done[k] |= (direction[k] != 0) & (new_direction != direction[k])
            direction[k] = np.where(done[k], direction[k], new_direction)
            step[k] = np.where(done[k], step[k], step[k] * np.where(direction[k] > 0, 2.0, 0.5))
        if all(np.all(d) for d in done.values()):
            break
    return rng, step["accept_prob"], nuis()


# Disk output helpers
#####################################################################
def save_chain_states(model, positions, group1, paths):
    """Append the current position of every chain to its caskade chain file
    (creating it on first use). Positions are unconstrained; set_values inside
    the ValidContext maps them to physical values, which is what is stored."""
    with ck.ValidContext(model):
        for pos, path in zip(positions, paths):
            model.set_values((pos, group1))
            if path.exists():
                model.append_state(str(path))
            else:
                model.save_state(str(path), appendable=True)


def write_trace(path, trace):
    """Flush the diagnostics to an npz atomically (write then rename)."""
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **{k: np.asarray(v) for k, v in trace.items()})
    os.replace(tmp, path)
