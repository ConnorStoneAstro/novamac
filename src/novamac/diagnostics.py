"""
Diagnostics that read a run's products (and the model) to say what went
wrong, for use at scale when a run misbehaves:

  describe_init(nm)   where the initial scene sits relative to the prior,
                      in the prior's own coordinate x = log I: the prior's
                      landmarks (sky pedestal, data_mean, sigma range), the
                      initial scene by percentile with the data's sky
                      statistics it came from, the prior-only reference
                      draws and the saved chain states by percentile. A
                      start whose sky sits e-folds above or below the
                      prior's pedestal is out of the prior's distribution
                      however much sigma_t noise is added on top; sigma_start
                      and the init settings are what to change.
  diagnose_stall(nm)  post-mortem for a run whose chain stalled: the step
                      size and mass in force during the last windows, the
                      target (values, gradient norms, numerical resolution)
                      at the final state around the stall, single HMC
                      transitions from the final state over a step-size
                      scan, and transitions with one block frozen at a time
                      (which block limits the step size).
"""

import caskade as ck
import h5py
import jax
import jax.numpy as jnp
import numpy as np

from ._backend import DTYPE, MATMUL_PRECISION
from .hmc import make_broadening_fn, make_chunk_runner, make_log_parts, make_sigma_fn, make_target
from .products import load_chain_state, num_saved_states, read_scene_history
from .sampler import calibrate_prior_offset, parameter_blocks
from .utils import log

PCT = [0.1, 1, 10, 50, 90, 99, 99.9]


def _fmt(x):
    return "  ".join(f"{v:7.2f}" for v in np.percentile(x, PCT))


def describe_init(nm):
    """See the module docstring. Returns the report lines (also logged)."""
    nm.prepare()
    lines = []
    out = lambda s: (lines.append(s), log(s))  # noqa: E731
    prior = nm.prior
    s = prior.transform.softening
    sde = prior.config["sde"]
    factor = nm.prior_scene_factor
    out(f"prior: softening s = {s:g} nJy/prior px, sky pedestal log(s log 2) = {np.log(s * np.log(2)):.3f}, "
        f"data_mean = {sde.get('data_mean')}, data_std = {sde.get('data_std')}, sigma in [{sde['sigma_min']:.4g}, {sde['sigma_max']:.4g}]")
    out(f"units: 1 data unit = {nm.njy_per_count:.4g} nJy (ZP {nm.zeropoint:g}); scale = {nm.scene_scale0:.4g}; prior scene factor {factor:g}"
        + (f" (the prior's landmarks above are {np.log(factor):.3f} higher in log I than the scene units below)" if factor != 1 else ""))
    for line in nm.init_info["report"]:
        out(line)
    out(f"data pixel {nm.data_pixel_scale:.4f} arcsec, scene pixel {nm.scene_pixel_scale:.4f} arcsec ({nm.data_pixel_scale / nm.scene_pixel_scale:.2f}x upsampling)")
    init_I = nm.likelihood_args["pix_init"] / nm.scene_scale0  # prior units after the transform
    head = "  ".join(f"{p:>7g}" for p in PCT)
    out(f"percentiles of log I, columns: {head}")
    out(f"  initial scene (transformed)   {_fmt(np.log(init_I))}")
    raw = nm.init_info["raw"]
    pos = raw[raw > 0]
    if pos.size:
        out(f"  raw init > 0 ({pos.size}/{raw.size} px), as data units/arcsec^2 -> log nJy/prior px: {_fmt(np.log(pos / nm.scene_scale0))}")
    data, mask = nm.data, nm.mask
    edge = np.ones(data.shape[-2:], bool)
    edge[4:-4, 4:-4] = False
    sky_level = np.array([np.median(d[edge & ~m]) if (edge & ~m).any() else np.nan for d, m in zip(data, mask)])
    sky_rms = np.array([1.4826 * np.median(np.abs(d[edge & ~m] - lv)) if (edge & ~m).any() else np.nan for d, m, lv in zip(data, mask, sky_level)])
    out(f"cutouts: median of the outer 4-px border per image: median {np.nanmedian(sky_level):.4g}, range [{np.nanmin(sky_level):.4g}, {np.nanmax(sky_level):.4g}] (data units/px); "
        f"robust rms {np.nanmedian(sky_rms):.4g}; initial sky (per arcsec^2) median {np.median(nm.sky):.4g}, range [{nm.sky.min():.4g}, {nm.sky.max():.4g}]")
    out(f"  the sky level per data pixel, as log nJy/prior px, would be "
        f"{np.log(max(np.nanmedian(sky_level), 1e-30) / nm.data_pixel_scale**2 / nm.scene_scale0):.2f}")
    if not nm.files.exists():
        out("no run products yet (the chain states and the prior-only reference draws are reported once a run exists)")
        return lines
    trace = nm.load_trace()
    if "pflow_reference" in trace:
        ref = trace["pflow_reference"]
        for c in range(ref.shape[0]):
            out(f"  pflow reference chain {c}       {_fmt(np.log(np.clip(ref[c], 1e-30, None)))}")
    C = int(trace["num_chains"])
    for c in range(C):
        with h5py.File(nm.files.chain(c), "r") as h5:
            n = num_saved_states(h5)
            scenes = read_scene_history(h5)
        for k in sorted({0, 1, n // 2, n - 1}):
            if k < n:
                out(f"  chain {c} saved state {k:2d} (t={float(trace['saved_t'][k]):.3f}) {_fmt(np.log(np.clip(scenes[k], 1e-30, None)))}")
        frac_dark = float(np.mean(np.log(np.clip(scenes[-1], 1e-30, None)) < np.log(s * np.log(2)) - 3))
        out(f"  chain {c}: fraction of final pixels more than 3 e-folds below the prior's sky pedestal: {frac_dark:.3%}")
    return lines


def diagnose_stall(nm, n_last=4):
    """See the module docstring. Returns the report lines (also logged)."""
    lines = []
    out = lambda s: (lines.append(s), log(s))  # noqa: E731
    trace = nm.load_trace()
    n_chunks = int(np.sum(trace["chunk_step"] <= int(trace["steps_done"])))
    C = int(trace["num_chains"])
    n_last = min(n_last, n_chunks)

    model = nm.build_model()
    blocks = {b.name: b.idx for b in parameter_blocks(model, nm.release_t, nm.release_latest_t)}
    blocks["scene"] = trace["scene_idx"]
    sde = model.meta.host_prior.sde
    data_std = nm.prior.config["sde"].get("data_std")
    tweedie = bool(trace["tweedie"])
    sigma_fn, broaden_fn = make_sigma_fn(sde), make_broadening_fn(sde, float(data_std) if tweedie else None)
    jj = jnp.asarray(trace["jj_diag"], DTYPE)

    # 1. adaptation state over the last windows
    inv_mass0 = trace["inv_mass0"].astype(np.float64)
    frac = float(trace["mass_sigma_fraction"])
    scene_mask = np.zeros(len(inv_mass0))
    scene_mask[trace["scene_idx"]] = 1.0
    inv_mass_at = lambda t: inv_mass0 + frac * float(sigma_fn(t)) ** 2 * scene_mask  # noqa: E731
    out("step size, acceptance and predictor displacement over the last windows (mass is fixed: inv_mass0 + frac * sigma_t^2 on the scene)")
    out(f"{'step':>6} {'t':>7} {'acc':>14} {'median log alpha':>18} {'eps':>16} {'pred rms':>14}  scene mass")
    for ci in range(n_chunks - n_last, n_chunks):
        s1, t = int(trace["chunk_step"][ci]), float(trace["chunk_t"][ci])
        s0 = s1 - int(trace["chunk_steps"])
        acc = np.nanmean(trace["accept_prob"][s0:s1], axis=0)
        la = np.nanmedian(trace["log_alpha"][s0:s1], axis=0)
        pr = np.nanmean(trace["predictor_rms"][s0:s1], axis=0)
        eps = trace["step_size"][ci]
        out(f"{s1:6d} {t:7.4f} {np.array2string(acc, precision=3):>14} {np.array2string(la, precision=3):>18} {np.array2string(eps, precision=4):>16} {np.array2string(pr, precision=3):>14}  {np.median(inv_mass_at(t)[trace['scene_idx']]):.3g}")
    out("inv_mass0 block medians: " + "  ".join(f"{k} {np.median(inv_mass0[idx]):.3g}" for k, idx in blocks.items() if len(idx)))
    n_bad = int(np.sum(inv_mass0 <= 0)) + int(np.sum(~np.isfinite(inv_mass0)))
    out(f"inv_mass0: {n_bad} coordinates are zero / non-finite, smallest positive {np.min(inv_mass0[inv_mass0 > 0]):.3g}")

    # 2. the target at the final state around the stall
    with ck.ValidContext(model):
        x_init, group1 = model.get_values()
    group1 = jnp.asarray(group1, DTYPE)
    # the prior's per-pixel energy offset, exactly as the sampler installs it (without it the
    # log density is ~1e7 and float32 resolves differences only to ~1)
    if "prior_offset_value" in trace:
        pix = model.models.pixelated_batch_model.model
        calibrate_prior_offset(model, np.asarray(x_init)[trace["scene_idx"]].reshape(pix.I.shape), jax.random.key(123), len(trace["prior_offset_sigma"]))
        out("prior energy offset installed (as in the sampler)")
    log_lik, log_prior, _ = make_log_parts(model, group1)
    if not bool(trace.get("likelihood", True)) or np.all(np.nan_to_num(trace["loglik"][: int(trace["steps_done"])]) == 0):
        log_lik = jax.jit(lambda x, extra_var: jnp.zeros((), dtype=x.dtype))
    target = make_target(log_lik, log_prior, sigma_fn, jnp.asarray(scene_mask, DTYPE), tweedie, nm.tweedie_exact_jacobian)
    finals = []
    for c in range(C):
        with h5py.File(nm.files.chain(c), "r") as h5:
            load_chain_state(model, h5, -1)
        with ck.ValidContext(model):
            x, _ = model.get_values()
        finals.append(jnp.asarray(x, DTYPE))
    x = finals[0]
    scene = jnp.asarray(trace["scene_idx"])
    acc_per_chunk = np.array([np.nanmean(trace["accept_prob"][int(e) - int(trace["chunk_steps"]) : int(e)]) for e in trace["chunk_step"][:n_chunks]])
    stalled = np.where(acc_per_chunk < 0.1)[0]
    ci_stall = int(stalled[0]) if len(stalled) else n_chunks - 1
    ci_before = max(ci_stall - 1, 0)
    t_stall = float(trace["t"][max(int(trace["chunk_step"][ci_stall]) - int(trace["chunk_steps"]), 0)])  # t at the start of the first stalled window
    t_before = float(trace["chunk_t"][ci_before])  # t at the end of the last healthy window
    out(f"first stalled window ends at step {int(trace['chunk_step'][ci_stall])} (starts at t={t_stall:.4f}); last healthy window ends at t={t_before:.4f}"
        + ("" if len(stalled) else " (no window with acceptance < 0.1: using the last two)"))
    out(f"final state of chain 0 (last chunk t={t_stall:.4f}, previous {t_before:.4f}); sigma_min={sde.sigma_min:.4g}")
    out(f"{'t':>8} {'sigma':>9} {'log L':>14} {'log prior':>14} {'|grad L|':>10} {'|grad p|':>10} {'|grad p| scene':>14} {'finite':>7}")
    for t in sorted({0.02, 0.01, t_before, t_stall * 0.5 + t_before * 0.5, t_stall, 0.001, 0.0}, reverse=True):
        t = float(t)
        ev = broaden_fn(t) * jj
        ll, gl = jax.value_and_grad(log_lik)(x, ev)
        lp, gp = jax.value_and_grad(log_prior)(x, t)
        finite = bool(jnp.isfinite(ll) & jnp.isfinite(lp) & jnp.all(jnp.isfinite(gl)) & jnp.all(jnp.isfinite(gp)))
        out(f"{t:8.4f} {float(sigma_fn(t)):9.4g} {float(ll):14.6g} {float(lp):14.6g} {float(jnp.linalg.norm(gl)):10.3g} {float(jnp.linalg.norm(gp)):10.3g} {float(jnp.linalg.norm(gp[scene])):14.3g} {str(finite):>7}")

    # 2b. numerical resolution of the log density at the final state (t = 0)
    out(f"log density resolution at the final state, t = 0 (the Metropolis test needs ~0.01); "
        f"matmul precision = {MATMUL_PRECISION}, backend = {jax.default_backend()}:")
    ev0 = broaden_fn(0.0) * jj
    lp_a = float(log_lik(x, ev0)) + float(log_prior(x, 0.0))
    lp_b = float(log_lik(x + 0.0, ev0)) + float(log_prior(x + 0.0, 0.0))
    out(f"  same state evaluated twice: {lp_a:.6f} vs {lp_b:.6f} (difference {lp_b - lp_a:.3g}; nonzero = GPU reduction nondeterminism)")
    g_full = jax.grad(lambda y: log_lik(y, ev0) + log_prior(y, 0.0))(x)
    for name in ("scene", "alignment", "psf", "ps_center"):
        if name not in blocks or len(blocks[name]) == 0:
            continue
        i = int(np.asarray(blocks[name])[0])
        for d in (1e-2, 1e-3, 1e-4):
            x2 = x.at[i].add(d)
            meas = float(log_lik(x2, ev0)) + float(log_prior(x2, 0.0)) - lp_a
            out(f"  {name:10s} coord {i}: shift {d:g} -> measured change {meas:+.5f}, gradient predicts {float(g_full[i]) * d:+.5f}")
    la_hold = trace["log_alpha"][int(trace["warmup_steps"]) + int(trace["anneal_steps"]) : int(trace["steps_done"])]
    if la_hold.size:
        la = la_hold[np.isfinite(la_hold)]
        if la.size:
            out(f"  hold log alpha: percentiles 5/25/50/75/95 = {np.round(np.percentile(la, [5, 25, 50, 75, 95]), 3)}; "
                f"fraction with |log alpha| < 0.01: {np.mean(np.abs(la) < 0.01):.3f}; fraction exactly 0: {np.mean(la == 0):.3f}")

    # 3. single transitions: which combination of (adaptation state, t) diverges?
    runner = make_chunk_runner(target, broaden_fn, sigma_fn, nm.num_integration_steps, 1, frac, 0.0)
    t_release = jnp.full(x.shape, jnp.inf, DTYPE)
    im0 = jnp.asarray(inv_mass0, DTYPE)
    sm = jnp.asarray(scene_mask, DTYPE)
    transition = lambda keys, eps, t, rel: runner(keys, jnp.stack(finals), eps, eps, im0, jnp.asarray([t], DTYPE), jnp.asarray([t], DTYPE), jj, rel, sm)[1]  # noqa: E731
    key = jax.random.key(0)
    out("single HMC transitions from the final states (4 tries per chain): mean log acceptance ratio, fraction divergent")
    out("  if a 10x or 100x smaller step size recovers acceptance the target just got stiffer; if nothing does, the target itself is broken")
    for label, ci in (("healthy window", ci_before), ("stalled window", ci_stall)):
        eps0 = np.asarray(trace["step_size"][ci], dtype=np.float64)
        for t in (t_before, t_stall, 0.0):
            for scale in (10.0, 1.0, 0.1, 0.01, 0.001):
                las, divs = [], []
                for _ in range(4):
                    key, k = jax.random.split(key)
                    tr = transition(jax.random.split(k, C), jnp.asarray(eps0 * scale, DTYPE), t, t_release)
                    las.append(np.asarray(tr["log_alpha"])[:, 0])
                    divs.append(np.asarray(tr["divergent"])[:, 0])
                out(f"  {label:16s} step size, t={t:.4f}, eps x{scale:<5g}: log alpha {np.array2string(np.mean(las, axis=0), precision=2)}, divergent {np.mean(divs, axis=0)}")

    # 4. which block limits the step size? transitions with one block frozen at a time
    out("transitions at the healthy window's tuning with one block frozen at a time, step size x1 / x10:")
    out("  a block whose freezing lets a 10x larger step be accepted is the one limiting the step size")
    eps0 = np.asarray(trace["step_size"][ci_before], dtype=np.float64)
    for name in ["none"] + [b for b in blocks if len(blocks[b])]:
        rel = np.full(x.shape, np.inf)
        if name != "none":
            rel[blocks[name]] = -np.inf
        rel = jnp.asarray(rel, DTYPE)
        res = []
        for scale in (1.0, 10.0):
            las = []
            for _ in range(4):
                key, k = jax.random.split(key)
                tr = transition(jax.random.split(k, C), jnp.asarray(eps0 * scale, DTYPE), t_before, rel)
                las.append(np.asarray(tr["log_alpha"])[:, 0])
            res.append(np.array2string(np.mean(las, axis=0), precision=2))
        out(f"  frozen {name:11s}: log alpha x1 {res[0]}, x10 {res[1]}")
    return lines
