"""
Diagnostic figures for a finished (or partially finished) run, from the
products on disk (products.py). Every function takes the Novamac instance
and writes PDFs next to the products:

  {name}_sample_progress.pdf   log density / likelihood / prior, acceptance
                               and step size, likelihood vs prior score on
                               the scene block, predictor displacement and
                               mass, divergences, all against HMC step with
                               the warmup / releases / hold / restarts marked
  {name}_sample_scene.pdf      the static scene (pixelated host, log scale)
                               at saved states through the run, one row per
                               chain, the point source position marked
  {name}_sample_tweedie.pdf    the same states denoised by Tweedie's
                               formula, x + sigma_t^2 score: the scene the
                               likelihood sees (before the -v_t/2 mean shift)
  {name}_sample_fluxes.pdf     light curve from every saved state, the
                               reference light curve overlaid, the t=0 states
                               highlighted
  {name}_sample_flux_trace.pdf point-source flux of each non-template image
                               vs saved step
  {name}_sample_alignment_trace.pdf  image alignment offsets dx, dy (arcsec,
                               one line per image) and the point source
                               centre vs saved step
  {name}_sample_nuisance_trace.pdf   scene scale, per-image sky level and
                               PSF n, Rd, q, PA vs saved step
and, at one saved state (the last of chain 0 by default), the image-level
fit diagnostics:
  {name}_fit_fluxes.pdf, {name}_fit_target_shifts.pdf, {name}_core_pix_{linear,log}.png,
  {name}_residual_histogram.pdf, {name}_target.pdf, {name}_fit_model.pdf, {name}_fit_residual.pdf
"""

import traceback

import astrophot as ap
import caskade as ck
import h5py
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ._backend import DTYPE  # noqa: E402
from .hmc import SCORE_FIELDS, block_indices, make_log_parts, make_sigma_fn  # noqa: E402
from .products import (  # noqa: E402
    collect_fluxes,
    conditional_fluxes,
    load_chain_state,
    read_param_history,
    read_scene_history,
)
from .utils import Timer, get_fluxes, log  # noqa: E402

CHAIN_COLORS = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]


def running_mean(x, window):
    """Running mean that ignores nan (nan where the window holds no finite value)."""
    window = max(1, min(window, len(x)))
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    num = np.convolve(np.where(ok, x, 0.0), np.ones(window), mode="same")
    den = np.convolve(ok.astype(float), np.ones(window), mode="same")
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def phase_marks(trace):
    """(step, label) of the schedule's phase boundaries."""
    warmup, anneal = int(trace["warmup_steps"]), int(trace["anneal_steps"])
    marks = [(warmup, "anneal start"), (warmup + anneal, "hold (t=0)")]
    if int(trace.get("likelihood_on_step", 0)) > 0:
        marks.append((int(trace["likelihood_on_step"]), "likelihood on"))
    if "restart_step" in trace:
        restarts = {}
        reasons = trace["restart_reason"] if "restart_reason" in trace else [""] * len(trace["restart_step"])
        for step, c, src, why in zip(trace["restart_step"], trace["restart_chain"], trace["restart_source"], reasons):
            restarts.setdefault(int(step), []).append(f"{int(c)}<-{int(src)}" + (f" ({why})" if len(str(why)) else ""))
        for step, names in restarts.items():
            marks.append((step, "restart chain " + ", ".join(names)))
    # blocks released in the same chunk share one label
    released = {}
    for name, t_rel, step in zip(trace["block_names"], trace["block_release_t"], trace["block_release_step"]):
        if step >= 0:
            released.setdefault((int(step), f"{t_rel:.2f}"), []).append(str(name))
    for (step, t_rel), names in released.items():
        marks.append((step, f"{', '.join(names)} released (t={t_rel})"))
    return sorted(marks)


def plot_progress(nm, trace=None):
    trace = nm.load_trace() if trace is None else trace
    done = int(trace["steps_done"])
    if done == 0:
        log("no steps recorded yet, skipping progress plot")
        return None
    steps = np.arange(1, done + 1)
    n_chains = int(trace["num_chains"])
    chunk_step = trace["chunk_step"]
    n_chunks = int(np.sum(chunk_step <= done))
    si = {f: i for i, f in enumerate(SCORE_FIELDS)}

    fig, axes = plt.subplots(6, 1, figsize=(11, 18), sharex=True)
    ax_lp, ax_parts, ax_acc, ax_score, ax_mass, ax_div = axes
    ax_prior = ax_parts.twinx()
    ax_eps = ax_acc.twinx()
    ax_cos = ax_score.twinx()

    ratio = lambda a, b: np.where((a > 0) & (b > 0), a / np.where(b > 0, b, 1.0), np.nan)  # noqa: E731
    offset_in = "prior_offset_value" in trace
    for c in range(n_chains):
        color = CHAIN_COLORS[c % len(CHAIN_COLORS)]
        ax_lp.plot(steps, trace["logdensity"][:done, c], lw=0.8, color=color, label=f"chain {c}")
        # change since the first step: the absolute values are dominated by
        # constants and the prior's 1/sigma scaling and tell nothing by eye
        ax_parts.plot(steps, trace["loglik"][:done, c] - trace["loglik"][0, c], lw=0.8, color=color, label=f"chain {c} log L")
        ax_prior.plot(steps, trace["logprior"][:done, c] - (0.0 if offset_in else trace["logprior"][0, c]), lw=0.8, ls="--", color=color, label=f"chain {c} log prior")
        ax_acc.plot(steps, running_mean(trace["accept_prob"][:done, c], 64), lw=0.8, color=color)
        ax_eps.plot(chunk_step[:n_chunks], trace["step_size"][:n_chunks, c], lw=0.8, ls=":", color=color)
        if "accept_prob_nuis" in trace and np.any(np.isfinite(trace["accept_prob_nuis"][:done, c])):
            an = trace["accept_prob_nuis"][:done, c]
            ax_acc.plot(steps, running_mean(np.where(np.isfinite(an), an, np.nan), 64), lw=0.8, ls="-.", color=color,
                        label="non-scene HMC acceptance (dash-dot)" if c == 0 else None)  # fmt: skip
            ax_eps.plot(chunk_step[:n_chunks], trace["step_size_nuis"][:n_chunks, c], lw=0.6, ls="--", color=color)
        if "gibbs_accept" in trace:
            ga = trace["gibbs_accept"][:n_chunks, c]
            ok = np.isfinite(ga)
            ax_acc.scatter(chunk_step[:n_chunks][ok], ga[ok], s=6, marker="^", color=color, alpha=0.6,
                           label="linear Gibbs: fraction of images accepted" if c == 0 else None)  # fmt: skip
        sc = trace["scores"][:n_chunks, c]
        # what drives the scene: |likelihood score| / |prior score| on the scene block
        ax_score.plot(chunk_step[:n_chunks], ratio(sc[:, si["lik_mnorm_scene"]], sc[:, si["prior_mnorm_scene"]]), lw=0.8, color=color, label=f"chain {c} metric weighted")
        ax_score.plot(chunk_step[:n_chunks], ratio(sc[:, si["lik_norm_scene"]], sc[:, si["prior_norm_scene"]]), lw=0.8, ls="--", color=color, label=f"chain {c} raw")
        ax_cos.plot(chunk_step[:n_chunks], sc[:, si["cos_scene"]], lw=0.6, ls=":", color=color)
        div_steps = steps[trace["divergent"][:done, c]]
        ax_div.plot(div_steps, np.full(len(div_steps), c), "|", color=color, ms=10)

    ax_lp.set_ylabel("log posterior density")
    ax_lp.legend(loc="lower right", fontsize=8)
    ax_parts.set_ylabel("log L - log L(step 0)")
    ax_parts.legend(loc="lower left", fontsize=8)
    ax_prior.set_ylabel("log prior relative to a typical noised scene (dashed)" if offset_in else "log prior - log prior(step 0) (dashed)")
    ax_prior.legend(loc="lower right", fontsize=8)
    ax_acc.set_ylabel("acceptance (running mean; dash-dot: non-scene HMC)")
    ax_acc.set_ylim(0, 1)
    ax_eps.set_ylabel("step size (dotted)")
    ax_eps.set_yscale("log")
    ax_score.axhline(1.0, color="k", lw=0.5)
    ax_score.set_ylabel("scene |score L| / |score prior|")
    ax_score.set_yscale("log")
    ax_score.legend(loc="upper right", fontsize=8)
    ax_cos.set_ylabel("cos(score_L, score_prior) (dotted)")
    ax_cos.set_ylim(-1, 1)
    # predictor displacement (solid, left) and the fixed mass matrix's block
    # medians (dotted, right)
    ax_massr = ax_mass.twinx()
    for c in range(n_chains):
        color = CHAIN_COLORS[c % len(CHAIN_COLORS)]
        pr = trace["predictor_rms"][:done, c]
        ax_mass.plot(steps, np.where(pr > 0, pr, np.nan), lw=0.6, color=color, label=f"chain {c} predictor rms")
    if "inv_mass0" in trace:
        for name in ["scene"] + [str(n) for n in trace["block_names"]]:
            idx = trace["scene_idx"] if name == "scene" else trace[f"block_idx_{name}"]
            med = np.full(n_chunks, float(np.median(trace["inv_mass0"][idx])))
            ax_massr.plot(chunk_step[:n_chunks], med, lw=0.8, ls=":", label=name)
    ax_mass.set_yscale("log")
    ax_mass.set_ylabel("predictor rms displacement (scene)")
    ax_mass.legend(loc="upper left", fontsize=7)
    ax_massr.set_yscale("log")
    ax_massr.set_ylabel("inv_mass0 block median (dotted)")
    ax_massr.legend(loc="upper right", fontsize=7, ncol=3)
    ax_div.set_ylabel("divergent transitions")
    ax_div.set_yticks(range(n_chains))
    ax_div.set_yticklabels([f"chain {c}" for c in range(n_chains)])
    ax_div.set_ylim(-0.5, n_chains - 0.5)
    ax_div.set_xlabel("HMC step")

    for step, label in phase_marks(trace):
        for ax in axes:
            ax.axvline(step, color="k", ls=":", lw=0.6)
        ax_lp.text(step, ax_lp.get_ylim()[1], label, fontsize=7, va="top", ha="right", rotation=90)

    fig.suptitle(f"{nm.name}: sampling progress")
    path = nm.files.figure("sample_progress")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _trace_panel(ax, saved_step, series, n_chains, trace, ylabel, log_scale=False, band=None):
    """series[c]: (n_saved, k) lines for chain c, drawn like the flux traces
    (one line per image, chain colour, phase marks); `band` = (lo, hi) shades
    a reference range."""
    if band is not None:
        ax.axhspan(band[0], band[1], color="0.9", zorder=0)
    for c in range(n_chains):
        color = CHAIN_COLORS[c % len(CHAIN_COLORS)]
        arr = np.asarray(series[c])
        for k in range(arr.shape[1]):
            ax.plot(saved_step[: arr.shape[0]], arr[:, k], lw=0.6, color=color, alpha=0.5)
        ax.plot([], [], color=color, label=f"chain {c}")
    for step, _ in phase_marks(trace):
        ax.axvline(step, color="k", ls=":", lw=0.6)
    if log_scale:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)


def plot_lightcurve(nm, fluxes=None, trace=None):
    """Flux vs MJD (or image index) for every saved state; the t = 0 states
    are the actual posterior draws and are highlighted, earlier (annealed)
    states are grey. The reference light curve, if given, is overlaid."""
    trace = nm.load_trace() if trace is None else trace
    n_chains = int(trace["num_chains"])
    fluxes = collect_fluxes(nm.files, n_chains) if fluxes is None else fluxes
    n_chains, n_saved, P, N = fluxes.shape
    saved_t = trace["saved_t"][:n_saved]
    keep = ~nm.is_template
    final = saved_t == 0
    x = np.arange(N) if nm.mjd is None else nm.mjd
    paths = []
    for p in range(P):
        fig, ax = plt.subplots(figsize=(8, 5))
        for c in range(n_chains):
            color = CHAIN_COLORS[c % len(CHAIN_COLORS)]
            for k in np.where(~final)[0]:
                ax.scatter(x[keep], fluxes[c, k, p, keep], s=4, c="0.6", alpha=0.3, lw=0)
            for k in np.where(final)[0]:
                ax.scatter(x[keep], fluxes[c, k, p, keep], s=12, c=color, alpha=0.8, lw=0)
            ax.scatter([], [], c=color, label=f"chain {c} (t=0 states)")
        ax.scatter([], [], c="0.6", label="annealed (t>0) states")
        if nm.reference_flux is not None:
            ax.scatter(x[keep], nm.reference_flux[keep], c="k", marker="x", label="reference", zorder=5)
        ax.set_yscale("log")
        ax.set_xlabel("MJD" if nm.mjd is not None else "image")
        ax.set_ylabel("Flux")
        ax.set_title(f"{nm.name}: point source {p} light curve")
        ax.legend()
        path = nm.files.figure("sample_fluxes" + ("" if p == 0 else f"_ps{p}"))
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_flux_traces(nm, fluxes=None, trace=None):
    """Flux of each non-template image against saved step."""
    trace = nm.load_trace() if trace is None else trace
    n_chains = int(trace["num_chains"])
    fluxes = collect_fluxes(nm.files, n_chains) if fluxes is None else fluxes
    n_chains, n_saved, P, N = fluxes.shape
    saved_step = trace["saved_step"][:n_saved]
    keep = np.where(~nm.is_template)[0]
    paths = []
    for p in range(P):
        fig, ax = plt.subplots(figsize=(10, 5))
        for c in range(n_chains):
            color = CHAIN_COLORS[c % len(CHAIN_COLORS)]
            for n in keep:
                ax.plot(saved_step, fluxes[c, :, p, n], lw=0.6, color=color, alpha=0.5)
            ax.plot([], [], color=color, label=f"chain {c}")
        for step, label in phase_marks(trace):
            ax.axvline(step, color="k", ls=":", lw=0.6)
        ax.set_yscale("log")
        ax.set_xlabel("HMC step")
        ax.set_ylabel("Flux (one line per non-template image)")
        ax.set_title(f"{nm.name}: point source {p} flux traces")
        ax.legend()
        path = nm.files.figure("sample_flux_trace" + ("" if p == 0 else f"_ps{p}"))
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_parameter_traces(nm, trace=None):
    """
    Per-image nuisance parameters against saved step, one line per image per
    chain, in the style of the flux traces:
      {name}_sample_alignment_trace.pdf  image alignment offsets dx, dy (crtan minus
                                         the input WCS, arcsec; grey band = one
                                         data pixel) and the point source centre
      {name}_sample_nuisance_trace.pdf   scene scale, sky level and the Moffat PSF n, Rd, q, PA
    """
    trace = nm.load_trace() if trace is None else trace
    n_chains = int(trace["num_chains"])
    names = nm.target_names
    crtan0 = nm.crtan
    data_pix = nm.data_pixel_scale
    P = nm.n_sources
    align, centre, sky, scale, psf = [], [], [], [], {k: [] for k in ("n", "Rd", "q", "PA")}
    for c in range(n_chains):
        with h5py.File(nm.files.chain(c), "r") as h5:
            crtan = np.stack([read_param_history(h5, f"images/{name}/crtan/value") for name in names], axis=1)  # (saved, N, 2)
            align.append(crtan - crtan0[None])
            centre.append(np.concatenate([read_param_history(h5, f"ps_batch_model_{p}/model/center/value") for p in range(P)], axis=1))
            sky.append(read_param_history(h5, "sky_batch_model/model/I0/value"))
            scale.append(read_param_history(h5, "pixelated_batch_model/model/scale/value")[:, None])
            for k in psf:
                psf[k].append(read_param_history(h5, f"_psf/{k}/value"))
    saved_step = trace["saved_step"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    _trace_panel(axes[0], saved_step, [a[:, :, 0] for a in align], n_chains, trace, "dx [arcsec]", band=(-data_pix / 2, data_pix / 2))
    _trace_panel(axes[1], saved_step, [a[:, :, 1] for a in align], n_chains, trace, "dy [arcsec]", band=(-data_pix / 2, data_pix / 2))
    _trace_panel(axes[2], saved_step, centre, n_chains, trace, "point source centre x, y [arcsec]")
    axes[0].set_title(f"{nm.name}: image alignment offsets (one line per image; grey = one data pixel) and point source centre")
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("HMC step")
    p1 = nm.files.figure("sample_alignment_trace")
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(6, 1, figsize=(10, 16), sharex=True)
    _trace_panel(axes[0], saved_step, scale, n_chains, trace, "scene scale (model = scale x I)")
    _trace_panel(axes[1], saved_step, sky, n_chains, trace, "sky level [data units / arcsec^2]")
    _trace_panel(axes[2], saved_step, psf["n"], n_chains, trace, "PSF n")
    _trace_panel(axes[3], saved_step, psf["Rd"], n_chains, trace, "PSF Rd [arcsec]")
    _trace_panel(axes[4], saved_step, psf["q"], n_chains, trace, "PSF q")
    _trace_panel(axes[5], saved_step, psf["PA"], n_chains, trace, "PSF PA [rad]")
    axes[0].set_title(f"{nm.name}: scene scale, per-image sky and PSF traces (one line per image)")
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("HMC step")
    p2 = nm.files.figure("sample_nuisance_trace")
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    return [p1, p2]


def _pick_states(n_saved, max_panels):
    return np.unique(np.round(np.linspace(0, n_saved - 1, min(max_panels, n_saved))).astype(int))


def _panel_grid(nm, images, pick, saved_step, saved_t, scales, reference, vmin, vmax, title, what, marks=None):
    """images (chains, n_saved, H, W) in physical flux, shown at the picked
    states (log10, common colour scale), one row per chain; `reference`
    (chains?, H, W) adds a last column; `marks` (chains, n_saved, P, 2) are
    point-source positions in scene pixels (col, row) at each saved state,
    drawn as small open circles (thin, translucent, so the scene underneath
    stays visible)."""
    n_chains = images.shape[0]
    n_cols = len(pick) + (1 if reference is not None else 0)
    fig, axes = plt.subplots(n_chains, n_cols, figsize=(1.6 * n_cols, 1.8 * n_chains + 0.6), squeeze=False)
    for c in range(n_chains):
        for ax, k in zip(axes[c], pick):
            img = np.log10(np.clip(images[c, k], 1e-30, None))
            # interpolation="none": the PDF carries the array itself; any
            # resampling (matplotlib's default) adds moire at the beat period
            ax.imshow(img, cmap="inferno", origin="lower", vmin=vmin, vmax=vmax, interpolation="none")
            if marks is not None and k < marks.shape[1]:
                ax.plot(marks[c, k, :, 0], marks[c, k, :, 1], ls="none", marker="o", ms=3.5, mfc="none", mec="w", mew=0.4, alpha=0.7)
                ax.set_xlim(-0.5, img.shape[1] - 0.5)
                ax.set_ylim(-0.5, img.shape[0] - 0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            label = f"step {int(saved_step[k])}\nt={saved_t[k]:.2f}\n" if c == 0 else ""
            if scales is not None:
                label += f"scale {float(scales[c][k]):.3g}"
            ax.set_title(label, fontsize=7)
        if reference is not None:
            ax = axes[c, -1]
            ax.imshow(np.log10(np.clip(reference[min(c, len(reference) - 1)], 1e-30, None)), cmap="inferno", origin="lower", vmin=vmin, vmax=vmax, interpolation="none")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("prior's own\nsampler (pflow)" if c == 0 else "", fontsize=7)
        axes[c, 0].set_ylabel(f"chain {c}", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path = nm.files.figure(what)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return path


def scene_pixel_marks(nm, centres, scene_shape):
    """Point-source centres (..., P, 2) in tangent-plane arcsec to scene
    pixel coordinates (..., P, 2) = (col, row): the pixelated model is
    centred on ps_center[0] with x along the columns and y along the rows
    (init_scene.py has the convention and its checks)."""
    H, W = scene_shape
    pix = nm.scene_pixel_scale
    cxy = nm.ps_center[0]
    col = (centres[..., 0] - cxy[0]) / pix + (W - 1) / 2
    row = (centres[..., 1] - cxy[1]) / pix + (H - 1) / 2
    return np.stack([col, row], axis=-1)


def _scene_states(nm, trace, max_panels):
    """The saved scenes, scales and point-source marks of every chain, the
    picked panels and the common colour scale shared by the scene and
    Tweedie figures."""
    n_chains = int(trace["num_chains"])
    P = nm.n_sources
    scenes, scales, centres = [], [], []
    for c in range(n_chains):
        with h5py.File(nm.files.chain(c), "r") as h5:
            scenes.append(read_scene_history(h5))
            scales.append(read_param_history(h5, "pixelated_batch_model/model/scale/value"))
            centres.append(np.stack([read_param_history(h5, f"ps_batch_model_{p}/model/center/value") for p in range(P)], axis=1))
    n_saved = min(sc.shape[0] for sc in scenes)
    scenes = np.stack([sc[:n_saved] for sc in scenes])  # (chains, saved, H, W)
    marks = scene_pixel_marks(nm, np.stack([ce[:n_saved] for ce in centres]), scenes.shape[-2:])  # (chains, saved, P, 2)
    pick = _pick_states(n_saved, max_panels)
    reference = trace["pflow_reference"] if "pflow_reference" in trace else None  # prior-only runs
    # common log colour scale from the final states (and the reference), so
    # the early noise is shown on the scale of the final galaxy
    final = np.log10(np.clip(scenes[:, -1] if reference is None else np.concatenate([scenes[:, -1], reference]), 1e-30, None))
    vmin, vmax = np.percentile(final, [1, 99.8])
    return scenes, scales, marks, pick, reference, vmin, vmax


def plot_scenes(nm, trace=None, max_panels=16):
    """The scene at evenly spaced saved states (always the first and last),
    with the point-source position(s) at that state marked by a small open
    circle."""
    trace = nm.load_trace() if trace is None else trace
    scenes, scales, marks, pick, reference, vmin, vmax = _scene_states(nm, trace, max_panels)
    n_saved = scenes.shape[1]
    return _panel_grid(
        nm, scenes, pick, trace["saved_step"][:n_saved], trace["saved_t"][:n_saved], scales, reference, vmin, vmax,
        f"{nm.name}: scene I (log10, prior units, common colour scale) through the annealing; model = scale x I; o = point source",
        "sample_scene", marks=marks,
    )  # fmt: skip


def plot_tweedie(nm, trace=None, max_panels=16):
    """
    The Tweedie estimate of the clean scene, x + sigma_t^2 * score(x, t), at
    the same saved states and on the same colour scale as the scene figure.
    This is the scene the likelihood sees during the annealing (before the
    -v_t/2 mean shift); at the t = 0 states the sampler evaluates the
    likelihood on x itself, and the panel then shows what the denoiser would
    still do at sigma_min. Read from the sampler's Tweedie file; recomputed
    (one prior gradient per chain and state) only for runs made without it.
    """
    trace = nm.load_trace() if trace is None else trace
    _, _, marks, pick, _, vmin, vmax = _scene_states(nm, trace, max_panels)
    n_chains = int(trace["num_chains"])
    saved_step = trace["saved_step"]
    saved_t = trace["saved_t"]
    tw = nm.files.tweedie
    if tw.exists():
        # written by the sampler at every save: (saved, chains, H, W)
        images = np.transpose(np.load(tw)["scene_hat"], (1, 0, 2, 3)).astype(np.float64)
        if images.shape[1] <= int(max(pick)):
            pick = pick[pick < images.shape[1]]
    else:  # a run made without it: recompute, one prior gradient per (chain, state)
        log("no Tweedie file from the sampler; recomputing the estimates")
        model = nm.build_model()
        with ck.ValidContext(model):
            _, group1 = model.get_values()
        _, log_prior, _ = make_log_parts(model, jnp.asarray(group1, DTYPE))
        score_fn = jax.jit(jax.grad(log_prior))
        sigma_fn = make_sigma_fn(model.meta.host_prior.sde)
        pix = model.models.pixelated_batch_model.model
        scene_idx = jnp.asarray(block_indices(model, [pix.I]))
        images = np.full((n_chains, int(max(pick)) + 1) + tuple(pix.I.shape), np.nan)
        for c in range(n_chains):
            with h5py.File(nm.files.chain(c), "r") as h5:
                for k in pick:
                    load_chain_state(model, h5, int(k))
                    with ck.ValidContext(model):
                        x, _ = model.get_values()
                    x = jnp.asarray(x, DTYPE)
                    t = float(saved_t[k])
                    x_hat = x[scene_idx] + float(sigma_fn(t)) ** 2 * score_fn(x, t)[scene_idx]
                    images[c, k] = np.exp(np.asarray(x_hat, dtype=np.float64)).reshape(pix.I.shape)
    return _panel_grid(
        nm, images, pick, saved_step, saved_t, None, None, vmin, vmax,
        f"{nm.name}: Tweedie estimate exp(x + sigma_t^2 score) (log10, scene figure's colour scale): "
        "the scene the likelihood sees before the -v_t/2 mean shift; o = point source",
        "sample_tweedie", marks=marks,
    )  # fmt: skip


# Image-level fit diagnostics at one saved state
#####################################################################
def image_grid(n_images):
    """Figure and flattened axes with enough panels for n_images."""
    H = max(int(np.sqrt(n_images)), 1)
    W = int(np.ceil(n_images / H))
    fig, axarr = plt.subplots(H, W, figsize=(W * 2, H * 2), squeeze=False)
    axes = axarr.flatten()
    for ax in axes:
        ax.axis("off")
    return fig, axes


def plot_fit_state(nm, chain=0, index=-1):
    """Light curve, alignment shifts, host images, residual histogram and the
    target / model / residual image grids at saved state `index` of `chain`."""
    model = nm.build_model()
    with h5py.File(nm.files.chain(chain), "r") as h5:
        load_chain_state(model, h5, index)
    files = nm.files
    is_template = nm.is_template
    x = np.arange(nm.n_images) if nm.mjd is None else nm.mjd
    paths = []

    # Light curve at this state
    fluxes = get_fluxes(model)  # (P, N)
    log(f"fluxes all finite: {np.all(np.isfinite(fluxes))}")
    keep = ~is_template
    plt.scatter(x[keep], fluxes[0, keep], label="Novamac", c="tab:red", marker="o")
    if nm.reference_flux is not None:
        plt.scatter(x[keep], nm.reference_flux[keep], label="reference", c="k", marker="x")
    plt.yscale("log")
    plt.xlabel("MJD" if nm.mjd is not None else "image")
    plt.ylabel("Flux")
    plt.legend()
    paths.append(files.figure("fit_fluxes"))
    plt.savefig(paths[-1], bbox_inches="tight")
    plt.close()

    # Image alignment offsets relative to the input WCS
    crtan = np.stack([t.crtan.npvalue for t in model.target.images], axis=0)
    shifts = crtan - nm.crtan
    ps_center = model.models.ps_batch_model_0.model.center.npvalue
    pix = nm.data_pixel_scale
    plt.scatter(shifts[:, 0], shifts[:, 1], c="k")
    plt.scatter([ps_center[0]], [ps_center[1]], c="r", marker="x", s=100, label="point source")
    plt.gca().add_patch(patches.Rectangle((-pix / 2, -pix / 2), pix, pix, linewidth=2, edgecolor="k", facecolor="none"))
    plt.axis("equal")
    plt.xlabel("delta x [arcsec]")
    plt.ylabel("delta y [arcsec]")
    plt.title("Fitted target position shifts (box = one data pixel)")
    plt.legend()
    paths.append(files.figure("fit_target_shifts"))
    plt.savefig(paths[-1], bbox_inches="tight")
    plt.close()

    # Pixelated host model
    core_pix = model.models.pixelated_batch_model.model.I.npvalue
    log(f"scene scale factor: {float(model.models.pixelated_batch_model.model.scale.npvalue):.4g} (model = scale x I)")
    for scale, img in [("linear", core_pix), ("log", np.log10(np.clip(core_pix, 1e-30, None)))]:
        plt.imshow(img, cmap="inferno", origin="lower", vmin=0 if scale == "linear" else None, interpolation="none")
        plt.colorbar()
        plt.title(f"Pixelated host model ({scale} scale)")
        paths.append(files.figure(f"core_pix_{scale}", "png"))
        plt.savefig(paths[-1], dpi=300)
        plt.close()

    # Residual histogram (unmasked pixels, normalised by sigma)
    with Timer("evaluate model"):
        model_images = model()
    residuals = (model.target.flatten("data") - model_images.flatten("data")) * np.sqrt(model.target.flatten("weight"))
    residuals = np.array(residuals[~model.target.flatten("mask")])
    hist, bins = np.histogram(np.clip(residuals, -5, 5), range=(-5, 5), bins=50, density=True)
    plt.bar(bins[:-1], hist, width=bins[1] - bins[0], align="edge", color="k", linewidth=0, label="Residuals")
    xx = np.linspace(-5, 5, 1000)
    plt.plot(xx, np.exp(-0.5 * xx**2) / np.sqrt(2 * np.pi), color="r", label="Normal dist")
    plt.legend()
    plt.xlabel("Residual [(data - model) / sigma]")
    plt.title(f"Residual histogram (std {np.std(residuals):.3f})")
    paths.append(files.figure("residual_histogram"))
    plt.savefig(paths[-1])
    plt.close()
    log(f"normalised residual std {np.std(residuals):.3f} over {residuals.size} unmasked pixels")

    # Image grids: target, model, residual
    n_images = len(model.target.images)
    with Timer("plot images"):
        for name, plotter in (
            ("target", lambda fig, axes: ap.plots.target_image(fig, axes, model.target)),
            ("fit_model", lambda fig, axes: ap.plots.model_image(fig, axes, model, showcbar=False)),
            ("fit_residual", lambda fig, axes: ap.plots.residual_image(fig, axes, model, showcbar=False, normalize_residuals=True, scaling="clip")),
        ):
            fig, axes = image_grid(n_images)
            plotter(fig, axes[:n_images])
            for ax, t in zip(axes, model.target.images):
                ax.set_title(t.name, fontsize=6)
            paths.append(files.figure(name))
            fig.savefig(paths[-1])
            plt.close(fig)
    return paths


def plot_all(nm, raise_errors=False):
    """Every figure from the products on disk; returns the paths written.
    Each figure is attempted even if an earlier one failed (the error is
    logged) unless raise_errors."""
    if not nm.files.exists():
        log(f"{nm.name}: no trace file, nothing to plot")
        return []
    trace = nm.load_trace()
    n_chains = int(trace["num_chains"])
    paths = []

    def attempt(what, fn):
        try:
            with Timer(what):
                out = fn()
            if out is not None:
                paths.extend(out if isinstance(out, list) else [out])
        except Exception:  # noqa: BLE001
            if raise_errors:
                raise
            log(f"{what} failed:\n{traceback.format_exc()}")

    attempt("progress plot", lambda: plot_progress(nm, trace))
    attempt("scene progression plot", lambda: plot_scenes(nm, trace))
    attempt("Tweedie progression plot", lambda: plot_tweedie(nm, trace))
    fluxes = collect_fluxes(nm.files, n_chains)
    log(f"fluxes shape (chains, saved, P, N) = {fluxes.shape}, all finite: {np.all(np.isfinite(fluxes))}")
    conditional = conditional_fluxes(trace)
    if conditional is not None:
        for p in sorted(k for k in conditional if isinstance(k, int)):
            imgs = sorted(conditional[p])
            log(f"source {p}: conditional flux over {conditional['n_states']} hold states, first/last image "
                f"{conditional[p][imgs[0]][0]:.4g} +- {conditional[p][imgs[0]][1]:.3g} / {conditional[p][imgs[-1]][0]:.4g} +- {conditional[p][imgs[-1]][1]:.3g}")
    attempt("light curve", lambda: plot_lightcurve(nm, fluxes, trace))
    attempt("flux traces", lambda: plot_flux_traces(nm, fluxes, trace))
    attempt("parameter traces", lambda: plot_parameter_traces(nm, trace))
    attempt("fit diagnostics at the last saved state", lambda: plot_fit_state(nm))
    return paths
