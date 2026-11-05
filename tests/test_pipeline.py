"""
End-to-end runs on the synthetic fixtures through every stage of the
sampler (warmup, anneal with every block released and the non-scene HMC on,
the Gibbs refresh, the hold, the restart check), then every plot and both
diagnostics, with the outputs checked for existence and finite values. Not
a statistical test: the prior is random weights. The posterior run takes a
couple of minutes on a CPU (jit compiles dominate); the prior-only and
frozen test modes are marked slow.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from conftest import FAST_SETTINGS
from novamac import Novamac
from novamac._backend import DTYPE
from novamac.products import RunFiles, collect_fluxes

FIGURES = (
    "sample_progress", "sample_scene", "sample_tweedie", "sample_fluxes", "sample_flux_trace",
    "sample_alignment_trace", "sample_nuisance_trace", "fit_fluxes", "fit_target_shifts",
    "residual_histogram", "target", "fit_model", "fit_residual", "fit_psf",
)  # fmt: skip


def _check_run(nm, trace):
    s = FAST_SETTINGS
    assert int(trace["steps_done"]) == s["warmup_steps"] + s["anneal_steps"] + s["hold_steps"]
    assert np.all(np.isfinite(trace["logdensity"]))
    assert nm.files.tweedie.exists() and nm.files.timings.exists()
    for c in range(s["num_chains"]):
        assert nm.files.chain(c).exists()
    fluxes = collect_fluxes(nm.files, s["num_chains"])
    assert fluxes.shape[0] == s["num_chains"] and fluxes.shape[1] == len(trace["saved_step"])
    assert np.all(np.isfinite(fluxes))


def test_posterior_run_plots_and_diagnostics(tmp_path, prior_dir, synthetic):
    inputs, truth = synthetic
    nm = Novamac(prior=prior_dir, out_dir=tmp_path / "out", name="syn_g", **FAST_SETTINGS, **inputs)
    trace = nm.sample()
    _check_run(nm, trace)
    releasable = np.isfinite(trace["block_t_earliest"])
    assert np.all(trace["block_release_step"][releasable] >= 0), "every releasable block released"
    assert np.any(np.isfinite(trace["accept_prob_nuis"])), "non-scene HMC ran"
    assert np.any(np.isfinite(trace["gibbs_accept"])), "Gibbs refresh ran"
    assert "prior_offset_value" in trace
    assert "settings" in trace
    # the draw phase: kernels fixed after hold_adapt_chunks chunks of the hold, no restarts after that
    s = FAST_SETTINGS
    hold_start = s["warmup_steps"] + s["anneal_steps"]
    expected = min(int(trace["steps_done"]), hold_start + Novamac.hold_adapt_chunks * s["chunk_steps"])
    assert int(trace["draws_from_step"]) == expected
    n_chunks = len(trace["chunk_step"])
    frozen = trace["chunk_step"] >= expected
    assert np.all(trace["step_size"][n_chunks - 1] == trace["step_size"][np.argmax(frozen)]) if frozen.any() else True
    assert all(int(st) <= expected for st in trace["restart_step"])
    # the re-noise proposals: at every renoise_every steps before the hold, with a finite ratio per chain
    rn = np.asarray(trace["renoise_step"], dtype=int)
    assert list(rn) == [st for st in range(s["renoise_every"], hold_start, s["renoise_every"])]
    assert np.asarray(trace["renoise_accept"]).shape == (len(rn), s["num_chains"])
    assert np.all(np.isfinite(np.asarray(trace["renoise_log_alpha"])))
    assert np.all(np.isfinite(trace["chunk_logdensity"]))

    paths = nm.plot_all(raise_errors=True)
    for what in FIGURES:
        ext = "png" if what.startswith("core_pix") else "pdf"
        assert nm.files.figure(what, ext).exists(), f"{what} written"
    assert nm.files.figure("core_pix_log", "png").exists()
    assert len(paths) >= len(FIGURES)

    lines = nm.describe_init()
    assert any("initial scene" in line for line in lines) and any("saved state" in line for line in lines)
    lines = nm.diagnose_stall(n_last=2)
    assert any("frozen" in line for line in lines)

    # a second process would construct the same Novamac and plot from the products
    again = Novamac(prior=prior_dir, out_dir=tmp_path / "out", name="syn_g", **FAST_SETTINGS, **inputs)
    assert again.files.exists() and again.files.num_chains() == FAST_SETTINGS["num_chains"]
    assert again.plot_progress().exists()
    assert RunFiles(tmp_path / "out", "syn_g").trace == nm.files.trace


@pytest.mark.slow
def test_prior_only_run(tmp_path, prior_dir, synthetic):
    inputs, _ = synthetic
    nm = Novamac(prior=prior_dir, out_dir=tmp_path / "out", name="syn_g_prioronly", likelihood=False, **FAST_SETTINGS, **inputs)
    trace = nm.sample()
    _check_run(nm, trace)
    assert "pflow_reference" in trace
    assert np.all(trace["block_release_step"] < 0), "nothing released without a likelihood"
    nm.plot_all(raise_errors=True)
    assert nm.files.figure("sample_scene").exists()


@pytest.mark.slow
def test_frozen_run(tmp_path, prior_dir, synthetic):
    inputs, _ = synthetic
    nm = Novamac(prior=prior_dir, out_dir=tmp_path / "out", name="syn_g_frozen", freeze_non_scene=True, **FAST_SETTINGS, **inputs)
    trace = nm.sample()
    _check_run(nm, trace)
    assert np.all(trace["block_release_step"] < 0)
    assert not np.any(np.isfinite(trace["accept_prob_nuis"]))
    nm.plot_all(raise_errors=True)


@pytest.mark.slow
def test_plateau_run(tmp_path, prior_dir, synthetic):
    inputs, _ = synthetic
    s = dict(FAST_SETTINGS, plateau_steps=16, plateau_at=[0.7, 0.3], frozen_blocks=("sky",))
    nm = Novamac(prior=prior_dir, out_dir=tmp_path / "out", name="syn_g_plateau", **s, **inputs)
    trace = nm.sample()
    assert int(trace["steps_done"]) == s["warmup_steps"] + s["anneal_steps"] + 2 * s["plateau_steps"] + s["hold_steps"]
    assert int(trace["anneal_steps"]) == s["anneal_steps"] + 2 * s["plateau_steps"]
    starts = np.asarray(trace["plateau_step"], dtype=int)
    assert len(starts) == 2 and starts[0] < starts[1]
    assert s["warmup_steps"] < starts[0] and starts[1] + s["plateau_steps"] <= s["warmup_steps"] + int(trace["anneal_steps"])
    t_start = float(trace["t_start"])
    assert 0 < trace["plateau_t"][1] <= 0.3 * t_start < trace["plateau_t"][0] <= 0.7 * t_start
    # the predictor is idle on the plateaus: no scene displacement by it
    for p0 in starts:
        assert np.allclose(trace["predictor_rms"][p0 : p0 + s["plateau_steps"]], 0.0)
    assert int(trace["draws_from_step"]) > starts[1] + s["plateau_steps"]
    assert np.all(np.isfinite(trace["logdensity"]))
    names = list(trace["block_names"]) if "block_names" in trace else []
    if "sky" in names:
        assert trace["block_release_step"][names.index("sky")] < 0, "the frozen sky block is never released"
    nm.plot_progress()
    assert nm.files.figure("sample_progress").exists()


def test_prior_t_min_remaps_the_prior_time(tmp_path, prior_dir, synthetic):
    """prior_t_min: the model's log prior at t is the plain model's at
    t_a = prior_t_min + (1 - prior_t_min) t, and the sampler's schedule
    (model.meta.prior_sde) has sigma_min = sigma(prior_t_min)."""
    import caskade as ck
    from novamac.hmc import make_log_parts, sampler_sde

    inputs, _ = synthetic
    models = {}
    for t_min in (0.0, 0.1):
        nm = Novamac(prior=prior_dir, out_dir=tmp_path / f"out{t_min}", name="syn_g", **dict(FAST_SETTINGS, prior_t_min=t_min), **inputs)
        models[t_min] = nm.build_model()
    plain, floor = models[0.0], models[0.1]
    host = plain.meta.host_prior.sde
    assert sampler_sde(plain).sigma_min == pytest.approx(host.sigma_min)
    assert sampler_sde(floor).sigma_min == pytest.approx(float(host.sigma(0.1)))
    assert sampler_sde(floor).sigma_max == pytest.approx(host.sigma_max)
    for t in (0.0, 0.5, 1.0):
        assert float(sampler_sde(floor).sigma(t)) == pytest.approx(float(host.sigma(0.1 + 0.9 * t)), rel=1e-6)
    with ck.ValidContext(plain):
        x0, group1 = plain.get_values()
    x, group1 = jnp.asarray(x0, DTYPE), jnp.asarray(group1, DTYPE)
    _, lp_plain, _ = make_log_parts(plain, group1)
    _, lp_floor, _ = make_log_parts(floor, group1)
    for t in (0.0, 0.5, 1.0):
        assert float(lp_floor(x, t)) == pytest.approx(float(lp_plain(x, 0.1 + 0.9 * t)), rel=1e-5)
    with pytest.raises(ValueError, match="prior_t_min"):
        Novamac(prior=prior_dir, out_dir=tmp_path / "bad", name="syn_g", **dict(FAST_SETTINGS, prior_t_min=1.0), **inputs).build_model()


def test_predictor_only_ramps_below_a_t_threshold(tmp_path, prior_dir, synthetic):
    """ramp_predictor_below: only the ramps starting at or below it are
    predictor-only, the ones above keep the corrector; with prior_t_min set."""
    inputs, _ = synthetic
    s = dict(
        FAST_SETTINGS, plateau_steps=16, plateau_at=[0.75, 0.5, 0.25], ramp_predictor_only=True,
        ramp_predictor_below=0.6, prior_t_min=0.1, frozen_blocks=("sky",),
    )  # fmt: skip
    nm = Novamac(prior=prior_dir, out_dir=tmp_path / "out", name="syn_g_ramps_low", **s, **inputs)
    trace = nm.sample()
    assert np.all(np.isfinite(trace["logdensity"])) and float(trace["prior_t_min"]) == 0.1
    t = np.asarray(trace["t"])
    t_prev = np.concatenate([[t[0]], t[:-1]])
    n = s["chunk_steps"]
    # a chunk is predictor-only when it ramps and starts at or below 0.6 (the
    # plateau t it ramps down from; a ramp that merely passes 0.6 is not), so
    # the first ramp (from t_start, above 0.6) keeps the corrector
    starts = np.array([t_prev[c] for c in range(0, len(t), n) if t[c] < t_prev[c]])
    assert starts.max() > 0.6 and 0 < np.sum(starts <= 0.6) < len(starts)
    pred = np.concatenate([np.repeat((t[c] < t_prev[c]) & (t_prev[c] <= 0.6), n) for c in range(0, len(t), n)])
    assert pred.sum() == n * np.sum(starts <= 0.6)
    acc = np.asarray(trace["accept_prob"])
    assert np.all(np.isnan(acc[pred])) and np.all(np.isfinite(acc[~pred]))
    assert int(np.sum(trace["predictor_chunks"])) == int(np.sum(starts <= 0.6))


def test_predictor_only_ramp_run(tmp_path, prior_dir, synthetic):
    """ramp_predictor_only: the chunks where t decreases are the prior's
    reverse diffusion alone (scene displaced, no acceptance, no Gibbs refresh,
    no re-noise proposal), the plateaus, warmup and hold sample as usual."""
    inputs, _ = synthetic
    s = dict(FAST_SETTINGS, plateau_steps=16, plateau_at=[0.75, 0.5, 0.25], ramp_predictor_only=True, frozen_blocks=("sky",))
    nm = Novamac(prior=prior_dir, out_dir=tmp_path / "out", name="syn_g_ramps", **s, **inputs)
    trace = nm.sample()
    total = s["warmup_steps"] + s["anneal_steps"] + 3 * s["plateau_steps"] + s["hold_steps"]
    assert int(trace["steps_done"]) == total and np.all(np.isfinite(trace["logdensity"]))
    t = np.asarray(trace["t"])
    ramp = np.concatenate([[False], t[1:] < t[:-1]])
    assert ramp.sum() == s["anneal_steps"]
    acc, prms = np.asarray(trace["accept_prob"]), np.asarray(trace["predictor_rms"])
    assert np.all(np.isnan(acc[ramp])) and np.all(prms[ramp] > 0), "ramp steps are predictor-only"
    assert np.all(np.isfinite(acc[~ramp])) and np.allclose(prms[~ramp], 0.0, atol=1e-5), "fixed-t steps are HMC without predictor"
    n = s["chunk_steps"]
    ramp_chunk_ends = {int(e) for e in np.arange(n, total + 1, n) if ramp[e - 1]}
    assert not ramp_chunk_ends & {int(e) for e in trace["renoise_step"]}, "no re-noise proposal on a ramp chunk"
    if "gibbs_accept" in trace:
        g = np.asarray(trace["gibbs_accept"], dtype=float)
        for e in ramp_chunk_ends:
            assert np.all(np.isnan(g[e // n - 1])), "no Gibbs refresh on a ramp chunk"


def test_nuisance_log_lik_matches_the_masked_full_gradient(tmp_path, prior_dir, synthetic):
    """make_nuisance_log_lik: same likelihood value as log_lik, its gradient
    with respect to the non-scene coordinates equals the full gradient's
    non-scene entries, and the model's parameter groups are restored."""
    import jax
    import caskade as ck
    from novamac.hmc import block_indices, make_log_parts, make_nuisance_log_lik

    inputs, _ = synthetic
    nm = Novamac(prior=prior_dir, out_dir=tmp_path / "out", name="syn_g", **FAST_SETTINGS, **inputs)
    model = nm.build_model()
    with ck.ValidContext(model):
        x0, group1 = model.get_values()
    x = jnp.asarray(x0, DTYPE)
    group1 = jnp.asarray(group1, DTYPE)
    pix = model.models.pixelated_batch_model.model
    scene_idx = block_indices(model, [pix.I])
    nuis_idx = np.setdiff1d(np.arange(x.shape[0]), scene_idx)
    log_lik, _, _ = make_log_parts(model, group1)
    log_lik_nuis = make_nuisance_log_lik(model, group1, pix.I)
    ev = jnp.zeros(())
    v_full, g_full = jax.value_and_grad(log_lik)(x, ev)
    vg = jax.jit(jax.value_and_grad(lambda y_n, scene: log_lik_nuis(y_n, scene, ev)))
    v_n, g_n = vg(x[nuis_idx], x[scene_idx])
    assert float(v_n) == pytest.approx(float(v_full), rel=1e-6)
    np.testing.assert_allclose(np.asarray(g_n), np.asarray(g_full)[nuis_idx], rtol=1e-5, atol=1e-6 * float(jnp.abs(g_full).max()))
    assert pix.I.group == 0 and model.dynamic_param_groups == (0, 1)
    assert float(log_lik(x + 0.01, ev)) == pytest.approx(float(jax.value_and_grad(log_lik)(x + 0.01, ev)[0]))


def test_scene_log_lik_matches_the_masked_full_gradient(tmp_path, prior_dir, synthetic):
    """make_scene_log_lik: same likelihood value as log_lik, its gradient with
    respect to the scene equals the full gradient's scene entries, and the
    model's parameter groups are restored."""
    import jax
    import caskade as ck
    from novamac.hmc import block_indices, make_log_parts, make_scene_log_lik

    inputs, _ = synthetic
    nm = Novamac(prior=prior_dir, out_dir=tmp_path / "out", name="syn_g", **FAST_SETTINGS, **inputs)
    model = nm.build_model()
    with ck.ValidContext(model):
        x0, group1 = model.get_values()
    x = jnp.asarray(x0, DTYPE) + 0.05
    group1 = jnp.asarray(group1, DTYPE)
    pix = model.models.pixelated_batch_model.model
    scene_idx = block_indices(model, [pix.I])
    nuis_idx = np.setdiff1d(np.arange(x.shape[0]), scene_idx)
    log_lik, _, _ = make_log_parts(model, group1)
    log_lik_scene = make_scene_log_lik(model, group1, pix.I)
    ev = jnp.zeros(())
    v_full, g_full = jax.value_and_grad(log_lik)(x, ev)
    vg = jax.jit(jax.value_and_grad(lambda s, y_n: log_lik_scene(s, y_n, ev)))
    v_s, g_s = vg(x[scene_idx], x[nuis_idx])
    assert float(v_s) == pytest.approx(float(v_full), rel=1e-6)
    np.testing.assert_allclose(np.asarray(g_s), np.asarray(g_full)[scene_idx], rtol=1e-5, atol=1e-6 * float(jnp.abs(g_full).max()))
    assert pix.I.group == 0 and model.dynamic_param_groups == (0, 1)
    assert all(p.group in (0, 1) for p in model.dynamic_params)
    assert float(log_lik(x + 0.01, ev)) == pytest.approx(float(jax.value_and_grad(log_lik)(x + 0.01, ev)[0]))
