"""
End-to-end runs on the synthetic fixtures through every stage of the
sampler (warmup, anneal with every block released and the non-scene HMC on,
the Gibbs refresh, the hold, the restart check), then every plot and both
diagnostics, with the outputs checked for existence and finite values. Not
a statistical test: the prior is random weights. The posterior run takes a
couple of minutes on a CPU (jit compiles dominate); the prior-only and
frozen test modes are marked slow.
"""

import numpy as np
import pytest

from conftest import FAST_SETTINGS
from novamac import Novamac
from novamac.products import RunFiles, collect_fluxes

FIGURES = (
    "sample_progress", "sample_scene", "sample_tweedie", "sample_fluxes", "sample_flux_trace",
    "sample_alignment_trace", "sample_nuisance_trace", "fit_fluxes", "fit_target_shifts",
    "residual_histogram", "target", "fit_model", "fit_residual",
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
