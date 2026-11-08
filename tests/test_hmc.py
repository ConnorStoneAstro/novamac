"""
The annealing machinery on small problems with known answers (CPU, seconds).

The toy anneal mirrors the real problem minus the nonlinearity:
    prior     x ~ N(mu_p, diag(sd_p^2)),  annealed: N(mu_p, diag(sd_p^2 + sigma_t^2))
    data      y = A x + n,  n ~ N(0, diag(sd_n^2))
    annealed likelihood: N(y; A x, diag(sd_n^2 + sigma_t^2 * diag(A A^T)))
with sigma_t = sigma_min (sigma_max / sigma_min)^t, the VE schedule. At t = 0
the posterior is Gaussian with known mean and covariance. Many independent
chains are run and only their final states are kept; the test is whether
those final states are distributed like the exact posterior.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from novamac.hmc import (
    adapt_step_size,
    fold_into_box,
    hutchinson_jj_diag,
    make_broadening_fn,
    make_chunk_runner,
    make_gauss_newton_fn,
    make_linear_gibbs_fn,
    make_predictor_runner,
    make_target,
    release_inverse_mass,
    strided_pixel_groups,
)

D = 5
SIGMA_MIN, SIGMA_MAX = 0.01, 10.0


def sigma_fn(t):
    return SIGMA_MIN * (SIGMA_MAX / SIGMA_MIN) ** t


def broaden_fn(t):
    return sigma_fn(t) ** 2  # linear model: the delta method is exact


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    prior_mean = np.array([0.0, 1.0, -1.0, 2.0, 0.5], np.float32)
    prior_sd = np.array([1.0, 3.0, 0.3, 2.0, 1.0], np.float32)
    A = rng.normal(size=(D, D)).astype(np.float32)
    A[0, 1] += 2.0  # a strongly correlated pair
    noise_sd = np.array([0.5, 0.2, 1.0, 0.3, 0.8], np.float32)
    x_true = prior_mean + prior_sd * rng.normal(size=D).astype(np.float32)
    y = A @ x_true + noise_sd * rng.normal(size=D).astype(np.float32)
    prec = np.diag(1 / prior_sd**2) + A.T @ np.diag(1 / noise_sd**2) @ A
    post_cov = np.linalg.inv(prec.astype(np.float64))
    post_mean = post_cov @ (A.T @ (y / noise_sd**2) + prior_mean / prior_sd**2)
    A_j, y_j, pm_j, ps_j, ns_j = (jnp.asarray(a) for a in (A, y, prior_mean, prior_sd, noise_sd))

    def log_lik(x, extra_var):
        return -0.5 * jnp.sum((y_j - A_j @ x) ** 2 / (ns_j**2 + extra_var))

    def log_prior(x, t):
        return -0.5 * jnp.sum((x - pm_j) ** 2 / (ps_j**2 + sigma_fn(t) ** 2))

    images = lambda x: A_j @ x  # noqa: E731
    return dict(A=A, y=y, prior_mean=prior_mean, prior_sd=prior_sd, noise_sd=noise_sd, x_true=x_true, post_cov=post_cov,
                post_mean=post_mean, log_lik=log_lik, log_prior=log_prior, images=images)  # fmt: skip


def test_hutchinson_matches_exact_jacobian_diagonal(toy):
    key = jax.random.key(1)
    x0 = jnp.asarray(toy["prior_mean"])
    jj = hutchinson_jj_diag(toy["images"], x0, np.arange(D), key, num_probes=1024, batch=64)
    exact = np.sum(toy["A"] ** 2, axis=1)
    assert np.allclose(np.asarray(jj), exact, rtol=0.2)


def test_gauss_newton_is_exact_on_decoupled_groups(toy):
    gn = make_gauss_newton_fn(toy["images"], [[i] for i in range(D)])
    w = jnp.asarray(1 / toy["noise_sd"] ** 2)
    curv = np.asarray(gn(jnp.asarray(toy["prior_mean"]), w))
    exact = np.diag(toy["A"].T @ np.diag(1 / toy["noise_sd"] ** 2) @ toy["A"])
    assert np.allclose(curv, exact, rtol=1e-4)


def test_adapt_step_size_and_release_mass():
    eps = np.array([1.0, 1.0, 1.0])
    out = adapt_step_size(eps, np.array([0.2, 0.65, 0.95]), target=0.65, tol=0.05, down=0.5, up=2.0)
    assert np.allclose(out, [0.5, 1.0, 2.0])
    assert np.allclose(release_inverse_mass(np.array([4.0, 0.0, -1.0]), np.array([9.0, 9.0, 9.0])), [0.25, 9.0, 9.0])


def test_strided_pixel_groups_partition():
    idx = np.arange(100, 164)  # an 8x8 grid of flat indices
    groups = strided_pixel_groups(idx, (8, 8), 4)
    assert len(groups) == 16
    assert sorted(np.concatenate(groups).tolist()) == idx.tolist()
    rows, cols = np.divmod(groups[0] - 100, 8)
    assert np.all(rows % 4 == 0) and np.all(cols % 4 == 0)


def test_target_tweedie_shift_vanishes_at_t0(toy):
    target = make_target(toy["log_lik"], toy["log_prior"], sigma_fn, jnp.ones(D, jnp.float32), tweedie=True)
    x = jnp.asarray(toy["prior_mean"]) + 0.3
    assert np.allclose(np.asarray(target.denoise(x, 0.0)), np.asarray(x))
    (val, (ll, lp, score)), force = target.value_and_grad(x, 0.0, broaden_fn(0.0))
    exact = jax.value_and_grad(lambda z: toy["log_lik"](z, broaden_fn(0.0)) + toy["log_prior"](z, 0.0))(x)
    assert np.isclose(float(val), float(exact[0]), rtol=1e-5)
    assert np.allclose(np.asarray(force), np.asarray(exact[1]), rtol=1e-4, atol=1e-5)
    # at t > 0 the Tweedie map on a Gaussian prior is the linear shrinkage toward the prior mean
    t = 0.5
    s2 = sigma_fn(t) ** 2
    expected = x + s2 * (-(x - toy["prior_mean"]) / (toy["prior_sd"] ** 2 + s2))
    assert np.allclose(np.asarray(target.denoise(x, t)), expected, rtol=1e-4)


def _run_anneal(toy, chains, warmup, anneal, chunk, leapfrog, seed=0, t_freeze=0.3, predictor=True):
    """The schedule, chunked runner, step-size search and adaptation as Novamac.sample uses them."""
    t_start = float(np.log(2.0 / SIGMA_MIN) / np.log(SIGMA_MAX / SIGMA_MIN))
    ts = np.concatenate([np.full(warmup, t_start), t_start * (1 - np.arange(1, anneal + 1) / anneal), np.zeros(1)]).astype(np.float32)
    ts_prev = np.concatenate([[ts[0]], ts[:-1]])
    total = len(ts)
    bounds = [(s, min(s + chunk, total)) for s in range(0, total, chunk)]
    t_release = np.full(D, np.inf, np.float32)
    t_release[4] = t_freeze
    scene_mask = np.ones(D, np.float32)
    scene_mask[4] = 0.0  # a non-diffused ("nuisance") coordinate: the predictor leaves it, HMC moves it once released
    target = make_target(toy["log_lik"], toy["log_prior"], sigma_fn, scene_mask, tweedie=True)
    gn = make_gauss_newton_fn(toy["images"], [[i] for i in range(D)])
    curv = np.asarray(gn(jnp.asarray(toy["prior_mean"]), jnp.asarray(1 / toy["noise_sd"] ** 2))) + 1 / toy["prior_sd"] ** 2
    inv_mass0 = 1.0 / curv
    jj = jnp.asarray(np.sum(toy["A"] ** 2, axis=1), jnp.float32)
    x0 = jnp.asarray(toy["prior_mean"]).at[4].set(float(toy["x_true"][4]))
    positions = jnp.tile(x0, (chains, 1))
    runners = {n: make_chunk_runner(target, broaden_fn, sigma_fn, leapfrog, n, 0.5, 0.0, predictor=predictor) for n in {b - a for a, b in bounds} | {1}}
    dev = lambda a: jnp.asarray(a, jnp.float32)  # noqa: E731
    run = lambda n, keys, x, eps, t_arr, tp_arr: runners[n](keys, x, dev(eps), dev(eps), dev(inv_mass0), dev(t_arr), dev(tp_arr), jj, dev(t_release), dev(scene_mask))  # noqa: E731
    key = jax.random.key(seed)
    step_size = np.full(chains, 0.1)
    direction = np.zeros(chains)
    done = np.zeros(chains, bool)
    for _ in range(30):
        key, k = jax.random.split(key)
        _, tr = run(1, jax.random.split(k, chains), positions, step_size, [t_start], [t_start])
        acc = np.asarray(tr["accept_prob"])[:, 0]
        new_dir = np.where(acc > 0.5, 1.0, -1.0)
        done |= (direction != 0) & (new_dir != direction)
        direction = np.where(done, direction, new_dir)
        step_size = np.where(done, step_size, step_size * np.where(direction > 0, 2.0, 0.5))
        if done.all():
            break
    frozen_moved = False
    for s0, s1 in bounds:
        key, k = jax.random.split(key)
        before = np.asarray(positions[:, 4])
        positions, tr = run(s1 - s0, jax.random.split(k, chains), positions, step_size, ts[s0:s1], ts_prev[s0:s1])
        moved = not np.allclose(np.asarray(positions[:, 4]), before)
        if ts[s1 - 1] > t_freeze:
            assert not moved, "a frozen coordinate moved"
        frozen_moved |= moved
        assert np.all(np.isfinite(np.asarray(tr["logdensity"])))
        acceptance = np.asarray(tr["accept_prob"]).mean(axis=1)
        step_size = adapt_step_size(step_size, acceptance, 0.65, 0.05, 0.5 if s1 <= warmup else 0.7, 1.5 if s1 <= warmup else 1.1)
    assert frozen_moved, "the released coordinate never moved"
    return np.asarray(positions, np.float64)


def test_annealed_final_states_match_exact_posterior(toy):
    C = 384
    X = _run_anneal(toy, chains=C, warmup=64, anneal=256, chunk=64, leapfrog=5)
    L = np.linalg.cholesky(toy["post_cov"])
    Z = np.linalg.solve(L, (X - toy["post_mean"]).T).T  # whitened: should be N(0, I)
    assert np.all(np.abs(Z.mean(0)) < 5 / np.sqrt(C)), Z.mean(0)
    assert np.all(np.abs(Z.std(0) - 1) < 0.2), Z.std(0)
    r2 = np.sum(Z**2, axis=1)
    assert abs(np.mean(r2 < 5.0) - 0.584) < 0.1  # chi2_5 cdf at 5 ~ 0.584


def test_linear_gibbs_draws_from_the_conditional():
    """A diagonal linear model: each 'image' pixel depends on one coordinate,
    so the Gibbs refresh is an exact Gaussian conditional draw of the
    likelihood, Metropolis-corrected by the (weak) prior. Iterated over many
    chains its draws must match the exact posterior; the conditional mean and
    covariance it reports must match the exact values."""
    K = 4
    a = jnp.asarray([2.0, 0.5, 1.0, 3.0])
    y = jnp.asarray([1.0, -2.0, 0.5, 3.0])
    noise_sd = 0.3
    prior_mean = np.zeros(K)
    prior_prec = np.full(K, 0.05)
    images = lambda x: a * x  # noqa: E731
    lin = {
        "idx": np.arange(K), "group": np.arange(K), "image": np.arange(K), "is_log": np.zeros(K, bool),
        "prior_mean": prior_mean, "prior_prec": prior_prec, "source": np.full(K, -1), "pix_image": np.arange(K), "n_images": K,
    }  # fmt: skip
    gibbs = make_linear_gibbs_fn(images, y, jnp.full(K, 1 / noise_sd**2), lambda t: 0.0 * t, lin)
    C = 4000
    key = jax.random.key(0)
    x = jnp.zeros((C, K), jnp.float32)
    x_new, frac, c_mean, c_cov = gibbs(jax.random.split(key, C), x, 0.0, jnp.zeros(K), jnp.ones(K))
    exact_prec = np.asarray(a) ** 2 / noise_sd**2
    exact_cov = 1 / exact_prec
    exact_mean = np.asarray(a) * np.asarray(y) / noise_sd**2 * exact_cov
    assert np.allclose(np.asarray(c_mean)[0], exact_mean, rtol=1e-3)
    assert np.allclose(np.diag(np.asarray(c_cov)[0]), exact_cov, rtol=1e-3)
    # the proposal is the likelihood's conditional; the MH test with the prior moves the
    # draws toward the posterior N(post_mean, post_var)
    post_prec = exact_prec + prior_prec
    post_mean = (exact_prec * exact_mean + prior_prec * prior_mean) / post_prec
    assert float(np.mean(np.asarray(frac))) > 0.5
    for i in range(30):  # iterate so every chain forgets its start whatever the acceptance
        key, k = jax.random.split(key)
        x_new, frac, _, _ = gibbs(jax.random.split(k, C), x_new, 0.0, jnp.zeros(K), jnp.ones(K))
    draws = np.asarray(x_new, np.float64)
    assert np.allclose(draws.mean(0), post_mean, atol=4 / np.sqrt(C * post_prec) + 0.02)
    assert np.allclose(draws.var(0), 1 / post_prec, rtol=0.15)


def test_fold_into_box_is_an_involution_with_momentum_parity():
    bound = 1.0
    mask = jnp.array([1.0, 1.0, 1.0, 1.0, 0.0])
    x = jnp.array([0.2, 1.5, -2.2, 4.2, 7.0])  # inside; one wall; one wall the other way; two walls; not a scene coordinate
    folded, flip = fold_into_box(x, bound, mask)
    assert np.allclose(np.asarray(folded), [0.2, 0.5, 0.2, 0.2, 7.0])
    assert np.asarray(flip).tolist() == [False, True, True, False, False]
    # reversibility: the folded point with the flipped momentum retraces the move
    x0, drift = 0.2, 4.0
    y, flip_y = fold_into_box(jnp.array([x0 + drift]), bound, jnp.ones(1))
    back, _ = fold_into_box(y - (-1.0 if flip_y[0] else 1.0) * drift, bound, jnp.ones(1))
    assert np.isclose(float(back[0]), x0)
    y, flip_y = fold_into_box(jnp.array([x0 + 1.3]), bound, jnp.ones(1))  # one wall: momentum flips
    back, _ = fold_into_box(y - (-1.0 if flip_y[0] else 1.0) * 1.3, bound, jnp.ones(1))
    assert np.isclose(float(back[0]), x0)
    # an infinite bound is a no-op
    same, flip_inf = fold_into_box(x, jnp.inf, mask)
    assert np.allclose(np.asarray(same), np.asarray(x)) and not np.any(np.asarray(flip_inf))


def test_broadening_is_zero_at_t0():
    class SDE:
        sigma_min, sigma_max = 0.01, 10.0

    broaden = make_broadening_fn(SDE(), data_std=1.0)
    assert float(broaden(0.0)) == 0.0
    assert float(broaden(0.3)) > 0.0


def test_linear_gibbs_ridge_term_keeps_the_weak_coordinate_exact():
    """Precisions 1e6 and 1: the ridge (1e-6 x the mean diagonal) would
    tighten the weak coordinate's proposal by a quarter; the acceptance
    ratio carries the matching term, so the draws keep the exact variance."""
    K = 2
    a = jnp.asarray([1000.0, 1.0])
    y = jnp.asarray([500.0, 0.7])
    lin = {
        "idx": np.arange(K), "group": np.arange(K), "image": np.arange(K), "is_log": np.zeros(K, bool),
        "prior_mean": np.zeros(K), "prior_prec": np.zeros(K), "source": np.full(K, -1), "pix_image": np.arange(K), "n_images": K,
    }  # fmt: skip
    gibbs = make_linear_gibbs_fn(lambda x: a * x, y, jnp.ones(K), lambda t: 0.0 * t, lin, extra_jitter=1e-6)
    C = 4000
    key = jax.random.key(1)
    x = jnp.zeros((C, K), jnp.float32)
    for _ in range(40):
        key, k = jax.random.split(key)
        x, frac, _, _ = gibbs(jax.random.split(k, C), x, 0.0, jnp.zeros(K), jnp.ones(K))
    draws = np.asarray(x, np.float64)
    assert float(np.mean(np.asarray(frac))) > 0.5
    assert abs(draws[:, 1].mean() - 0.7) < 0.08
    assert abs(draws[:, 1].var() - 1.0) < 0.1


def test_nuisance_step_trace_describes_the_returned_state(toy):
    """With the non-scene sub-step on, the logged density must be that of the
    state the step returns, after an accepted nuisance move."""
    scene_mask = jnp.array([1.0, 1.0, 1.0, 1.0, 0.0])
    pm, ps = toy["prior_mean"], toy["prior_sd"]
    target = make_target(toy["log_lik"], toy["log_prior"], sigma_fn, scene_mask, tweedie=False)
    nuisance_log_prior = lambda x: -0.5 * (x[4] - pm[4]) ** 2 / ps[4] ** 2  # the t = 0 prior of the nuisance coordinate  # noqa: E731
    runner = make_chunk_runner(target, broaden_fn, sigma_fn, 3, 8, 0.0, 0.0, nuisance_log_prior=nuisance_log_prior, nuisance_steps=3)
    C = 16
    x0 = jnp.tile(jnp.asarray(pm) + 0.5, (C, 1))
    jj = jnp.asarray(np.sum(toy["A"] ** 2, axis=1), jnp.float32)
    zeros = jnp.zeros(8, jnp.float32)
    x, tr = runner(jax.random.split(jax.random.key(3), C), x0, jnp.full(C, 0.05), jnp.full(C, 0.3), jnp.full(D, 0.05), zeros, zeros, jj,
                   jnp.full(D, jnp.inf), scene_mask)  # fmt: skip
    assert np.asarray(tr["accept_prob_nuis"]).mean() > 0.2
    moved = ~np.isclose(np.asarray(x[:, 4]), float(pm[4] + 0.5))
    assert moved.any(), "no nuisance move was accepted"
    ev = broaden_fn(0.0) * jj  # the toy's own broadening (sigma_min^2 at t = 0), as the runner used it
    lp_true = np.array([float(toy["log_lik"](x[c], ev) + toy["log_prior"](x[c], 0.0)) for c in range(C)])
    assert np.allclose(np.asarray(tr["logdensity"])[:, -1], lp_true, rtol=1e-4, atol=1e-3)
    assert np.allclose(np.asarray(tr["loglik"])[:, -1], [float(toy["log_lik"](x[c], ev)) for c in range(C)], rtol=1e-4, atol=1e-3)


def test_renoise_move_keeps_a_gaussian_target_invariant():
    """A Gaussian prior on x_t: clean pixels ~ N(0, 1), so x_t ~ N(0, 1 + sigma^2),
    the Tweedie estimate is x / (1 + sigma^2) and the clean scene scatters by
    sigma^2 / (1 + sigma^2) about it (the residual variance with s = 1). With
    that residual the re-noise proposal is the exact Gibbs move and is always
    accepted; many moves on many chains must leave the N(0, 1 + sigma^2)
    marginal in place and relax a cold start to it. Without the residual the
    proposal contracts onto the Tweedie estimate and is mostly rejected."""
    from novamac.hmc import Target, make_renoise_fn

    sigma = 0.8
    dim, n_scene, chains, moves = 6, 4, 2000, 20
    scene_mask = jnp.asarray([1.0] * n_scene + [0.0] * (dim - n_scene))
    sigma_fn = lambda t: sigma  # noqa: E731
    var_t = 1.0 + sigma**2
    log_prior = lambda x, t: -0.5 * jnp.sum(scene_mask * x**2) / var_t  # noqa: E731
    log_lik = lambda x, ev: 0.0 * jnp.sum(x)  # noqa: E731
    target = Target(log_lik, log_prior, sigma_fn, scene_mask, tweedie=True)
    residual = lambda t: sigma**2 / var_t  # noqa: E731  (var(clean | x_t) for s = 1)
    renoise = make_renoise_fn(target, lambda t: 0.0, sigma_fn, scene_mask, residual)
    key = jax.random.key(3)
    # start far from the target: every scene pixel at +4
    x0 = jnp.concatenate([jnp.full((chains, n_scene), 4.0), jnp.ones((chains, dim - n_scene))], axis=1)
    x, accepted = x0, []
    for _ in range(moves):
        key, k = jax.random.split(key)
        x, acc, log_alpha, _, parts = renoise(jax.random.split(k, chains), x, 0.5, jnp.zeros(1))
        assert bool(jnp.all(jnp.isfinite(log_alpha))) and parts.shape == (chains, 3)
        assert np.allclose(np.asarray(parts).sum(axis=1), np.asarray(log_alpha), atol=1e-3)
        accepted.append(float(acc.mean()))
    scene = np.asarray(x[:, :n_scene])
    assert min(accepted) > 0.95, "the Gibbs proposal is (numerically) always accepted"
    assert abs(scene.mean()) < 0.05, "relaxed from the cold start"
    assert abs(scene.var() / var_t - 1.0) < 0.05, "the marginal of x_t is N(0, 1 + sigma^2)"
    assert np.allclose(np.asarray(x[:, n_scene:]), 1.0), "non-scene coordinates untouched"
    # the bare re-noise (no residual scatter) is a valid but poor proposal
    bare = make_renoise_fn(target, lambda t: 0.0, sigma_fn, scene_mask)
    key, k = jax.random.split(key)
    _, acc, _, _, _ = bare(jax.random.split(k, chains), x, 0.5, jnp.zeros(1))
    assert float(acc.mean()) < 0.8


def test_scalar_jacobian_is_exact_for_an_isotropic_gaussian_prior(toy):
    """For a Gaussian prior of scale s_a the denoiser's Jacobian is
    s_a^2 / (sigma_t^2 + s_a^2) I, so the "scalar" force equals the "exact"
    one (and differs from the "identity" one where sigma_t is large)."""
    s_a = 1.5
    mu = jnp.asarray(toy["prior_mean"])

    def log_prior(x, t):
        return -0.5 * jnp.sum((x - mu) ** 2 / (s_a**2 + sigma_fn(t) ** 2))

    mask = jnp.ones(D)
    targets = {
        j: make_target(toy["log_lik"], log_prior, sigma_fn, mask, True, j, None, s_a if j == "scalar" else None)
        for j in ("identity", "scalar", "exact")
    }
    x = jnp.asarray(toy["x_true"]) + 0.3
    for t in (1.0, 0.7, 0.4, 0.1):
        parts = {j: targets[j].parts(x, t, broaden_fn(t)) for j in targets}
        for j in ("scalar", "exact"):  # same density whatever the force
            assert np.allclose(parts[j][0], parts["identity"][0]) and np.allclose(parts[j][1], parts["identity"][1])
        g_scalar, g_exact, g_id = (np.asarray(parts[j][2]) for j in ("scalar", "exact", "identity"))
        assert np.allclose(g_scalar, g_exact, rtol=1e-4, atol=1e-5), (t, g_scalar, g_exact)
        factor = s_a**2 / (sigma_fn(t) ** 2 + s_a**2)
        assert np.allclose(g_scalar, factor * g_id, rtol=1e-5)
        assert float(targets["scalar"].jacobian_factor(t)) == pytest.approx(factor, rel=1e-5)
    with pytest.raises(ValueError):
        make_target(toy["log_lik"], log_prior, sigma_fn, mask, True, "scalar")
    with pytest.raises(ValueError):
        make_target(toy["log_lik"], log_prior, sigma_fn, mask, True, "bogus")


def test_predictor_runner_is_the_prior_flow_and_leaves_the_rest_alone(toy):
    """With eta = 0 the predictor-only runner integrates the probability-flow
    ODE of the toy's Gaussian prior N(pm, ps^2 + sigma_t^2), whose exact flow
    from sigma_a to sigma_b scales x - pm by sqrt((ps^2 + sigma_b^2) /
    (ps^2 + sigma_a^2)) on the scene coordinates; the non-scene coordinate
    must not move, the likelihood is never evaluated, and the trace reports
    log p_t at the returned state with nan log density and acceptance."""
    scene_mask = jnp.array([1.0, 1.0, 1.0, 1.0, 0.0])
    pm, ps = toy["prior_mean"], toy["prior_sd"]
    target = make_target(toy["log_lik"], toy["log_prior"], sigma_fn, scene_mask, tweedie=False)
    n = 64
    ts = np.linspace(0.9, 0.3, n + 1)
    runner = make_predictor_runner(target, broaden_fn, sigma_fn, n, predictor_noise=0.0)
    C = 3
    x0 = jnp.asarray(np.tile(pm + 2.0 * ps, (C, 1)) + np.arange(C)[:, None] * 0.1, jnp.float32)
    jj = jnp.asarray(np.sum(toy["A"] ** 2, axis=1), jnp.float32)
    dev = lambda a: jnp.asarray(a, jnp.float32)  # noqa: E731
    x, tr = runner(jax.random.split(jax.random.key(5), C), x0, jnp.full(C, 0.1), jnp.full(C, 0.1), jnp.ones(D), dev(ts[1:]), dev(ts[:-1]), jj,
                   jnp.full(D, jnp.inf), scene_mask)  # fmt: skip
    sa, sb = float(sigma_fn(ts[0])), float(sigma_fn(ts[-1]))
    scale = np.sqrt((ps**2 + sb**2) / (ps**2 + sa**2))
    expected = pm + (np.asarray(x0) - pm) * scale
    x = np.asarray(x)
    assert np.allclose(x[:, :4], expected[:, :4], rtol=2e-2, atol=1e-3), "scene coordinates do not follow the prior flow"
    assert np.array_equal(x[:, 4], np.asarray(x0)[:, 4]), "the non-scene coordinate moved"
    assert set(tr) == {"logdensity", "loglik", "logprior", "accept_prob", "log_alpha", "divergent", "predictor_rms",
                       "accept_prob_nuis", "log_alpha_nuis", "divergent_nuis"}  # fmt: skip
    assert np.all(np.isnan(np.asarray(tr["accept_prob"]))) and np.all(np.asarray(tr["predictor_rms"]) > 0)
    assert np.all(np.isnan(np.asarray(tr["logdensity"]))) and np.all(np.isnan(np.asarray(tr["loglik"])))
    lpr_true = np.array([float(toy["log_prior"](x[c], ts[-1])) for c in range(C)])
    assert np.allclose(np.asarray(tr["logprior"])[:, -1], lpr_true, rtol=1e-4, atol=1e-3)


def _nuisance_setup(toy, tweedie=True):
    """The toy with coordinate 4 as the non-scene block: target, its scene-only
    variant, the nuisance prior and a starting state for 16 chains."""
    scene_mask = jnp.array([1.0, 1.0, 1.0, 1.0, 0.0])
    scene_idx, nuis_idx = np.array([0, 1, 2, 3]), np.array([4])
    pm, ps = toy["prior_mean"], toy["prior_sd"]
    pm_j, ps_j = jnp.asarray(pm), jnp.asarray(ps)
    # the prior: the scene coordinates broadened by sigma(t), the non-scene
    # coordinate at its fixed width, and nuisance_log_prior exactly that
    # non-scene part (as novamac's model has it; the reuse of the prior across
    # the non-scene move assumes it)
    log_prior = lambda x, t: -0.5 * jnp.sum((x - pm_j) ** 2 / (ps_j**2 + sigma_fn(t) ** 2 * scene_mask))  # noqa: E731
    nuisance_log_prior = lambda x: -0.5 * (x[4] - pm[4]) ** 2 / ps[4] ** 2  # noqa: E731
    lik_scene = lambda s, n, ev: toy["log_lik"](jnp.concatenate([s, n]), ev)  # noqa: E731
    target = make_target(toy["log_lik"], log_prior, sigma_fn, scene_mask, tweedie=tweedie)
    target_scene = make_target(toy["log_lik"], log_prior, sigma_fn, scene_mask, tweedie=tweedie,
                               lik_scene=lik_scene, scene_idx=scene_idx, nuisance_idx=nuis_idx)  # fmt: skip
    C = 16
    x0 = jnp.tile(jnp.asarray(pm) + 0.5, (C, 1))
    jj = jnp.asarray(np.sum(toy["A"] ** 2, axis=1), jnp.float32)
    return dict(scene_mask=scene_mask, target=target, target_scene=target_scene, nuisance_log_prior=nuisance_log_prior, C=C, x0=x0, jj=jj)


def test_scene_only_force_matches_the_masked_gradient(toy):
    """A Target with lik_scene returns the same density and the same force on
    the scene coordinates as the full one, and zero force on the others."""
    su = _nuisance_setup(toy)
    x = su["x0"][0] + 0.1
    ev = broaden_fn(0.3) * su["jj"]
    (v_full, aux_full), g_full = su["target"].value_and_grad(x, 0.3, ev)
    (v_s, aux_s), g_s = su["target_scene"].value_and_grad(x, 0.3, ev)
    assert float(v_s) == pytest.approx(float(v_full), rel=1e-6)
    assert float(aux_s[0]) == pytest.approx(float(aux_full[0]), rel=1e-6)
    # the force = likelihood gradient (scene only in the scene target) + prior score (both)
    expected = (g_full - aux_full[2]) * su["scene_mask"] + aux_full[2]
    np.testing.assert_allclose(np.asarray(g_s), np.asarray(expected), rtol=1e-5, atol=1e-6)


def _run(su, target, ts, ts_prev, **kw):
    runner = make_chunk_runner(target, broaden_fn, sigma_fn, 3, len(ts), 0.5, 1.0, nuisance_log_prior=su["nuisance_log_prior"],
                               nuisance_steps=3, **kw)  # fmt: skip
    C, D_ = su["C"], su["x0"].shape[1]
    return runner(jax.random.split(jax.random.key(5), C), su["x0"], jnp.full(C, 0.05), jnp.full(C, 0.3), jnp.full(D_, 0.05),
                  jnp.asarray(ts, jnp.float32), jnp.asarray(ts_prev, jnp.float32), su["jj"], jnp.full(D_, jnp.inf), su["scene_mask"])  # fmt: skip


def test_reuse_prior_reproduces_the_fresh_evaluation(toy):
    """At constant t the corrector's initial evaluation with the carried prior
    value and score (reuse_prior) gives the same chains and trace as
    evaluating the prior afresh, with the non-scene sub-step moving between
    steps."""
    su = _nuisance_setup(toy)
    ts = np.full(6, 0.2)
    x_a, tr_a = _run(su, su["target_scene"], ts, ts, predictor=False, reuse_prior=False, carry_score=False)
    x_b, tr_b = _run(su, su["target_scene"], ts, ts, predictor=False, reuse_prior=True)
    assert np.asarray(tr_b["accept_prob_nuis"]).mean() > 0.2 and np.asarray(tr_b["accept_prob"]).mean() > 0.2
    np.testing.assert_allclose(np.asarray(x_b), np.asarray(x_a), rtol=1e-4, atol=1e-4)
    for k in ("logdensity", "loglik", "logprior", "accept_prob", "accept_prob_nuis"):
        np.testing.assert_allclose(np.asarray(tr_b[k]), np.asarray(tr_a[k]), rtol=1e-4, atol=1e-4, err_msg=k)


def test_carried_score_reproduces_the_predictor(toy):
    """In a ramp the predictor's first Heun stage with the score carried from
    the corrector gives the same chains and trace as evaluating it afresh."""
    su = _nuisance_setup(toy)
    ts = np.linspace(0.5, 0.1, 6)
    ts_prev = np.concatenate([[0.6], ts[:-1]])
    x_a, tr_a = _run(su, su["target_scene"], ts, ts_prev, predictor=True, carry_score=False)
    x_b, tr_b = _run(su, su["target_scene"], ts, ts_prev, predictor=True, carry_score=True)
    assert np.asarray(tr_b["predictor_rms"]).mean() > 0
    np.testing.assert_allclose(np.asarray(x_b), np.asarray(x_a), rtol=1e-4, atol=1e-4)
    for k in ("logdensity", "predictor_rms", "accept_prob"):
        np.testing.assert_allclose(np.asarray(tr_b[k]), np.asarray(tr_a[k]), rtol=1e-4, atol=1e-4, err_msg=k)
