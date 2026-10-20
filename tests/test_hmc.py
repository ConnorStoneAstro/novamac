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
    hutchinson_jj_diag,
    make_chunk_runner,
    make_gauss_newton_fn,
    make_linear_gibbs_fn,
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
