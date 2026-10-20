"""
Predictor-corrector annealed sampling of the novamac posterior with a
diffusion (VE-SDE) host prior.

The idea
--------
The host-galaxy prior is a score/energy model trained at noise levels
sigma(t), t in [0, 1]. The annealed target at time t is

    log pi_t(theta) = log L_t(theta) + log p_t(theta),

with the prior at noise level sigma(t) and the likelihood broadened
consistently: a Gaussian perturbation of the scene's log-pixels x by sigma_t
is a log-normal perturbation of the fluxes, so data pixel i gets the extra
variance (exp(sigma_t^2) - 1) * jj_i with jj = diag(J_x J_x^T), J_x =
d(images)/dx, estimated once with Hutchinson probes at the initial scene (the
sensible linearisation point; the annealed state at large t is mostly noise).
For sigma_t << 1 that is the familiar sigma_t^2 * jj; for sigma_t >~ 1 it
saturates and the likelihood goes flat. The log-normal mean shift
exp(sigma_t^2 / 2) is ignored; it vanishes as t -> 0.

Each step from t_prev to t is a predictor-corrector pair:
  * predictor: one Heun step of the reverse VE diffusion of the scene from
    sigma(t_prev) to sigma(t) driven by the posterior score (prior + broadened
    likelihood), with a tunable noise fraction (0 = probability-flow ODE,
    1 = reverse SDE). Only the diffused coordinates (the scene) move.
  * corrector: one exact Metropolis-corrected HMC transition for pi_t.
Only the final state is meant to be a sample of the t = 0 posterior.

Mechanics
---------
* Chain-vmapped, jitted chunks of `chunk_steps` steps (`make_chunk_runner`).
* An exact Gibbs-type refresh of the linear parameters, sky levels and point
  source fluxes (`make_linear_gibbs_fn`), run once per chunk.
* The annealed target itself (`Target` / `make_target`): with Tweedie on, the
  likelihood is evaluated on the denoised scene x + sigma_t^2 * score.
  Step size and the schedule are plain arguments; the host loop adapts the
  step size between chunks without recompiling.
* Diagonal mass matrix, fixed in shape: inv_mass(t) = inv_mass0 +
  mass_sigma_frac * sigma(t)^2 on the scene, where inv_mass0 is a
  Gauss-Newton estimate of the t = 0 posterior variance made once at setup
  (`make_gauss_newton_fn`). Nothing about the mass is learned from the chain,
  so it cannot drift; a bad mass only shows up as a small step size.
* Every coordinate has a release time t_release and is frozen (zero
  velocity, zero kinetic energy, masked out of the kicks) while t > t_release;
  +inf = always free, -inf = always frozen.
* `make_score_fn` returns the norms of the likelihood and prior scores (full
  vector, scene block, metric-weighted scene block) and their cosine on the
  scene block, to see whether one term is annealing out before the other.

Everything here works in caskade's unconstrained (ValidContext) space; the
scene block there is exactly log(I), the prior's own coordinate.
"""

import caskade as ck
import jax
import jax.numpy as jnp
import numpy as np

# Names of the per-chunk score diagnostics returned by make_score_fn
SCORE_FIELDS = (
    "lik_norm",  # ||grad log L|| over the full parameter vector
    "prior_norm",  # ||grad log p|| over the full parameter vector
    "lik_norm_scene",  # same restricted to the scene (pixel) block
    "prior_norm_scene",
    "lik_mnorm_scene",  # metric-weighted: sqrt(sum inv_mass * g^2) on the scene block
    "prior_mnorm_scene",
    "cos_scene",  # cosine between the two scene-block scores
)


# Model wrappers
#####################################################################
def make_log_parts(model, group1):
    """
    Jitted log_lik(x, extra_var), log_prior(x, t) and images(x) with x the
    group-0 parameter vector in caskade's unconstrained space.

    The ValidContext is entered *inside* each function on purpose: caskade's
    valid_context is a plain Python flag read when the forward methods are
    traced, so this guarantees every trace works in unconstrained space no
    matter where the function is first called from.
    """

    def call(method, x, **kwargs):
        with ck.ValidContext(model):
            return method(params=(x, group1), **kwargs)

    log_lik = jax.jit(lambda x, extra_var: call(model.log_likelihood, x, extra_var=extra_var))
    log_prior = jax.jit(lambda x, t: call(model.log_prior, x, t=t))
    images = jax.jit(lambda x: call(model.model_image_flat, x))
    return log_lik, log_prior, images


def make_nuisance_log_prior(model, group1):
    """Jitted log_prior_nuisance(x): every prior term except the host prior
    on the scene (Gaussian, no network), or None if the model has no such
    method. Used by the non-scene HMC sub-step, which holds the scene fixed."""
    if not hasattr(model, "log_prior_nuisance"):
        return None

    def call(x):
        with ck.ValidContext(model):
            return model.log_prior_nuisance(params=(x, group1))

    return jax.jit(call)



def flat_index(model, group=0):
    """
    One entry per element of the flat parameter vector, in order:
    (Param, index-within-param). Used to locate parameter blocks.
    """
    return model._array_inspection(group)


def block_indices(model, params, element=None, group=0):
    """Flat-vector indices belonging to any of the given Param objects,
    optionally only the entries with a given index-within-param (e.g.
    element=(0,) for the x component of every image's crtan)."""
    ids = {id(p) for p in params}
    return np.array(
        [i for i, (p, ix) in enumerate(flat_index(model, group))
         if id(p) in ids and (element is None or tuple(ix) == tuple(element))],
        dtype=int,
    )  # fmt: skip


def make_sigma_fn(sde):
    """sigma(t) of the prior's VE schedule as a pure jnp function."""
    sigma_min, sigma_max = float(sde.sigma_min), float(sde.sigma_max)
    return lambda t: sigma_min * (sigma_max / sigma_min) ** t


# exp(sigma^2) - 1 is capped at exp(MAX_LOG_BROADENING) so float32 never
# overflows to inf (inf * 0 = nan on data pixels with jj = 0).
MAX_LOG_BROADENING = 60.0


def make_broadening_fn(sde, data_std=None):
    """
    broaden(t) = exp(v(t)) - 1: the factor multiplying jj in the likelihood's
    extra variance (see the module docstring), the second moment of a
    log-normal pixel factor with log variance v(t).

    v(t) = sigma(t)^2 when the likelihood is evaluated on the noisy x_t
    itself. With `data_std` given, the likelihood is evaluated on the
    Tweedie-denoised scene (see make_target) and what is left is the
    denoiser's residual uncertainty, approximated as in PiGDM by
    v(t) = sigma^2 s^2 / (sigma^2 + s^2) with s the prior's data scale;
    v -> sigma^2 at small sigma and saturates at s^2.
    """
    log_variance = make_log_variance_fn(sde, data_std)
    # zero at t = 0: the hold samples the plain likelihood (the Tweedie shift
    # and the mean correction switch off there too), not one broadened by
    # the residual sigma_min noise
    return lambda t: jnp.where(t > 0, jnp.expm1(jnp.minimum(log_variance(t), MAX_LOG_BROADENING)), 0.0)


def make_log_variance_fn(sde, data_std=None):
    """v(t): the residual log-variance of the scene the likelihood sees
    (see make_broadening_fn): sigma(t)^2, or the Tweedie residual
    sigma^2 s^2 / (sigma^2 + s^2) with `data_std` given."""
    sigma_fn = make_sigma_fn(sde)

    def log_variance(t):
        s2 = sigma_fn(t) ** 2
        if data_std is None:
            return s2
        return s2 * data_std**2 / (s2 + data_std**2)

    return log_variance


# The annealing target
#####################################################################
class Target:
    """
    The annealed log density pi_t(x) = log p_t(x) + log L(y | x_hat(x)) and
    its force, built by make_target. With `tweedie` on, the likelihood is
    evaluated on the Tweedie estimate of the clean scene,

        x_hat = x + sigma_t^2 * grad log p_t(x)   (scene coordinates, t > 0),

    rather than on the noisy x_t itself (PiGDM-style). Evaluating the
    likelihood on x_t forces the sigma_t noise out of the scene wherever the
    data constrain it and leaves it in the PSF margin, which is not a noised
    version of the t = 0 posterior; through the denoiser the likelihood is
    insensitive to the noise directions, so the noise stays everywhere. At
    t = 0 (the hold) the plain likelihood is used, so the reported states are
    draws from the true posterior.

    The force is score + grad_x_hat log L by default, i.e. the denoiser's
    Jacobian is taken as the identity. That costs nothing: the score is the
    prior's gradient the corrector needs anyway. It does not cost
    correctness either: the Metropolis test uses the exact log density, and
    a leapfrog with any position-dependent force is reversible and volume
    preserving, so HMC remains exact for pi_t; only the acceptance suffers
    where the Jacobian is far from the identity (large sigma_t). With
    `exact_jacobian` the missing term sigma^2 * H v (one Hessian-vector
    product of the prior per evaluation) is added.

    Mean correction (`log_var_fn` given): the scene the likelihood renders is
    shifted by -v(t)/2 in log I, v the residual log-variance of the scene it
    sees (the broadening's log variance: sigma_t^2, or the Tweedie residual).
    The data constrain the local average of exp(x) over the scene pixels and
    the PSF under each data pixel, and for a field with log-mean mu and
    residual pixel scatter v that average is exp(mu + v/2); without the
    shift, matching the data pushes mu down by v/2 wherever the data have a
    say and leaves the PSF margin alone, which is the dark box the size of
    the cutout seen in annealed scenes. With it the rendered flux is the mean
    over the residual scatter and the log-level is unbiased. Zero at t = 0.
    """

    def __init__(self, log_lik, log_prior, sigma_fn, scene_mask, tweedie=True, exact_jacobian=False, log_var_fn=None):
        self.log_lik, self.log_prior, self.sigma_fn = log_lik, log_prior, sigma_fn
        self.scene_mask = jnp.asarray(scene_mask)
        self.tweedie, self.exact_jacobian = bool(tweedie), bool(exact_jacobian)
        self.log_var_fn = log_var_fn
        self.prior_vg = jax.value_and_grad(log_prior)
        self.lik_vg = jax.value_and_grad(log_lik)
        self.prior_grad = jax.grad(log_prior)

    def shift(self, t, score):
        """(scene the likelihood sees) - x: the Tweedie step sigma_t^2 * score
        and the log-normal mean correction -v(t)/2, on the scene for t > 0."""
        out = jnp.zeros_like(score)
        if self.tweedie:
            out = out + self.sigma_fn(t) ** 2 * score
        if self.log_var_fn is not None:
            out = out - 0.5 * self.log_var_fn(t)
        return jnp.where(t > 0, out * self.scene_mask, 0.0)

    def denoise(self, x, t):
        """The scene the likelihood sees at time t."""
        if not self.tweedie and self.log_var_fn is None:
            return x
        score = self.prior_grad(x, t) if self.tweedie else jnp.zeros_like(x)
        return x + self.shift(t, score)

    def parts(self, x, t, extra_var):
        """(log L, log p_t, grad log L wrt x (force part), score)."""
        lp, score = self.prior_vg(x, t)
        x_hat = x + self.shift(t, score)
        ll, g_ll = self.lik_vg(x_hat, extra_var)
        if self.tweedie and self.exact_jacobian:
            # (d x_hat / d x)^T g = g + sigma^2 H (M g) on the scene, H the prior's Hessian
            v = g_ll * self.scene_mask
            hvp = jax.jvp(lambda y: self.prior_grad(y, t), (x,), (v,))[1]
            g_ll = g_ll + jnp.where(t > 0, self.sigma_fn(t) ** 2 * hvp * self.scene_mask, 0.0)
        return ll, lp, g_ll, score

    def value_and_grad(self, x, t, extra_var):
        """((log pi_t, (log L, log p_t, prior score)), force), the shape
        jax.value_and_grad(has_aux=True) gives; the score is kept so the
        non-scene sub-step can form x_hat without another prior evaluation."""
        ll, lp, g_ll, score = self.parts(x, t, extra_var)
        return (ll + lp, (ll, lp, score)), g_ll + score


def make_target(log_lik, log_prior, sigma_fn, scene_mask, tweedie=True, exact_jacobian=False, log_var_fn=None):
    return Target(log_lik, log_prior, sigma_fn, scene_mask, tweedie, exact_jacobian, log_var_fn)


# Jacobian diagonal
#####################################################################
def hutchinson_jj_diag(images, x0, scene_idx, rng_key, num_probes=256, batch=32):
    """
    diag(J J^T) with J = d images / d x[scene_idx], evaluated at x0, from
    Rademacher probe JVPs: E[(J z)^2] = diag(J J^T) for z with iid +-1 entries.
    Relative error per pixel ~ sqrt(2 / num_probes).
    """
    assert num_probes % batch == 0, "num_probes must be a multiple of batch"
    scene_idx = jnp.asarray(scene_idx)
    xs0 = x0[scene_idx]

    def f(xs):
        return images(x0.at[scene_idx].set(xs))

    def probe(key):
        z = jax.random.rademacher(key, xs0.shape, dtype=xs0.dtype)
        return jax.jvp(f, (xs0,), (z,))[1] ** 2

    vprobe = jax.jit(lambda keys: jnp.sum(jax.vmap(probe)(keys), axis=0))
    keys = jax.random.split(rng_key, num_probes)
    acc = 0.0
    for i in range(0, num_probes, batch):
        acc = acc + vprobe(keys[i : i + batch])
    return acc / num_probes


# Curvature groups for a pixel grid
#####################################################################
def strided_pixel_groups(scene_idx, shape, stride):
    """
    Partition the flat indices of a (rows, cols) pixel grid (row-major, as
    caskade flattens it) into stride^2 groups of pixels that are at least
    `stride` pixels apart in both directions. Pixels that far apart share no
    data pixels (up to PSF wings), so each group can be probed at once;
    residual coupling biases the estimated curvature up, i.e. the implied
    variance down, which for a mass matrix is the safe direction.
    """
    scene_idx = np.asarray(scene_idx)
    rows, cols = np.divmod(np.arange(len(scene_idx)), shape[1])
    return [
        scene_idx[(rows % stride == a) & (cols % stride == b)]
        for a in range(stride) for b in range(stride)
        if np.any((rows % stride == a) & (cols % stride == b))
    ]  # fmt: skip


# Gauss-Newton curvature of the likelihood (for the setup mass matrix)
#####################################################################
def make_gauss_newton_fn(images, groups):
    """
    gauss_newton(x, weight) -> diag(J^T W J) for the coordinates in `groups`
    (in the order of np.concatenate(groups)), with J = d images / d x at x
    and W = diag(weight) over the flattened data (1 / variance, 0 on masked
    pixels). That is the likelihood's curvature in the Gauss-Newton
    approximation, which ignores the residual x second-derivative term and
    so is positive wherever a coordinate touches any data.

    Each group must contain coordinates that share no data pixels: it is
    probed with one JVP (J z for the group's indicator z) and one VJP
    (J^T W J z), whose entries on the group are the diagonal plus cross terms
    that vanish without shared pixels. The VJP linearisation is built once
    and reused for every group.
    """
    groups = [jnp.asarray(g) for g in groups]

    @jax.jit
    def gauss_newton(x, weight):
        _, pullback = jax.vjp(images, x)
        out = []
        for g in groups:
            probe = jnp.zeros_like(x).at[g].set(1.0)
            jz = jax.jvp(images, (x,), (probe,))[1]
            (jtwjz,) = pullback(weight * jz)
            out.append(jtwjz[g])
        return jnp.concatenate(out)

    return gauss_newton


# Curvature of the log density for a block (release criterion)
#####################################################################
def make_block_curvature_fn(target, broaden_fn, groups):
    """
    curvature(x, t, jj) -> negative diagonal Hessian of the log density at x
    (one chain, x of shape (dim,)) for the coordinates in `groups`, returned
    in the order of np.concatenate(groups). The groups are non-scene
    coordinates, which enter the likelihood directly, so the Hessian is taken
    at the denoised scene (target.denoise) with the scene held fixed.

    Each group must be a set of coordinates that are mutually decoupled in the
    Hessian (zero off-diagonal entries between them), because the whole group
    is probed with ONE Hessian-vector product: with a probe that is 1 on every
    coordinate of the group, (H z)_i = sum_j H_ij = H_ii for i in the group.
    For the novamac model this holds for "the PSF parameter n of every
    image", "Rd of every image", etc.: the likelihood is a sum over images and
    each image's PSF only enters its own term. Probing coordinate by
    coordinate instead (hundreds of HVPs batched together) needs hundreds of
    model passes' worth of memory and was what ran the GPU out of memory.

    The groups are evaluated one at a time, each HVP a separate jitted call,
    so the peak memory is that of a single forward-over-reverse pass.
    """
    groups = [jnp.asarray(g) for g in groups]
    log_lik, log_prior = target.log_lik, target.log_prior

    def logp(x, t, extra_var):
        return log_lik(x, extra_var) + log_prior(x, t)

    @jax.jit
    def hvp_diag(x, t, jj, probe):
        extra_var = broaden_fn(t) * jj
        x_hat = jax.lax.stop_gradient(target.denoise(x, t))
        grad = lambda y: jax.grad(logp)(y, t, extra_var)
        return -jax.jvp(grad, (x_hat,), (probe,))[1]

    def curvature(x, t, jj):
        out = []
        for g in groups:
            probe = jnp.zeros_like(x).at[g].set(1.0)
            out.append(hvp_diag(x, t, jj, probe)[g])
        return jnp.concatenate(out)

    return curvature


# Reflecting box on the scene coordinates
#####################################################################
def fold_into_box(x, bound, scene_mask):
    """
    Fold the scene coordinates of x (scene_mask > 0) back into [-bound, bound]
    by reflection at the walls, however many walls a jump crossed (a triangle
    wave of period 4 * bound), so the map is its own inverse and the leapfrog
    stays reversible. Also returns, per coordinate, whether an odd number of
    walls was crossed, which is when the momentum must flip. No-op for an
    infinite bound.
    """
    finite = jnp.isfinite(bound)
    b = jnp.where(finite, bound, 1.0)
    u = jnp.mod(x + b, 4.0 * b)
    folded = jnp.where(u <= 2.0 * b, u - b, 3.0 * b - u)
    crossings = jnp.floor((x + b) / (2.0 * b))
    in_box = (scene_mask > 0) & finite
    flip = in_box & (jnp.mod(crossings, 2.0) != 0)
    return jnp.where(in_box, folded, x), flip


# Predictor-corrector chunk
#####################################################################
def make_chunk_runner(
    target,
    broaden_fn,
    sigma_fn,
    num_integration_steps,
    chunk_steps,
    mass_sigma_frac=0.5,
    predictor_noise=1.0,
    max_abs_scene=jnp.inf,
    predictor=True,
    nuisance_log_prior=None,
    nuisance_steps=10,
):
    """
    Build the jitted, chain-vmapped function that advances every chain by
    `chunk_steps` predictor-corrector steps on `target` (a Target: the
    annealed log density, its force and the Tweedie map).

        run(keys, x, step_size, step_size_nuis, inv_mass0, ts, ts_prev, jj, t_release, scene_mask)
            -> (x, trace)

    keys, x, step_size, step_size_nuis carry a leading chain axis; inv_mass0 (dim,) is the
    t = 0 variance estimate shared by the chains, ts / ts_prev (chunk_steps,)
    are the diffusion time of each step and of the step before it (equal
    where the target does not move), jj is diag(J J^T) over the flattened
    data, t_release (dim,) is the time each coordinate becomes free, and
    scene_mask (dim,) is 1 on the diffused (scene) coordinates.

    Per step, with sigma_cur = sigma(t_prev) and sigma_new = sigma(t):
      predictor (scene only, skipped when sigma_new >= sigma_cur): a Heun
        step of dx = -(1 + eta)/2 * d(sigma^2) * score + sqrt(eta * |d(sigma^2)|) * z
        with eta = predictor_noise and the PRIOR score at the current noise
        level, i.e. exactly the prior's reverse VE diffusion (eta = 1) or
        probability-flow ODE (eta = 0). The likelihood enters only through
        the corrector: its score in log I is proportional to exp(x), so an
        unadjusted explicit step with it sends a pixel that is a few e-folds
        too bright straight into float32 overflow, whereas the Metropolis
        step just rejects such a move;
      corrector: one HMC transition for pi_t with inverse mass
        inv_mass0 + mass_sigma_frac * sigma_new^2 * scene_mask.

    trace is a dict of per-step arrays with shape (chains, chunk_steps):
        logdensity, loglik, logprior, accept_prob, log_alpha (log Metropolis
        ratio, -inf if non-finite), divergent, predictor_rms (rms predictor
        displacement over the scene).

    predictor=False skips the predictor altogether (corrector-only annealing).

    With `nuisance_log_prior` given (the Gaussian prior on everything but the
    scene), each step is predictor -> HMC on the scene coordinates -> HMC on
    the released non-scene coordinates with the scene held fixed, the latter
    with its own step size `step_size_nuis` and `nuisance_steps` leapfrog
    steps and its own acceptance (trace keys *_nuis). Each sub-step is a
    valid kernel for the joint target with the other block fixed. The
    non-scene step sees the same scene the scene step does, x_hat (Tweedie
    + mean correction), formed from the score the scene step already
    computed, and its leapfrog needs only likelihood renders. Without it
    (None) one HMC moves every released coordinate, as before.

    max_abs_scene truncates the target to |x| <= max_abs_scene on the scene
    coordinates: the leapfrog reflects at the box (fold the coordinate back
    across however many walls it crossed, flip its momentum for an odd
    number, which keeps the integrator volume preserving and reversible, so
    no rejections at the wall; fold_into_box), the predictor reflects too,
    and the density is -inf outside as a backstop. The scene is stored as
    exp(x) in float32, which overflows at |x| ~ 88, so this keeps the chain
    on representable values when the noise level is large (sigma ~ 27 at
    t = 1 puts pixels beyond 88 at only 3.3 sigma). A hard wall without
    reflection would not do: with many pixels near the wall almost every
    proposal crosses it somewhere. Leave it at inf where the scene stays small.
    """
    L = num_integration_steps
    bound = jnp.asarray(max_abs_scene)

    def _value_and_grad(x, t, extra_var, scene_mask):
        (val, aux), g = target.value_and_grad(x, t, extra_var)
        inside = jnp.all(jnp.abs(x) * scene_mask <= bound)
        val = jnp.where(inside, val, -jnp.inf)
        return (val, aux), g

    prior_grad = target.prior_grad  # the predictor follows the prior score only

    def single_chain(key, x, step_size, step_size_nuis, inv_mass0, ts, ts_prev, jj, t_release, scene_mask):
        n_scene = jnp.maximum(jnp.sum(scene_mask), 1.0)
        value_and_grad = lambda x, t, ev: _value_and_grad(x, t, ev, scene_mask)  # noqa: E731
        if nuisance_log_prior is not None:
            # the non-scene sub-step's target: likelihood at the (fixed) denoised
            # scene + the Gaussian nuisance prior; no host prior evaluation
            def _nuis_logp(y, t, ev):
                ll = target.log_lik(y, ev)
                lpn = nuisance_log_prior(y)
                return ll + lpn, (ll, lpn, jnp.zeros_like(y))
            nuis_value_and_grad = jax.value_and_grad(_nuis_logp, has_aux=True)

        def reflect_scene(x):
            # scene coordinates folded back into the box (no-op when bound = inf);
            # also returns which coordinates need their momentum flipped
            return fold_into_box(x, bound, scene_mask)

        def predictor_step(key, x, t_prev, t):
            s_cur, s_new = sigma_fn(t_prev), sigma_fn(t)
            dvar = jnp.maximum(s_cur**2 - s_new**2, 0.0)  # decrease of the noise variance
            drift_coef = 0.5 * (1.0 + predictor_noise) * dvar
            noise = jnp.sqrt(predictor_noise * dvar) * jax.random.normal(key, x.shape, x.dtype) * scene_mask
            g_cur = prior_grad(x, t_prev)
            x_euler, _ = reflect_scene(x + drift_coef * g_cur * scene_mask + noise)
            g_new = prior_grad(x_euler, t)
            x_new, _ = reflect_scene(x + 0.5 * drift_coef * (g_cur + g_new) * scene_mask + noise)
            # a non-finite predictor step is dropped rather than propagated
            ok = jnp.all(jnp.isfinite(x_new))
            x_new = jnp.where(ok, x_new, x)
            return x_new, jnp.sqrt(jnp.sum(((x_new - x) * scene_mask) ** 2) / n_scene)

        def corrector(key, x, t, extra_var, inv_mass, active, step_size, n_leap, vg):
            """One HMC transition on the coordinates in `active` with step size
            `step_size`, `n_leap` leapfrog steps and target value_and_grad `vg`."""
            velocity_scale = inv_mass * active

            def kinetic(p):
                return 0.5 * jnp.sum(velocity_scale * p * p)

            (lp0, aux0), g0 = vg(x, t, extra_var)
            key_momentum, key_accept = jax.random.split(key)
            p = jax.random.normal(key_momentum, x.shape, x.dtype) / jnp.sqrt(inv_mass) * active
            ke0 = kinetic(p)

            # Leapfrog: initial half kick, then n_leap x (drift, kick) with the
            # final kick halved so consecutive half kicks are merged.
            p = p + 0.5 * step_size * g0 * active

            def leap(i, state):
                x, p, _, _ = state
                x, reflected = reflect_scene(x + step_size * velocity_scale * p)
                p = jnp.where(reflected, -p, p)
                (lp, aux), g = vg(x, t, extra_var)
                p = p + step_size * g * active * jnp.where(i == n_leap - 1, 0.5, 1.0)
                return x, p, lp, aux

            x1, p1, lp1, aux1 = jax.lax.fori_loop(0, n_leap, leap, (x, p, lp0, aux0))
            ke1 = kinetic(p1)

            # lp is the log density with the prior's typical per-pixel energy
            # level subtracted inside its sum (the sampler calibrates it per
            # sigma), so the float32 totals stay small enough for this
            # difference to be meaningful; the raw totals are ~1e7.
            log_alpha = (lp1 - ke1) - (lp0 - ke0)
            log_alpha = jnp.where(jnp.isfinite(log_alpha), log_alpha, -jnp.inf)
            # Recovery, not a Metropolis move: a chain whose current density
            # is non-finite (which no valid state has) takes the first finite
            # proposal instead of rejecting forever.
            log_alpha = jnp.where(jnp.isfinite(lp0) | ~jnp.isfinite(lp1 - ke1), log_alpha, jnp.inf)
            accept_prob = jnp.minimum(1.0, jnp.exp(log_alpha))
            accept = jax.random.uniform(key_accept, dtype=x.dtype) < accept_prob
            divergent = log_alpha < -1000.0

            x = jnp.where(accept, x1, x)
            lp = jnp.where(accept, lp1, lp0)
            ll = jnp.where(accept, aux1[0], aux0[0])
            lpr = jnp.where(accept, aux1[1], aux0[1])
            score = jnp.where(accept, aux1[2], aux0[2])
            return x, (lp, ll, lpr, accept_prob, log_alpha, divergent), score, aux0

        def step(x, inputs):
            key, t, t_prev = inputs
            key_pred, key_corr, key_nuis = jax.random.split(key, 3)
            if predictor:
                x, pred_rms = predictor_step(key_pred, x, t_prev, t)
            else:
                pred_rms = jnp.zeros((), x.dtype)  # corrector only: HMC tracks the moving target by itself
            extra_var = broaden_fn(t) * jj
            # Frozen coordinates get no momentum, no velocity and no kicks,
            # which is HMC restricted to the active subspace.
            active = (t <= t_release).astype(x.dtype)
            inv_mass = inv_mass0 + mass_sigma_frac * sigma_fn(t) ** 2 * scene_mask
            if nuisance_log_prior is None:
                x, out, _, _ = corrector(key_corr, x, t, extra_var, inv_mass, active, step_size, L, value_and_grad)
                out_n = (jnp.nan, jnp.nan, False)
            else:
                # scene sub-step on the scene coordinates only ...
                x, out, score, _ = corrector(key_corr, x, t, extra_var, inv_mass, active * scene_mask, step_size, L, value_and_grad)
                # ... then the released non-scene coordinates with the scene held
                # at x_hat, the scene the likelihood sees (free: the score is the
                # one the scene step ended on)
                x_hat = x + target.shift(t, score)
                y, out_n6, _, aux0_n = corrector(
                    key_nuis, x_hat, t, extra_var, inv_mass, active * (1.0 - scene_mask), step_size_nuis, nuisance_steps,
                    nuis_value_and_grad,
                )
                x = jnp.where(scene_mask > 0, x, y)  # scene coordinates keep their raw values
                # the logged density describes the state returned: the likelihood
                # after the non-scene move, and the prior with its nuisance part
                # swapped (the host prior term is the scene step's, unchanged)
                ll_new, lpn_new, lpn_old = out_n6[1], out_n6[2], aux0_n[1]
                lpr_new = out[2] - lpn_old + lpn_new
                out = (ll_new + lpr_new, ll_new, lpr_new) + out[3:]
                out_n = (out_n6[3], out_n6[4], out_n6[5])
            return x, out + (pred_rms,) + out_n

        keys = jax.random.split(key, chunk_steps)
        x, (lp, ll, lpr, acc, la, div, prms, acc_n, la_n, div_n) = jax.lax.scan(step, x, (keys, ts, ts_prev))
        trace = {
            "logdensity": lp, "loglik": ll, "logprior": lpr,
            "accept_prob": acc, "log_alpha": la, "divergent": div, "predictor_rms": prms,
            "accept_prob_nuis": acc_n, "log_alpha_nuis": la_n, "divergent_nuis": div_n,
        }  # fmt: skip
        return x, trace

    return jax.jit(jax.vmap(single_chain, in_axes=(0, 0, 0, 0, None, None, None, None, None, None)))


# Score diagnostics
#####################################################################
def make_score_fn(target, broaden_fn, scene_idx):
    """
    Jitted, chain-vmapped: score(x, t, jj, inv_mass) -> (scores, logdensity)
    with scores (chains, len(SCORE_FIELDS)) and logdensity (chains,) the
    target's value at x (the states handed in, e.g. after the Gibbs refresh).
    The likelihood part is the force the corrector uses (at the denoised scene).
    """
    scene_idx = jnp.asarray(scene_idx)

    def single(x, t, jj, inv_mass):
        ll, lp, gl, gp = target.parts(x, t, broaden_fn(t) * jj)
        gls, gps, ms = gl[scene_idx], gp[scene_idx], inv_mass[scene_idx]
        norm = lambda v: jnp.sqrt(jnp.sum(v * v))
        mnorm = lambda v: jnp.sqrt(jnp.sum(ms * v * v))
        cos = jnp.sum(gls * gps) / (norm(gls) * norm(gps) + jnp.finfo(x.dtype).tiny)
        return jnp.stack([norm(gl), norm(gp), norm(gls), norm(gps), mnorm(gls), mnorm(gps), cos]), ll + lp

    return jax.jit(jax.vmap(single, in_axes=(0, None, None, None)))


# Step size adaptation (host side)
#####################################################################
def adapt_step_size(step_size, acceptance, target=0.65, tol=0.05, down=0.7, up=1.1):
    """Multiplicative step size update from the mean acceptance of the last
    window: shrink if below target - tol, grow if above target + tol."""
    step_size = np.asarray(step_size, dtype=float)
    acceptance = np.asarray(acceptance, dtype=float)
    return np.where(acceptance < target - tol, step_size * down,
                    np.where(acceptance > target + tol, step_size * up, step_size))  # fmt: skip


def release_inverse_mass(curvature, fallback):
    """Inverse mass for a block from its curvature: 1 / curvature where the
    curvature is positive, the fallback elsewhere."""
    curvature = np.asarray(curvature, dtype=np.float64)
    return np.where(curvature > 0, 1.0 / np.maximum(curvature, 1e-30), fallback)


# Exact refresh of the linear parameters (sky levels, point source fluxes)
#####################################################################
def make_linear_gibbs_fn(images, data, weight, broaden_fn, lin, denoise=None, extra_jitter=1e-6):
    """
    Build the jitted, chain-vmapped Gibbs-type refresh of the coordinates
    that enter the model image linearly in their physical value: the per-image
    sky level (identity transform) and the point source fluxes (log transform
    in the unconstrained vector).

        gibbs(keys, x, t, jj, active) -> (x_new, accepted_fraction, c_mean, c_cov)

    `lin` is a dict of per-coordinate arrays (K,): idx (flat index in x),
    group (coordinates in one group have disjoint image support, e.g. the sky
    of every image, or source p's flux in every image), image (which image
    the coordinate belongs to), is_log (True for log-transformed values),
    prior_mean / prior_prec (Gaussian prior in the unconstrained coordinate,
    precision 0 where there is none); plus pix_image (NHW,), the image index
    of every flattened data pixel, and n_images. The Metropolis-Hastings test
    is made per image (the conditional is block diagonal across images), and
    accepted_fraction is the fraction of images with a released coordinate
    whose refresh was accepted. `active` (K,) is 1 for coordinates that
    are currently free; the others stay put and act as part of the base.

    With everything else fixed, image = base + Phi c, so the likelihood is
    exactly Gaussian in the physical values c: precision A = Phi^T W Phi,
    mean c_hat = A^{-1} Phi^T W r with r the data minus the base and W the
    (broadened) inverse variance. Phi comes from one JVP per group (the
    columns of a group are separated by image afterwards). The draw
    c' ~ N(c_hat, A^{-1}) is an independence proposal; the likelihood
    ratio cancels against the proposal exactly, leaving the prior ratio and,
    for log coordinates, the Jacobian prod(c / c') of the change of
    variables, so the move is an exact Metropolis-Hastings step for the
    target in the unconstrained coordinates. A proposal that sends a flux
    below zero is rejected for that image only (the model forbids it; for a
    faint epoch that happens often and must not block the other epochs).
    A ridge `extra_jitter` x the mean diagonal is added to A so a column
    without support stays solvable; the proposal then differs from the
    likelihood conditional by exp(-jitter |c|^2 / 2), and the acceptance
    ratio carries the matching term, so the move stays exact. c_mean and
    c_cov are the mean and covariance of that Gaussian proposal (the
    likelihood's conditional of the physical values given the rest of the
    state, before the prior and the positivity constraint), not posterior
    moments. `denoise(x, t)` (Target.denoise) gives the scene the likelihood
    sees; the base image and the columns are rendered there.
    """
    lin_idx = jnp.asarray(lin["idx"])
    lin_group = np.asarray(lin["group"])
    lin_image = jnp.asarray(lin["image"])
    lin_is_log = jnp.asarray(lin["is_log"])
    prior_mean = jnp.asarray(lin["prior_mean"])
    prior_prec = jnp.asarray(lin["prior_prec"])
    pix_image = jnp.asarray(lin["pix_image"])
    n_images = int(lin["n_images"])
    n_groups = int(lin_group.max()) + 1
    group_of = jnp.asarray(lin_group)
    K = len(lin["idx"])

    def columns(x_hat):
        """Phi (K, NHW): the image response per unit physical value, and c (K,)."""
        c = jnp.where(lin_is_log, jnp.exp(x_hat[lin_idx]), x_hat[lin_idx])
        dcdx = jnp.where(lin_is_log, c, 1.0)
        rows = []
        for g in range(n_groups):
            v = jnp.zeros_like(x_hat).at[lin_idx].add(jnp.where(group_of == g, 1.0, 0.0).astype(x_hat.dtype))
            rows.append(jax.jvp(images, (x_hat,), (v,))[1])
        J = jnp.stack(rows)  # (G, NHW), d image / d x along each group
        same_image = (pix_image[None, :] == lin_image[:, None]).astype(x_hat.dtype)
        phi = J[group_of] * same_image / dcdx[:, None]
        return phi, c

    def lp_lin(xl):
        return -0.5 * jnp.sum(prior_prec * (xl - prior_mean) ** 2)

    def single(key, x, t, jj, active):
        key_z, key_u = jax.random.split(key)
        # the linear coordinates are not scene coordinates, so x_hat differs from x only on the scene
        x_hat = jax.lax.stop_gradient(denoise(x, t)) if denoise is not None else x
        phi, c = columns(x_hat)
        w = weight / (1.0 + broaden_fn(t) * jj * weight)  # 1 / (var + extra_var); masked pixels stay 0
        phi_a = phi * active[:, None]
        r = data - images(x_hat) + phi_a.T @ c  # data minus the base image
        pw = phi_a * w[None, :]
        A = pw @ phi_a.T + jnp.diag(1.0 - active)
        jitter = extra_jitter * jnp.trace(A) / K + 1e-30
        A = A + jnp.diag(active) * jitter  # a column without support stays solvable
        b = pw @ r + (1.0 - active) * c
        L = jnp.linalg.cholesky(A)
        c_mean = jax.scipy.linalg.cho_solve((L, True), b)
        c_cov = jax.scipy.linalg.cho_solve((L, True), jnp.eye(K, dtype=x.dtype))
        z = jax.random.normal(key_z, (K,), x.dtype) * active
        c_new = c_mean + jax.scipy.linalg.solve_triangular(L, z, trans=1, lower=True)  # L^{-T} z ~ N(0, A^{-1})
        x_lin = x[lin_idx]
        x_lin_new = jnp.where(lin_is_log, jnp.log(jnp.where(c_new > 0, c_new, 1.0)), c_new)
        x_lin_new = jnp.where(active > 0, x_lin_new, x_lin)
        # Metropolis-Hastings test per image: the conditional is block diagonal
        # across images, so the joint draw is a set of independent per-image
        # proposals, and one epoch whose flux is consistent with zero (a
        # negative draw) must not veto every other epoch's refresh.
        per_coord = (
            -0.5 * prior_prec * ((x_lin_new - prior_mean) ** 2 - (x_lin - prior_mean) ** 2)
            + jnp.where(lin_is_log & (active > 0), x_lin - x_lin_new, 0.0)  # Jacobian of c = exp(x)
            + 0.5 * jitter * (c_new**2 - c**2) * active  # likelihood / proposal mismatch from the ridge
        )
        bad = (active > 0) & ((lin_is_log & (c_new <= 0)) | ~jnp.isfinite(c_new) | ~jnp.isfinite(per_coord))
        log_alpha = jax.ops.segment_sum(jnp.where(bad, 0.0, per_coord), lin_image, n_images)
        n_bad = jax.ops.segment_sum(bad.astype(x.dtype), lin_image, n_images)
        n_active = jax.ops.segment_sum(active, lin_image, n_images)
        log_alpha = jnp.where(n_bad > 0, -jnp.inf, log_alpha)
        u = jax.random.uniform(key_u, (n_images,), dtype=x.dtype)
        accept_image = (jnp.log(u) < log_alpha) & (n_active > 0)
        accept_coord = accept_image[lin_image]
        x_out = x.at[lin_idx].set(jnp.where(accept_coord, x_lin_new, x_lin))
        # fraction of the images with a released coordinate whose refresh was accepted
        frac = jnp.sum(accept_image) / jnp.maximum(jnp.sum(n_active > 0), 1.0)
        return x_out, frac, c_mean, c_cov

    return jax.jit(jax.vmap(single, in_axes=(0, 0, None, None, None)))
