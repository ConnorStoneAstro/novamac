import jax


def build_posterior(model: "ap.Model"):
    """
    Taking a model generated with `build_likelihood_model` and augmented with a
    prior using `build_prior`, this will construct the log posterior density and
    its gradient functions. These are returned as JAX jitted functions, so their
    first use will be quite slow and then fast afterwards.

    `t` is the diffusion time of the host prior (0 = sharpest, 1 = broadest)
    and `extra_var` is added to the data variance in the likelihood; together
    they define an annealed posterior. Note the returned functions follow the
    scipy-minimize convention (negative log posterior); a sampler that wants
    the log density must negate them.

    Args:
      - model: The full constructed AstroPhot model, augmented with a prior
        using `build_prior`.

    Returns:
      - x0: The initial parameter values
      - f: A jitted function that takes in parameter values and returns the log
        posterior density
      - df: A jitted function that takes in parameter values and returns the
        gradient and log posterior density
    """

    def neg_log_posterior_density(params, t=0.0, extra_var=0.0):
        ll = model.log_likelihood(extra_var=extra_var, params=params)
        lp = model.log_prior(t=t, params=params)
        return -(ll + lp)

    f = jax.jit(neg_log_posterior_density)
    df = jax.jit(jax.grad(neg_log_posterior_density))
    x0 = model.get_values()

    return x0, f, df
