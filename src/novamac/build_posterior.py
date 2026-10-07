import jax


def build_posterior(model: "ap.Model"):
    """
    Taking a model generated with `build_likelihood_model` and augmented with a
    prior using `build_prior`, this will construct the log posterior density and
    its gradient functions. These are returned as JAX jitted functions, so their
    first use will be quite slow and then fast afterwards.

    A `beta` parameter is used to control the annealing schedule with beta=0
    indicating no annealing and beta=1 indicating maximum annealing.

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

    def neg_log_posterior_density(params, beta=0.0):
        ll = model.log_likelihood(beta=beta, params=params)
        lp = model.log_prior(beta=beta, params=params)
        return -(ll + lp)

    f = jax.jit(neg_log_posterior_density)
    df = jax.jit(jax.grad(neg_log_posterior_density))
    x0 = model.get_values()

    return x0, f, df
