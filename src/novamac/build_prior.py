# construct the prior model
import types

import astrophot as ap
import jax.numpy as jnp

from .host_prior import load_prior


def build_prior(
    model: ap.Model, prior_loc: str, wcs_prior_width: float = 0.3, psf_position_prior: float = 0.5
):
    """
    Taking a model generated with `build_likelihood_model` this will construct a
    prior function and add it to the model. This will return an unnormalized log
    prior density.

    Args:
      - model: The full constructed AstroPhot model
      - prior_loc: The location of the prior, a path to the checkpoint directory (str)
      - wcs_prior_width: The width of the Gaussian prior on the WCS parameters
        in tangent plane coordinates, in arcseconds (float)
      - psf_position_prior: The width of the Gaussian prior on the PSF positions
        in tangent plane coordinates, in arcseconds (float)
    """
    P = model.meta.shapes["P"]
    crtan0 = jnp.array(model.meta.crtan0)
    ps_center0 = jnp.array(model.meta.ps_center0)
    host_prior = load_prior(prior_loc)

    @ap.param.forward
    def log_prior(self, beta=0.0):
        lp = 0.0
        # Prior on image positions (crtan, for image alignment)
        lp = lp - 0.5 * ap.backend.sum(
            ((self.sky_batch_model.target.crtan - crtan0) / wcs_prior_width) ** 2
        )
        # Prior on point source positions
        for p in range(P):
            lp = lp - 0.5 * ap.backend.sum(  # Prior on point source position
                (
                    (self.models[f"ps_batch_model_{p}"].model.center.value - ps_center0[p])
                    / psf_position_prior
                )
                ** 2
            )
        # Prior on pixel fluxes
        log_pix = jnp.log(self.models.pixelated_batch_model.model.I.value)
        lp = lp + host_prior.energy(log_pix, host_prior.sde.sigma(beta))
        return lp

    model.log_prior = types.MethodType(log_prior, model)
