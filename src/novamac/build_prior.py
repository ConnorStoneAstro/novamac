# construct the prior model
import types

import astrophot as ap
import jax.numpy as jnp

from .host_prior import load_prior


def build_prior(
    model: ap.Model,
    host_prior_loc: str,
    wcs_prior_width: float = 0.3,
    ps_position_prior_width: float = 0.5,
    logflux_prior_width: float = 2.0,
    psf_n_prior_width: float = 0.5,
    psf_Rd_prior_width: float = 0.2,
    psf_q_prior_width: float = 0.1,
    psf_PA_prior_width: float = 0.2,
):
    """
    Taking a model generated with `build_likelihood_model` this will construct a
    prior function and add it to the model. This will return an unnormalized log
    prior density.

    Args:
      - model: The full constructed AstroPhot model
      - host_prior_loc: The location of the host prior, a path to the checkpoint directory (str)
      - wcs_prior_width: The width of the Gaussian prior on the WCS parameters
        in tangent plane coordinates, in arcseconds (float)
      - ps_position_prior_width: The width of the Gaussian prior on the PS positions
        in tangent plane coordinates, in arcseconds (float)
      - logflux_prior_width: The width of the Gaussian prior on the log of the pixel fluxes (float)
    """
    # Collect prior configuration
    P = model.meta.shapes["P"]
    crtan0 = jnp.array(model.meta.crtan0)
    ps_center0 = jnp.array(model.meta.ps_center0)
    log_flux0 = jnp.log(jnp.array(model.meta.ps_flux0))
    psf_n0 = jnp.array(model.meta.psf_n0)
    psf_Rd0 = jnp.array(model.meta.psf_Rd0)
    psf_q0 = jnp.array(model.meta.psf_q0)
    psf_PA0 = jnp.array(model.meta.psf_PA0)
    not_template = jnp.array(~model.meta.is_template)
    host_prior = load_prior(host_prior_loc)

    @ap.param.forward
    def log_prior(self, beta=0.0):
        lp = 0.0

        # Prior on image positions (crtan, for image alignment)
        lp = lp - 0.5 * ap.backend.sum(
            ((self.sky_batch_model.target.crtan - crtan0) / wcs_prior_width) ** 2
        )

        for p in range(P):
            # Prior on point source positions
            lp = lp - 0.5 * ap.backend.sum(
                (
                    (self.models[f"ps_batch_model_{p}"].model.center.value - ps_center0[p])
                    / ps_position_prior_width
                )
                ** 2
            )

            # Prior on point source log fluxes
            lp = lp - 0.5 * ap.backend.sum(
                (
                    (jnp.log(self.models[f"ps_batch_model_{p}"].model.flux.value) - log_flux0[p])
                    * not_template
                    / logflux_prior_width
                )
                ** 2
            )

        # PSF prior
        lp = lp - 0.5 * (model.pixelated_batch_model.model.psf.n.value - psf_n0) ** 2 / psf_n_prior_width**2 # fmt: skip
        lp = lp - 0.5 * (model.pixelated_batch_model.model.psf.Rd.value - psf_Rd0) ** 2 / psf_Rd_prior_width**2 # fmt: skip
        lp = lp - 0.5 * (model.pixelated_batch_model.model.psf.q.value - psf_q0) ** 2 / psf_q_prior_width**2 # fmt: skip
        lp = lp - 0.5 * (model.pixelated_batch_model.model.psf.PA.value - psf_PA0) ** 2 / psf_PA_prior_width**2 # fmt: skip

        # Prior on pixel fluxes
        log_pix = jnp.log(self.models.pixelated_batch_model.model.I.value)
        lp = lp + host_prior.log_prob(log_pix, host_prior.sde.sigma(beta))
        return lp

    model.log_prior = types.MethodType(log_prior, model)
