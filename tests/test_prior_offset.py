"""The prior energy offset table: interpolation between its sigma nodes must
follow the power law of the energy, for either sign of the energy."""

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from novamac.sampler import calibrate_prior_offset


class _PowerLawPrior:
    """energy(x, sigma) = sign * n_pix * (s^2 + sigma^2) / sigma^2: the 'dae' level."""

    def __init__(self, sign):
        self.sign, self.sde = sign, types.SimpleNamespace(sigma_min=0.02, sigma_max=10.0)

    def energy(self, x, sigma):
        return self.sign * x.size * (0.1**2 + sigma**2) / sigma**2


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_offset_interpolates_the_power_law(sign):
    prior = _PowerLawPrior(sign)
    model = types.SimpleNamespace(meta=types.SimpleNamespace(host_prior=prior))
    sigmas, values = calibrate_prior_offset(model, np.zeros((8, 8)), jax.random.PRNGKey(0), n_sigmas=64)
    assert np.all(np.sign(values) == sign)
    # halfway between nodes in log sigma, where linear interpolation errs most
    mids = np.sqrt(sigmas[:-1] * sigmas[1:])
    got = np.array([float(model.meta.prior_energy_offset(jnp.asarray(s))) for s in mids])
    exact = sign * (0.1**2 + mids**2) / mids**2
    assert np.allclose(got, exact, rtol=5e-3), np.max(np.abs(got / exact - 1))
