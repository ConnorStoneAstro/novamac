"""The prior energy offset table: interpolation between its sigma nodes must
follow the power law of the energy, for either sign of the energy."""

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from novamac.sampler import calibrate_prior_offset


class _PowerLawPrior:
    """energy(x, sigma) = sign * n_pix * (s^2 + sigma^2) / sigma^2: the 'dae' level,
    with pixel_offset taken off `levels` times per pixel (1: the ncsnpp_energy
    prior; the encoder prior subtracts it on every pyramid level)."""

    def __init__(self, sign, levels=1):
        self.sign, self.levels = sign, levels
        self.sde = types.SimpleNamespace(sigma_min=0.02, sigma_max=10.0)

    def energy(self, x, sigma, pixel_offset=0.0):
        return self.sign * x.size * (0.1**2 + sigma**2) / sigma**2 - self.levels * x.size * pixel_offset


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


@pytest.mark.parametrize("levels", [1, 5])
def test_offset_is_divided_by_the_priors_multiplicity(levels):
    """A prior that takes pixel_offset off `levels` times per pixel gets 1/levels
    of the per-pixel energy, so its total with the offset is ~0 at every node."""
    prior = _PowerLawPrior(1.0, levels)
    model = types.SimpleNamespace(meta=types.SimpleNamespace(host_prior=prior))
    scene = np.zeros((8, 8))
    sigmas, values = calibrate_prior_offset(model, scene, jax.random.PRNGKey(0), n_sigmas=16)
    assert model.meta.prior_offset_multiplicity == levels
    for sg, val in zip(sigmas, values):
        assert np.isclose(val * levels, (0.1**2 + sg**2) / sg**2)
        raw = prior.energy(scene, sg)
        assert abs(prior.energy(scene, sg, float(model.meta.prior_energy_offset(jnp.asarray(sg))))) < 1e-5 * abs(raw)  # float32 table
