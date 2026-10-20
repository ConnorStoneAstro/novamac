import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CASKADE_BACKEND", "jax")

import pytest  # noqa: E402

import novamac  # noqa: E402,F401  (pins the backend / precision before anything else imports jax)
from novamac.testing import make_random_prior, make_synthetic_inputs  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: longer end-to-end runs, skipped by default (pytest -m slow)")


@pytest.fixture(scope="session")
def prior_dir(tmp_path_factory):
    """A random 16x16 host prior checkpoint (untrained: exercises the code paths only)."""
    return make_random_prior(tmp_path_factory.mktemp("prior"), size=16)


@pytest.fixture(scope="session")
def synthetic():
    """(inputs, truth) of the model-consistent synthetic cutouts on the 16x16 grid."""
    return make_synthetic_inputs(size=16)


# Settings that make a full run take a couple of minutes on a CPU: tiny
# chunks, few leapfrog steps, few probes, every block released early so the
# release machinery, the non-scene HMC and the Gibbs refresh all run.
FAST_SETTINGS = dict(
    num_chains=2,
    num_integration_steps=2,
    nuisance_integration_steps=2,
    chunk_steps=8,
    warmup_steps=16,
    anneal_steps=32,
    hold_steps=16,
    save_every=16,
    hutchinson_probes=8,
    hutchinson_batch=4,
    gn_stride=4,
    prior_offset_sigmas=8,
    pflow_reference_steps=8,
    release_t={"scene_scale": None, "sky": 0.6, "alignment": 0.6, "ps_flux": 0.6, "ps_center": 0.5, "psf": 0.4},
    release_latest_t={"ps_flux": 0.5},
    release_probe_every=1,
    sigma_start=1.0,
    verbose=False,
)
