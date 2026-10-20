"""
novamac: Bayesian transient scene modelling. `Novamac` takes one band's
cutouts, calibrations and a trained host-galaxy diffusion prior, samples the
posterior over the static scene, the transient light curve and the nuisance
parameters with an annealed HMC, and plots and diagnoses the result.

Importing the package pins the caskade backend to jax and jax to float32
with full-precision matmuls (see _backend.py); import novamac before
anything that imports caskade or astrophot.
"""

from . import _backend  # noqa: F401  (must be first: pins the caskade backend before astrophot is imported)
from .build_likelihood import build_likelihood
from .build_prior import build_prior
from .build_posterior import build_posterior
from .utils import get_fluxes, log, enable_console_logging
from . import host_prior
from . import init_scene, hmc, products, plots, diagnostics
from .inference import Novamac

_backend.configure()  # after AstroPhot (which turns x64 on at import)

__all__ = (
    "Novamac",
    "build_likelihood",
    "build_prior",
    "build_posterior",
    "get_fluxes",
    "host_prior",
    "init_scene",
    "hmc",
    "products",
    "plots",
    "diagnostics",
    "log",
    "enable_console_logging",
)
