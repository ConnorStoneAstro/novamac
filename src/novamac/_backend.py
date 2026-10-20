"""
Backend and precision pinning. Imported first by the package.

caskade picks torch over jax when both are importable, which breaks every
jax trace with a confusing "Input params type ... Tensor" error, so the
backend is pinned to jax here, before caskade is imported anywhere.

Precision: AstroPhot's jax backend turns x64 on unconditionally at import.
The host prior checkpoints are float32 and float32 is faster, so
`configure()` (called at the end of the package import, after AstroPhot)
turns it back off and everything is built in float32. On Ampere GPUs JAX
runs float32 matmuls and convolutions in TF32 (10 mantissa bits) unless
told otherwise; the host prior's energy multiplies ||h||^2 by
(s^2 + sigma^2) / sigma^2 ~ 850 at sigma_min, so rounding noise of ~1e-5 in
the network's output becomes ~1 in the log density, which is what the
Metropolis test has to resolve. "highest" forces full float32 arithmetic
(slower convolutions). `Novamac.diagnose_stall` measures the resulting
resolution.
"""

import os
import warnings

os.environ.setdefault("CASKADE_BACKEND", "jax")

import caskade as ck  # noqa: E402

if ck.backend.backend != "jax":
    raise ImportError(
        f"caskade backend is '{ck.backend.backend}' but novamac needs jax. "
        "Export CASKADE_BACKEND=jax before starting python (novamac sets it if unset, "
        "but caskade must not have been imported before novamac)."
    )

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

DTYPE = jnp.float32
MATMUL_PRECISION = "highest"


def configure(matmul_precision: str = MATMUL_PRECISION):
    """Pin jax to float32 and set the matmul precision (see the module
    docstring). Called once at package import; call again to change the
    precision ("highest", "high" or "default")."""
    global MATMUL_PRECISION
    MATMUL_PRECISION = matmul_precision
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_default_matmul_precision", matmul_precision)
    # AstroPhot asks for int64 index arrays; int32 is what it gets and is fine
    warnings.filterwarnings("ignore", message="Explicitly requested dtype int64")
