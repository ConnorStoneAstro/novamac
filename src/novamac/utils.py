import json
import logging
import sys
from pathlib import Path
from time import time

import numpy as np

logger = logging.getLogger("novamac")


def log(*args):
    """Log a line through the "novamac" logger (INFO). `enable_console_logging`
    turns the logger into timestamped, flushed prints, the right thing for
    SLURM logs (python buffers stdout when it is redirected to a file)."""
    logger.info(" ".join(str(a) for a in args))


def enable_console_logging(level=logging.INFO, stream=None):
    """Attach one timestamped stdout handler to the "novamac" logger (idempotent)."""
    logger.setLevel(level)
    for h in logger.handlers:
        if getattr(h, "_novamac_console", False):
            return
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    handler._novamac_console = True
    logger.addHandler(handler)
    logger.propagate = False


class Timer:
    """Context manager that times a block, logs the elapsed time and records
    it in `timings[name]` (a dict you can later dump to JSON).

        with Timer("warmup", timings):
            ...
    """

    def __init__(self, name: str, timings: dict | None = None, quiet: bool = False):
        self.name = name
        self.timings = timings
        self.quiet = quiet

    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time() - self.start
        if self.timings is not None:
            self.timings[self.name] = self.timings.get(self.name, 0.0) + self.elapsed
        if not self.quiet:
            log(f"{self.name}: {self.elapsed:.1f} s")


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def get_fluxes(model):
    """
    Extract the fluxes for the point sources from a model as constructed with
    `build_likelihood_model`. The fluxes are organized in the same order
    as the input data, so that they can be directly compared to the observed
    fluxes.
    Returns:
      - fluxes: A numpy array of shape (P, N) containing the fluxes for each point source in each image.
    """
    return np.stack(
        tuple(
            model.models[f"ps_batch_model_{p}"].model.flux.npvalue
            for p in range(model.meta.shapes["P"])
        ),
        axis=0,
    )  # shape (P, N)
