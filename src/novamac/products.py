"""
The files a run writes and the readers for them.

All products of a run live in one directory under one name prefix:

  {name}_chain{c}.h5      caskade state file for chain c. Saved state k is
                          the chain position at step trace["saved_step"][k]
                          (diffusion time trace["saved_t"][k]). Index 0 is
                          the starting point. Physical parameter values.
  {name}_trace.npz        per step: t, logdensity, loglik, logprior,
                          accept_prob, log_alpha, divergent, predictor_rms
                          (and the *_nuis versions for the non-scene
                          sub-step). Per chunk: step_size, scores
                          (hmc.SCORE_FIELDS), the Gibbs acceptance and
                          conditionals. Plus inv_mass0, the Jacobian
                          diagonal, block indices, the release / probe /
                          restart history and the settings. Rewritten after
                          every save.
  {name}_tweedie.npz      scene_hat (saved, chains, H, W): the Tweedie
                          estimate exp(x + sigma_t^2 score) of the scene at
                          every saved state, in the scene's flux units.
  {name}_timings.json     wall-clock seconds per stage.
  {name}_<figure>.pdf     the plots (novamac.plots).
"""

import json
import warnings
from pathlib import Path

import h5py
import numpy as np
from caskade.warnings import InvalidValueWarning


class RunFiles:
    """Paths of one run's products: `RunFiles(out_dir, name)`."""

    def __init__(self, out_dir, name: str):
        self.out_dir = Path(out_dir)
        self.name = str(name)

    def mkdir(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        return self.out_dir

    def path(self, suffix: str) -> Path:
        return self.out_dir / f"{self.name}_{suffix}"

    def chain(self, c: int) -> Path:
        return self.path(f"chain{c}.h5")

    @property
    def trace(self) -> Path:
        return self.path("trace.npz")

    @property
    def tweedie(self) -> Path:
        return self.path("tweedie.npz")

    @property
    def timings(self) -> Path:
        return self.path("timings.json")

    def figure(self, what: str, ext: str = "pdf") -> Path:
        return self.path(f"{what}.{ext}")

    def exists(self) -> bool:
        return self.trace.exists()

    def load_trace(self) -> dict:
        return dict(np.load(self.trace, allow_pickle=False))

    def load_timings(self) -> dict:
        return json.loads(self.timings.read_text())

    def num_chains(self) -> int:
        return int(self.load_trace()["num_chains"])

    def __repr__(self):
        return f"RunFiles({str(self.out_dir)!r}, {self.name!r})"


# Chain file helpers
#####################################################################
def num_saved_states(h5file) -> int:
    """Number of states appended to an open chain file (h5py.File). Every
    param dataset has the same leading sample axis, so read it off the first
    appendable one found."""
    found = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.attrs.get("appendable", False):
            found.append(obj.shape[0])
            return True  # stop walking

    h5file.visititems(visit)
    if not found:
        raise ValueError(f"{h5file.filename} holds no appendable parameter datasets")
    return found[0]


def load_chain_state(model, h5file, index: int):
    """Set the model parameters to saved state `index` of an open chain file
    (an h5py.File). Negative indices count from the end."""
    # caskade restores the valid range one bound at a time, which raises a
    # spurious "outside the valid range" warning for cyclic params like PA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InvalidValueWarning)
        model.load_state(h5file, index=index)


def read_param_history(h5, suffix):
    """All saved states of one parameter, read straight from the chain file:
    the dataset whose path ends with `suffix` (e.g. "ps_batch_model_0/model/
    flux/value"). caskade stores appendable values with the sample index as
    the leading axis, so the result has shape (n_saved, *param_shape)."""
    found = []

    def visit(name, obj):
        if name.endswith(suffix) and isinstance(obj, h5py.Dataset):
            found.append(obj[()])
            return True

    h5.visititems(visit)
    if not found:
        raise ValueError(f"{h5.filename}: no dataset ending in {suffix}")
    return np.asarray(found[0])


def read_scene_history(h5):
    """All saved states of the pixelated scene (n_saved, H, W), physical flux
    (the prior's units; the model renders scale x I)."""
    return read_param_history(h5, "pixelated_batch_model/model/I/value")


def collect_fluxes(files: RunFiles, n_chains: int):
    """
    Point-source fluxes at every saved state of every chain, read from the
    chain files without building the model: (chains, saved, P, N). The flux
    is a pointer parameter (allflux x template indicator) and caskade saved
    its evaluated value, so it is already zero on template images.
    """
    fluxes = []
    for c in range(n_chains):
        with h5py.File(files.chain(c), "r") as h5:
            per_ps = []
            p = 0
            while True:
                try:
                    per_ps.append(read_param_history(h5, f"ps_batch_model_{p}/model/flux/value"))
                except ValueError:
                    break
                p += 1
            fluxes.append(np.stack(per_ps, axis=1))  # (saved, P, N)
    n_saved = min(f.shape[0] for f in fluxes)  # chains can differ by one if the job died mid-save
    return np.stack([f[:n_saved] for f in fluxes])


def conditional_fluxes(trace):
    """
    Rao-Blackwellised flux estimate from the linear Gibbs conditionals stored
    at the end of every chunk of the hold phase (t = 0): for each source p a
    dict image -> (mean, std) where the mean averages the conditional means
    over chains and hold chunks, and the variance is the average conditional
    variance plus the spread of the conditional means across those states
    (law of total variance). None if the run stored no conditionals or has
    no hold chunk yet.
    """
    if "cond_mean" not in trace:
        return None
    n_chunks = int(np.sum(trace["chunk_step"] <= int(trace["steps_done"])))
    hold = np.where((trace["chunk_t"][:n_chunks] == 0) & np.all(np.isfinite(trace["cond_mean"][:n_chunks]), axis=(1, 2)))[0]
    if len(hold) == 0:
        return None
    means = trace["cond_mean"][hold].astype(np.float64)  # (S, C, K)
    var = np.einsum("sckk->sck", trace["cond_cov"][hold].astype(np.float64))  # (S, C, K)
    mean = means.reshape(-1, means.shape[-1]).mean(axis=0)
    total_var = var.reshape(-1, var.shape[-1]).mean(axis=0) + means.reshape(-1, means.shape[-1]).var(axis=0)
    out = {}
    for j, (src, img) in enumerate(zip(trace["lin_source"], trace["lin_image"])):
        if src >= 0:
            out.setdefault(int(src), {})[int(img)] = (mean[j], np.sqrt(total_var[j]))
    out["n_states"] = len(hold) * means.shape[1]
    return out
