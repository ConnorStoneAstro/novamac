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

    @property
    def chains(self) -> Path:
        """The caskade chain file: every saved state of every chain, the chains
        as the batch axis of each sampled parameter (sampler.save_chain_states)."""
        return self.path("chains.h5")

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
    """Number of states appended to the open chain file (h5py.File). Every
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


def load_chain_state(model, h5file, index: int, chain: int):
    """Set the model parameters to saved state `index` of chain `chain` from
    the open chain file (an h5py.File). Negative indices count from the end.
    caskade loads the state with the chains as the batch axis of every
    sampled parameter; the chain's values are then selected so the model is
    an ordinary single-state model again."""
    # caskade restores the valid range one bound at a time, which raises a
    # spurious "outside the valid range" warning for cyclic params like PA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InvalidValueWarning)
        model.load_state(h5file, index=index)
    n_chains = chain_axis_length(h5file)
    for p in model.dynamic_params:
        v = p.value
        if v is not None and v.ndim >= 1 and v.shape[0] == n_chains and not (p.group == 1):
            p.value = v[chain]


def chain_axis_length(h5) -> int:
    """The number of chains in the chain file: the batch axis of the scene."""
    found = []

    def visit(name, obj):
        if name.endswith("pixelated_batch_model/model/I/value") and isinstance(obj, h5py.Dataset):
            found.append(obj.shape[1])
            return True

    h5.visititems(visit)
    if not found:
        raise ValueError(f"{h5.filename}: no scene dataset")
    return found[0]


def read_param_history(h5, suffix, n_chains=None):
    """All saved states of one parameter, read straight from the chain file:
    the dataset whose path ends with `suffix` (e.g. "ps_batch_model_0/model/
    flux/value"). caskade stores appendable values with the sample index as
    the leading axis and the sampler sets the chains as the batch axis of the
    sampled parameters, so a sampled parameter comes back as (n_saved,
    n_chains, *param_shape) and a static one as (n_saved, *param_shape); with
    `n_chains` given a static parameter is repeated over a chain axis so
    every result has the chain axis."""
    found = []

    def visit(name, obj):
        if name.endswith(suffix) and isinstance(obj, h5py.Dataset):
            found.append((obj[()], str(obj.attrs.get("node_type", ""))))
            return True

    h5.visititems(visit)
    if not found:
        raise ValueError(f"{h5.filename}: no dataset ending in {suffix}")
    value, node_type = found[0]
    value = np.asarray(value)
    if n_chains is not None and ("static" in node_type or value.ndim < 2 or value.shape[1] != n_chains):
        value = np.broadcast_to(value[:, None], (value.shape[0], n_chains) + value.shape[1:])
    return value


def read_scene_history(h5):
    """All saved states of the pixelated scene (n_saved, n_chains, H, W),
    physical flux (the prior's units; the model renders scale x I)."""
    return read_param_history(h5, "pixelated_batch_model/model/I/value")


def collect_fluxes(files: RunFiles, n_chains: int = None):
    """
    Point-source fluxes at every saved state of every chain, read from the
    chain file without building the model: (chains, saved, P, N). The flux
    is a pointer parameter (allflux x template indicator) and caskade saved
    its evaluated value, so it is already zero on template images.
    """
    with h5py.File(files.chains, "r") as h5:
        n_chains = chain_axis_length(h5) if n_chains is None else n_chains
        per_ps = []
        p = 0
        while True:
            try:
                per_ps.append(read_param_history(h5, f"ps_batch_model_{p}/model/flux/value", n_chains))
            except ValueError:
                break
            p += 1
    return np.transpose(np.stack(per_ps, axis=2), (1, 0, 2, 3))  # (saved, chains, P, N) -> (chains, saved, P, N)


def draw_states(trace):
    """Which saved states are posterior draws: those saved once every kernel
    was fixed (step >= draws_from_step; for an older trace, the t = 0 states)."""
    saved_step = np.asarray(trace["saved_step"])
    if "draws_from_step" in trace:
        return saved_step >= int(trace["draws_from_step"])
    return np.asarray(trace["saved_t"]) == 0


def summarize_draws(fluxes, trace):
    """Mean and standard deviation of the point-source fluxes over the
    posterior draws (chains x draw states): (P, N) each, or None without
    any draw. `fluxes` is collect_fluxes' (chains, saved, P, N)."""
    draws = draw_states(trace)[: fluxes.shape[1]]
    if not draws.any():
        return None
    f = fluxes[:, draws].reshape(-1, *fluxes.shape[2:])
    return f.mean(axis=0), f.std(axis=0), f.shape[0]


def proposal_flux_moments(trace):
    """
    The linear Gibbs proposal's moments of the point-source fluxes, averaged
    over the chunks of the draw phase: for each source p a dict
    image -> (mean, std) where the mean averages the proposal means over
    chains and chunks, and the variance is the average proposal variance
    plus the spread of the means across those states. These are the
    LIKELIHOOD's conditional moments of the physical flux given the rest of
    the state, before the log-flux prior, its Jacobian and the positivity
    constraint, so they are not posterior moments (a faint epoch can have a
    negative mean); use summarize_draws for the posterior. None if the run
    stored no proposals or has no draw chunk yet.
    """
    if "cond_mean" not in trace:
        return None
    n_chunks = int(np.sum(trace["chunk_step"] <= int(trace["steps_done"])))
    draws_from = int(trace["draws_from_step"]) if "draws_from_step" in trace else None
    in_draws = trace["chunk_step"][:n_chunks] >= draws_from if draws_from is not None else trace["chunk_t"][:n_chunks] == 0
    hold = np.where(in_draws & np.all(np.isfinite(trace["cond_mean"][:n_chunks]), axis=(1, 2)))[0]
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
