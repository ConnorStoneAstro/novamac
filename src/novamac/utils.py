import numpy as np


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
