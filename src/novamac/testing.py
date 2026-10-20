"""
Synthetic inputs for tests and for trying the package without data: a
random (untrained) host prior checkpoint in novamac's format, and a set of
cutouts generated with novamac's own forward model so that the data are
model-consistent. The prior's samples mean nothing; it exercises the code
paths (energy, score, Tweedie, offset calibration, pflow reference) only.

    from novamac.testing import make_random_prior, make_synthetic_inputs
    prior_dir = make_random_prior("/tmp/prior")
    inputs, truth = make_synthetic_inputs()
    nm = Novamac(prior=prior_dir, out_dir="/tmp/out", **inputs)
"""

import json
from pathlib import Path

import equinox as eqx
import jax
import numpy as np

from .build_likelihood import build_likelihood
from .host_prior._net import NCSNppEnergy
from .host_prior._prior import LogFluxTransform
from .host_prior._support import NCSNppConfig


def make_random_prior(directory, size=16, nf=8, softening=10.0, sigma_min=0.01, sigma_max=10.0, data_std=1.0, seed=0):
    """Write a random ncsnpp_energy ("dae") checkpoint on a size x size grid
    to `directory` (config.json, model.eqx, ema.eqx, state.json) and return
    the directory."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    cfg = {
        "architecture": "ncsnpp_energy",
        "sde": {"sigma_min": sigma_min, "sigma_max": sigma_max, "data_mean": 0.0, "data_std": data_std},
        "transform": {"softening": softening},
        "patch": {"out_size": size, "pool_factor": 1},
        "ncsnpp": {"in_channels": 1, "nf": nf, "ch_mult": [1, 2], "num_blocks": 1, "attention": False, "energy_form": "dae"},
    }
    (d / "config.json").write_text(json.dumps(cfg, indent=1))
    net_cfg = NCSNppConfig.from_dict(cfg["ncsnpp"])
    net_cfg.check()
    transform = LogFluxTransform(softening=softening)
    model = NCSNppEnergy(
        net_cfg, sigma_min=sigma_min, sigma_max=sigma_max, data_std=data_std, input_offset=transform.sky_level,
        key=jax.random.key(seed),
    )  # fmt: skip
    eqx.tree_serialise_leaves(d / "model.eqx", model)
    eqx.tree_serialise_leaves(d / "ema.eqx", model)
    (d / "state.json").write_text(json.dumps({"step": 0}))
    return d


def make_synthetic_inputs(size=16, pix_scale=0.2628, n_template=5, n_sn=3, seed=0, zeropoint=27.5, noise=0.5,
                          residual_sky=0.3, fluxes=(150.0, 400.0, 80.0), ps_center=(0.4, -0.3), masked_blob=True):
    """
    Model-consistent cutouts: a host (exponential profile on a flat floor)
    plus a point source in the n_sn transient epochs, rendered through
    novamac's forward model with a Moffat PSF, small random image offsets,
    a residual sky and Gaussian noise. The scene grid is size x size at
    pix_scale, the same as the data (no upsampling), so the random prior
    from make_random_prior(size=size) matches.

    Returns (inputs, truth): `inputs` are keyword arguments for Novamac
    (everything but the prior and the output location, including
    scene_pixel_scale), `truth` holds the scene (data units per arcsec^2),
    the point-source fluxes (P, N), the sky and the noise.
    """
    N = n_template + n_sn
    rng = np.random.default_rng(seed)
    is_template = np.array([True] * n_template + [False] * n_sn)
    r, c = np.mgrid[:size, :size]
    x = (c - (size - 1) / 2) * pix_scale
    y = (r - (size - 1) / 2) * pix_scale
    scene = 40.0 * np.exp(-np.hypot(x, y) / 1.2) + 1.0  # data units / arcsec^2
    ps_center = np.asarray(ps_center, dtype=float)[None, :]
    ps_flux = np.zeros((1, N))
    ps_flux[0, ~is_template] = np.asarray(fluxes, dtype=float)[:n_sn]
    crtan = rng.uniform(-0.1, 0.1, (N, 2))
    crpix = np.full((N, 2), (size - 1) / 2)
    CD = np.stack([np.diag([pix_scale, pix_scale])] * N)
    psf = dict(psf_n=np.full(N, 3.0), psf_Rd=np.full(N, 1.2), psf_q=np.full(N, 0.95), psf_PA=np.zeros(N))
    args = dict(
        band="g", psf_upsample=1, Hp=9, Wp=9, pix_scale=pix_scale,
        data=np.zeros((N, size, size)), weight=np.ones((N, size, size)), mask=np.zeros((N, size, size), bool),
        is_template=is_template, crpix=crpix, crtan=crtan, crval=np.array([10.0, -30.0]), CD=CD,
        **psf, ps_center0=ps_center, ps_flux0=np.maximum(ps_flux, 1.0), sky_I0=np.zeros(N), pix_init=scene, var=None,
        scene_scale0=1.0, zp=float(zeropoint),
    )  # fmt: skip
    m = build_likelihood(**args)
    m.models.ps_batch_model_0.model.flux.allflux.value = np.maximum(ps_flux[0], 1.0)
    clean = np.asarray(m().flatten("data")).reshape(N, size, size)
    data = clean + residual_sky + rng.normal(0, noise, clean.shape)
    weight = np.full(data.shape, 1 / noise**2)
    mask = np.zeros(data.shape, bool)
    if masked_blob:
        mask[min(2, N - 1), 3:5, 3:5] = True  # a masked blob in one image
    inputs = dict(
        data=data, weight=weight, mask=mask, is_template=is_template, crpix=crpix, crtan=crtan, crval=np.array([10.0, -30.0]),
        CD=CD, **psf, psf_shape=(9, 9), psf_upsample=1, zeropoint=float(zeropoint), ps_center=ps_center, band="g",
        scene_pixel_scale=pix_scale, mjd=56000 + np.arange(N) * 3.0, reference_flux=ps_flux[0],
    )  # fmt: skip
    truth = dict(scene=scene, fluxes=ps_flux, sky=residual_sky, noise=noise, clean=clean)
    return inputs, truth
