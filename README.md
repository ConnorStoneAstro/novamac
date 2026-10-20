# Novamac

Bayesian transient scene modelling: given cutouts of a transient's host
across epochs, a trained diffusion prior on host-galaxy scenes and the usual
calibrations (WCS, PSF, zero point), `novamac` samples the joint posterior
over the static scene, the transient's light curve and the nuisance
parameters (image alignment, PSF shape, sky levels, source position, the
scene's overall scale) with an annealed Hamiltonian Monte Carlo that runs the
prior's reverse diffusion and the likelihood together.

## Use

```python
from novamac import Novamac

nm = Novamac(
    data=data, weight=weight, mask=mask, is_template=is_template,       # (N, H, W), (N,)
    crpix=crpix, crtan=crtan, crval=crval, CD=CD,                        # WCS per image
    psf_n=psf_n, psf_Rd=psf_Rd, psf_q=psf_q, psf_PA=psf_PA,              # Moffat PSF per image
    zeropoint=27.5, ps_center=[[x, y]],                                  # units; transient position (arcsec)
    prior="path/to/host_prior/final",                                    # checkpoint directory
    out_dir="fits/01234567", name="01234567_g",
    num_chains=4, anneal_steps=4096,                                     # any sampler setting
)
nm.sample()      # writes chain files, trace, Tweedie estimates, timings to out_dir
nm.plot_all()    # every diagnostic figure, from the products on disk
```

Everything the inference needs is given at construction; reading a
particular survey's data and preparing it (cutting cutouts, calibrations,
units) stays outside the package. Optional inputs are estimated from the data
when not given: the initial scene (a WCS-based drizzled stack of the template
epochs, `novamac.init_scene`), the per-image sky (border median) and the
initial point-source fluxes (aperture photometry of the data minus the
initial scene). Every sampler setting is a documented class attribute of
`Novamac` (`Novamac.setting_names()`); pass any as a keyword to override per
instance, or set it on the class for every instance.

Diagnostics for use at scale: `nm.describe_init()` (where the initial scene
sits relative to the prior) and `nm.diagnose_stall()` (post-mortem of a
stalled chain). Test modes: `likelihood=False` samples the prior alone,
`freeze_non_scene=True` keeps every non-scene block frozen.

`novamac.testing` makes a random (untrained) prior checkpoint and
model-consistent synthetic cutouts, enough to run the whole pipeline without
data.

## Layout

| module | role |
|---|---|
| `inference.py` | the `Novamac` class: inputs, settings, model construction, the sampling loop, and the entry points to plots and diagnostics |
| `hmc.py` | the sampler: annealed target (Tweedie-denoised likelihood, mean correction), chain-vmapped predictor-corrector chunks with frozen-coordinate masks and the non-scene sub-step, Hutchinson Jacobian diagonal, Gauss-Newton curvature, the exact linear Gibbs refresh. The module docstring explains the method. |
| `sampler.py` | schedule, parameter blocks and release rules, prior energy offset, mass matrix, step size search, chain-file writers |
| `init_scene.py` | the initial estimates from the data: drizzle stack, soft threshold, the prior's flux transform, border sky, aperture fluxes; the AstroPhot array convention |
| `products.py` | the files a run writes and their readers |
| `plots.py`, `diagnostics.py` | figures and post-mortems from the products |
| `build_likelihood.py`, `build_prior.py` | the AstroPhot model and its priors (the host prior from `host_prior`, Gaussian priors on the rest) |
| `host_prior/` | the vendored host-galaxy diffusion prior loader (`load_prior`, `Prior`) |
| `testing.py` | synthetic prior and cutouts |
| `_backend.py` | pins caskade to jax and jax to float32 with full-precision matmuls; imported first |

## Tests

```bash
pip install -e ".[test]"
pytest            # unit tests and one short end-to-end run (a few minutes on a CPU)
pytest -m slow    # the prior-only and frozen end-to-end runs as well
```

CI (`.github/workflows/tests.yml`) runs the default set on a CPU.
