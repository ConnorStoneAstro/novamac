"""Moffat parameters from PSF images, and Novamac taking PSF images directly."""

import astrophot as ap
import numpy as np

from novamac import Novamac, fit_moffat_psf, fit_moffat_psfs
from novamac.testing import make_synthetic_inputs


def render_moffat(n, Rd, q, PA, size=31, upsample=1):
    target = ap.PSFImage(data=np.zeros((size * upsample, size * upsample)), upsample=upsample)
    model = ap.Model(name="truth", model_type="moffat ellipse psf model", target=target, n=n, Rd=Rd, q=q, PA=PA)
    return np.asarray(model().data)


def test_fit_moffat_psf_recovers_the_parameters():
    truth = dict(n=2.5, Rd=2.0, q=0.8, PA=0.7)
    img = render_moffat(**truth)
    rng = np.random.default_rng(0)
    noisy = img + rng.normal(0, 1e-4 * img.max(), img.shape)
    fit = fit_moffat_psf(noisy)
    assert abs(fit["n"] / truth["n"] - 1) < 0.05
    assert abs(fit["Rd"] / truth["Rd"] - 1) < 0.05
    assert abs(fit["q"] - truth["q"]) < 0.03
    assert min(abs(fit["PA"] - truth["PA"]), np.pi - abs(fit["PA"] - truth["PA"])) < 0.05
    assert fit["fwhm"] > 0 and np.isfinite(fit["chi2"])
    assert fit["residual"].shape == img.shape and np.abs(fit["residual"]).max() < 0.02 * img.max()


def test_fit_moffat_psf_circular_and_upsampled():
    img = render_moffat(n=3.0, Rd=1.5, q=1.0, PA=0.0, size=21, upsample=3)  # 63 px: PSF images must be odd
    fit = fit_moffat_psf(img, upsample=3, ellipse=False)
    assert abs(fit["Rd"] / 1.5 - 1) < 0.05 and abs(fit["n"] / 3.0 - 1) < 0.05
    assert fit["q"] == 1.0 and fit["PA"] == 0.0


def test_novamac_takes_psf_images(prior_dir, tmp_path):
    inputs, _ = make_synthetic_inputs(size=16, n_template=2, n_sn=1)  # three epochs: each PSF fit takes seconds
    psf_images = np.stack([render_moffat(inputs["psf_n"][i], inputs["psf_Rd"][i], inputs["psf_q"][i], inputs["psf_PA"][i], size=9)
                           for i in range(len(inputs["data"]))])  # fmt: skip
    fitted = fit_moffat_psfs(psf_images[:2])  # the batch helper's arrays (Novamac below fits every epoch itself)
    assert fitted["psf_Rd"].shape == (2,) and np.allclose(fitted["psf_Rd"], inputs["psf_Rd"][:2], rtol=0.1)
    kwargs = {k: v for k, v in inputs.items() if not k.startswith("psf_")}
    nm = Novamac(prior=prior_dir, out_dir=tmp_path, verbose=False, psf_images=psf_images, psf_upsample=1, **kwargs)
    assert nm.psf_shape == (9, 9)
    assert np.allclose(nm.psf_Rd, inputs["psf_Rd"], rtol=0.1) and np.allclose(nm.psf_n, inputs["psf_n"], rtol=0.1)
    nm.build_model()
