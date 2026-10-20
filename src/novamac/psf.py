"""
Moffat PSF parameters from PSF images.

novamac's forward model uses AstroPhot's elliptical Moffat PSF (n, Rd, q,
PA per image), while most pipelines hand out PSF images. These helpers fit
that model to an image with AstroPhot's own PSF model and Levenberg-Marquardt
fitter, so the parameters mean exactly what the forward model takes:

    I(R) = I0 / (1 + (R / Rd)^2)^n   on an ellipse of axis ratio q at angle PA

with Rd in pixels of the image the PSF belongs to (AstroPhot PSF models work
in pixel space; an oversampled PSF image is declared with `upsample` and Rd
stays in the target image's pixels), PA in radians.

    from novamac import fit_moffat_psf, fit_moffat_psfs
    params = fit_moffat_psf(psf_image)                 # dict n, Rd, q, PA (+ the fit)
    psf = fit_moffat_psfs(psf_images)                  # arrays psf_n, psf_Rd, psf_q, psf_PA for Novamac
    nm = Novamac(..., **psf)                           # or Novamac(..., psf_images=psf_images)
"""

import astrophot as ap
import numpy as np

from .utils import log


def fit_moffat_psf(psf, upsample=1, variance=None, ellipse=True, init=None, max_iter=100, relative_tolerance=1e-4, verbose=0):
    """
    Fit a Moffat PSF model to one PSF image.

    psf        (h, w) image of the PSF, any normalisation (it is normalised to
               unit sum), background-subtracted, the PSF centred (odd sizes
               are best)
    upsample   the image's oversampling relative to the target image: Rd is
               returned in target pixels
    variance   (h, w) per-pixel variance for the fit's weights; None uses a
               flat variance, i.e. a plain least-squares fit of the shape
    ellipse    elliptical (q, PA free) or circular (q = 1, PA = 0) Moffat
    init       dict of starting values (n, Rd, q, PA); AstroPhot initialises
               the rest from the image
    max_iter, relative_tolerance, verbose   passed to AstroPhot's LM fitter
    Returns a dict with n, Rd (target pixels), q, PA (rad), fwhm (target
    pixels, along the major axis), chi2 (reduced, of the fit), message (the
    fitter's), model (the fitted AstroPhot model) and residual (data minus
    model, normalised PSF units).
    """
    psf = np.asarray(psf, dtype=float)
    if psf.ndim != 2:
        raise ValueError(f"psf must be a 2D image, got shape {psf.shape}")
    if not np.all(np.isfinite(psf)):
        raise ValueError("psf has non-finite pixels")
    if variance is None:
        variance = np.full(psf.shape, (1e-3 * float(psf.max())) ** 2)
    target = ap.PSFImage(data=psf, upsample=int(upsample), variance=np.asarray(variance, dtype=float))
    target.normalize()
    model = ap.Model(
        name="psf", model_type="moffat ellipse psf model" if ellipse else "moffat psf model", target=target,
        **(init or {}),
    )  # fmt: skip
    model.initialize()
    result = ap.fit.LM(model, max_iter=max_iter, relative_tolerance=relative_tolerance, verbose=verbose).fit()
    out = {"n": float(model.n.npvalue), "Rd": float(model.Rd.npvalue)}
    if ellipse:
        out["q"] = float(model.q.npvalue)
        out["PA"] = float(np.mod(model.PA.npvalue, np.pi))
    else:
        out["q"], out["PA"] = 1.0, 0.0
    out["fwhm"] = 2.0 * out["Rd"] * np.sqrt(2.0 ** (1.0 / out["n"]) - 1.0)
    out["chi2"] = float(result.loss_history[-1]) if len(result.loss_history) else np.nan
    out["message"] = str(result.message)
    out["model"] = model
    out["residual"] = np.asarray(target.data) - np.asarray(model().data)
    return out


def fit_moffat_psfs(psf_images, upsample=1, variance=None, ellipse=True, init=None, max_iter=100, relative_tolerance=1e-4, verbose=0):
    """
    fit_moffat_psf for a stack of PSF images (N, h, w) (or a list of 2D
    images of any sizes), one per epoch. Returns the arrays Novamac takes:
    {"psf_n", "psf_Rd", "psf_q", "psf_PA"}, each (N,). `variance` may be
    None, one (h, w) array for all, or a list per image.
    """
    out = {k: [] for k in ("psf_n", "psf_Rd", "psf_q", "psf_PA")}
    for i, img in enumerate(psf_images):
        var = variance[i] if isinstance(variance, (list, tuple)) or (isinstance(variance, np.ndarray) and variance.ndim == 3) else variance
        fit = fit_moffat_psf(img, upsample=upsample, variance=var, ellipse=ellipse, init=init, max_iter=max_iter,
                             relative_tolerance=relative_tolerance, verbose=verbose)  # fmt: skip
        for k in ("n", "Rd", "q", "PA"):
            out[f"psf_{k}"].append(fit[k])
        log(f"PSF {i}: n {fit['n']:.3f}, Rd {fit['Rd']:.3f} px, q {fit['q']:.3f}, PA {fit['PA']:.3f} rad "
            f"(FWHM {fit['fwhm']:.2f} px, chi2 {fit['chi2']:.3g}; {fit['message']})")
    return {k: np.array(v) for k, v in out.items()}
