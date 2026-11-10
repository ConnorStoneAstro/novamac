"""
Analytic diagonals of the scene -> data Jacobian.

The scene enters every image linearly: image_n = scale * R_n (K_n * I), with I
the scene intensities, K_n the image's Moffat PSF rendered on the scene grid
(sum 1), * a convolution and R_n the resampling of the scene onto the image's
pixel grid through its WCS (align.resample_scene: flux per data pixel, which
carries one factor of the data pixel area). In the sampler's log-intensity
coordinate x = log I the Jacobian is J_{(n,p), i} = I_i scale (R_n K_n)_{p,i},
and the sampler needs two of its diagonals:

  scene curvature  diag(J^T W J)_i   = I_i^2 scale^2 sum_n sum_p w_{n,p} (R_n K_n)_{p,i}^2
  data broadening  diag(J J^T)_{n,p} = scale^2 sum_i I_i^2 (R_n K_n)_{p,i}^2

The first is the Gauss-Newton term of the mass matrix (the squared S/N at
which the whole stack detects pixel i's flux on its own), the second the
PiGDM-style broadening of the likelihood (hmc.make_broadening_fn). Replacing
the square of the resampled kernel by the resampled square of the kernel,
(R_n K_n)^2 ~= area_n R_n (K_n^2) (R_n / area_n averages sub x sub bilinear
samples whose weights sum to one, so this is Jensen's inequality: exact when
K^2 is flat across the scene samples under a data pixel, otherwise an
overestimate), both become one FFT per image:

  curvature_i = I_i^2 scale^2 G_i,   G_i = sum_n area_n [(R_n^T w_n) (x) K_n^2]_i
  jj_{n,p}    = scale^2 area_n [R_n (K_n^2 * I^2)]_p

with (x) a correlation. The bilinear interpolation itself smooths the kernel
(the samples sit at fractional scene-pixel offsets), which matters when the
PSF is only a pixel or two wide, so K_n^2 stands for the expectation, over the
offsets of that image's pixels, of the smoothed kernel squared
(sampled_kernels_sq). G depends on the PSFs, the weights (masks, coverage)
and the WCS only, so it is built once (one resampler adjoint and one FFT per
image); both diagonals are then functions of the squared intensities alone and
cost one FFT per image to refresh. Validation (2026-11): on the synthetic
test cutouts (1x upsampling, Rd 1.2 px, pure shifts) the curvature matches
exact single-pixel Gauss-Newton probes to 0.5% away from the scene's edge
(where the renderer's treatment of PSF spill off the scene is not modelled),
and diag(JJ^T) matches Hutchinson to 1% on the bright pixels; on a
Rubin-like mock (44 rotated cutouts, 3x upsampled scene) the curvature is
within 1.00-1.12x of the exact probes (median 1.02) from the brightest to the
faintest pixels and diag(JJ^T) within 10% of Hutchinson.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.signal import fftconvolve

from ._backend import DTYPE
from .align import resample_scene, scene_coordinates
from .init_scene import cd_arcsec


def moffat_kernels(psf_n, psf_Rd, psf_q, psf_PA, psf_shape, upsample):
    """(N, kh, kw) elliptical Moffat PSFs on the scene grid, each normalised to
    sum 1: the Moffat parameters are in data pixels (n, Rd, axis ratio q, PA
    in radians), `psf_shape` the PSF footprint in data pixels and `upsample`
    the number of scene pixels per data pixel. Each kernel pixel is integrated
    with Simpson's rule on 3 x 3 points, as AstroPhot renders the PSF
    (sampling_mode "simpsons"): at Rd ~ 1 pixel the point-sampled Moffat is
    much too peaked (sum K^2 twice the rendered PSF's)."""
    kh = (int(psf_shape[0] * upsample) | 1, int(psf_shape[1] * upsample) | 1)
    yy, xx = (jnp.asarray(a, DTYPE) for a in np.mgrid[-(kh[0] // 2) : kh[0] // 2 + 1, -(kh[1] // 2) : kh[1] // 2 + 1])
    n, Rd, q, PA = (jnp.asarray(a, DTYPE)[:, None, None] for a in (psf_n, psf_Rd, psf_q, psf_PA))
    K = jnp.zeros((len(psf_n),) + kh, DTYPE)
    for dy, wy in zip((-0.5, 0.0, 0.5), (1.0, 4.0, 1.0)):
        for dx, wx in zip((-0.5, 0.0, 0.5), (1.0, 4.0, 1.0)):
            x, y = (xx + dx) / upsample, (yy + dy) / upsample  # data pixels
            xr = x * jnp.cos(PA) + y * jnp.sin(PA)
            yr = -x * jnp.sin(PA) + y * jnp.cos(PA)
            K = K + (wy * wx / 36.0) * (1.0 + (xr**2 + (yr / q) ** 2) / Rd**2) ** (-n)
    return K / jnp.sum(K, axis=(1, 2), keepdims=True)


def sampling_offset_histogram(crpix, crtan, CD, center, scene_pixel_scale, scene_shape, data_shape, sub=3, m=8):
    """(N, m*m) histogram, per image, of the fractional scene-pixel offsets
    (row, column) at which the resampler samples the scene for that image
    (align.scene_coordinates, sub x sub samples per pixel): one bin for a
    pure shift, spread over all of them for a rotated or rescaled cutout."""
    CD = cd_arcsec(CD)
    hist = np.empty((len(crpix), m * m))
    for n in range(len(crpix)):
        r, c = scene_coordinates(scene_shape, *(jnp.asarray(a[n], DTYPE) for a in (crpix, crtan, CD)), jnp.asarray(center, DTYPE),
                                 float(scene_pixel_scale), tuple(data_shape), sub)  # fmt: skip
        fy, fx = (np.asarray(a, dtype=np.float64).ravel() % 1.0 for a in (r, c))
        counts, _, _ = np.histogram2d(fy, fx, bins=m, range=[[0, 1], [0, 1]])
        hist[n] = counts.ravel() / counts.sum()
    return hist


def sampled_kernels_sq(kernels, offset_hist):
    """(N, kh, kw): for each image the expectation over its sampling offsets
    f (sampling_offset_histogram, m x m bins) of the squared kernel as
    bilinear interpolation sees it from the pixel the sample falls in,
    A_f(d) = sum_k b_k(f) K(d + k) over the 2 x 2 corners k."""
    m = int(round(np.sqrt(offset_hist.shape[1])))
    f = (np.arange(m) + 0.5) / m
    fy, fx = (a.ravel() for a in np.meshgrid(f, f, indexing="ij"))
    weights = jnp.asarray(np.stack([(1 - fy) * (1 - fx), (1 - fy) * fx, fy * (1 - fx), fy * fx], axis=1), DTYPE)  # (m*m, 4)
    kh, kw = kernels.shape[1:]
    padded = jnp.pad(kernels, ((0, 0), (0, 1), (0, 1)))
    # K(d + k) for k = (0, 0), (0, 1), (1, 0), (1, 1)
    shifted = jnp.stack([padded[:, :kh, :kw], padded[:, :kh, 1:], padded[:, 1:, :kw], padded[:, 1:, 1:]], axis=1)  # (N, 4, kh, kw)

    def expected_sq(k4, p):
        return jnp.einsum("b,bij->ij", p, jnp.einsum("bk,kij->bij", weights, k4) ** 2)

    return jax.vmap(expected_sq)(shifted, jnp.asarray(offset_hist, DTYPE))


class SceneJacobian:
    """The two Jacobian diagonals above for one set of cutouts (see the
    module docstring). `crpix`, `crtan`, `CD` (N, ...) are the images' WCS,
    `center` the scene centre on the tangent plane (arcsec), `scene_pixel_scale`
    and `data_shape` / `scene_shape` the grids, `weight` and `mask` (N, H, W)
    the data inverse variance and mask, the psf_* arrays (N,) the Moffat PSFs
    (data pixels) with their footprint `psf_shape`, `sub` the resampler's
    sub-sampling (the scene pixels per data pixel)."""

    def __init__(self, crpix, crtan, CD, center, scene_pixel_scale, scene_shape, data_shape, weight, mask,
                 psf_n, psf_Rd, psf_q, psf_PA, psf_shape, sub=3):  # fmt: skip
        CD = cd_arcsec(CD)
        self.n_images, self.data_shape, self.scene_shape = len(crpix), tuple(data_shape), tuple(scene_shape)
        self.area = jnp.asarray(np.abs(np.linalg.det(CD)), DTYPE)  # data pixel area per image, arcsec^2
        upsample = float(np.median(np.sqrt(np.abs(np.linalg.det(CD)))) / scene_pixel_scale)
        crpix_j, crtan_j, CD_j = (jnp.asarray(a, DTYPE) for a in (crpix, crtan, CD))
        center_j, pix_scale = jnp.asarray(center, DTYPE), float(scene_pixel_scale)

        # The kernels below are what bilinear interpolation sees from the scene
        # pixel each sample falls in, so the scene is read at those pixels
        # (order 0), not interpolated again
        def resample(scene, n):
            return resample_scene(scene, crpix_j[n], crtan_j[n], CD_j[n], center_j, pix_scale, self.data_shape, sub=sub, order=0)

        self._resample = resample
        kernels = moffat_kernels(psf_n, psf_Rd, psf_q, psf_PA, psf_shape, upsample)
        hist = sampling_offset_histogram(crpix, crtan, CD, center, scene_pixel_scale, scene_shape, data_shape, sub)
        self.kernels_sq = sampled_kernels_sq(kernels, hist)
        self.weights = jnp.asarray(np.where(np.asarray(mask, dtype=bool) | ~np.isfinite(weight) | (weight <= 0), 0.0, weight), DTYPE)
        self.build()

    def build(self):
        """G (the curvature per unit I^2 scale^2) and the jj function from the
        current kernels_sq and weights."""
        resample, n_images = self._resample, self.n_images

        def geometry(weights):
            def one(n, w):
                rtw = jax.vjp(lambda s: resample(s, n), jnp.zeros(self.scene_shape, DTYPE))[1](w)[0]
                return self.area[n] * fftconvolve(rtw, self.kernels_sq[n][::-1, ::-1], mode="same")

            return jnp.sum(jax.vmap(one)(jnp.arange(n_images), weights), axis=0)

        self.G = np.asarray(jax.jit(geometry)(self.weights), dtype=np.float64)  # (Hs, Ws)

        def jj(I_sq):
            spread = jax.vmap(lambda k: fftconvolve(I_sq, k, mode="same"))(self.kernels_sq)  # (N, Hs, Ws): K^2 * I^2
            return (self.area[:, None, None] * jax.vmap(resample)(spread, jnp.arange(n_images))).ravel()

        self._jj = jax.jit(jj)

    def curvature(self, I_sq, scale):
        """diag(J^T W J) in log I per scene pixel (Hs, Ws), from the squared
        intensities `I_sq` (Hs, Ws) and the scene scale."""
        return float(scale) ** 2 * np.asarray(I_sq, dtype=np.float64) * self.G

    def jj_diag(self, I_sq, scale):
        """diag(J J^T) per data pixel, flattened in the data order (N, H, W),
        from the squared intensities `I_sq` (Hs, Ws) and the scene scale."""
        return float(scale) ** 2 * np.asarray(self._jj(jnp.asarray(I_sq, DTYPE)), dtype=np.float64)
