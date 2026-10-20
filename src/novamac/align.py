"""
Initial image alignment: a bounded refinement of each image's tangent-plane
offset (crtan) against the initial scene, before sampling starts.

The input WCS can be off by a fraction of a pixel per epoch. The sampler
corrects that as the scene settles, but a start that is misaligned by a
good fraction of a pixel costs the early annealing (the stacked initial
scene is blurred by the misalignment, and the likelihood pushes every
epoch's residual into the scene). This step moves each image's crtan a
little toward what the data prefer, and is deliberately conservative so
that it cannot bake a pathological start in:

  * the comparison is direct: the raw stack (an image-like, PSF-blurred
    estimate of the sky, before the threshold and the prior's transform)
    resampled into each image's pixel grid through the WCS, plus that
    image's sky level, against the data. It does not go through the forward
    model, which would blur the already-blurred stack a second time and
    leave a mismatch that has nothing to do with alignment;
  * each template is compared with the stack of the OTHER templates
    (leave-one-out stacks, Novamac.build_model): a stack containing the
    image's own noise correlates with it at zero shift, and in a faint
    field that term outweighs the alignment signal and pins the image where
    it started;
  * pixels the likelihood ignores and, in the transient epochs, an aperture
    around the point source are left out, so only the static scene aligns;
  * a fixed number of Adam steps on the chi-square, not a search for the
    optimum: the aim is to remove the gross misalignment, the exact
    alignment comes from the sampler once the scene is in place;
  * the shift is clipped to +-max_shift data pixels of the input WCS;
  * the shift common to every image is removed: all images moving together
    is the scene moving, which the scene pixels absorb, and the prior pins
    the images, so only the relative alignment is a start worth changing;
  * each image's shift is kept only if its own chi-square decreased by at
    least `min_dchi2` (two fitted offsets: 6 is a 95% improvement), else
    that image keeps the input WCS, so noise is not chased;
  * the alignment prior stays centred on the input WCS (build_prior), so a
    bad refinement can only move the starting point, not the target.

Afterwards the initial scene is stacked again with the refined offsets
(Novamac.build_model), which sharpens it.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.ndimage import map_coordinates

from ._backend import DTYPE
from .init_scene import cd_arcsec
from .utils import log


def resample_scene(scene, crpix, crtan, CD, center, pix_scale, shape, sub=3):
    """
    The scene (H, W) in surface brightness, resampled into an image of
    `shape` with the WCS (crpix, crtan, CD in arcsec) by bilinear
    interpolation at sub x sub points per data pixel, in flux per data
    pixel. Differentiable in crtan. Conventions as in init_scene.
    """
    H, W = scene.shape
    h, w = shape
    rows, cols = jnp.meshgrid(jnp.arange(h, dtype=DTYPE), jnp.arange(w, dtype=DTYPE), indexing="ij")
    off = (jnp.arange(sub, dtype=DTYPE) + 0.5) / sub - 0.5
    dr, dc = jnp.meshgrid(off, off, indexing="ij")
    rr = rows[..., None] + dr.ravel()
    cc = cols[..., None] + dc.ravel()
    x = CD[0, 0] * (cc - crpix[0]) + CD[0, 1] * (rr - crpix[1]) + crtan[0]
    y = CD[1, 0] * (cc - crpix[0]) + CD[1, 1] * (rr - crpix[1]) + crtan[1]
    c = (x - center[0]) / pix_scale + (W - 1) / 2
    r = (y - center[1]) / pix_scale + (H - 1) / 2
    values = map_coordinates(scene, [r.ravel(), c.ravel()], order=1, mode="nearest").reshape(h, w, sub * sub)
    area = jnp.abs(CD[0, 0] * CD[1, 1] - CD[0, 1] * CD[1, 0])
    return values.mean(axis=-1) * area


def refine_alignment(scene, data, weight, exclude, crpix, crtan, CD, center, pix_scale, sky=None, max_shift=1.0,
                     steps=100, rate=0.05, pixel_scale=None, remove_common=True, min_dchi2=6.0, beta1=0.9, beta2=0.999):
    """
    Refine every image's crtan by `steps` Adam steps on its chi-square
    against `scene` resampled into its pixels (see the module docstring),
    the step size `rate` and the bound `max_shift` both in data pixels
    (`pixel_scale` arcsec per data pixel; the median of the CD matrices
    when None). `scene` is one (H, W) scene for every image or (N, H, W)
    one per image (leave-one-out stacks). `exclude` (N, H, W) marks pixels left out (masked, bad
    weight, the transient's aperture); `sky` (N,) is the per-pixel level
    the stack had subtracted (added back to the resampled scene).

    Returns a dict: crtan (N, 2) the refined offsets (the input where
    rejected), shift (N, 2) in arcsec, chi2_before and chi2_after (N,),
    accepted (N,) and report (lines).
    """
    data = np.asarray(data, dtype=float)
    N, h, w = data.shape
    CD = cd_arcsec(CD)
    if pixel_scale is None:
        pixel_scale = float(np.median(np.sqrt(np.abs(np.linalg.det(CD)))))
    use = ~np.asarray(exclude, dtype=bool) & np.isfinite(data) & np.isfinite(weight) & (np.asarray(weight) > 0)
    wgt = jnp.asarray(np.where(use, weight, 0.0), DTYPE)
    dat = jnp.asarray(np.where(use, data, 0.0), DTYPE)
    sky = np.zeros(N) if sky is None else np.asarray(sky, dtype=float)
    scene = np.asarray(scene, dtype=float)
    scene_j = jnp.asarray(np.broadcast_to(scene, (N,) + scene.shape[-2:]) if scene.ndim == 2 else scene, DTYPE)  # (N, H, W)
    crpix_j, CD_j, sky_j = jnp.asarray(crpix, DTYPE), jnp.asarray(CD, DTYPE), jnp.asarray(sky, DTYPE)
    center_j = jnp.asarray(center, DTYPE)
    a0 = jnp.asarray(crtan, DTYPE)  # (N, 2)
    bound = float(max_shift) * float(pixel_scale)
    lr = float(rate) * float(pixel_scale)
    lo, hi = a0 - bound, a0 + bound

    def chi2_image(n, ct):
        model = resample_scene(scene_j[n], crpix_j[n], ct, CD_j[n], center_j, float(pix_scale), (h, w)) + sky_j[n]
        return jnp.sum(wgt[n] * (dat[n] - model) ** 2)

    chi2_per_image = jax.vmap(chi2_image, in_axes=(0, 0))
    grad = jax.grad(lambda a: jnp.sum(chi2_per_image(jnp.arange(N), a)))

    @jax.jit
    def run(a0):
        def body(i, state):
            a, m, v = state
            g = grad(a)
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g * g
            m_hat = m / (1 - beta1 ** (i + 1))
            v_hat = v / (1 - beta2 ** (i + 1))
            a = a - lr * m_hat / (jnp.sqrt(v_hat) + 1e-12)
            a = jnp.clip(a, lo, hi)
            a = jnp.where(jnp.isfinite(a), a, a0)
            return a, m, v

        a, _, _ = jax.lax.fori_loop(0, int(steps), body, (a0, jnp.zeros_like(a0), jnp.zeros_like(a0)))
        return a, chi2_per_image(jnp.arange(N), a0)

    a, chi2_before = (np.asarray(v, dtype=np.float64) for v in run(a0))
    a0 = np.asarray(a0, dtype=np.float64)
    if remove_common:
        a = a - np.mean(a - a0, axis=0)
        a = np.clip(a, np.asarray(lo), np.asarray(hi))
    chi2_after = np.asarray(chi2_per_image(jnp.arange(N), jnp.asarray(a, DTYPE)), dtype=np.float64)
    accepted = np.isfinite(chi2_after) & (chi2_before - chi2_after >= float(min_dchi2))
    shift = np.where(accepted[:, None], a - a0, 0.0)
    refined = a0 + shift
    at_bound = accepted & (np.max(np.abs(shift), axis=1) >= 0.999 * bound)
    shift_px = shift / pixel_scale
    report = [
        f"initial alignment: {int(accepted.sum())} of {N} images shifted after {steps} Adam steps (bound {max_shift:g} data px); "
        f"shifts |dx|,|dy| median {np.median(np.abs(shift_px[accepted, 0])) if accepted.any() else 0:.3f}, "
        f"{np.median(np.abs(shift_px[accepted, 1])) if accepted.any() else 0:.3f} px, max {np.max(np.abs(shift_px)):.3f} px; "
        f"chi2 {np.sum(chi2_before):.6g} -> {np.sum(np.where(accepted, chi2_after, chi2_before)):.6g} on the pixels compared"
        + (f"; {int(at_bound.sum())} image(s) at the bound (kept: the sampler decides)" if at_bound.any() else "")
    ]
    log(report[0])
    return {"crtan": refined, "shift": shift, "chi2_before": chi2_before, "chi2_after": chi2_after, "accepted": accepted, "report": report}
