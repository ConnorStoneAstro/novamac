"""
Resampling of a scene onto an image's pixel grid through its WCS (crpix,
crtan, CD), used by the initial-scene estimate (init_scene) to compare a
raw template stack with each epoch.
"""
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

from ._backend import DTYPE


def scene_coordinates(scene_shape, crpix, crtan, CD, center, pix_scale, shape, sub=3):
    """The scene-grid (row, column) coordinates, each (h, w, sub * sub), at
    which an image of `shape` with the WCS (crpix, crtan, CD in arcsec)
    samples a scene of `scene_shape` centred on `center` (arcsec) with
    `pix_scale`: sub x sub points per data pixel. Conventions as in init_scene."""
    H, W = scene_shape
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
    return r, c


def resample_scene(scene, crpix, crtan, CD, center, pix_scale, shape, sub=3, order=1):
    """
    The scene (H, W) in surface brightness, resampled into an image of
    `shape` with the WCS (crpix, crtan, CD in arcsec) by bilinear
    interpolation (`order` 1; 0 takes the scene pixel each sample falls in)
    at sub x sub points per data pixel, in flux per data pixel.
    Differentiable in crtan. Conventions as in init_scene.
    """
    h, w = shape
    r, c = scene_coordinates(scene.shape, crpix, crtan, CD, center, pix_scale, shape, sub)
    if order == 0:  # floor: the pixel the sample falls in (map_coordinates rounds to nearest)
        r, c = r - 0.5, c - 0.5
    values = map_coordinates(scene, [r.ravel(), c.ravel()], order=order, mode="nearest").reshape(h, w, sub * sub)
    area = jnp.abs(CD[0, 0] * CD[1, 1] - CD[0, 1] * CD[1, 0])
    return values.mean(axis=-1) * area
