"""
Initial estimates from the data: the initial scene (a WCS-based stack of
the template visits on the scene grid), the per-image sky level, and
aperture point-source fluxes.

The initial scene
-----------------
The start of the annealing is x_start = x_init + sigma(t_start) * z. The
Tweedie denoiser then removes Gaussian noise as sigma_t falls, but anything
in x_init that is not Gaussian noise it keeps as structure. A single visit
interpolated to the scene grid is full of exactly that: its sky noise
becomes blocks, the softplus keeps the positive ones as compact bright
blocks in log space, and the denoiser reads them as faint sources that the
likelihood cannot remove (they sit below the stacked noise of all epochs).
So the init has to be a clean estimate of the static scene:

  1. stack_templates: inverse-variance-weighted mean of the template visits
     (no point source; all images if fewer than `min_templates`), drizzled
     onto the scene grid through each image's WCS. A data pixel's light is
     not put at its centre: the pixel is cut into sub x sub subpixels, fine
     enough that several fall into every scene pixel, each carrying its
     share of the pixel's flux, and each is deposited in the scene pixel its
     centre maps to. Masked pixels and subpixels landing outside the scene
     contribute nothing, and the number of images with a valid pixel under
     each scene pixel is kept (Rubin visits are rotated against each other,
     so the corners are covered by fewer images; DES cutouts share a grid
     and the count is flat inside the cutout). The PSF margin of the scene,
     outside every cutout, has count 0.
  2. soft_threshold: pixels below a few times the stack's own noise are
     blended toward zero flux, which the prior's transform then puts on its
     sky pedestal, so the sky starts with no texture at all and faint real
     sources below the threshold start absent (the data and the prior can
     create them). The noise is measured from the stack itself (robust
     scatter of its faint pixels, clipped), not from the variance maps, and
     scaled per pixel by the coverage.
  3. prior_transform: the scene goes through the prior's own flux transform,
     s * softplus(f / s) with f in the prior's units, exactly as the prior's
     training images did: bright pixels are untouched, the sky lands on the
     prior's pedestal s * log 2 and negative sky pixels compress smoothly
     toward zero instead of becoming deep pits in log I that the prior has
     never seen.

Conventions (AstroPhot, established empirically by rendering a single bright
scene pixel through novamac's forward model while moving the model centre,
the image offset crtan and the CD matrix): in the data array data[row, col]
and in the scene array I[row, col] alike, the first tangent-plane coordinate
x runs along the COLUMN axis and y along the ROW axis. Pixel coordinates are
(i, j) = (col, row), crpix = (reference col, reference row), and
(x, y) = CD @ (col - crpix[0], row - crpix[1]) + crtan with CD in arcsec per
pixel. The scene element I[row, col] sits at x = (col - (W - 1) / 2) * pixscale + cx,
y = (row - (H - 1) / 2) * pixscale + cy with (cx, cy) the model centre and
PA = 0. Checks: a model centre x + 1" moves the rendered scene + 1" / pixscale
columns, an image crtan x + 1" moves it the same amount the other way, and
with aligned grids the rendered image equals the scene array element for
element. tests/test_init_scene.py has the round trip through the forward
model that catches a transposed axis.
"""

import numpy as np


def cd_arcsec(CD):
    """CD matrices in arcsec per pixel (cutout files may carry degrees)."""
    CD = np.asarray(CD, dtype=float)
    if np.sqrt(abs(np.linalg.det(CD[0]))) < 1e-2:  # degrees per pixel
        CD = CD * 3600.0
    return CD


def data_pixel_scale(CD):
    """Median data pixel scale in arcsec from the CD matrices."""
    CD = cd_arcsec(CD)
    return float(np.median(np.sqrt(np.abs(np.linalg.det(CD)))))


def border_sky(data, mask, width=4):
    """Per-image residual sky: median of the unmasked outer `width` pixels."""
    N, H, W = data.shape
    edge = np.ones((H, W), bool)
    if width > 0:
        edge[width:-width, width:-width] = False
    out = np.zeros(N)
    for n in range(N):
        sel = edge & ~mask[n]
        out[n] = np.median(data[n][sel]) if sel.any() else 0.0
    return out


def stack_templates(data, weight, mask, is_template, crpix, crtan, CD, scene_shape, pix_scale, center, sky=None,
                    min_templates=4, subsample=5):
    """
    Weighted mean of the template images on the scene grid, in data units
    per arcsec^2 (the pixelated model's surface-brightness convention), by
    drizzling: every data pixel is split into sub x sub subpixels with
    `subsample` subpixels per scene pixel across the data pixel (so sub =
    ceil(subsample * data_pixel / scene_pixel)), each carrying 1 / sub^2 of
    the pixel's flux and weight, and deposited in the scene pixel its centre
    maps to. Light therefore spreads over the pixel's true footprint,
    rotated or not, instead of sitting at its centre.

    Template images are used if there are at least `min_templates` of them,
    otherwise every image (the point source is then in the stack).

    Returns (stack, count, n_used, used_templates): stack (H, W), count
    (H, W) = number of images with a valid pixel under each scene pixel,
    n_used = number of images stacked, used_templates = whether only
    templates were used. Where count is 0 the stack is 0.
    """
    data = np.asarray(data, dtype=float)
    weight = np.asarray(weight, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    is_template = np.asarray(is_template, dtype=bool)
    crpix, crtan = np.asarray(crpix, dtype=float), np.asarray(crtan, dtype=float)
    CD = cd_arcsec(CD)
    used_templates = int(is_template.sum()) >= min_templates
    use = np.where(is_template)[0] if used_templates else np.arange(len(data))
    if sky is None:
        sky = np.zeros(len(data))

    H, W = scene_shape
    num = np.zeros(H * W)
    den = np.zeros(H * W)
    count = np.zeros(H * W, dtype=int)
    for n in use:
        area = abs(np.linalg.det(CD[n]))  # arcsec^2 per data pixel
        data_pixel = np.sqrt(area)
        sub = int(np.ceil(subsample * data_pixel / pix_scale))
        valid = (~mask[n]) & np.isfinite(data[n]) & np.isfinite(weight[n]) & (weight[n] > 0)
        if not valid.any():
            continue
        rows, cols = np.nonzero(valid)
        w = weight[n][valid] / sub**2
        d = (data[n][valid] - sky[n]) / area  # surface brightness
        # subpixel centres within a pixel: offsets in (-0.5, 0.5)
        off = (np.arange(sub) + 0.5) / sub - 0.5
        dr, dc = np.meshgrid(off, off, indexing="ij")
        row_sub = (rows[:, None] + dr.ravel()[None, :]).ravel()  # (n_valid * sub^2,)
        col_sub = (cols[:, None] + dc.ravel()[None, :]).ravel()
        # pixel coordinates are (col, row); see the module docstring
        xy = CD[n] @ np.stack([col_sub - crpix[n][0], row_sub - crpix[n][1]]) + crtan[n][:, None]  # tangent plane, arcsec
        # scene pixel containing each subpixel centre: x along columns, y along rows
        c = np.rint((xy[0] - center[0]) / pix_scale + (W - 1) / 2).astype(int)
        r = np.rint((xy[1] - center[1]) / pix_scale + (H - 1) / 2).astype(int)
        inside = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        flat = (r * W + c)[inside]
        w_sub = np.repeat(w, sub**2)[inside]
        d_sub = np.repeat(d, sub**2)[inside]
        num_n = np.bincount(flat, weights=w_sub * d_sub, minlength=H * W)
        den_n = np.bincount(flat, weights=w_sub, minlength=H * W)
        num += num_n
        den += den_n
        count += den_n > 0
    stack = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    return stack.reshape(H, W), count.reshape(H, W), len(use), used_templates


def robust_noise(values, n_iter=5, clip=3.0):
    """Robust scatter of `values` about their median, iterating a sigma clip
    so that sources do not inflate it (1.4826 x MAD)."""
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    sigma = 1.4826 * np.median(np.abs(v - np.median(v)))
    for _ in range(n_iter):
        keep = np.abs(v - np.median(v)) < clip * max(sigma, 1e-30)
        if keep.sum() < 10:
            break
        v = v[keep]
        sigma = 1.4826 * np.median(np.abs(v - np.median(v)))
    return float(sigma)


def soft_threshold(f, count, k=3.0, width=1.0):
    """
    Blend f toward 0 below the noise: f * w with w = sigmoid((f - k s_i) / (width s_i)),
    s_i the noise measured on the fully covered pixels scaled by
    sqrt(count_max / count_i); pixels with no coverage go to 0.

    Returns (f_clean, sigma_full, fraction_zeroed) where fraction_zeroed is
    the fraction of covered pixels with w < 0.5.
    """
    f = np.asarray(f, dtype=float)
    count = np.asarray(count)
    full = count == count.max()
    sigma = robust_noise(f[full]) if full.any() else robust_noise(f)
    if not np.isfinite(sigma) or sigma <= 0:
        return f, sigma, 0.0
    s_i = sigma * np.sqrt(count.max() / np.maximum(count, 1))
    w = 1.0 / (1.0 + np.exp(-np.clip((f - k * s_i) / (width * s_i), -50, 50)))
    w = np.where(count > 0, w, 0.0)
    covered = count > 0
    frac = float(np.mean(w[covered] < 0.5)) if covered.any() else 1.0
    return f * w, sigma, frac


def prior_transform(scene, scene_scale, softening, prior_scene_factor=1.0):
    """
    The scene (data units per arcsec^2) through the prior's flux transform,
    s * softplus(f / s) with f = scene / scene_scale * prior_scene_factor in
    the prior's units (nJy per prior pixel, times the factor the prior is
    shown), back in the scene's units. Returns (softened scene, number of
    negative input pixels).
    """
    f_prior = np.asarray(scene, dtype=float) / scene_scale * prior_scene_factor
    softened = softening * np.logaddexp(0.0, f_prior / softening) / prior_scene_factor
    return softened * scene_scale, int(np.sum(f_prior < 0))


def moffat_fwhm(n, Rd):
    """FWHM of a Moffat profile with index n and scale radius Rd (same units as Rd;
    AstroPhot's PSF Rd is in pixels of the image the PSF belongs to)."""
    return 2.0 * np.asarray(Rd, dtype=float) * np.sqrt(2.0 ** (1.0 / np.asarray(n, dtype=float)) - 1.0)


def moffat_enclosed(r, n, Rd):
    """Fraction of a circular Moffat's flux inside radius r."""
    return 1.0 - (1.0 + (r / Rd) ** 2) ** (1.0 - n)


def source_pixel_position(crpix, crtan, CD, position):
    """(col, row) of a tangent-plane position (x, y) in each image: the
    inverse of (x, y) = CD @ (col - crpix[0], row - crpix[1]) + crtan."""
    CD = cd_arcsec(CD)
    crpix, crtan = np.asarray(crpix, dtype=float), np.asarray(crtan, dtype=float)
    out = np.zeros((len(CD), 2))
    for n in range(len(CD)):
        out[n] = np.linalg.solve(CD[n], np.asarray(position, dtype=float) - crtan[n]) + crpix[n]
    return out


def aperture_fluxes(residual, weight, mask, crpix, crtan, CD, position, psf_n, psf_Rd, radius_factor=1.5, min_radius=2.5):
    """
    Aperture flux of a point source at tangent-plane `position` in every
    image of `residual` (data with the static scene and sky removed), in the
    data's flux units: the sum of the unmasked pixels within
    radius_factor x the Moffat FWHM of the source (at least `min_radius`
    pixels, so a sharp PSF still gets a few pixels), corrected by the
    fraction of a circular Moffat enclosed in that radius. psf_Rd is in
    pixels, as AstroPhot's PSF models have it. Returns (flux, error) with
    the error from the weights (1 / variance). Pixels the likelihood ignores
    (masked, or with zero / non-finite weight) are left out, so an epoch
    without a detection is consistent with zero within its error.
    """
    residual = np.asarray(residual, dtype=float)
    weight = np.asarray(weight, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    N, H, W = residual.shape
    CD = cd_arcsec(CD)
    pos = source_pixel_position(crpix, crtan, CD, position)
    rows, cols = np.mgrid[:H, :W]
    flux = np.zeros(N)
    err = np.zeros(N)
    for n in range(N):
        fwhm = float(moffat_fwhm(psf_n[n], psf_Rd[n]))  # pixels
        r_pix = max(radius_factor * fwhm, min_radius)
        usable = ~mask[n] & np.isfinite(residual[n]) & np.isfinite(weight[n]) & (weight[n] > 0)  # as the likelihood counts them
        inside = ((cols - pos[n, 0]) ** 2 + (rows - pos[n, 1]) ** 2 <= r_pix**2) & usable
        if not inside.any():
            continue
        enclosed = float(moffat_enclosed(r_pix, psf_n[n], psf_Rd[n]))
        flux[n] = residual[n][inside].sum() / enclosed
        w = weight[n][inside]
        var = np.where(w > 0, 1.0 / np.where(w > 0, w, 1.0), 0.0)
        err[n] = np.sqrt(var.sum()) / enclosed
    return flux, err


def prepare_initial_scene(data, weight, mask, is_template, crpix, crtan, CD, scene_shape, pix_scale, center, *,
                          scene_scale, softening, prior_scene_factor=1.0, init_image=None, sky=None,
                          threshold_sigma=3.0, min_templates=4, subsample=5):
    """
    The sampler's initial scene: `init_image` (data units per arcsec^2 on
    the scene grid) if given, else the drizzled template stack; then the
    soft threshold (None disables) and the prior's transform.

    Returns (scene, info) with info holding `count` (coverage per scene
    pixel), `raw` (the scene before threshold and transform), `noise` (the
    threshold's noise estimate) and `report` (lines for the log).
    """
    H, W = scene_shape
    report = []
    if init_image is not None:
        raw = np.asarray(init_image, dtype=float)
        if raw.shape != (H, W):
            raise ValueError(f"init_image has shape {raw.shape}, the scene grid is {(H, W)}")
        count = np.ones((H, W), dtype=int)
        report.append("initial scene supplied by the caller")
    else:
        raw, count, n_used, used_templates = stack_templates(
            data, weight, mask, is_template, crpix, crtan, CD, (H, W), pix_scale, center, sky,
            min_templates=min_templates, subsample=subsample,
        )  # fmt: skip
        covered = count > 0
        report.append(
            f"initial scene drizzled from {n_used} {'template' if used_templates else 'image (fewer than ' + str(min_templates) + ' templates: all images, point source included)'}"
            f"{'s' if used_templates and n_used != 1 else ''} on the scene grid ({subsample} subpixels per scene pixel): "
            f"coverage {int(count.max())} images at most, "
            f"{100 * (1 - covered.mean()):.0f}% of the scene (the PSF margin) outside every image"
            + (f"; per-image border sky median {np.median(sky):.3g} (range [{np.min(sky):.3g}, {np.max(sky):.3g}]) subtracted" if sky is not None else "")
        )
    scene = raw.copy()
    noise = np.nan
    if threshold_sigma is not None:
        scene, noise, frac = soft_threshold(scene, count, threshold_sigma)
        report.append(
            f"soft threshold at {threshold_sigma:g} x the init's own noise ({noise:.3g} data units/arcsec^2 = "
            f"{noise / scene_scale:.3g} nJy/prior px): {100 * frac:.0f}% of the covered pixels start on the sky pedestal"
        )
    scene, n_neg = prior_transform(scene, scene_scale, softening, prior_scene_factor)
    pedestal = softening * np.log(2.0)
    report.append(
        f"initial scene through the prior's transform (softening {softening:g} nJy/prior px, pedestal {pedestal:.3g}"
        + (f", prior scene factor {prior_scene_factor:g}: pedestal {pedestal / prior_scene_factor:.3g} in scene units" if prior_scene_factor != 1 else "")
        + f"): {n_neg} of {scene.size} pixels were negative; log I now in [{np.log(scene.min() / scene_scale):.2f}, {np.log(scene.max() / scene_scale):.2f}]"
    )
    return scene, {"count": count, "raw": raw, "noise": noise, "report": report}
