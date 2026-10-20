"""The initial estimates from the data: the drizzle stack (self-consistent
synthetic check and the round trip through novamac's forward model, which
catches a transposed axis), the soft threshold, the aperture fluxes."""

import numpy as np
import pytest

from novamac import Novamac, build_likelihood
from novamac.sampler import flat_data_weight
from novamac.init_scene import (
    aperture_fluxes,
    border_sky,
    moffat_enclosed,
    moffat_fwhm,
    prior_transform,
    soft_threshold,
    source_pixel_position,
    stack_templates,
)


def _bilinear(img, rr, cc):
    """Bilinear sample of img at fractional (row, col), clamped at the edges."""
    H, W = img.shape
    rr = np.clip(rr, 0, H - 1)
    cc = np.clip(cc, 0, W - 1)
    r0 = np.floor(rr).astype(int)
    c0 = np.floor(cc).astype(int)
    r1 = np.minimum(r0 + 1, H - 1)
    c1 = np.minimum(c0 + 1, W - 1)
    fr, fc = rr - r0, cc - c0
    return (img[r0, c0] * (1 - fr) * (1 - fc) + img[r1, c0] * fr * (1 - fc)
            + img[r0, c1] * (1 - fr) * fc + img[r1, c1] * fr * fc)  # fmt: skip


@pytest.fixture(scope="module")
def rotated_images():
    """Rotated, offset, masked images of a known scene (two Gaussians)."""
    rng = np.random.default_rng(0)
    H = W = 96
    pix = 0.1
    center = (0.3, -0.2)
    r, c = np.mgrid[:H, :W]  # x along the columns, y along the rows
    x = (c - (W - 1) / 2) * pix + center[0]
    y = (r - (H - 1) / 2) * pix + center[1]
    truth = 50 * np.exp(-((x - 0.5) ** 2 + (y + 0.4) ** 2) / (2 * 0.9**2)) + 8 * np.exp(-((x + 2.0) ** 2 + (y - 1.5) ** 2) / (2 * 0.3**2))

    def make_image(theta, dx, dy, n=30, scale=0.26, sky=0.7, noise=0.4, sub=7):
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        CD = R * scale
        crpix = np.array([(n - 1) / 2, (n - 1) / 2])
        crtan = np.array([dx, dy])
        row, col = np.mgrid[:n, :n]
        off = (np.arange(sub) + 0.5) / sub - 0.5
        dr, dc = np.meshgrid(off, off, indexing="ij")
        rr = (row[..., None] + dr.ravel()).ravel()
        cc = (col[..., None] + dc.ravel()).ravel()
        xy = CD @ np.stack([cc - crpix[0], rr - crpix[1]]) + crtan[:, None]  # (col, row) pixel coordinates
        sr = (xy[1] - center[1]) / pix + (H - 1) / 2
        sc = (xy[0] - center[0]) / pix + (W - 1) / 2
        sb = _bilinear(truth, sr, sc).reshape(n, n, sub * sub).mean(-1)
        area = abs(np.linalg.det(CD))
        data = sb * area + sky + rng.normal(0, noise, (n, n))
        return data, np.full((n, n), 1 / noise**2), np.zeros((n, n), bool), crpix, crtan, CD

    N = 12
    imgs = [make_image(th, dx, dy) for th, dx, dy in zip(rng.uniform(0, 2 * np.pi, N), rng.uniform(-0.4, 0.4, N), rng.uniform(-0.4, 0.4, N))]
    data, weight, mask = (np.stack([m[k] for m in imgs]) for k in range(3))
    crpix, crtan, CD = (np.stack([m[k] for m in imgs]) for k in (3, 4, 5))
    mask[3, 10:14, 10:14] = True
    is_template = np.ones(N, bool)
    is_template[0] = False
    return dict(truth=truth, x=x, data=data, weight=weight, mask=mask, crpix=crpix, crtan=crtan, CD=CD, is_template=is_template,
                H=H, W=W, pix=pix, center=center, noise=0.4)  # fmt: skip


def test_stack_recovers_scene(rotated_images):
    d = rotated_images
    N = len(d["data"])
    sky = border_sky(d["data"], d["mask"], 4)
    assert abs(sky[1] - 0.7) < 0.1
    stack, count, n_used, used_templates = stack_templates(
        d["data"], d["weight"], d["mask"], d["is_template"], d["crpix"], d["crtan"], d["CD"], (d["H"], d["W"]), d["pix"], d["center"], sky
    )
    assert used_templates and n_used == N - 1
    assert count.max() == N - 1 and (count == 0).any()
    corners = [(0, 0), (0, d["W"] - 1), (d["H"] - 1, 0), (d["H"] - 1, d["W"] - 1)]
    assert all(count[i, j] < count.max() for i, j in corners), "rotated images cover the corners with fewer images"
    full = count == count.max()
    err = (stack - d["truth"])[full]
    single_noise = d["noise"] / abs(np.linalg.det(d["CD"][0]))
    assert np.sqrt(np.mean(err**2)) < 1.5 * single_noise / np.sqrt(n_used)
    assert abs(np.mean(err)) < 0.3, "surface brightness conserved"
    assert abs(stack[full].max() / d["truth"][full].max() - 1) < 0.1
    few = stack_templates(
        d["data"], d["weight"], d["mask"], np.array([True] * 3 + [False] * (N - 3)), d["crpix"], d["crtan"], d["CD"],
        (d["H"], d["W"]), d["pix"], d["center"], sky,
    )  # fmt: skip
    assert not few[3] and few[2] == N, "fewer than 4 templates: all images used"


def test_soft_threshold(rotated_images):
    d = rotated_images
    sky = border_sky(d["data"], d["mask"], 4)
    stack, count, _, _ = stack_templates(
        d["data"], d["weight"], d["mask"], d["is_template"], d["crpix"], d["crtan"], d["CD"], (d["H"], d["W"]), d["pix"], d["center"], sky
    )
    cleaned, noise, frac = soft_threshold(stack, count, 3.0)
    peak_faint = np.unravel_index(np.argmax(np.where(d["x"] < -1, d["truth"], 0)), d["truth"].shape)
    assert 0.6 < frac < 0.95
    assert cleaned[peak_faint] > 0.5 * d["truth"][peak_faint], "faint source kept"
    assert np.all(cleaned[count == 0] == 0)


def test_prior_transform():
    scene = np.array([[-5.0, 0.0, 1000.0]])
    out, n_neg = prior_transform(scene, scene_scale=2.0, softening=10.0)
    assert n_neg == 1
    assert np.all(out > 0)
    assert np.isclose(out[0, 1], 2.0 * 10.0 * np.log(2))  # zero flux lands on the pedestal s log 2
    assert np.isclose(out[0, 2], 1000.0, rtol=1e-6)  # bright pixels untouched


def test_stack_roundtrip_through_forward_model():
    """A scene with two marked pixels rendered into rotated, offset images
    of an off-centre model must stack back with the marks in place."""
    SIZE, PIX, N = 32, 0.25, 6
    truth = np.ones((SIZE, SIZE))
    marks = [(10, 20), (22, 7)]
    truth[marks[0]] = 300.0
    truth[marks[1]] = 120.0
    center = np.array([0.5, -0.3])
    rng = np.random.default_rng(1)
    crtan = rng.uniform(-0.4, 0.4, (N, 2))
    theta = rng.uniform(0, 2 * np.pi, N)
    CD = np.stack([np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]) * PIX for t in theta])
    crpix = np.full((N, 2), (SIZE - 1) / 2)
    args = dict(
        band="g", psf_upsample=1, Hp=3, Wp=3, pix_scale=PIX, data=np.zeros((N, SIZE, SIZE)), weight=np.ones((N, SIZE, SIZE)),
        mask=np.zeros((N, SIZE, SIZE), bool), is_template=np.ones(N, bool), crpix=crpix, crtan=crtan, crval=np.array([10.0, -30.0]), CD=CD,
        psf_n=np.full(N, 3.0), psf_Rd=np.full(N, 0.1), psf_q=np.full(N, 0.99), psf_PA=np.zeros(N), ps_center0=center[None],
        ps_flux0=np.ones((1, N)), sky_I0=np.zeros(N), pix_init=truth, var=None, scene_scale0=1.0,
    )  # fmt: skip
    m = build_likelihood(**args)
    m.models.ps_batch_model_0.model.flux.allflux.value = np.full(N, 1e-6)
    img = np.asarray(m().flatten("data")).reshape(N, SIZE, SIZE)
    stack, count, _, _ = stack_templates(img, np.ones_like(img), np.zeros_like(img, bool), np.ones(N, bool), crpix, crtan, CD, (SIZE, SIZE), PIX, center, None)

    def centroid(arr, p, k=2):
        sl = (slice(p[0] - k, p[0] + k + 1), slice(p[1] - k, p[1] + k + 1))
        sub = np.clip(arr[sl] - 1.0, 0, None)
        rr, cc = np.mgrid[sl]
        return float((sub * rr).sum() / sub.sum()), float((sub * cc).sum() / sub.sum())

    for p in marks:
        cen = centroid(stack, p)
        assert abs(cen[0] - p[0]) < 0.1 and abs(cen[1] - p[1]) < 0.1, f"mark at {p} stacks to {cen}"
    flux_ratio = (stack[8:13, 18:23].sum() - 25) / 299.0
    assert abs(flux_ratio - 1) < 0.05, "flux conserved"


def test_source_pixel_position_inverts_wcs():
    rng = np.random.default_rng(3)
    N = 4
    theta = rng.uniform(0, 2 * np.pi, N)
    CD = np.stack([np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]) * 0.2 for t in theta])
    crpix = rng.uniform(5, 15, (N, 2))
    crtan = rng.uniform(-1, 1, (N, 2))
    pos = np.array([0.7, -0.4])
    pix = source_pixel_position(crpix, crtan, CD, pos)
    for n in range(N):
        xy = CD[n] @ (pix[n] - crpix[n]) + crtan[n]
        assert np.allclose(xy, pos)


def test_aperture_fluxes_recover_a_moffat():
    """A circular Moffat of known flux, sampled analytically into pixels."""
    N, H, W, pix = 3, 41, 41, 0.25
    n, Rd = 3.0, 3.2  # Rd in pixels, as AstroPhot's PSF models have it
    flux_true = np.array([500.0, 0.0, 120.0])
    rng = np.random.default_rng(5)
    crpix = np.full((N, 2), 20.0) + rng.uniform(-2, 2, (N, 2))
    crtan = rng.uniform(-0.3, 0.3, (N, 2))
    CD = np.stack([np.diag([pix, pix])] * N)
    pos = np.array([0.35, -0.2])
    ppix = source_pixel_position(crpix, crtan, CD, pos)
    rows, cols = np.mgrid[:H, :W]
    data = np.zeros((N, H, W))
    for i in range(N):
        r = np.hypot(cols - ppix[i, 0], rows - ppix[i, 1])  # pixels
        profile = (n - 1) / (np.pi * Rd**2) * (1 + (r / Rd) ** 2) ** (-n)  # per pixel
        data[i] = flux_true[i] * profile + rng.normal(0, 0.05, (H, W))
    weight = np.full(data.shape, 400.0)
    hot = (int(round(ppix[0, 1])) + 1, int(round(ppix[0, 0])))
    data[0][hot] += 1e4  # a bad pixel inside the aperture ...
    weight[0][hot] = 0.0  # ... that the likelihood ignores must not count
    flux, err = aperture_fluxes(data, weight, np.zeros(data.shape, bool), crpix, crtan, CD, pos,
                                np.full(N, n), np.full(N, Rd), radius_factor=1.5)  # fmt: skip
    assert np.allclose(flux[[0, 2]], flux_true[[0, 2]], rtol=0.08)
    assert abs(flux[1]) < 3 * err[1]
    assert err[0] > 0
    fwhm = moffat_fwhm(n, Rd)
    assert 0.8 < moffat_enclosed(1.5 * fwhm, n, Rd) < 1.0


def test_novamac_estimates_initial_values(tmp_path, prior_dir, synthetic):
    """With no fluxes, sky or scene given, Novamac estimates them from the data."""
    inputs, truth = synthetic
    nm = Novamac(prior=prior_dir, out_dir=tmp_path, name="init", verbose=False, **inputs)
    assert nm.scene_shape == (16, 16)
    nm.build_model()
    sn = ~inputs["is_template"]
    assert nm.ps_flux.shape == (1, len(sn))
    assert np.allclose(nm.ps_flux[0, sn], truth["fluxes"][0, sn], rtol=0.35)
    assert np.all(nm.ps_flux > 0)
    # the border median holds the residual sky plus the host's outer light here (the
    # host fills this tiny field), in flux per arcsec^2 like the model's sky
    area = inputs["CD"][0, 0, 0] ** 2
    assert np.all(nm.sky * area > truth["sky"]) and np.all(nm.sky * area < truth["sky"] + 1.0)
    scene = nm.likelihood_args["pix_init"]
    assert scene.shape == nm.scene_shape and np.all(scene > 0)
    assert nm.init_info["count"].max() == int(inputs["is_template"].sum())
    # the raw stack plus the border level it subtracted recovers the host where it
    # is bright; the prepared scene is that stack thresholded and put on the
    # prior's pedestal (the host fills this tiny field, so the border level is
    # mostly host light and the prepared scene sits low: the sky block carries it)
    bright = truth["scene"] > 10
    raw = nm.init_info["raw"]
    area = inputs["CD"][0, 0, 0] ** 2
    host_in_border = nm.sky.mean() - truth["sky"] / area  # the border level minus the true residual sky
    # per pixel the stack of 5 noisy templates scatters by ~30% here, so compare the mean
    assert abs(np.mean((raw + host_in_border)[bright]) / np.mean(truth["scene"][bright]) - 1) < 0.15
    pedestal = nm.prior.transform.softening * np.log(2) * nm.scene_scale0  # where zero flux lands
    assert np.all(scene > 0.5 * pedestal) and np.any(np.isclose(scene, pedestal, rtol=0.05))
    assert "drizzled" in nm.init_info["report"][0]
    assert len(nm.geometry_report()) == 5


def test_novamac_rejects_bad_inputs(prior_dir, synthetic):
    inputs, _ = synthetic
    with pytest.raises(TypeError):
        Novamac(prior=prior_dir, verbose=False, not_a_setting=1, **inputs)
    bad = dict(inputs)
    bad["psf_n"] = bad["psf_n"][:-1]
    with pytest.raises(ValueError):
        Novamac(prior=prior_dir, verbose=False, **bad)
    both = dict(inputs)
    both["variance"] = 1.0 / both["weight"]
    with pytest.raises(ValueError):
        Novamac(prior=prior_dir, verbose=False, **both)


def test_settings_are_per_instance(prior_dir, synthetic):
    inputs, _ = synthetic
    a = Novamac(prior=prior_dir, verbose=False, num_chains=7, release_t={"psf": 0.1}, **inputs)
    b = Novamac(prior=prior_dir, verbose=False, **inputs)
    assert a.num_chains == 7 and b.num_chains == Novamac.num_chains
    assert a.release_t["psf"] == 0.1 and a.release_t["sky"] == Novamac.release_t["sky"]
    assert b.release_t["psf"] == Novamac.release_t["psf"]
    assert "num_chains" in a.settings() and a.settings()["release_t"]["psf"] == 0.1


def test_masked_nan_does_not_reach_the_flat_data(prior_dir, synthetic, tmp_path):
    inputs, _ = synthetic
    bad = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in inputs.items()}
    bad["data"][2, 3, 3] = np.nan  # under the fixture's masked blob
    assert bad["mask"][2, 3, 3]
    nm = Novamac(prior=prior_dir, out_dir=tmp_path, verbose=False, **bad)
    data, weight = flat_data_weight(nm.build_model())
    assert np.all(np.isfinite(data)) and np.all(weight[~np.isfinite(bad["data"].ravel())] == 0)
