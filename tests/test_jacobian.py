"""The analytic Jacobian diagonals (jacobian.SceneJacobian) against autodiff
on the synthetic model: exact single-pixel Gauss-Newton probes for the scene
curvature, Hutchinson for diag(JJ^T)."""

import caskade as ck
import jax
import jax.numpy as jnp
import numpy as np

from novamac import Novamac
from novamac._backend import DTYPE
from novamac.hmc import block_indices, hutchinson_jj_diag, make_gauss_newton_fn, make_log_parts
from novamac.jacobian import SceneJacobian
from novamac.sampler import flat_data_weight


def test_analytic_diagonals_match_autodiff(prior_dir, synthetic, tmp_path):
    inputs, _ = synthetic
    nm = Novamac(prior=prior_dir, out_dir=tmp_path, verbose=False, **inputs)
    model = nm.build_model()
    pix = model.models.pixelated_batch_model.model
    with ck.ValidContext(model):
        x0, g1 = model.get_values()
    x0, g1 = jnp.asarray(x0, DTYPE), jnp.asarray(g1, DTYPE)
    _, _, images = make_log_parts(model, g1)
    scene_idx = block_indices(model, [pix.I])
    jac = SceneJacobian(
        nm.crpix, nm.crtan, nm.CD, nm.ps_center[0], nm.scene_pixel_scale, pix.I.shape, nm.data.shape[1:], nm.weight, nm.mask,
        nm.psf_n, nm.psf_Rd, nm.psf_q, nm.psf_PA, nm.psf_shape, sub=nm.psf_upsample,
    )  # fmt: skip
    I_sq = np.exp(2.0 * np.asarray(x0, dtype=np.float64)[scene_idx]).reshape(pix.I.shape)
    scale = float(pix.scale.npvalue)

    # scene curvature against exact single-pixel probes, bright and faint pixels
    # away from the scene's edge (the 16-px test scene is smaller than cutout +
    # PSF, and the renderer's treatment of the spill is not modelled)
    H, W = pix.I.shape
    rows, cols = np.divmod(np.arange(H * W), W)
    interior = np.where((rows >= 4) & (rows < H - 4) & (cols >= 4) & (cols < W - 4))[0]
    order = interior[np.argsort(I_sq.ravel()[interior])]
    sample = np.unique(np.concatenate([order[-6:], order[:: len(order) // 6][:6]]))
    gn = make_gauss_newton_fn(images, [scene_idx[[int(i)]] for i in sample])
    _, w = flat_data_weight(model)
    exact = np.asarray(gn(x0, jnp.asarray(w, DTYPE)))
    ratio = jac.curvature(I_sq, scale).ravel()[sample] / exact
    assert np.all((ratio > 0.8) & (ratio < 1.25)), ratio

    # diag(JJ^T) against Hutchinson on the data pixels away from the cutout edge
    # (the same scene-spill caveat), where the scene contributes
    jj_h = np.asarray(hutchinson_jj_diag(images, x0, scene_idx, jax.random.key(0), num_probes=512, batch=64))
    jj_a = jac.jj_diag(I_sq, scale)
    assert jj_a.shape == jj_h.shape
    inner = np.zeros(nm.data.shape, dtype=bool)
    inner[:, 4:-4, 4:-4] = True
    good = inner.ravel() & (jj_h > 1e-2 * jj_h.max())
    r = jj_a[good] / jj_h[good]
    assert 0.85 < np.median(r) < 1.15 and np.all((r > 0.6) & (r < 1.6)), (np.median(r), r.min(), r.max())
