"""The initial alignment refinement: images whose input WCS is off by a
known fraction of a pixel are pulled back, within the bound, only when
their chi-square improves, and the alignment prior stays on the input WCS."""

import numpy as np

from novamac import Novamac
from novamac.testing import make_synthetic_inputs


def test_alignment_recovers_known_offsets(prior_dir, tmp_path):
    inputs, truth = make_synthetic_inputs(size=16, n_template=5, n_sn=3, noise=0.2)
    pix = inputs["CD"][0, 0, 0]
    offset = np.zeros_like(inputs["crtan"])
    offset[1] = (0.4 * pix, -0.3 * pix)  # a template off by a fraction of a pixel
    offset[6] = (-0.5 * pix, 0.2 * pix)  # a transient epoch too
    wrong = dict(inputs)
    wrong["crtan"] = inputs["crtan"] + offset
    nm = Novamac(prior=prior_dir, out_dir=tmp_path, verbose=False, init_align_steps=150, **wrong)
    model = nm.build_model()
    info = nm.align_info
    assert info is not None and info["accepted"][1] and info["accepted"][6]
    recovered = nm.crtan_init - wrong["crtan"]  # should undo the offsets
    assert np.allclose(recovered[1], -offset[1], atol=0.12 * pix), recovered[1] / pix
    # a transient epoch aligns on the pixels outside the transient's aperture only, so less precisely
    assert np.allclose(recovered[6], -offset[6], atol=0.3 * pix), recovered[6] / pix
    assert np.all(np.abs(recovered) <= Novamac.init_align_max_shift * pix + 1e-9)
    untouched = np.abs(recovered[inputs["is_template"]][[0, 2, 3, 4]])
    assert np.all(untouched < 0.4 * pix), f"untouched epochs stay close: {untouched / pix} px"
    assert np.sum(np.where(info["accepted"], info["chi2_after"], info["chi2_before"])) < np.sum(info["chi2_before"])
    # the start moved, the prior's centre and the plots' reference did not
    assert np.allclose(model.meta.crtan0, wrong["crtan"]) and np.allclose(nm.crtan, wrong["crtan"])
    started = np.stack([t.crtan.npvalue for t in model.target.images])
    assert np.allclose(started, nm.crtan_init)
    assert any("restacked" in line for line in nm.init_info["report"])


def test_alignment_can_be_disabled(prior_dir, synthetic, tmp_path):
    inputs, _ = synthetic
    nm = Novamac(prior=prior_dir, out_dir=tmp_path, verbose=False, init_align=False, **inputs)
    model = nm.build_model()
    assert nm.align_info is None and np.allclose(nm.crtan_init, inputs["crtan"])
    assert np.allclose(np.stack([t.crtan.npvalue for t in model.target.images]), inputs["crtan"])
