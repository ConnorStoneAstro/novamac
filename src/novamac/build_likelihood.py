from typing import Union
import types
import numpy as np
import astrophot as ap


def build_likelihood(
    data: np.ndarray,  # (N, H, W)
    var: np.ndarray,  # (N, H, W)
    weight: np.ndarray,  # (N, H, W)
    mask: np.ndarray,  # (N, H, W)
    band: str,  # ()
    is_template: np.ndarray,  # (N,)
    crpix: np.ndarray,  # (N, 2)
    crtan: np.ndarray,  # (N, 2)
    crval: np.ndarray,  # (2,)
    CD: np.ndarray,  # (N, 2, 2)
    psf_n: np.ndarray,  # (N,)
    psf_Rd: np.ndarray,  # (N,)
    psf_q: np.ndarray,  # (N,)
    psf_PA: np.ndarray,  # (N,)
    psf_upsample: int,  # ()
    Hp: int,  # ()
    Wp: int,  # ()
    ps_center0: np.ndarray,  # (P, 2)
    ps_flux0: np.ndarray,  # (P, N)
    sky_I0: np.ndarray,  # (N,)
    pix_scale: float,  # ()
    pix_init: np.ndarray,  # (Hpix, Wpix)
    zp: Union[float, np.ndarray] = 22.5,  # (N,)
    name: str = "model",
    print_graph: bool = False,
    target_names: list = None,
):
    """
    Shape legend:
    - N: number of observed cutouts
    - H: Height of object cutout (pixels)
    - W: Width of object cutout (pixels)
    - Hp: Height of PSF image (pixels)
    - Wp: Width of PSF image (pixels)
    - Hpix: Height of pixelated model image (pixels)
    - Wpix: Width of pixelated model image (pixels)
    - P: number of point sources

    Args:
    - data: The observed data cutouts (N, H, W)
    - var: The variance cutouts of the data (N, H, W), same shapes as data. Only one of var or weight should be provided (other is None).
    - weight: The weight cutouts for the data (N, H, W), same shapes as data. Only one of var or weight should be provided (other is None).
    - mask: The mask cutouts for the data (N, H, W), same shapes as data, True for masked pixels
    - band: The band name string, e.g. "g", "r", "i"
    - is_template: Bool, whether each image is a template or not, templates have no point source flux (N,)
    - crpix: The reference pixel coordinates in the data (N, 2)
    - crtan: The reference tangent plane coordinates (N, 2)
    - crval: The reference world coordinate (2,)
    - CD: The coordinate transformation matrix, pixel scale matrix (N, 2, 2)
    - psf_n: The Moffat PSF shape parameter n for each image (N,)
    - psf_Rd: The Moffat PSF scale radius Rd for each image (N,)
    - psf_q: The Moffat PSF axis ratio q for each image (N,)
    - psf_PA: The Moffat PSF position angle PA for each image (N,)
    - psf_upsample: The PSF upsample factor, 1 means same pixelscale as data, 2 means twice the resolution (pixelscale / 2)
    - Hp: The height of the PSF image in pixels (before upsampling)
    - Wp: The width of the PSF image in pixels (before upsampling)
    - ps_center0: The initial center position of the point source(s), in tangent plane (P, 2)
    - ps_flux0: The initial flux of the point source(s). Values for template images will be ignored but must be provided (P, N)
    - sky_I0: The initial sky background for each image (N,)
    - pix_scale: The pixel scale of the pixelated model image in arcsec/pix
    - pix_init: The initial values for the pixelated model fluxes (Hpix, Wpix)
    - zp: The photometric zeropoint for each image, default is 22.5 (N,)
    - name: The name of the model (str)
    - print_graph: Whether to print the model parameter directed acyclic graph to a PDF file (bool)
    - target_names: Optional list of names for the target images, default is None which will use "target_0", "target_1", etc. (list of str)

    Returns:
    - model: The full constructed AstroPhot model
    """
    if ps_center0.shape == (2,):
        ps_center0 = ps_center0[None, :]
    assert ps_center0.ndim == 2 and ps_center0.shape[1] == 2, "ps_center0 must have shape (P, 2)"

    # Collect shapes
    # ----------------------------------------------------------------
    N = len(data)  # number of cutout images
    P = len(ps_center0)  # number of point sources

    # Get weight cutouts
    # ----------------------------------------------------------------
    assert (weight is None) ^ (var is None), "Only one of weight or var should be provided!"
    if weight is None:
        weight = 1.0 / var
    else:
        var = 1.0 / weight

    # Construct Target Images
    # ----------------------------------------------------------------
    target = []
    for n in range(N):
        # All images, in original order
        target.append(
            ap.TargetImage(
                name=f"target_{n}" if target_names is None else target_names[n],
                data=data[n],
                weight=weight[n],
                mask=mask[n],
                crpix=crpix[n],
                crtan=crtan[n],
                crval=crval,
                CD=CD[n],
                zeropoint=zp if isinstance(zp, float) else zp[n],
            )
        )
        target[-1].crtan.to_dynamic()  # allow image alignment
    batch_target = ap.TargetImageBatch(name="batch_target", images=target)
    target = ap.TargetImageList(name="target", images=target)

    # Construct PSF Model
    # ----------------------------------------------------------------
    psf_model = ap.Model(
        name="psf_model",
        model_type="moffat ellipse psf model",
        n=psf_n,
        Rd=psf_Rd,
        q=psf_q,
        PA=psf_PA,
        sampling_mode="simpsons",
        integrate_mode="none",
        target=ap.PSFImage(
            data=np.zeros((Hp * psf_upsample, Wp * psf_upsample)), upsample=psf_upsample
        ),
    )

    # Construct Models
    # ----------------------------------------------------------------
    # Sky
    sky_model = ap.Model(
        name=f"sky",
        model_type="flat sky model",
        I0=sky_I0,
        target=batch_target.images[0],
        integrate_mode="none",
        sampling_mode="midpoint",
    )
    sky_model.I0.valid = None
    sky_batch_model = ap.Model(
        name="sky_batch_model",
        model_type="batch scene model",
        model=sky_model,
        target=batch_target,
    )
    # Pixelated Scene
    pixelated_model = ap.Model(
        name="pixelated",
        model_type="pixelated model",
        center=ps_center0[0],
        I=pix_init,
        pixelscale=pix_scale,
        PA=0.0,
        target=batch_target.images[0],
        integrate_mode="none",
        sampling_mode="simpsons",
        psf=psf_model,
        jacobian_maxparams=32,
    )
    pixelated_model.scale.to_static(1.0)
    pixelated_model.I.valid = (0.0, None)
    pixelated_model.pixelscale.to_static()  # fix pixelated model pixelscale
    pixelated_model.PA.to_static()  # fix pixelated model PA
    pixelated_model.center.to_static()  # fix pixelated model center
    pixelated_batch_model = ap.Model(
        name="pixelated_batch_model",
        model_type="batch scene model",
        model=pixelated_model,
        target=batch_target,
    )

    # Only apply point source in non-template images
    # point source(s)
    ps_components = []
    template_indicator = ap.Param(
        "template_indicator",
        (~is_template).astype(np.float64),
        shape=(),
        dynamic=True,
        description="an indicator value to say if an image is a template image which can be vmapped",
        group=1,
    )
    for p in range(P):
        allflux = ap.Param(
            "allflux",
            ps_flux0[p],
            shape=(),
            dynamic=True,
            description="A param to accept fluxes for all images (actual flux will only take non-template-image values).",
            valid=(0, None),
        )
        ps_model = ap.Model(
            name=f"ps_{p}",
            model_type="point model",
            center=ps_center0[p],
            flux=lambda p: p.allflux.value * p.template_indicator.value,
            target=batch_target.images[0],
            integrate_mode="none",
            sampling_mode="simpsons",
            psf=psf_model,
        )
        ps_model.flux.allflux = allflux
        ps_model.flux.template_indicator = template_indicator
        ps_components.append(
            ap.Model(
                name=f"ps_batch_model_{p}",
                model_type="batch scene model",
                model=ps_model,
                target=batch_target,
            )
        )

    # Full model combining all bands
    model = ap.Model(
        name=name,
        model_type="group model",
        models=[sky_batch_model, pixelated_batch_model] + ps_components,
        target=target,
    )
    model.initialize()
    model.meta.shapes = {"N": N, "P": P}
    model.meta.band = band
    model.meta.is_template = is_template
    model.meta.psf_n0 = psf_n
    model.meta.psf_Rd0 = psf_Rd
    model.meta.psf_q0 = psf_q
    model.meta.psf_PA0 = psf_PA
    model.meta.crpix0 = crpix
    model.meta.crtan0 = crtan
    model.meta.crval0 = crval
    model.meta.CD0 = CD
    model.meta.ps_center0 = ps_center0
    model.meta.ps_flux0 = ps_flux0
    model.meta.sky_I0 = sky_I0

    @ap.param.forward
    def log_likelihood(self, beta=0.0):
        """
        Compute the negative log likelihood of the model wrt the target image in the appropriate window.
        """

        model = self().flatten("data")
        data = self.target[self.window]
        weight = 1 / (data.flatten("variance") * (1 + 5 * beta))
        mask = data.flatten("mask")
        data = data.flatten("data")
        ll = -0.5 * ap.backend.sum((data - model) ** 2 * weight * (~mask))

        return ll

    model.log_likelihood = types.MethodType(log_likelihood, model)

    # Print model
    # ----------------------------------------------------------------
    if print_graph:
        model.graphviz(saveto=f"{name}_graph.pdf")

    return model
