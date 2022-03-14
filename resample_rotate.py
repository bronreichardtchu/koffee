from mpdaf.obj import Image, Cube
import matplotlib

# import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
import warnings

# warnings.filterwarnings("ignore", module="matplotlib\..*")
warnings.filterwarnings("ignore", module="astropy")

matplotlib.rcParams.update(
    {
        "image.aspect": "equal",
        "image.origin": "lower",
        "image.interpolation": "nearest",
        "image.cmap": "Blues_r",
        "axes.facecolor": "whitesmoke",
    }
)


def run():
    output_append_str = "_resampled"
    co_settings = {
        #"filename": "ac_co.0.fits",
        "filename": "../../../data/IRAS08/ac_co_beam_mom0.fits",
        "hdu_ext": 0,
        "shift_dx": 0,  # may be decimal. Adds this many pixels to the reference pixel position.
        "shift_dy": 0,  # may be decimal
        "shift_reference_size": "output",  # "input" or "output". are you shifting number of pixels by the output pixel size or input?
        "rotate": 0,
        # comment out whichever you don't want.
        #     'output_pix_factor': [1,1.357848/0.291456], #applied after rotation. Rebin by this factor for [y, x] dimensions.
        "output_pix_arcsec": [
            1.357848,
            1.357848,
        ],  # [dy,dx] preferenced over 'output_pix_factor'
        "resample_settings": {"antialias": True, "window": "gaussian"},
    }
    iras_settings = {
        #"filename": "IRAS08_star_formation_rate_shifted.fits",
        "filename": "../../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2022-01-28_resolved/IRAS08_star_formation_rate_shifted.fits",
        "hdu_ext": 0,
        "shift_dx": 0,  # may be decimal. Adds this many pixels to the reference pixel position.
        "shift_dy": 0,  # may be decimal
        "rotate": 90,
        # comment out whichever you don't want.
        #     'output_pix_factor': [1,1.357848/0.291456], #applied after rotation. Rebin by this factor for [y, x] dimensions.
        "output_pix_arcsec": [
            1.357848,
            1.357848,
        ],  # preferenced over 'output_pix_factor'
        "resample_settings": {"antialias": False},
    }

    # I introduced the "this_settings" because I kept forgetting to change it in some functions...

    # rotate and resample iras:
    this_settings = iras_settings

    iras_orig = open_im(this_settings)
    iras_rot = rotate_im(iras_orig, this_settings)
    iras_resample = resample_im(iras_rot, this_settings, fill_mask_zeros=True)

    # crop the iras image
    border_slices = crop_border(iras_resample, border=5)
    iras_resample_crop = iras_resample[border_slices]

    # resample co, temporary, we will re-do the resample later, after calculating shift.:
    this_settings = co_settings

    co_orig = open_im(this_settings)
    co_resample = resample_im(co_orig, this_settings)

    # calculate shift between the 2:
    # we will be shifting co, so use that as the main image.
    shift = co_resample.estimate_coordinate_offset(iras_resample)

    # enter the shift into co_settings, then shift and resample co.
    this_settings = co_settings
    this_settings["shift_dy"], this_settings["shift_dx"] = shift
    co_shift = shift_im(co_orig, this_settings, inplace=False)
    co_resample = resample_im(co_shift, this_settings)

    # find iras center ra/dec  on co image and extract a rectangle with same number of pixels.
    iras_center = iras_resample_crop.wcs.get_center()  # [arranged as [dec, ra]
    iras_shape = iras_resample_crop.shape
    co_cutout = co_resample.subimage(iras_center, size=iras_shape[::-1], unit_size=None)

    # save the images
    iras_resample_crop.write(
        iras_settings["filename"][:-5] + output_append_str + ".fits", savemask="nan"
    )
    co_cutout.write(
        co_settings["filename"][:-5] + output_append_str + ".fits", savemask="nan"
    )

    #############################################################

    # now do the other 2 iras images. I'll just be updating the filename
    # so that I know I will be using the same settings.

    #iras_settings.update({"filename": "IRAS08_mass_outflow_rate_shifted.fits"})
    iras_settings.update({"filename": "../../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2022-01-28_resolved/IRAS08_mass_outflow_rate_shifted.fits"})
    # rotate and resample iras:
    this_settings = iras_settings

    iras_orig = open_im(this_settings)
    iras_rot = rotate_im(iras_orig, this_settings)
    iras_resample = resample_im(iras_rot, this_settings, fill_mask_zeros=True)

    # instead of crop, we will do subimage to ensure it's the same.
    iras_resample_crop = iras_resample.subimage(
        iras_center, size=iras_shape[::-1], unit_size=None
    )
    iras_resample_crop.write(
        iras_settings["filename"][:-5] + output_append_str + ".fits", savemask="nan"
    )

    #iras_settings.update({"filename": "IRAS08_outflow_velocity_shifted.fits"})
    iras_settings.update({"filename": "../../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2022-01-28_resolved/IRAS08_outflow_velocity_shifted.fits"})
    # rotate and resample iras:
    this_settings = iras_settings

    iras_orig = open_im(this_settings)
    iras_rot = rotate_im(iras_orig, this_settings)
    iras_resample = resample_im(iras_rot, this_settings, fill_mask_zeros=True)

    # instead of crop, we will do subimage to ensure it's the same.
    iras_resample_crop = iras_resample.subimage(
        iras_center, size=iras_shape[::-1], unit_size=None
    )

    # This has conserved "flux", but since this is not flux, we want to convert this to an average,
    # and I think that means multiplying by original pixel area/ new pixel area.

    factor = np.prod(iras_orig.wcs.get_step(u.arcsec)) / np.prod(
        iras_resample_crop.wcs.get_step(u.arcsec)
    )
    iras_resample_crop.data *= factor

    iras_resample_crop.write(
        iras_settings["filename"][:-5] + output_append_str + ".fits", savemask="nan"
    )
    print("script has finished")


########################################
def open_im(settings):
    try:
        im = Image(settings["filename"], ext=settings["hdu_ext"])
    except ValueError:
        im = Cube(settings["filename"], ext=settings["hdu_ext"])[0]
    return im


def plot_aspect(im, **kwargs):
    wcs_step = im.wcs.get_step()
    im.plot(aspect=wcs_step[0] / wcs_step[1], **kwargs)


def shift_im(in_im, settings, inplace=False):

    dx = settings["shift_dx"]
    dy = settings["shift_dy"]

    if settings.get("shift_reference_size") == "output":

        input_pix_arcsec = in_im.get_step(u.arcsec)  # [dy,dx]
        output_pix_arcsec = get_outstep(settings, input_pix_arcsec)
        dy *= output_pix_arcsec[0] / input_pix_arcsec[0]
        dx *= output_pix_arcsec[1] / input_pix_arcsec[1]

    # shift the reference pixel.

    if inplace:
        im = in_im
    else:
        im = in_im.copy()

    im.wcs.set_crpix1(im.wcs.get_crpix1() + dx)
    im.wcs.set_crpix2(im.wcs.get_crpix2() + dy)
    return im


def get_outstep(settings, in_arcsec):
    if settings.get("output_pix_arcsec", None) is not None:
        return settings.get("output_pix_arcsec")
    elif settings.get("output_pix_factor", None) is not None:
        factor = np.array(settings["output_pix_factor"])
        return in_arcsec * factor
    else:
        return in_arcsec


def resample_im(in_im, settings, fill_mask_zeros=False):
    im = in_im.copy()
    if fill_mask_zeros is True:
        im.data = im.data.filled(fill_value=0)
    # resample image. Do we do factor or do we do absolute?
    in_arcsec = im.get_step(u.arcsec)

    out_arcsec = get_outstep(settings, in_arcsec)
    if np.allclose(out_arcsec, in_arcsec):
        return im.copy()

    npix_out = (im.shape * in_arcsec / out_arcsec).astype(int)

    resample_kwargs = settings.get("resample_settings", {})

    #     return im.regrid(npix_out,None,None, [abs(out_arcsec[0]),-abs(out_arcsec[1])],flux=True,antialias=False,window='rectangle' )
    im_resampled = im.resample(npix_out, None, out_arcsec, flux=True, **resample_kwargs)

    if fill_mask_zeros is True:
        im_resampled.mask = ~(im_resampled.data > 0).data

    return im_resampled


def rotate_im(im, settings):
    center = im.wcs.get_center()
    square_im = im.subimage(center, size=max(im.shape) + 10, unit_size=None)

    return square_im.rotate(
        settings["rotate"],
        regrid=True,
        flux=True,
        pivot=(np.array(square_im.shape) / 2).astype(int),
    )


def crop_border(im, border):
    slices = im.copy().crop()
    return tuple(
        [
            slice(max(0, s.start - border), min(dim, s.stop + border))
            for s, dim in zip(slices, im.shape)
        ]
    )


if __name__ == "__main__":
    run()
