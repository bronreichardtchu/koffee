#!/usr/bin/env python
"""

Copyright (C) 2014-2017, Michele Cappellari
E-mail: michele.cappellari_at_physics.ox.ac.uk

(Adapted by Bronwyn Reichardt Chu, 2019)

Updated versions of the software are available from my web page
http://purl.org/cappellari/software

See example at the bottom for usage instructions.

MODIFICATION HISTORY:
    V1.0.0: Created to emulate my IDL procedure with the same name.
        Michele Cappellari, Oxford, 28 March 2014
    V1.0.1: Fixed treatment of optional parameters. MC, Oxford, 6 June 2014
    V1.0.2: Avoid potential runtime warning. MC, Oxford, 2 October 2014
    V1.0.3: Return axis. MC, Oxford, 26 March 2015
    V1.0.4: Return image instead of axis. MC, Oxford, 15 July 2015
    V1.0.5: Removes white gaps from rotated images using edgecolors.
        MC, Oxford, 5 October 2015
    V1.0.6: Pass kwargs to graphics functions.
        MC, Campos do Jordao, Brazil, 23 November 2015
    V1.0.7: Check that input (x,y) come from an axis-aligned image.
        MC, Oxford, 28 January 2016
    V1.0.8: Fixed deprecation warning in Numpy 1.11. MC, Oxford, 22 April 2016
    V1.1.0: Fixed program stop with kwargs. Included `colorbar` keyword.
        MC, Oxford, 18 May 2016
    V1.1.1: Use interpolation='nearest' to avoid crash on MacOS.
        MC, Oxford, 14 June 2016
    V1.1.2: Specify origin=`upper` in imshow() for consistent results with older
        Matplotlib version. Thanks to Guillermo Bosch for reporting the issue.
        MC, Oxford, 6 January 2017
    V1.1.3: Simplified passing of default keywords. MC, Oxford, 20 February 2017
    V1.1.4: Use register_sauron_colormap(). MC, Oxford, 29 March 2017
    V1.1.5: Request `pixelsize` when dataset is large. Thanks to Davor
        Krajnovic (Potsdam) for the feedback. MC, Oxford, 10 July 2017
    V1.1.6: Fixed new incompatibility with Matplotlib 2.1.
        MC, Oxford, 9 November 2017
    V1.1.7: Changed imports for plotbin as a package. MC, Oxford, 17 April 2018

    V2.0: Changes made by Bron.  Changing pixelsize to have x and y values to
        allow for KCWI's rectangular pixels.

"""
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator
from scipy.spatial import distance
import numpy as np

#from plotbin.sauron_colormap import register_sauron_colormap

##############################################################################

def display_pixels(x, y, val, pixelsize_x=None, pixelsize_y=None, vmin=None,
                    vmax=None, angle=None, colorbar=False, label=None,
                    nticks=7, cmap='viridis', axes=None, **kwargs):
    """
    Display vectors of rectangular pixels at coordinates (x,y) coloured with "val".
    An optional rotation around the origin can be applied to the whole image.

    The pixels are assumed to be taken from a regular cartesian grid with
    constant spacing (like an axis-aligned image), but not all elements of
    the grid are required (missing data are OK).

    This routine is designed to be fast even with large images and to produce
    minimal file sizes when the output is saved in a vector format like PDF.

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        flattened x-coordinate array (same shape as val)

    y : :obj:'~numpy.ndarray'
        flattened y-coordinate array (same shape as val)

    val : :obj:'~numpy.ndarray'
        flattened array of values to be mapped.  Can have some missing data.

    pixelsize_x : float
        size of the pixels in the x direction.  If None, calculates the minimum
        distance between unique values in the x array.  (Default=None)

    pixelsize_y : float
        size of the pixels in the y direction.  If None, calculates the minimum
        distance between unique values in the y array.  (Default=None)

    vmin : float
        minimum value plotted in colormap.  If None, uses np.min(val)
        (Default=None)

    vmax : float
        maximum value plotted in colormap.  If None, uses np.max(val)
        (Default=None)

    angle : float

    colorbar : boolean
        whether to include a colorbar (Default=False)

    label : string
        the label for the colorbar, if colorbar=True

    nticks : int
        the number of ticks along the colorbar, if colorbar=True

    cmap : string
        colormap to use (Default='viridis').

    axes :
        matplotlib axes to add the map to.  (Default=None)

    **kwargs

    Returns:
    -------
    A matplotlib instance of a map using the x and y coordinates given.

    """
    x, y, val = map(np.ravel, [x, y, val])

    assert x.size == y.size == val.size, 'The vectors (x, y, val) must have the same size'

    #if cmap in ['sauron', 'sauron_r']:
        #register_sauron_colormap()

    if vmin is None:
        vmin = np.min(val)

    if vmax is None:
        vmax = np.max(val)

    if pixelsize_x is None:
        if x.size < 1e4:
            pixelsize_x = np.min(distance.pdist(np.column_stack([np.unique(x), np.unique(x)])))
        else:
            raise ValueError("Dataset is large: Provide `pixelsize_x`")

    if pixelsize_y is None:
        if y.size < 1e4:
            pixelsize_y = np.min(distance.pdist(np.column_stack([np.unique(y), np.unique(y)])))
        else:
            raise ValueError("Dataset is large: Provide `pixelsize_y`")

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    x1 = (x - xmin)/pixelsize_x
    y1 = (y - ymin)/pixelsize_y
    nx = int(round((xmax - xmin)/pixelsize_x) + 1)
    ny = int(round((ymax - ymin)/pixelsize_y) + 1)
    mask = np.ones((nx, ny), dtype=bool)
    img = np.empty((nx, ny))
    j = np.round(x1).astype(int)
    k = np.round(y1).astype(int)
    mask[j, k] = 0
    img[j, k] = val
    img = np.ma.masked_array(img, mask)

    #assert np.all(np.abs(np.append(j - x1, k - y1)) < 0.1), \
    #    'The coordinates (x, y) must come from an axis-aligned image'

    if axes == None:
        ax = plt.gca()
    else:
        ax = axes

    if (angle is None) or (angle == 0):

        img = ax.imshow(np.rot90(img), interpolation='nearest',
                        origin='upper',
                        cmap=cmap, vmin=vmin, vmax=vmax,
                        extent=[xmin-pixelsize_x/2, xmax+pixelsize_x/2,
                                ymin-pixelsize_y/2, ymax+pixelsize_y/2], **kwargs)

    else:

        x, y = np.ogrid[xmin-pixelsize_x/2 : xmax+pixelsize_x/2 : (nx+1)*1j,
                        ymin-pixelsize_y/2 : ymax+pixelsize_y/2 : (ny+1)*1j]
        ang = np.radians(angle)
        x, y = x*np.cos(ang) - y*np.sin(ang), x*np.sin(ang) + y*np.cos(ang)
        img = ax.pcolormesh(x, y, img, cmap=cmap, vmin=vmin, vmax=vmax,
                            edgecolors="face", **kwargs)
        ax.axis('image')
        mask1 = np.ones_like(x, dtype=bool)
        mask1[:-1, :-1] *= mask  # Flag the four corners of the mesh
        mask1[:-1, 1:] *= mask
        mask1[1:, :-1] *= mask
        mask1[1:, 1:] *= mask
        x0, x1 = np.min(x[~mask1]), np.max(x[~mask1])
        y0, y1 = np.min(y[~mask1]), np.max(y[~mask1])
        ax.set_xlim([x0, x1])
        ax.set_ylim([y0, y1])

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        ticks = MaxNLocator(nticks).tick_values(vmin, vmax)
        cbar = plt.colorbar(img, cax=cax, ticks=ticks)
        cbar.solids.set_edgecolor("face")  # Remove gaps in PDF http://stackoverflow.com/a/15021541
        if label:
            cbar.set_label(label)
        plt.sca(ax)  # Activate main plot before returning


    ax.minorticks_on()
    ax.tick_params(length=10, width=1, which='major')
    ax.tick_params(length=5, which='minor') #width=1,

    return img

##############################################################################

# Usage example for display_pixels()

if __name__ == '__main__':

    n = 50  # 1 arcsec pixels
    x = np.linspace(-20, 20, n)
    y = np.linspace(-20, 20, n)
    xx, yy = np.meshgrid(x,y)
    counts = xx**2 - 2*yy**2
    w = xx**2 + 2*yy**2 < 10.1**2

    plt.clf()
    ax = display_pixels(xx[w], yy[w], counts[w], pixelsize=x[1]-x[0],
                        angle=20, colorbar=True)
    plt.pause(1)
