#Daniel's reprojection script

#Here's my reprojection script, it's written to be run on a bunch of
#galaxies at once so it might be a bit opaque but the key parts are:
#   1. line 23: arearatio = dimshdr['PXSCL'] / dimshdr['SLSCL']
#       we take the ratio of the dimensions of the rectangular spaxels,
#       we're going to use this later to scale our flux appropriately
#   2. line 26: mImgtbl
#       creates the .tbl file with all the relevant wcs stuff from the
#       input images (inputs: the folder containing the images and a
#       file name and location for the output .tbl file. Only have the
#       images you want to reproject in that folder)
#   3. line 28: mMakeHdr
#       creates the hdr file that will be applied to the reprojected images.
#       (inputs: the location and name of the .tbl file from the last step
#       and a file name and location for the output .hdr file)
#   4. lines 32-36: mProjectCube
#       this is where we actually reproject the cubes. Function takes one
#       image at a time so you have to loop over them. (inputs: the location
#       and file name of an image to be reprojected, location to output the
#       reprojected cube, location and name of the .hdr file to use, and
#       some keywords that Nikki told me to use:
#       drizzle=1.0, energyMode=False,
#       fluxscale=the ratio of our rectangular spaxel dimensions
#   5. line 38-: This is me fixing the new header.
#       I just update the old one with the values from the new header
#       and then replace the whole new header with the old one. You
#       might also need to delete the CDX_X keywords from the old header
#       since the Montage output produces CDELT ones and having CDX_X
#       based on the old pixel dimensions and CDELT based on the new ones
#       is going to really mess up a lot of stuff.


import numpy as np
from astropy.io import fits
from MontagePy.main import *
import glob

types = ['icubes', 'icubed', 'vcubes']

gal_list = [('ugc1385_red', 0.018749166), ('ugc1385_blue', 0.018749166)]

for gal in gal_list:
    for exptype in types:
        print(gal[0])
        images = np.array([])
        for file in glob.glob(gal[0] + '/aligned/*.fits'):
            images = np.append(images, file[-23:-20])
        images = list(set(images))
        images.sort()
        location = gal[0]
        img = glob.glob(gal[0] + '/aligned/*_' + exptype + '_aligned.fits')[1][-34:-23]

        dims = fits.open(gal[0] + '/aligned/' + img + images[0] + '_' + exptype + '_aligned.fits')
        dimshdr = dims[0].header
        arearatio = dimshdr['PXSCL'] / dimshdr['SLSCL']

        print('Creating Table')
        rtn = mImgtbl(gal[0] + '/aligned', gal[0] + '/aligned/cubes_c1.tbl')
        print('Creating Header')
        rtn2 = mMakeHdr(gal[0] + '/aligned/cubes_c1.tbl', gal[0] + '/aligned/cubes_c1.hdr')

        print('Reprojecting Cubes')

        for num in images:
            proj1 = mProjectCube(gal[0] + '/aligned/' + img + num + '_' + exptype + '_aligned.fits',
                                 gal[0] + '/reprojected/' + exptype + '/' + img + num + '_' + exptype + '_reprojected.fits',
                                 gal[0] + '/aligned/cubes_c1.hdr', drizzle=1.0, energyMode=False, fluxScale=arearatio)
            print(proj1)

        print('Fixing Headers')

        origcube = fits.open(gal[0] + '/aligned/' + img + images[0] + '_' + exptype + '_aligned.fits')
        orighdr = origcube[0].header

        for num in images:

            projcube = fits.open(gal[0] + '/reprojected/' + exptype + '/' + img + num + '_' + exptype + '_reprojected.fits')
            projhdr = projcube[0].header
            orighdr['BITPIX'] = projhdr.get('BITPIX')
            orighdr['NAXIS'] = projhdr.get('NAXIS')
            orighdr['NAXIS1'] = projhdr.get('NAXIS1')
            orighdr['NAXIS2'] = projhdr.get('NAXIS2')
            orighdr['NAXIS3'] = projhdr.get('NAXIS3')
            orighdr['CTYPE1'] = projhdr.get('CTYPE1')
            orighdr['CTYPE2'] = projhdr.get('CTYPE2')
            orighdr['EQUINOX'] = projhdr.get('EQUINOX')
            orighdr['CRVAL1'] = projhdr.get('CRVAL1')
            orighdr['CRVAL2'] = projhdr.get('CRVAL2')
            orighdr['CRVAL3'] = projhdr.get('CRVAL3')
            orighdr['CRPIX1'] = projhdr.get('CRPIX1')
            orighdr['CRPIX2'] = projhdr.get('CRPIX2')
            orighdr['CRPIX3'] = projhdr.get('CRPIX3')
            orighdr['CDELT1'] = projhdr.get('CDELT1')
            orighdr['CDELT2'] = projhdr.get('CDELT2')
            orighdr['CDELT3'] = orighdr.get('CD3_3')
            orighdr['CROTA2'] = projhdr.get('CROTA2')
            projcube[0].header = orighdr
            projcube.writeto(gal[0] + '/reprojected/' + exptype + '/' + img + num + '_' + exptype + '_reprojected_hdrfix.fits')
