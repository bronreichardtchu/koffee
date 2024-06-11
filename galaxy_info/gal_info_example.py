"""
NAME:
	gal_info_example.py

AUTHOR:
	Bronwyn Reichardt Chu
	Swinburne
	2023

EMAIL:
	<breichardtchu@swin.edu.au>

PURPOSE:
	Example on how to create the info dictionaries for galaxies
	Written on MacOS Mojave 10.14.5, with Python 3.7

"""

gal_red = {
    #the filepath to the data
    'data_filepath' : 'data/galaxy/gal_red.fits',
    #the filepath to the variance cube - if this is None, uses the second extension
    #from the data cube file
    'var_filepath' : None,
    #name of the galaxy, used in saving files
    'gal_name' : 'gal_example_red',
    #galaxy redshift
    'z' : 0.02,
    #whether the cube is the red or blue arrangement from KCWI - this changes
    #where the continuum band is calculated
    'cube_colour' : 'red',
    #filepath to the ssp models.
    #Options are MILES (comes with ppxf), Walcher09, BPASS and BC03 models
    'ssp_filepath' : 'models/p_walcher09/*',
    #filepath for where to save the results
    #use whatever you want, but more descriptors helps you know what you did later
    'results_folder' : 'code_outputs/ppxf_gal_example/gal_example_red_ppxf_date_models_deg6_mdeg0_emlines_masked350/',
    #whether to crop the edges of the cubes off - this was specifically written
    #to match the blue cube to the red cube for IRAS08, so don't use for any
    #other galaxy
    'data_crop' : False,
    'var_crop' : False,
    #whether to crop the edges of the wavelength range.  Takes off 150A at the
    #beginning and the end
    'lamda_crop' : False,
    #whether to apply the MW correction when reading the cube in.  If you've
    #already done MW correction and continuum subtraction and are reading the saved
    #cube in for whatever reason, turn this to False.
    'mw_correction' : True,
    # The extinction value from the Milky Way in the direction of the input galaxy.  
    # The default value is 0.2511, which is for IRAS 08339+6517.
    'Av_mw' : 0.2511,
    #the FWHM of the galaxy
    'fwhm_gal' : 1.7,
    #the FWHM of the tempate models MILES: 2.5A, Walcher09: 1.0A, BPASS: 1.0A,
    #BC03: 3A
    'fwhm_temp' : 1.0,
    #the sampling of the templates MILES: 0.9A, Walcher09: 0.9A, BPASS: 1.0A,
    #BC03: 1.0A
    'cdelt_temp' : 0.9,
    #whether to include the emission lines in the fit or not.  If this is True,
    #then ppxf uses the 'fwhm_emlines' to create gaussians which kind of fit the
    #emission lines.  If it's False, ppxf will use goodpixels to create a mask
    #for the emission lines.  The maskwidth can be controlled using 'maskwidth'
    'em_lines' : False,
    #the FWHM of the emission lines if 'em_lines' is True
    'fwhm_emlines' : 3.0,
    #you can use this to set the reddening effect on the emission lines, or None
    #see ppxf docs for more info
    'gas_reddening' : None,
    #the reddening/extinction for the galaxy continuum.  This is usually 0.13
    #set this to None when 'mdegree' > 0, since reddening can't be calculated
    #when multiplicative polynomials are included
    'reddening' : 0.13,
    #degree of additive polynomials to include (-1 for none)
    'degree' : 4,
    #degree of multiplicative polynomials to include
    'mdegree' : 0,
    #signal-to-noise ratio below which not to fit ppxf
    'sn_cut' : 3,
    #whether to do everything in vacuum wavelengths (rather than air)
    'vacuum' : True,
    #whether to include extra emission lines in the fitting (if 'em_lines' is True)
    'extra_em_lines' : False,
    #whether to tie the Balmer line ratio to the expected value when 'em_lines'
    #is True
    'tie_balmer' : True,
    #the width of the goodPixels mask - the ppxf default is 800km/s
    'maskwidth' : 350,
    #whether to plot and show every single fit (will crash your computer if the
    #cube is large).  The fit plots are saved anyway.
    'plot' : False,
    #whether to print out all of the ppxf fit information for every spaxel
    'quiet' : True,
    #whether to keep track of the normalisation done for ppxf fitting, and undo
    #it at the end
    'unnormalised' : True
}

gal_blue = {
    #the filepath to the data
    'data_filepath' : 'data/galaxy/gal_blue.fits',
    #the filepath to the variance cube - if this is None, uses the second extension
    #from the data cube file
    'var_filepath' : None,
    #name of the galaxy, used in saving files
    'gal_name' : 'gal_example_blue',
    #galaxy redshift
    'z' : 0.02,
    #whether the cube is the red or blue arrangement from KCWI - this changes
    #where the continuum band is calculated
    'cube_colour' : 'blue',
    #filepath to the ssp models.
    #Options are MILES (comes with ppxf), Walcher09, BPASS and BC03 models
    'ssp_filepath' : 'models/p_walcher09/*',
    #filepath for where to save the results
    #use whatever you want, but more descriptors helps you know what you did later
    'results_folder' : 'code_outputs/ppxf_gal_example/gal_example_blue_ppxf_date_models_deg6_mdeg0_emlines_masked350/',
    #whether to crop the edges of the cubes off - this was specifically written
    #to match the blue cube to the red cube for IRAS08, so don't use for any
    #other galaxy
    'data_crop' : False,
    'var_crop' : False,
    #whether to crop the edges of the wavelength range.  Takes off 150A at the
    #beginning and the end
    'lamda_crop' : False,
    #whether to apply the MW correction when reading the cube in.  If you've
    #already done MW correction and continuum subtraction and are reading the saved
    #cube in for whatever reason, turn this to False.
    'mw_correction' : True,
    # The extinction value from the Milky Way in the direction of the input galaxy.  
    # The default value is 0.2511, which is for IRAS 08339+6517.
    'Av_mw' : 0.2511,
    #the FWHM of the galaxy
    'fwhm_gal' : 1.7,
    #the FWHM of the tempate models MILES: 2.5A, Walcher09: 1.0A, BPASS: 1.0A,
    #BC03: 3A
    'fwhm_temp' : 1.0,
    #the sampling of the templates MILES: 0.9A, Walcher09: 0.9A, BPASS: 1.0A,
    #BC03: 1.0A
    'cdelt_temp' : 0.9,
    #whether to include the emission lines in the fit or not.  If this is True,
    #then ppxf uses the 'fwhm_emlines' to create gaussians which kind of fit the
    #emission lines.  If it's False, ppxf will use goodpixels to create a mask
    #for the emission lines.  The maskwidth can be controlled using 'maskwidth'
    'em_lines' : False,
    #the FWHM of the emission lines if 'em_lines' is True
    'fwhm_emlines' : 3.0,
    #you can use this to set the reddening effect on the emission lines, or None
    #see ppxf docs for more info
    'gas_reddening' : None,
    #the reddening/extinction for the galaxy continuum.  This is usually 0.13
    #set this to None when 'mdegree' > 0, since reddening can't be calculated
    #when multiplicative polynomials are included
    'reddening' : 0.13,
    #degree of additive polynomials to include (-1 for none)
    'degree' : 6,
    #degree of multiplicative polynomials to include
    'mdegree' : 0,
    #signal-to-noise ratio below which not to fit ppxf
    'sn_cut' : 3,
    #whether to do everything in vacuum wavelengths (rather than air)
    'vacuum' : True,
    #whether to include extra emission lines in the fitting (if 'em_lines' is True)
    'extra_em_lines' : False,
    #whether to tie the Balmer line ratio to the expected value when 'em_lines'
    #is True
    'tie_balmer' : True,
    #the width of the goodPixels mask - the ppxf default is 800km/s
    'maskwidth' : 350,
    #whether to plot and show every single fit (will crash your computer if the
    #cube is large).  The fit plots are saved anyway.
    'plot' : False,
    #whether to print out all of the ppxf fit information for every spaxel
    'quiet' : True,
    #whether to keep track of the normalisation done for ppxf fitting, and undo
    #it at the end
    'unnormalised' : True
}
