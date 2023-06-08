"""
NAME:
	J152527.py

AUTHOR:
	Bronwyn Reichardt Chu
	Swinburne
	2021

EMAIL:
	<breichardtchu@swin.edu.au>

PURPOSE:
	Keeps the info for J152527
	Written on MacOS Mojave 10.14.5, with Python 3.7

"""

J152527_red = {
    #'data_filepath' : '/Users/breichardtchu/Documents/data/J152527/J152527_red_test_data.fits',
    'data_filepath' : '/Users/breichardtchu/Documents/data/J152527/J152527_red_binned_3_by_3.fits',
    #'var_filepath' : '/Users/breichardtchu/Documents/data/J152527/J152527_red_var_binned_3_by_3.fits',
    'var_filepath' : None,
    'gal_name' : 'J152527_red',
    'z' : 0.03575,
    'cube_colour' : 'red',
    'ssp_filepath' : '/Users/breichardtchu/Documents/models/p_walcher09/*',
    'results_folder' : '/Users/breichardtchu/Documents/code_outputs/ppxf_J152527/J152527_red_ppxf_16Jan2023_walcher09_deg15_mdeg6_emlines_masked350_final/',
    'data_crop' : False,
    'var_crop' : False,
    'lamda_crop' : False,
    'mw_correction' : True,
    'fwhm_gal' : 1.7,
    'fwhm_temp' : 1.0,
    'cdelt_temp' : 0.9,
    'em_lines' : False,
    'fwhm_emlines' : 3.0,
    'gas_reddening' : None,
    'reddening' : None, # usually 0.13, must be None when mdegree > 0
    'degree' : 15,
    'mdegree' : 6,
    'sn_cut' : 3,
    'vacuum' : True,
    'extra_em_lines' : False,
    'tie_balmer' : True,
    'maskwidth' : 350, # ppxf default is 800km/s
    'plot' : False,
    'quiet' : True,
    'unnormalised' : True
}

J152527_blue = {
    'data_filepath' : '/Users/breichardtchu/Documents/data/J152527/J152527_blue_binned_3_by_3.fits',
    'var_filepath' : '/Users/breichardtchu/Documents/data/J152527/J152527_blue_var_binned_3_by_3.fits',
    'gal_name' : 'J152527_blue',
    'z' : 0.03575,
    'cube_colour' : 'blue',
    'ssp_filepath' : '/Users/breichardtchu/Documents/models/bc03/templates/ssp_*',
    'results_folder' : '/Users/breichardtchu/Documents/code_outputs/ppxf_J152527/J152527_blue_ppxf_9August2021_walcher09_deg6_OIIsn/',
    'data_crop' : False,
    'var_crop' : False,
    'lamda_crop' : False,
    'mw_correction' : True,
    'fwhm_gal' : 1.7,
    'fwhm_temp' : 1.0,
    'cdelt_temp' : 0.9,
    'em_lines' : True,
    'fwhm_emlines' : 3.0,
    'gas_reddening' : None,
    'reddening' : 0.13,
    'degree' : -1,
    'mdegree' : 0,
    'vacuum' : True,
    'extra_em_lines' : False,
    'tie_balmer' : True,
    'plot' : False,
    'quiet' : True,
    'unnormalised' : True
}
