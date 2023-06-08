"""
NAME:
	NGC0695.py

AUTHOR:
	Bronwyn Reichardt Chu
	Swinburne
	2021

EMAIL:
	<breichardtchu@swin.edu.au>

PURPOSE:
	Keeps the info for NGC 695
	Written on MacOS Mojave 10.14.5, with Python 3.7

"""

ngc0695_red = {
    'data_filepath' : '/Users/breichardtchu/Documents/data/ngc0695/NGC0695_red_binned_3_by_3_cropped.fits',
    #'data_filepath' : '/Users/breichardtchu/Documents/data/ngc0695/NGC0695_red_test_data.fits',
    'var_filepath' : None,
    'gal_name' : 'ngc0695_red',
    'z' : 0.03247,
    'cube_colour' : 'red',
    'ssp_filepath' : '/Users/breichardtchu/Documents/models/p_walcher09/*',
    #'ssp_filepath' : '/Users/breichardtchu/Documents/OneDrive - Swinburne University/python_scripts/ppxf/miles_models/Mun1.3*.fits',
    'results_folder' : '/Users/breichardtchu/Documents/code_outputs/ppxf_ngc0695/ngc0695_red_ppxf_24Jan2023_Walcher09_deg26_mdeg8_emlines_masked350_final/',
    'data_crop' : False,
    'var_crop' : False,
    'lamda_crop' : False,
    'mw_correction' : True,
    'fwhm_gal' : 1.7,
    'fwhm_temp' : 1.0, # Walcher09
    'cdelt_temp' : 0.9, # Walcher09
    #'fwhm_temp' : 2.51, # MILES
    #'cdelt_temp' : 0.9, # MILES
    'em_lines' : False,
    'fwhm_emlines' : 3.0,
    'gas_reddening' : None,
    'reddening' : None, # usually 0.13, must be None when mdegree > 0
    'degree' : 26,
    'mdegree' : 8,
    'sn_cut' : 3,
    'vacuum' : True,
    'extra_em_lines' : False,
    'tie_balmer' : True,
    'maskwidth' : 350, # ppxf default is 800km/s
    'plot' : False,
    'quiet' : True,
    'unnormalised' : True
}
