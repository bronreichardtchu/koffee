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
    'data_filepath' : '/Users/breichardtchu/Documents/data/ngc0695/ngc0695_red_binned_3_by_3.fits',
    'var_filepath' : None,
    'gal_name' : 'ngc0695_red',
    'z' : 0.03247,
    'cube_colour' : 'red',
    #'ssp_filepath' : '/Users/breichardtchu/Documents/models/p_walcher09/*',
    'ssp_filepath' : '/Users/breichardtchu/Documents/OneDrive - Swinburne University/python_scripts/ppxf/miles_models/Mun1.3*.fits',
    'results_folder' : '/Users/breichardtchu/Documents/code_outputs/ppxf_ngc0695/ngc0695_red_ppxf_1June2022_MILES_deg9_masked/',
    'data_crop' : False,
    'var_crop' : False,
    'lamda_crop' : False,
    'mw_correction' : True,
    'fwhm_gal' : 1.7,
    #'fwhm_temp' : 1.0, # Walcher09
    #'cdelt_temp' : 0.9, # Walcher09
    'fwhm_temp' : 2.51, # MILES
    'cdelt_temp' : 0.9, # MILES
    #'em_lines' : True,
    'em_lines' : False,
    'fwhm_emlines' : 3.0,
    'gas_reddening' : None,
    'reddening' : 0.13,
    #'reddening' : None,
    'degree' : 9,
    'mdegree' : 0,
    'sn_cut' : 5,
    'vacuum' : True,
    'extra_em_lines' : False,
    'tie_balmer' : True,
    'plot' : False,
    'quiet' : True,
    'unnormalised' : True
}
