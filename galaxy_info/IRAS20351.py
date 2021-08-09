"""
NAME:
	IRAS20351.py

AUTHOR:
	Bronwyn Reichardt Chu
	Swinburne
	2021

EMAIL:
	<breichardtchu@swin.edu.au>

PURPOSE:
	Keeps the info for IRAS20351
	Written on MacOS Mojave 10.14.5, with Python 3.7

"""

IRAS20351_red = {
    'data_filepath' : '/Users/breichardtchu/Documents/data/IRAS20351/IRAS20351_red_binned_3_by_3.fits',
    'var_filepath' : '/Users/breichardtchu/Documents/data/IRAS20351/IRAS20351_red_var_binned_3_by_3.fits',
    'gal_name' : 'IRAS20351_red',
    'z' : 0.03370,
    'cube_colour' : 'red',
    'ssp_filepath' : '/Users/breichardtchu/Documents/models/p_walcher09/*',
    'results_folder' : '/Users/breichardtchu/Documents/code_outputs/ppxf_IRAS20351/IRAS20351_red_ppxf_26July2021_walcher09_deg6/',
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
    'degree' : 6,
    'mdegree' : 0,
    'vacuum' : True,
    'extra_em_lines' : False,
    'tie_balmer' : True,
    'plot' : False,
    'quiet' : True,
    'unnormalised' : True
}

IRAS20351_blue = {
    'data_filepath' : '/Users/breichardtchu/Documents/data/IRAS20351/IRAS20351_blue_binned_3_by_3.fits',
    'var_filepath' : '/Users/breichardtchu/Documents/data/IRAS20351/IRAS20351_blue_var_binned_3_by_3.fits',
    'gal_name' : 'IRAS20351_blue',
    'z' : 0.03370,
    'cube_colour' : 'blue',
    'ssp_filepath' : '/Users/breichardtchu/Documents/models/bc03/templates/ssp_*',
    'results_folder' : '/Users/breichardtchu/Documents/code_outputs/ppxf_IRAS20351/IRAS20351_blue_ppxf_14May2021/',
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
    'degree' : 6,
    'mdegree' : 0,
    'vacuum' : True,
    'extra_em_lines' : False,
    'tie_balmer' : True,
    'plot' : False,
    'quiet' : True,
    'unnormalised' : True
}
