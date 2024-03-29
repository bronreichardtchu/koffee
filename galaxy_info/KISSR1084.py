"""
NAME:
	J164905.py

AUTHOR:
	Bronwyn Reichardt Chu
	Swinburne
	2021

EMAIL:
	<breichardtchu@swin.edu.au>

PURPOSE:
	Keeps the info for J164905
	Written on MacOS Mojave 10.14.5, with Python 3.7

"""

KISSR1084_red = {
    'data_filepath' : '/Users/breichardtchu/Documents/data/KISSR1084/KISSR1084_red_binned_3_by_3.fits',
    #'var_filepath' : '/Users/breichardtchu/Documents/data/J164905/J164905_red_var_binned_3_by_3.fits',
    #'data_filepath' : '/Users/breichardtchu/Documents/data/KISSR1084/KISSR1084_red_test_data.fits',
    'var_filepath' : None,
    'gal_name' : 'KISSR1084_red',
    'z' : 0.03205,
    'cube_colour' : 'red',
    'ssp_filepath' : '/Users/breichardtchu/Documents/models/p_walcher09/*',
    'results_folder' : '/Users/breichardtchu/Documents/code_outputs/ppxf_J164905/KISSR1084_red_ppxf_13Dec2022_walcher09_deg10_mdeg6_extra_emlines_final/',
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
    'reddening' : None, # usually 0.13, must be None when mdegree > 0
    'degree' : 10,
    'mdegree' : 6,
    'sn_cut' : 3,
    'vacuum' : True,
    'extra_em_lines' : True,
    'tie_balmer' : True,
    'maskwidth' : 250, # ppxf default is 800km/s
    'plot' : False,
    'quiet' : True,
    'unnormalised' : True
}

J164905_blue = {
    'data_filepath' : '/Users/breichardtchu/Documents/data/J164905/J164905_blue_binned_3_by_3.fits',
    'var_filepath' : '/Users/breichardtchu/Documents/data/J164905/J164905_blue_var_binned_3_by_3.fits',
    'gal_name' : 'J164905_blue',
    'z' : 0.03205,
    'cube_colour' : 'blue',
    'ssp_filepath' : '/Users/breichardtchu/Documents/models/bc03/templates/ssp_*',
    'results_folder' : '/Users/breichardtchu/Documents/code_outputs/ppxf_J164905/J164905_blue_ppxf_10Aug2021_deg6/',
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
