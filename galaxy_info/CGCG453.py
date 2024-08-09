"""
NAME:
	CGCG453.py

AUTHOR:
	Bronwyn Reichardt Chu
	Swinburne
	2021

EMAIL:
	<breichardtchu@swin.edu.au>

PURPOSE:
	Keeps the info for CGCG453
	Written on MacOS Mojave 10.14.5, with Python 3.7

"""

cgcg453_red = {
    #'data_filepath' : '/Users/breichardtchu/Documents/data/cgcg453-062/cgcg453_red_test_data_broad_region.fits',
    #'data_filepath' : '/Users/breichardtchu/Documents/data/cgcg453-062/cgcg453_red_binned_3_by_3.fits',
    #'var_filepath' : '/Users/breichardtchu/Documents/data/cgcg453/cgcg453_red_var_binned_3_by_3.fits',
    'data_filepath' : '/Volumes/BronsData/cgcg453-062/cgcg453_red_binned_3_by_3_cropped.fits',
    #'data_filepath' : '/Volumes/BronsData/cgcg453-062/cgcg453_red_binned_3_by_3_cropped_broad_region.fits',
    #'data_filepath' : '/Volumes/BronsData/cgcg453-062/cgcg453_red_test_data.fits',
    'var_filepath' : None,
    'gal_name' : 'cgcg453_red',
    'z' : 0.02510,
    'cube_colour' : 'red',
    #'ssp_filepath' : '/Users/breichardtchu/Documents/models/p_walcher09/*',
    'ssp_filepath' : '/Volumes/BronsData/models/p_walcher09/*',
    #'results_folder' : '/Users/breichardtchu/Documents/code_outputs/ppxf_cgcg453/cgcg453_red_ppxf_10Jan2023_test_broad_walcher09_deg10_mdeg6_emlines_masked500/',
    'results_folder' : '/Volumes/BronsData/code_outputs/ppxf_results/CGCG453-062_red_ppxf_31July2024_walcher09_deg-1_mdeg25_emlines_masked500/',
    'data_crop' : False,
    'var_crop' : False,
    'lamda_crop' : False,
    'mw_correction' : True,
    'Av_mw' : 0.330,
    'fwhm_gal' : 1.7,
    'fwhm_temp' : 1.0,
    'cdelt_temp' : 0.9,
    'em_lines' : False,
    'fwhm_emlines' : 3.0,
    'gas_reddening' : None,
    'reddening' : None, # usually 0.13, must be None when mdegree > 0
    'degree' : -1,
    'mdegree' : 25,
    'sn_cut' : 3,
    'vacuum' : True,
    'extra_em_lines' : True,
    'tie_balmer' : True,
    'maskwidth' : 500, #250, #500, # ppxf default is 800 km/s; only used if em_lines = False
    'plot' : False,
    'quiet' : True,
    'unnormalised' : True
}

cgcg453_blue = {
    #'data_filepath' : '/Users/breichardtchu/Documents/data/cgcg453/cgcg453_blue_binned_3_by_3.fits',
    #'var_filepath' : '/Users/breichardtchu/Documents/data/cgcg453/cgcg453_blue_var_binned_3_by_3.fits',
    'data_filepath' : '/Volumes/BronsData/cgcg453-062/cgcg453_blue_binned_3_by_3_cropped.fits',
    'var_filepath' : None,
    'gal_name' : 'cgcg453_blue',
    'z' : 0.02510,
    'cube_colour' : 'blue',
    #'ssp_filepath' : '/Users/breichardtchu/Documents/models/bc03/templates/ssp_*',
    'ssp_filepath' : '/Volumes/BronsData/models/p_walcher09/*',
    #'results_folder' : '/Users/breichardtchu/Documents/code_outputs/ppxf_cgcg453/cgcg453_blue_ppxf_10Aug2021_walcher09_deg6/',
    'results_folder' : '/Volumes/BronsData/code_outputs/ppxf_results/CGCG453-062_blue_ppxf_14Jun2024_walcher09_deg6_mdeg0_emlines/',
    'data_crop' : False,
    'var_crop' : False,
    'lamda_crop' : False,
    'mw_correction' : True,
    'Av_mw' : 0.395,
    'fwhm_gal' : 1.7,
    'fwhm_temp' : 1.0,
    'cdelt_temp' : 0.9,
    'em_lines' : True,
    'fwhm_emlines' : 3.0,
    'gas_reddening' : None,
    'reddening' : 0.13,
    'degree' : 6,
    'mdegree' : 0,
    'sn_cut' : 3,
    'vacuum' : True,
    'extra_em_lines' : False,
    'tie_balmer' : True,
    'maskwidth' : 500, #250, #500, # ppxf default is 800 km/s; only used if em_lines = False
    'plot' : False,
    'quiet' : True,
    'unnormalised' : True
}
