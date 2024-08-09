"""
NAME:
	UGC1385.py

AUTHOR:
	Bronwyn Reichardt Chu
	Swinburne
	2021

EMAIL:
	<breichardtchu@swin.edu.au>

PURPOSE:
	Keeps the info for UGC1385
	Written on MacOS Mojave 10.14.5, with Python 3.7

"""

ugc1385_red = {
    #'data_filepath' : '/Users/breichardtchu/Documents/data/ugc1385/ugc1385_red_test_data.fits',
    #'data_filepath' : '/Users/breichardtchu/Documents/data/ugc1385/ugc1385_red_binned_3_by_3.fits',
    #'var_filepath' : '/Users/breichardtchu/Documents/data/ugc1385/ugc1385_red_var_binned_3_by_3.fits',
    'data_filepath' : '/Volumes/BronsData/ugc1385/ugc1385_red_binned_3_by_3.fits',
    #'data_filepath' : '/Volumes/BronsData/ugc1385/ugc1385_red_test_data.fits',
    'var_filepath' : None,
    'gal_name' : 'ugc1385_red',
    'z' : 0.01875,
    'cube_colour' : 'red',
    #'ssp_filepath' : '/Users/breichardtchu/Documents/models/p_walcher09/*',
    'ssp_filepath' : '/Volumes/BronsData/models/p_walcher09/*',
    #'results_folder' : '/Users/breichardtchu/Documents/code_outputs/ppxf_ugc1385/ugc1385_red_ppxf_17Jan2023_test_walcher09_deg25_mdeg6_emlines_masked350/',
    'results_folder' : '/Volumes/BronsData/code_outputs/ppxf_results/UGC01385_red_ppxf_31July2024_walcher09_deg-1_mdeg25_emlines_masked350/',
    'data_crop' : False,
    'var_crop' : False,
    'lamda_crop' : False,
    'mw_correction' : True,
    'Av_mw' : 0.263,
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
    'extra_em_lines' : False,
    'tie_balmer' : True,
    'maskwidth' : 350, # ppxf default is 800km/s
    'plot' : False,
    'quiet' : True,
    'unnormalised' : True
}

ugc1385_blue = {
    #'data_filepath' : '/Users/breichardtchu/Documents/data/ugc1385/ugc1385_blue_binned_3_by_3.fits',
    #'var_filepath' : '/Users/breichardtchu/Documents/data/ugc1385/ugc1385_blue_var_binned_3_by_3.fits',
    'data_filepath' : '/Volumes/BronsData/ugc1385/ugc1385_blue_binned_3_by_3_cropped.fits',
    'var_filepath' : None,
    'gal_name' : 'ugc1385_blue',
    'z' : 0.01875,
    'cube_colour' : 'blue',
    #'ssp_filepath' : '/Users/breichardtchu/Documents/models/bc03/templates/ssp_*',
    'ssp_filepath' : '/Volumes/BronsData/models/p_walcher09/*',
    #'results_folder' : '/Users/breichardtchu/Documents/code_outputs/ppxf_ugc1385/ugc1385_blue_ppxf_16Aug2021_walcher09_deg6/',
    'results_folder' : '/Volumes/BronsData/code_outputs/ppxf_results/UGC01385_blue_ppxf_13Jun2024_walcher09_deg6_mdeg0_emlines/',
    'data_crop' : False,
    'var_crop' : False,
    'lamda_crop' : False,
    'mw_correction' : True,
    'Av_mw' : 0.316,
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
    'maskwidth' : 350, # ppxf default is 800km/s
    'plot' : False,
    'quiet' : True,
    'unnormalised' : True
}
