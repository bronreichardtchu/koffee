"""
NAME:
	IRAS08.py

AUTHOR:
	Bronwyn Reichardt Chu
	Swinburne
	2021

EMAIL:
	<breichardtchu@swin.edu.au>

PURPOSE:
	Keeps the info for IRAS08
	Written on MacOS Mojave 10.14.5, with Python 3.7

"""

IRAS08_red = {
    #'data_filepath' : '/Users/breichardtchu/Documents/data/IRAS08/IRAS08_red_cubes/IRAS08339_metacube.fits',
    #'var_filepath' : '/Users/breichardtchu/Documents/data/IRAS08/IRAS08_red_cubes/kb180215_00081_vcubes.fits',
    'data_filepath' : '/Volumes/BronsData/IRAS08/IRAS08_red_cubes/IRAS08339_metacube.fits',
    #'data_filepath' : '/Volumes/BronsData/IRAS08/IRAS08_red_cubes/IRAS08339_metacube_test.fits',
    'var_filepath' : '/Volumes/BronsData/IRAS08/IRAS08_red_cubes/kb180215_00081_vcubes.fits',
    'gal_name' : 'IRAS08339_red',
    'z' : 0.018950,
    'cube_colour' : 'red',
    #'ssp_filepath' : '/Users/breichardtchu/Documents/models/BPASS_v2.2.1_Tuatara/BPASSv2.2.1_bin-imf135_300/spectra-bin-imf135_300*',
    'ssp_filepath' : '/Volumes/BronsData/models/BPASS_v2.2.1_Tuatara/BPASSv2.2.1_bin-imf135_300/spectra-bin-imf135_300*',
    #'results_folder' : '/Users/breichardtchu/Documents/code_outputs/ppxf_IRAS08/IRAS08_ppxf_trial/',
    'results_folder' : '/Volumes/BronsData/code_outputs/ppxf_results/IRAS08339_red_ppxf_01Aug2024_BPASS_deg-1_mdeg4_emlines_masked500/',
    'data_crop' : False,
    'var_crop' : True,
    'lamda_crop' : False,
    'mw_correction' : True,
    'Av_mw' : 0.304,
    'fwhm_gal' : 1.7,
    'fwhm_temp' : 1.0,
    'cdelt_temp' : 1.0,
    'em_lines' : False,
    'fwhm_emlines' : 3.5,
    'gas_reddening' : None,
    'reddening' : None, # usually 0.13, must be None when mdegree > 0
    'degree' : -1,
    'mdegree' : 4,
    'sn_cut' : 3,
    'vacuum' : True,
    'extra_em_lines' : False,
    'tie_balmer' : True,
    'maskwidth' : 500, # ppxf default is 800 km/s; only used if em_lines = False
    'plot' : False,
    'quiet' : True,
    'unnormalised' : True
}

IRAS08_blue = {
    'data_filepath' : '/Users/breichardtchu/Documents/data/IRAS08/IRAS08_blue_cubes/IRAS08_blue_combined.fits',
    'var_filepath' : '/Users/breichardtchu/Documents/data/IRAS08/IRAS08_blue_cubes/IRAS08_blue_combined_var.fits',
    'gal_name' : 'IRAS08339_blue',
    'z' : 0.018950,
    'cube_colour' : 'blue',
    'ssp_filepath' : '/Users/breichardtchu/Documents/models/BPASS_v2.2.1_Tuatara/BPASSv2.2.1_bin-imf135_300/spectra-bin-imf135_300*',
    'results_folder' : '/Users/breichardtchu/Documents/code_outputs/ppxf_IRAS08/IRAS08blue_ppxf_4March2021_combined/',
    'data_crop' : True,
    'var_crop' : True,
    'lamda_crop' : True,
    'mw_correction' : True,
    'Av_mw' : 0.365,
    'fwhm_gal' : 1.7,
    'fwhm_temp' : 1.0,
    'cdelt_temp' : 1.0,
    'em_lines' : True,
    'fwhm_emlines' : 3.5,
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
