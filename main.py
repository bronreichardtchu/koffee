"""
NAME:
	main.py

AUTHOR:
	Bronwyn Reichardt Chu
	Swinburne
	2021

EMAIL:
	<breichardtchu@swin.edu.au>

PURPOSE:
	To run through everything.
	Written on MacOS Mojave 10.14.5, with Python 3.7

"""
import pathlib
import numpy as np

import galaxy_class as galclass
import apply_ppxf

import importlib
importlib.reload(apply_ppxf)
importlib.reload(galclass)

def main(galaxy_dict, run_cont_subtraction=True, run_ext_correction=True):
    """
    Runs through all of the things.
    """
    #create the galaxy class
    gal = galclass.Galaxy(galaxy_dict['data_filepath'], galaxy_dict['var_filepath'], galaxy_dict['gal_name'], galaxy_dict['z'], galaxy_dict['cube_colour'], galaxy_dict['ssp_filepath'], galaxy_dict['results_folder'], data_crop=galaxy_dict['data_crop'], var_crop=galaxy_dict['var_crop'], lamda_crop=galaxy_dict['lamda_crop'], mw_correction=galaxy_dict['mw_correction'])

    #check if the results folder exists, create it if it doesn't
    pathlib.Path(gal.results_folder).mkdir(exist_ok=True)

    #run the cube preparation script
    gal.prepare_cube()

    #run the continuum subtraction
    if run_cont_subtraction == True:
        #set the ppxf variables
        gal.set_ppxf_variables(galaxy_dict['fwhm_gal'], galaxy_dict['fwhm_temp'], galaxy_dict['cdelt_temp'], galaxy_dict['em_lines'], galaxy_dict['fwhm_emlines'], galaxy_dict['gas_reddening'], galaxy_dict['reddening'], galaxy_dict['degree'], galaxy_dict['mdegree'], sn_cut=galaxy_dict['sn_cut'], vacuum=galaxy_dict['vacuum'], extra_em_lines=galaxy_dict['extra_em_lines'], tie_balmer=galaxy_dict['tie_balmer'], maskwidth=galaxy_dict['maskwidth'], plot=galaxy_dict['plot'], quiet=galaxy_dict['quiet'], unnormalised=galaxy_dict['unnormalised'])

        #run ppxf
        apply_ppxf.main_parallelised(gal.lamdas, gal.data_flat, noise_flat=np.sqrt(abs(gal.variance_flat)), xx_flat=gal.xx_flat, yy_flat=gal.yy_flat, ssp_filepath=gal.ssp_filepath, z=gal.redshift, results_folder=gal.results_folder, galaxy_name=gal.galaxy_name, cube_colour=gal.cube_colour, sn_cut=gal.sn_cut, fwhm_gal=gal.fwhm_gal, fwhm_temp=gal.fwhm_temp, cdelt_temp=gal.cdelt_temp, em_lines=gal.em_lines, fwhm_emlines=gal.fwhm_emlines, gas_reddening=gal.gas_reddening, reddening=gal.reddening, degree=gal.degree, mdegree=gal.mdegree, vacuum=gal.vacuum, extra_em_lines=gal.extra_em_lines, tie_balmer=gal.tie_balmer, maskwidth=gal.maskwidth, plot=gal.plot, quiet=gal.quiet)


        #combine the results and save
        apply_ppxf.combine_results(lamdas=gal.lamdas, data_flat=gal.data_flat, final_shape=[gal.lamdas.shape[0], gal.data.shape[1], gal.data.shape[2]], results_folder=gal.results_folder, galaxy_name=gal.galaxy_name, header_file=gal.data_filepath, unnormalised=gal.unnormalised, em_lines=gal.em_lines)

    #run the extinction correction
    if run_ext_correction == True:
        if run_cont_subtraction == False:
            gal.unnormalised = galaxy_dict['unnormalised']

        gal.extinction_correction(filename=None, sn_cut=3)


    return gal


#======================================================================================================

if __name__ == '__main__':
    #get the galaxy info
    #from galaxy_info import NGC0695
    from galaxy_info import KISSR1084

    #run the continuum subtraction
    #ngc695_class = main(NGC0695.ngc0695_red, run_cont_subtraction=True, run_ext_correction=True)
    kissr1084_class = main(KISSR1084.KISSR1084_red, run_cont_subtraction=True, run_ext_correction=True)
