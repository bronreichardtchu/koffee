"""
NAME:
	galaxy_classes.py

AUTHOR:
	Bronwyn Reichardt Chu
	Swinburne
	2021

EMAIL:
	<breichardtchu@swin.edu.au>

PURPOSE:
	To create a class to define all the attributes of the galaxies
	Written on MacOS Mojave 10.14.5, with Python 3.7

"""

#define the galaxy class
class Galaxy:
    """
    Contains all the attributes of the galaxy we need to run stuff on
    """
    def __init__(self, data_filepath, var_filepath, gal_name, z, ssp_filepath, results_folder):
        """
        Creates the basic galaxy class

        Parameters
        ----------
        data_filepath : str
            path to the data cube file

        var_filepath : str or None
            path to the variance cube file, or None.

        gal_name : str
            galaxy name or descriptor

        z : float
            redshift

        ssp_filepath : str
            the general filepath to the SSP models

        results_folder : str
            where to save the results
        """
        self.data_filepath = data_filepath
        self.var_filepath = var_filepath
        self.gal_name = gal_name
        self.redshift = z
        self.ssp_filepath = ssp_filepath
        self.results_folder = results_folder

    #ppxf variables
    def set_ppxf_variables(self, cube_colour, fwhm_gal, fwhm_temp, cdelt_temp, em_lines, fwhm_emlines, gas_reddening, reddening, degree, mdegree, vacuum=True, extra_em_lines=False, tie_balmer=True, plot=False, quiet=True):
        """
        Sets the variables you need to run the ppxf continuum subtraction

        Parameters
        ----------
        cube_colour : str
            'red' or 'blue' - which cube is being fit; to use in creating the
            coordinate arrays, etc.

        fwhm_gal : float
            the FWHM of the galaxy data

        fwhm_temp : float
            the FWHM of the SSP templates

        cdelt_temp : float
            the sampling of the template.  This is: 0.05 for Conroy models,
            1.0 for BPASS, 1.0 for BC03, 0.9 for Walcher09.  Default=1.0.

        em_lines : boolean
            whether or not to include emission lines in the fit

        fwhm_emlines : float
            the FWHM of the fitted emission lines

        gas_reddening : float
            Set this keyword to an initial estimate of the gas reddening
            E(B-V) >= 0 to fit a positive reddening together with the kinematics
            and the templates. The fit assumes by default the extinction curve
            of Calzetti et al. (2000, ApJ, 533, 682) but any other prescription
            can be passed via the`reddening_func` keyword.

        reddening : float
            Set this keyword to an initial estimate of the reddening E(B-V) >= 0
            to fit a positive reddening together with the kinematics and the
            templates. The fit assumes by default the extinction curve of
            Calzetti et al. (2000, ApJ, 533, 682) but any other prescription can
            be passed via the`reddening_func` keyword.
            - IMPORTANT: The MDEGREE keyword cannot be used when REDDENING is set.

        degree : int
            degree of the *additive* Legendre polynomial used to correct the
            template continuum shape during the fit (ppxf default: 4).  Set
            DEGREE = -1 not to include any additive polynomial.

        mdegree : int
            degree of the *multiplicative* Legendre polynomial (with mean of 1)
            used to correct the continuum shape during the fit (default: 0).
            The zero degree multiplicative polynomial is always included in the
            fit as it corresponds to the weights assigned to the templates. Note
            that the computation time is longer with multiplicative polynomials
            than with the same number of additive polynomials.
            - IMPORTANT: Multiplicative polynomials cannot be used when the
            REDDENING keyword is set, as they are degenerate with the reddening.

        vacuum : boolean
            whether to make wavelengths in vacuum or air wavelengths
            (default=True, default in ppxf is False)

        extra_em_lines : bool
            set to True to include extra emission lines often found in
            KCWI data (OII 4317, [OIII]4363, OII4414 and [NeIII]3868).
            (default=False)

        tie_balmer : bool
            ties the Balmer lines according to a theoretical decrement
            (case B recombination T=1e4 K, n=100 cm^-3) (default=True)

        plot : bool
            whether to show all the plots made (Default=False, only set to True
            if fitting only a few spectra)

        quiet : bool
            Whether to print the results of ppxf fitting or not (Default=False)
        """
        self.cube_colour = cube_colour
        self.fwhm_gal = fwhm_gal
        self.fwhm_temp = fwhm_temp
        self.cdelt_temp = cdelt_temp
        self.em_lines = em_lines
        self.fwhm_emlines = fwhm_emlines
        self.gas_reddening = gas_reddening
        self.reddening = reddening
        self.degree = degree
        self.mdegree = mdegree
        self.vacuum = vacuum
        self.extra_em_lines = extra_em_lines
        self.tie_balmer = tie_balmer
        self.plot = plot
        self.quiet = quiet






#IRAS08




#J164905
