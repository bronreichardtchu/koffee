"""
NAME:
	calculate_equivalent_width.py

AUTHOR:
	Bronwyn Reichardt Chu
	Swinburne
	2020

EMAIL:
	<breichardtchu@swin.edu.au>

PURPOSE:
	To calculate the equivalent width of emission lines
	Written on MacOS Mojave 10.14.5, with Python 3.7

MODIFICATION HISTORY:
		v.1.0 - first created November 2020

"""
import numpy as np

from astropy.cosmology import WMAP9 as cosmo
from astropy.constants import c
from astropy import units
