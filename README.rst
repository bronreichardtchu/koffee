KOFFEE: Keck Outflow Fitter For Emission linEs
==============================================

KOFFEE takes IFU data of starbursting disk galaxies and uses gaussian decomposition to seperate emission from the galaxy from emission originating from the outflowing gas.  It is not assumed that every single spaxel should contain an outflow.  Additional tests such as Bayesian Information Criterion comparisons are applied to determine which spaxels within the cube truly require a double gaussian fit, and which spaxels can be more simply fit with a single gaussian.  More details about the fitting process can be found in my paper.

The code was written to be used on data from KCWI on Keck II.  I have not tested whether it can be used on data from any other instrument.  KOFFEE is provided as is, I may or may not respond to emails about bugs in the code, depending on how busy I am!
