#!/usr/bin/env python

"""
Module name: cakecut.py 
Description: Perform baseline test of cakecutting procedure 

Idea: I can define a deterministic log-cut of the cake, and feed in permutations of the cake; this achieves the same result 

Idea: Avoid memory intensive functions if possible. 

"""

import halla 
import numpy as np
from numpy import array 
from numpy.random import shuffle 
import math 
from halla.stats import pca, discretize, permutation_test_by_representative 
from halla.test import rand, randmix, uniformly_spaced_gaussian
from halla.distance import mi, norm_mi 
from pylab import * 




if __name__ == "__main__":
	rand_sample = rand( (100,1) ).flatten()  
	rand_mixture = array( uniformly_spaced_gaussian( 100 ) )

	p_val_plot( rand_mixture, rand_mixture )





