"""
Wrappers for testing procedures, random data generation, etc 
"""

import numpy as np 
from numpy import array 
import scipy as sp 
from scipy.stats import percentileofscore

def rand( tShape = (10,10), pDist = np.random.normal ):
	
	H = pDist #base measure 
	
	iRow, iCol = tShape

	assert( iRow != 0 and iCol !=0 ) 
	
	return array( [[H() for i in range(iRow)] for j in range(iCol)] )