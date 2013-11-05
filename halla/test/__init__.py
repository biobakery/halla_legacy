"""
Wrappers for testing procedures, random data generation, etc 
"""

import numpy as np 
from numpy import array 
import scipy as sp 
from scipy.stats import percentileofscore
from numpy.random import normal, multinomial 


#================================================================================
# Base cases 
#================================================================================

def rand( tShape = (10,10), pDist = normal ):
	"""
	Returns a tShape-dimensional matrix given by base distribution pDist 
	Order: Row, Col 
	"""	
	H = pDist #base measure 
	
	iRow, iCol = tShape

	assert( iRow != 0 and iCol !=0 ) 
	
	return array( [[H() for i in range(iRow)] for j in range(iCol)] )

def randmix( N, pDist = normal, atParam, tPi ):
	"""
	Returns N copies drawn from a mixture distribution $H$ 
	Input: N <- number of components
		pDist <- pointer to base distribution H 
		atParam <- length $k$ parameters to distribution pDist, $\theta$  
		tPi <- length $k$ tuple (vector) to categorical rv Z_n 

	Output: N copies from mixture distribution $\sum_{k=1}^{K} \pi_k H(.| \theta )$ 
	""" 
	
	assert( len( atParam ) == len( tPi ) )
	
	aOut = [] 

	K = len( atParam ) 
	H = pDist 
	for n in range(N):
		Zn = multinomial( 1, tPi )
		aOut.append( H( *atParam[Zn] ) )

#================================================================================
# Special cases 
#================================================================================

		
