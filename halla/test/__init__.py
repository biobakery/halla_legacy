"""
Wrappers for testing procedures, random data generation, etc 
"""

import numpy as np 
from numpy import array 
import scipy as sp 
from scipy.stats import percentileofscore
from numpy.random import normal, multinomial 
from itertools import compress 

#================================================================================
# Base cases 
#================================================================================

def randmat( tShape = (10,10), pDist = np.random.normal ):
	"""
	Returns a tShape-dimensional matrix given by base distribution pDist 
	Order: Row, Col 
	"""	
	H = pDist #base measure 
	
	iRow, iCol = tShape

	assert( iRow != 0 and iCol !=0 ) 
	
	return array( [[H() for i in range(iCol)] for j in range(iRow)] )

def randmix( N, pDist, atParam, tPi ):
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
		# multinomial returns boolean vector 
		aParam = [x for x in compress( atParam, multinomial( 1, tPi ) )][0]
		aOut.append( H( *aParam ) )
	return aOut

#================================================================================
# Special cases 
#================================================================================

def uniformly_spaced_gaussian( N, K = 4, fD = 2.0, tPi = (0.25,0.25,0.25,0.25) ):
	"""
	Generate uniformly spaced Gaussian, with spacing fD in the mean.
	Constant 1.0 variance 

	"""
	return randmix( N, pDist = normal, atParam = [(m,1.0) for m in [fD*i for i in range(K)]], tPi = tPi )


#================================================================================
# Pipelines  
#================================================================================

