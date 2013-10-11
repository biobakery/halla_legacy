#!/usr/bin/env python 
"""
unified statistics module 
"""


import numpy as np 
import scipy as sp 



############ Decomposition ###############

def pca( pArray, iComponents = 2 ):
	 """
	 Input: N x D matrix 
	 Output: D x N matrix 
	 """
	 from sklearn.decomposition import PCA
	 pPCA = PCA( n_components = iComponents )
	 return pPCA.fit_transform( pArray.T ).T 

def mca( pArray, iComponents = 2):
	pass

def discretize( pArray ):
	def _discretize_continuous( astrValues, iN = None ):
		"""
		>>> CDatum._discretize_continuous( [0] )
		[0]

		>>> CDatum._discretize_continuous( [0, 1] )
		[0, 0]

		>>> CDatum._discretize_continuous( [0, 1], 2 )
		[0, 1]

		>>> CDatum._discretize_continuous( [1, 0], 2 )
		[1, 0]

		>>> CDatum._discretize_continuous( [0.2, 0.1, 0.3], 3 )
		[1, 0, 2]

		>>> CDatum._discretize_continuous( [0.2, 0.1, 0.3], 1 )
		[0, 0, 0]

		>>> CDatum._discretize_continuous( [0.2, 0.1, 0.3], 2 )
		[0, 0, 1]

		>>> CDatum._discretize_continuous( [0.4, 0.2, 0.1, 0.3], 2 )
		[1, 0, 0, 1]

		>>> CDatum._discretize_continuous( [4, 0.2, 0.1, 0.3], 2 )
		[1, 0, 0, 1]

		>>> CDatum._discretize_continuous( [0.4, 0.2, 0.1, 0.3, 0.5] )
		[1, 0, 0, 0, 1]

		>>> CDatum._discretize_continuous( [0.4, 0.2, 0.1, 0.3, 0.5], 3 )
		[1, 0, 0, 1, 2]

		>>> CDatum._discretize_continuous( [0.4, 0.2, 0.6, 0.1, 0.3, 0.5] )
		[1, 0, 1, 0, 0, 1]

		>>> CDatum._discretize_continuous( [0.4, 0.2, 0.6, 0.1, 0.3, 0.5], 3 )
		[1, 0, 2, 0, 1, 2]

		>>> CDatum._discretize_continuous( [0.4, 0.2, 0.6, 0.1, 0.3, 0.5], 0 )
		[3, 1, 5, 0, 2, 4]

		>>> CDatum._discretize_continuous( [0.4, 0.2, 0.6, 0.1, 0.3, 0.5], 6 )
		[3, 1, 5, 0, 2, 4]

		>>> CDatum._discretize_continuous( [0.4, 0.2, 0.6, 0.1, 0.3, 0.5], 60 )
		[3, 1, 5, 0, 2, 4]

		>>> CDatum._discretize_continuous( [0, 0, 0, 0, 0, 0, 1, 2], 2 )
		[0, 0, 0, 0, 0, 0, 1, 1]

		>>> CDatum._discretize_continuous( [0, 0, 0, 0, 1, 2, 2, 2, 2, 3], 3 )
		[0, 0, 0, 0, 1, 1, 1, 1, 1, 2]

		>>> CDatum._discretize_continuous( [0.1, 0, 0, 0, 0, 0, 0, 0, 0] )
		[1, 0, 0, 0, 0, 0, 0, 0, 0]
		
		>>> CDatum._discretize_continuous( [0.992299, 1, 1, 0.999696, 0.999605, 0.663081, 0.978293, 0.987621, 0.997237, 0.999915, 0.984792, 0.998338, 0.999207, 0.98051, 0.997984, 0.999219, 0.579824, 0.998983, 0.720498, 1, 0.803619, 0.970992, 1, 0.952881, 0.999866, 0.997153, 0.014053, 0.998049, 0.977727, 0.971233, 0.995309, 0.0010376, 1, 0.989373, 0.989161, 0.91637, 1, 0.99977, 0.960816, 0.998025, 1, 0.998852, 0.960849, 0.957963, 0.998733, 0.999426, 0.876182, 0.998509, 0.988527, 0.998265, 0.943673] )
		[3, 6, 6, 5, 5, 0, 2, 2, 3, 5, 2, 4, 4, 2, 3, 5, 0, 4, 0, 6, 0, 1, 6, 1, 5, 3, 0, 3, 2, 1, 3, 0, 6, 3, 2, 0, 6, 5, 1, 3, 6, 4, 1, 1, 4, 5, 0, 4, 2, 4, 1]
		"""

		if iN == None:
			# Default to rounded sqrt(n) if no bin count requested
			iN = int(len( astrValues )**0.5 + 0.5)
		elif iN == 0:
			iN = len( astrValues )
		else:
			iN = min( iN, len( set(astrValues) ) )
			
		# This is still a bit buggy since ( [0, 0, 0, 1, 2, 2, 2, 2], 3 ) will exhibit suboptimal behavior
		aiIndices = sorted( range( len( astrValues ) ), cmp = lambda i, j: cmp( astrValues[i], astrValues[j] ) )
		astrRet = [None] * len( astrValues )
		iPrev = 0
		for i, iIndex in enumerate( aiIndices ):
			# If you're on a tie, you can't increase the bin number
			# Otherwise, increase by at most one
			iPrev = astrRet[iIndex] = iPrev if ( i and ( astrValues[iIndex] == astrValues[aiIndices[i - 1]] ) ) else \
				min( iPrev + 1, int(iN * i / float(len( astrValues ))) )
		
		return astrRet
	return array([CDatum._discretize_continuous(line) for line in pArray])

