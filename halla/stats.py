#!/usr/bin/env python 
"""
unified statistics module 
"""


import numpy as np 
from numpy import array 
import scipy as sp 
from scipy.stats import percentileofscore
from distance import mi, l2 
import rpy 


#=========================================================
# Decomposition wrappers 
#=========================================================

def pca( pArray, iComponents = 1 ):
	 """
	 Input: N x D matrix 
	 Output: D x N matrix 
	 """
	 from sklearn.decomposition import PCA
	 pPCA = PCA( n_components = iComponents )
	 return pPCA.fit_transform( pArray.T ).T 

def mca( pArray, iComponents = 1 ):
	pass 

#=========================================================
# Statistical test 
#=========================================================

def permutation_test( pArray1, pArray2, metric = "mi", decomposition = "pca", iIter = 100):
	
	pHashDecomposition = {"mca": mca, "pca": pca}
	pHashMetric = {"mi": mi}

	def _permutation( pVec ):
		return np.random.permutation( pVec )

	pDe = pHashDecomposition[decomposition]
	pMe = pHashMetric["mi"] 

	pRep1, pRep2 = [ discretize( pDe( pA ) )[0] for pA in [pArray1,pArray2] ]

	dMI = pMe( pRep1, pRep2 ) 

	# WLOG, permute pArray1 instead of 2, or both. Can fix later with added theory. 
	pArrayPerm = np.array( [ pMe( _permutation( pRep1 ), pRep2 ) for i in xrange( iIter ) ] )

	dPPerm = percentileofscore( pArrayPerm, dMI ) / 100 	

	return dPPerm

#=========================================================
# Density estimation 
#=========================================================

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
	return array([_discretize_continuous(line) for line in pArray])


#=========================================================
# FDR correcting procedure  
#=========================================================

def bh( afPVAL, fQ = 0.05 ):
	"""
	Implement the benjamini-hochberg hierarchical hypothesis testing criterion 
	In practice, used for implementing Yekutieli criterion PER layer 

	latex: $q$ BH procedure on $\mathcal{T}_t$:

	\begin{enumerate}
		\item $P_{(1)}^{t} \leq \cdots \leq P_{(m_t)^{t} $
		\item $r+t := \max\{i: P_{(i)}^{t} \leq i \cdot q / m_t \}
		\item If $r_t >0,$ reject $r_t$ hypotheses corresponding to $P_{(1)}^t, \ldots, P_{(r_t)}^t$
	\end{enumerate}

	Then FDR is approximately 

	\begin{equation}
		FDR = q \cdot \delta^{*} \cdot(observed no. of idscoveries + observed no. of families tested)/(observed no. of discoveries+1)
	\end{equation}

	Universal bound: the full tree FDR is $< q \cdot \delta^{*} \cdot 2$ 

	INPUT 

	afPVAL: list of p-values 

	OUTPUT 

	abOUT: boolean vector corresponding to which hypothesis test rejected, corresponding to p-value 

	"""

	afPVAL_args = np.argsort( afPVAL ) # permutation \pi
	afPVAL_reverse = np.argsort( afPVAL_args ) # unique inverse permutation \pi \; \pi \circ \pi^{-1} = 1
	afPVAL_sorted = array( afPVAL )[afPVAL_args]

	def _find_max( afPVAL_sorted, fQ ):
		dummyMax = -1 
		for i, pval in enumerate(afPVAL_sorted):
			fVal = i * fQ * 1.0/len(afPVAL_sorted)
			if pval <= fVal:
				dummyMax = i
		return dummyMax

	rt = _find_max( afPVAL_sorted , fQ )

	return array( [1] * (rt + 1) + [0] * ( len(afPVAL) - (rt+1) ) )[ afPVAL_reverse ]

