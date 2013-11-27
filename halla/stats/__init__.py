#!/usr/bin/env python 
"""
unified statistics module 
"""

# native python 

import math 
from itertools import compress 

# External dependencies 

import numpy as np 
from numpy import array 
import scipy as sp 
from scipy.stats import percentileofscore
import rpy 
from numpy.random import shuffle, binomial, normal, multinomial 

# Internal dependencies 

from halla.distance import mi, l2 

# doesn't play nice with other python extensions like numpy
# remember to cast to native python objects before passing!
# good for prototyping, not good for optimizing.  

#=========================================================
# Feature Selection 
#=========================================================

def pca( pArray, iComponents = 1 ):
	 """
	 Input: N x D matrix 
	 Output: D x N matrix 

	 """
	 from sklearn.decomposition import PCA
	 pPCA = PCA( n_components = iComponents )
	 ## doing this matrix inversion twice doesn't seem to be a good idea 
	 return pPCA.fit_transform( pArray.T ).T 

def mca( pArray, iComponents = 1 ):
	"""
	Input: D x N STRING DISCRETIZED matrix #Caution! must pass in strings  
	Output: D x N FLOAT matrix 
	"""
	aastrData = [map(str,l) for l in pArray.T]
	astrKeys = ["Dim " + str(i) for i in range(1,iComponents+1)]
	r = rpy.r 
	r.library( "FactoMineR" )
	residues = r.MCA( aastrData ) 
	return array( [residues["var"]["eta2"][x] for x in astrKeys] )	


def get_medoid( pArray, iAxis = 0, pMetric = l2 ):
	"""
	Input: numpy array 
	Output: float
	
	For lack of better way, compute centroid, then compute medoid 
	by looking at an element that is closest to the centroid. 

	Can define arbitrary metric passed in as a function to pMetric 

	"""

	d = pMetric 

	pArray = ( pArray.T if bool(iAxis) else pArray  ) 

	mean_vec = np.mean(pArray, 0) 
	
	pArrayCenter = pArray - ( mean_vec * np.ones(pArray.shape) )

	return pArray[np.argsort( map( np.linalg.norm, pArrayCenter) )[0],:]


def get_representative( pArray, pMethod = None ):
	hash_method = {None: get_medoid, "pca": pca, "mca": mca}
	return hash_method[pMethod]( pArray )

#=========================================================
# Statistical test 
#=========================================================

def permutation_test_by_representative( pArray1, pArray2, metric = "mi", decomposition = "pca", iIter = 100):
	"""
	Input: 
	pArray1, pArray2, metric = "mi", decomposition = "pca", iIter = 100

	metric = {"mca": mca, "pca": pca} 
	"""
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

	#print pArrayPerm 

	dPPerm = percentileofscore( pArrayPerm, dMI ) / 100 	

	return dPPerm


#=========================================================
# Cake Cutting 
#=========================================================
"""
Think about the differences between pdf and cdf 
"""

def uniform_cut( cake_length, iCuts = 10 ):
	"""
	Cut cake uniformly

	Note
	------

	Code still observes sub-optimal behavior; fix. 
	
	"""

	cake = range(cake_length)
	aOut = [] 
	iSize = int( math.floor( float(cake_length)/iCuts ) ) + 1
	while cake:
		aOut.append(cake[:iSize]) ; cake = cake[iSize:]
	return aOut 

def cumulative_uniform_cut( cake_length, iCuts = 10):
	assert( cake_length > iCuts )

	aOut = [] 

	iSize = int( math.floor( float(cake_length)/iCuts ) )

	for iStep in range(1,iSize+1):
		if iStep!= iSize:
			aOut.append( range(cake_length)[:iStep*iCuts] )
		else:
			aOut.append( range(cake_length)[:] )

	return aOut 

def log_cut( cake_length, iBase = 2 ):
	"""
	Input: cake_length <- length of array, iBase <- base of logarithm 

	Output: array of indices corresponding to the slice 

	Note: Probably don't want size-1 cake slices, but for proof-of-concept, this should be okay. 
	Avoid the "all" case 

	"""

	aOut = [] 

	iLength = cake_length 

	iSize = int( math.floor( math.log( iLength , iBase ) ) )
	aSize = [2**i for i in range(iSize)] 

	iStart = 0 
	for item in aSize:
		iStop =  iStart + item 
		if iStop == iLength - 1:
			iStop += 1 
			# ensure that the rest of the cake gets included in the tail case  
		aOut.append( array( range(iStart, iStop) ) ) 
		iStart = iStop 

	aOut.reverse() #bigger ones first 
	return aOut 

def cumulative_log_cut( cake_length, iBase = 2 ):
	"""
	Input: cake_length <- length of array, iBase <- base of logarithm 

	Output: array of indices corresponding to the slice 

	Note: Probably don't want size-1 cake slices, but for proof-of-concept, this should be okay. 
	Avoid the "all" case 

	"""

	aOut = [] 

	iLength = cake_length 

	iSize = int( math.floor( math.log( iLength , iBase ) ) )
	aSize = [2**i for i in range(iSize+1)] 

	aOut = [ range(cake_length)[:x] for x in aSize]
	aOut.reverse()
	return map( array, aOut )

def tables_to_probability( aaTable ):
	if not aaTable:
		raise Exception("Empty table.")
	
	aOut = [] 
	iN = sum( [len(x) for x in aaTable] )
	#iN = reduce( lambda x,y: len(x)+y, aaTable, 0 ) #number of elements 

	for aTable in aaTable:
		iB = len(aTable)
		aOut.append(float(iB)/(iN+1))

	#probability of creating new table 
	aOut.append(1.0/(iN+1))

	return aOut 

def CRP_cut( cake_length ):
	
	iLength = cake_length 

	aOut = None 

	for i in range(iLength):
		if not aOut:
			aOut = [[i]] 
			continue 
		else:
			pBool = multinomial( 1, tables_to_probability( aOut ) ) 
		
			iK = [x for x in compress( range(len(pBool)), pBool )][0]
			if iK+1 > len(aOut): # create new table 
				aOut.append([i])
			else:
				aOut[iK].append(i) # seat the customer on an existing table 

	return map( array, aOut ) # return as numpy array, so we can vectorize 

def cumulative_CRP_cut( cake_length ):
	aTmp = sorted( CRP_cut( cake_length ), key=lambda x: -1*len(x) )
	iLength = len( aTmp )

	return [ np.hstack( aTmp[:i] ) for i in range(1,iLength) ]

def PY_cut( cake_length ):
	""" 
	random cut generated by pitman-yor process prior
	"""

	pass

def IBP_cut( cake_length ):
	"""
	random cut generated by Indian Buffet Process prior
	"""

	pass 
	

def p_val_plot( pArray1, pArray2, pCut = log_cut, iIter = 100 ):
	"""
	Returns p value plot of combinatorial cuts 

	In practice, works best when arrays are of similar size, since I implement the minimum ... 
	For future think about implementing the correct step function 

	"""

	aOut = None 
	D1, D2 = pArray1[:], pArray2[:]

	for i in range(iIter):
		shuffle(D1)
		shuffle(D2)

		print "shuffled data"
		print D1 

		len1, len2 = len( D1 ), len( D2 )
		cut1, cut2 = sorted(pCut( len1 ), key= lambda x: -1.0* len(x)), sorted( pCut( len2 ), key = lambda x: -1.0 * len(x) )
		lencut1, lencut2 = len(cut1), len(cut2)
		iMin = min( lencut1, lencut2 )
		if not aOut:
			aOut = [[] for _ in range(iMin)] 

		print "cut1"
		print cut1
		print "cut2"
		print cut2

		for j in range(iMin):
			dP = permutation_test_by_representative( pArray1[cut1[j]], pArray2[cut2[j]] )

			print "pval"
			print dP 

			#be careful when using CRP/PYP, don't know the cluster size in advance 
			try: 
				aOut[j].append( dP )
			except IndexError: 
				aOut += [[] for _ in range(j-len(aOut)+1)]
				aOut[j].append( dP )

	return aOut 


#=========================================================
# Density estimation 
#=========================================================

def discretize( pArray, iN = None, method = None, aiSkip = [] ):
	"""
	>>> discretize( [0.1, 0.2, 0.3, 0.4] )
	[0, 0, 1, 1]

	>>> discretize( [0.01, 0.04, 0.09, 0.16] )
	[0, 0, 1, 1]

	>>> discretize( [-0.1, -0.2, -0.3, -0.4] )
	[1, 1, 0, 0]

	>>> discretize( [0.25, 0.5, 0.75, 1.00] )
	[0, 0, 1, 1]

	>>> discretize( [0.015625, 0.125, 0.421875, 1] )
	[0, 0, 1, 1]

	>>> discretize( [0] )
	[0]

	>>> discretize( [0, 1] )
	[0, 0]

	>>> discretize( [0, 1], 2 )
	[0, 1]

	>>> discretize( [1, 0], 2 )
	[1, 0]

	>>> discretize( [0.2, 0.1, 0.3], 3 )
	[1, 0, 2]

	>>> discretize( [0.2, 0.1, 0.3], 1 )
	[0, 0, 0]

	>>> discretize( [0.2, 0.1, 0.3], 2 )
	[0, 0, 1]

	>>> discretize( [0.4, 0.2, 0.1, 0.3], 2 )
	[1, 0, 0, 1]

	>>> discretize( [4, 0.2, 0.1, 0.3], 2 )
	[1, 0, 0, 1]

	>>> discretize( [0.4, 0.2, 0.1, 0.3, 0.5] )
	[1, 0, 0, 0, 1]

	>>> discretize( [0.4, 0.2, 0.1, 0.3, 0.5], 3 )
	[1, 0, 0, 1, 2]

	>>> discretize( [0.4, 0.2, 0.6, 0.1, 0.3, 0.5] )
	[1, 0, 1, 0, 0, 1]

	>>> discretize( [0.4, 0.2, 0.6, 0.1, 0.3, 0.5], 3 )
	[1, 0, 2, 0, 1, 2]

	>>> discretize( [0.4, 0.2, 0.6, 0.1, 0.3, 0.5], 0 )
	[3, 1, 5, 0, 2, 4]

	>>> discretize( [0.4, 0.2, 0.6, 0.1, 0.3, 0.5], 6 )
	[3, 1, 5, 0, 2, 4]

	>>> discretize( [0.4, 0.2, 0.6, 0.1, 0.3, 0.5], 60 )
	[3, 1, 5, 0, 2, 4]

	>>> discretize( [0, 0, 0, 0, 0, 0, 1, 2], 2 )
	[0, 0, 0, 0, 0, 0, 1, 1]

	>>> discretize( [0, 0, 0, 0, 1, 2, 2, 2, 2, 3], 3 )
	[0, 0, 0, 0, 1, 1, 1, 1, 1, 2]

	>>> discretize( [0.1, 0, 0, 0, 0, 0, 0, 0, 0] )
	[1, 0, 0, 0, 0, 0, 0, 0, 0]

	>>> discretize( [0.992299, 1, 1, 0.999696, 0.999605, 0.663081, 0.978293, 0.987621, 0.997237, 0.999915, 0.984792, 0.998338, 0.999207, 0.98051, 0.997984, 0.999219, 0.579824, 0.998983, 0.720498, 1, 0.803619, 0.970992, 1, 0.952881, 0.999866, 0.997153, 0.014053, 0.998049, 0.977727, 0.971233, 0.995309, 0.0010376, 1, 0.989373, 0.989161, 0.91637, 1, 0.99977, 0.960816, 0.998025, 1, 0.998852, 0.960849, 0.957963, 0.998733, 0.999426, 0.876182, 0.998509, 0.988527, 0.998265, 0.943673] )
	[3, 6, 6, 5, 5, 0, 2, 2, 3, 5, 2, 4, 4, 2, 3, 5, 0, 4, 0, 6, 0, 1, 6, 1, 5, 3, 0, 3, 2, 1, 3, 0, 6, 3, 2, 0, 6, 5, 1, 3, 6, 4, 1, 1, 4, 5, 0, 4, 2, 4, 1]
	
	>>> x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])
	>>> y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0],[0.015625,0.125,0.421875,1.0]])
	>>> dx = discretize( x, iN = None, method = None, aiSkip = [1,3] )
	>>> dx
	array([[ 0.,  0.,  1.,  1.],
	       [ 1.,  1.,  1.,  0.],
	       [ 0.,  0.,  1.,  1.],
	       [ 0.,  0.,  0.,  1.]])
	>>> dy = discretize( y, iN = None, method = None, aiSkip = [1] )
	>>> dy 
	array([[ 1.,  1.,  0.,  0.],
	       [ 1.,  1.,  0.,  0.],
	       [ 0.,  0.,  1.,  1.],
	       [ 0.,  0.,  1.,  1.]])

	"""


	def _discretize_continuous( astrValues, iN = iN ): 
		
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

	try:
		iRow1, iCol = pArray.shape 

		aOut = [] 
		
		for i, line in enumerate( pArray ):
			if i in aiSkip:
				aOut.append( line )
			else:
				aOut.append( _discretize_continuous( line ) )

		return array( aOut )

	except Exception:
		return _discretize_continuous( pArray )

def discretize2d( pX, pY, method = None ):
	pass 


#=========================================================
# FDR correcting procedure  
#=========================================================

def bh( afPVAL, fQ = 1.0 ):
	"""
	Implement the benjamini-hochberg hierarchical hypothesis testing criterion 
	In practice, used for implementing Yekutieli criterion *per layer*.  

	When BH is performed per layer, FDR is approximately 

	.. math::
		FDR = q \cdot \delta^{*} \cdot(m_0 + m_1)/(m_0+1)

	where :math:`m_0` is the observed number of discoveries and :math:`m_1` is the observed number of families tested. 

	Universal bound: the full tree FDR is :math:`< q \cdot \delta^{*} \cdot 2`

	
	`afPVAL`
	 list of p-values 
 
	`abOUT`
	 boolean vector corresponding to which hypothesis test rejected, corresponding to p-value 

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

	if len(afPVAL) == 1:
		return [1] if afPVAL[0] < fQ else [0]
	else:
		return array( [1] * (rt + 1) + [0] * ( len(afPVAL) - (rt+1) ) )[ afPVAL_reverse ]

