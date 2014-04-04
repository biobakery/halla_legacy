#!/usr/bin/env python 
"""
unified statistics module 
"""

# native python 

import math 
import itertools 
from itertools import compress 
import sys 

# External dependencies 

import pylab as pl
import scipy 
import numpy 
from numpy import array 
from scipy.stats import percentileofscore
from numpy.random import shuffle, binomial, normal, multinomial 

# ML plug-in 
import sklearn 
from sklearn.metrics import roc_curve, auc 

# Internal dependencies 
import halla 
from halla.distance import mi, l2, norm_mi, adj_mi 

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

	 try:
	 	iRow, iCol = pArray.shape 
	 	pPCA = PCA( n_components = iComponents )
		## doing this matrix inversion twice doesn't seem to be a good idea 
		return pPCA.fit_transform( pArray.T ).T 

	 except ValueError:
	 	iRow = pArray.shape
	 	iCol = None 

	 	return pArray

def kpca( pArray, iComponents = 1):
	from sklearn.decomposition import KernelPCA
	
	if pArray.ndim == 1:
		pArray = array([pArray])

	kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
	
	return kpca.fit_transform( pArray.T ).T 

def cca( pArray1, pArray2, iComponents = 1 ):
	"""
	Input N X D matrix 
	Output: D x N matrix 
	"""
	from sklearn.cross_decomposition import CCA

	pArray1, pArray2 = array(pArray1).T, array(pArray2).T

	pCCA = CCA( n_components = iComponents )
	pCCA.fit( pArray1, pArray2 )
	X_c, Y_c = pCCA.transform( pArray1, pArray2 )
	X_cout, Y_cout = X_c.T, Y_c.T
	
	#print X_cout 
	if X_cout.ndim > 1:
		X_cout = X_cout[0]
	if Y_cout.ndim > 1:
		Y_cout = Y_cout[0]
	#print X_cout 

	#return array(X_cout), array(Y_cout)
	return X_cout, Y_cout 

def cca_score( pArray1, pArray2, strMethod = "pearson", bPval = 1, bParam = False ):
	#from sklearn.cross_decomposition import CCA
	import scipy.stats 

	pArray1 = array(pArray1)
	pArray2 = array(pArray2)

	if pArray1.ndim == 1:
		pArray1 = array([pArray1])
	if pArray2.ndim == 1:
		pArray2 = array([pArray2])

	X_c, Y_c = cca( pArray1, pArray2, iComponents = 1 )
	
	if X_c.ndim > 1:
		X_c = list(X_c[0])
	if Y_c.ndim > 1:
		Y_c = list(Y_c[0])
	
	return scipy.stats.pearsonr( X_c, Y_c )[0]

def cca_score_norm_mi( pArray1, pArray2 ):
	import scipy.stats 

	pArray1 = array(pArray1)
	pArray2 = array(pArray2)

	if pArray1.ndim == 1:
		pArray1 = array([pArray1])
	if pArray2.ndim == 1:
		pArray2 = array([pArray2])

	X_c, Y_c = cca( pArray1, pArray2, iComponents = 1 )
	
	if X_c.ndim > 1:
		X_c = list(X_c[0])
	if Y_c.ndim > 1:
		Y_c = list(Y_c[0])
	
	X_cd = halla.stats.discretize( X_c )
	Y_cd = halla.stats.discretize( Y_c )

	return halla.distance.norm_mi( X_cd, Y_cd )
	#return scipy.stats.pearsonr( X_c, Y_c )[0]

def kernel_cca( ):
	pass

def kernel_cca_score( ):
	pass 

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

	mean_vec = numpy.mean(pArray, 0) 
	
	pArrayCenter = pArray - ( mean_vec * numpy.ones(pArray.shape) )

	return pArray[numpy.argsort( map( numpy.linalg.norm, pArrayCenter) )[0],:]


def get_representative( pArray, pMethod = None ):
	hash_method = {None: get_medoid, "pca": pca, }
	return hash_method[pMethod]( pArray )


#=========================================================
# Multiple comparison adjustment 
#=========================================================

def bh( afPVAL ):
	"""
	Implement the benjamini-hochberg hierarchical hypothesis testing criterion 
	In practice, used for implementing Yekutieli criterion *per layer*.  


	Parameters
	-------------

		afPVAL : list 
		 	list of p-values 


	Returns 
	--------------

		abOUT : list 
			boolean vector corresponding to which hypothesis test rejected, corresponding to p-value 


	Notes
	---------

		Reject up to highest i: max_i { p_(i) <= i*q/m }

		=> p_(i)*m/i <= q 

		Therefore, it is equivalent to setting the new p-value to the q-value := p_(i)*m/i 

		When BH is performed per layer, FDR is approximately 

		.. math::
			FDR = q \cdot \delta^{*} \cdot(m_0 + m_1)/(m_0+1)

		where :math:`m_0` is the observed number of discoveries and :math:`m_1` is the observed number of families tested. 

		Universal bound: the full tree FDR is :math:`< q \cdot \delta^{*} \cdot 2`

		At ties, update ranking as to remove redundancy from list 
 
	"""

	afPVAL_reduced = list(set(afPVAL)) ##duplicate elements removed 
	iLenReduced = len(afPVAL_reduced)

	pRank = scipy.stats.rankdata( afPVAL, method = "dense" ) ##the "dense" method ranks ties as if the list did not contain any redundancies 
	## source: http://docs.scipy.org/doc/scipy-dev/reference/generated/scipy.stats.rankdata.html

	aOut = [] 

	for i, fP in enumerate(afPVAL):
		fAdjusted = fP*1.0*pRank[i]/iLenReduced
		
		aOut.append(fAdjusted)

	assert( all(map(lambda x: x <= 1.0, aOut)) ) ##sanity check 

	return aOut 

def p_adjust( pval, method = "BH" ):
	"""
	
	Parameters
	--------------

		pval : list or float 

		method : str 
			{"bh","BH"}
	
	Returns 
	----------------

		afPVAL : list of float 

	"""

	try:
		pval[0]
	except (TypeError,IndexError):
		pval = [pval]

	return bh( pval ) 

#=========================================================
# Statistical test 
#=========================================================

def association_by_representative( pArray1, pArray2, metric = "norm_mi", decomposition = "pca", iIter = 100 ):
	"""
	Returns the inverse of the strength of the association (smaller scores for better association)
	"""

	pass 

## BUGBUG: why am I getting a p-value of this 1.0099009901? 

def permutation_test_by_representative( pArray1, pArray2, metric = "norm_mi", decomposition = "pca", iIter = 100 ):
	"""
	Input: 
	pArray1, pArray2, metric = "mi", decomposition = "pca", iIter = 100

	metric = {pca": pca} 
	"""

	#numpy.random.seed(0)

	strMetric = metric 
	pHashDecomposition = {"pca": pca, "kpca": kpca }
	pHashMetric = halla.distance.c_hash_metric 
	
	def _permutation( pVec ):
		return numpy.random.permutation( pVec )

	pDe = pHashDecomposition[decomposition]
	pMe = pHashMetric[strMetric] 

	## implicit assumption is that the arrays do not need to be discretized prior to input to the function
	pRep1, pRep2 = [ discretize( pDe( pA ) )[0] for pA in [pArray1,pArray2] ] if bool(halla.distance.c_hash_association_method_discretize[strMetric]) else [pDe( pA ) for pA in [pArray1, pArray2]]

	fAssociation = pMe( pRep1, pRep2 ) 

	aDist = numpy.array( [ pMe( _permutation( pRep1 ), pRep2 ) for _ in xrange( iIter ) ] )
	# WLOG, permute pArray1 instead of 2, or both. Can fix later with added theory. 
	## BUGBUG: currently this permutes the discretized values; we may be using information, but better than doing iIter iterations of PCA

	fPercentile = percentileofscore( aDist, fAssociation, kind="strict" ) ##source: Good 2000 
	### \frac{ \sharp\{\rho(\hat{X},Y) \geq \rho(X,Y) \} +1  }{ k + 1 }
	### k number of iterations, \hat{X} is randomized version of X 
	### PercentileofScore function ('strict') is essentially calculating the additive inverse (1-x) of the wanted quantity above 
	### consult scipy documentation at: http://docs.scipy.org/doc/scipy-0.7.x/reference/generated/scipy.stats.percentileofscore.html

	fP = ((1.0-fPercentile/100.0)*iIter + 1)/(iIter+1)

	assert(fP <= 1.0)

	return fP

def permutation_test_by_cca( pArray1, pArray2, metric = "norm_mi", iIter = 100 ):

	#numpy.random.seed(0)

	pArray1 = array(pArray1)
	pArray2 = array(pArray2)

	if pArray1.ndim == 1:
		pArray1 = array([pArray1])
	if pArray2.ndim == 1:
		pArray2 = array([pArray2])

	def _permutation( pVec ):
		return numpy.random.permutation( pVec )
	def _permute_matrix( X ):
		return array([numpy.random.permutation(x) for x in X])

	#pDe = pHashDecomposition[decomposition]
	#pMe = pHashMetric[strMetric] 

	## implicit assumption is that the arrays do not need to be discretized prior to input to the function
	#pRep1, pRep2 = [ discretize( pDe( pA ) )[0] for pA in [pArray1,pArray2] ] if bool(halla.distance.c_hash_association_method_discretize[strMetric]) else [pDe( pA ) for pA in [pArray1, pArray2]]

	X_c, Y_c = cca( pArray1, pArray2 )
	
	if metric == "norm_mi":
		fAssociation = cca_score_norm_mi( pArray1, pArray2 ) 
		pMetric = cca_score_norm_mi 
	elif metric == "pearson":
		fAssociation = cca_score( pArray1, pArray2 ) 
		pMetric = cca_score
	
	aDist = [pMetric( _permute_matrix(pArray1), pArray2 ) for _ in xrange(iIter)]
	#aDist = numpy.array([ halla.distance.norm_mi( _permutation( X_c ), Y_c ) for _ in xrange(iIter) ])
	#aDist = numpy.array( [ pMe( _permutation( pRep1 ), pRep2 ) for _ in xrange( iIter ) ] )
	# WLOG, permute pArray1 instead of 2, or both. Can fix later with added theory. 
	## BUGBUG: currently this permutes the discretized values; we may be using information, but better than doing iIter iterations of PCA

	fPercentile = percentileofscore( aDist, fAssociation, kind="strict" ) ##source: Good 2000 
	### \frac{ \sharp\{\rho(\hat{X},Y) \geq \rho(X,Y) \} +1  }{ k + 1 }
	### k number of iterations, \hat{X} is randomized version of X 
	### PercentileofScore function ('strict') is essentially calculating the additive inverse (1-x) of the wanted quantity above 
	### consult scipy documentation at: http://docs.scipy.org/doc/scipy-0.7.x/reference/generated/scipy.stats.percentileofscore.html

	fP = ((1.0-fPercentile/100.0)*iIter + 1)/(iIter+1)

	return fP

def permutation_test_by_copula( ):
	pass 

def permutation_test_by_average( pArray1, pArray2, metric = "norm_mid", iIter = 100 ):

	#numpy.random.seed(0)

	pHashDecomposition = {"pca": pca}
	
	pHashMetric = halla.distance.c_hash_metric

	def _permutation( pVec ):
		return numpy.random.permutation( pVec )

	strMetric = metric 
	pMe = pHashMetric[strMetric] 
	
	pFun = lambda x,y: numpy.average( [pMe(i,j) for i,j in itertools.product( x, y )] )

	dVal = pFun( pArray1, pArray2 )

	# WLOG, permute pArray1 instead of 2, or both. Can fix later with added theory. 
	pArrayPerm = numpy.array( [ pFun( array( [_permutation( x ) for x in pArray1] ), pArray2 ) for i in xrange( iIter ) ] )

	dPPerm = percentileofscore( pArrayPerm, dVal ) / 100 	

	return dPPerm


def parametric_test( pArray1, pArray2 ):
	
	#numpy.random.seed(0)

	pMe1 = lambda x,y: halla.distance.cor( x,y, method = "pearson", pval = True)
	pMe2 = lambda x,y: halla.distance.cor( x,y, method = "spearman", pval = True)

	pVal1 = [pMe1(i,j)[1] for i,j in itertools.product( pArray1, pArray2 )]
 	pVal2 = [pMe2(i,j)[1] for i,j in itertools.product( pArray1, pArray2 )]

	return numpy.average(pVal1), numpy.average(pVal2)


def parametric_test_by_cca( pArray1, pArray2, iIter = 100 ):
	
	#numpy.random.seed(0)

	pArray1 = array(pArray1)
	pArray2 = array(pArray2)

	if pArray1.ndim == 1:
		pArray1 = array([pArray1])
	if pArray2.ndim == 1:
		pArray2 = array([pArray2])

	def _permutation( pVec ):
		return numpy.random.permutation( pVec )
	def _permute_matrix( X ):
		return array([numpy.random.permutation(x) for x in X])

	X_c, Y_c = cca( pArray1, pArray2 )
	
	fAssociation = cca_score( pArray1, pArray2 ) 
	pMetric = cca_score

	fP = scipy.stats.pearsonr( X_c, Y_c )[1]

	return fP


def permutation_test_by_pca( pArray1, pArray2, iIter = 100 ):
	return permutation_test_by_representative( pArray1, pArray2, iIter = iIter )

def permutation_test_by_pca_norm_mi( pArray1, pArray2, iIter = 100 ):
	return permutation_test_by_representative( pArray1, pArray2, iIter = iIter )

def permutation_test_by_cca_pearson( pArray1, pArray2, iIter = 100 ):
	return permutation_test_by_cca( pArray1, pArray2, metric = "pearson", iIter = iIter )

def permutation_test_by_cca_norm_mi( pArray1, pArray2, iIter = 100 ):
	return permutation_test_by_cca( pArray1, pArray2, metric = "norm_mi", iIter = iIter )

def permutation_test_by_kpca_pearson( pArray1, pArray2, iIter = 100 ):
	return permutation_test_by_representative( pArray1, pArray2, decomposition = "kpca", metric = "pearson", iIter = iIter )

def permutation_test_by_kpca_norm_mi( pArray1, pArray2, iIter = 100 ):
	return permutation_test_by_representative( pArray1, pArray2, decomposition = "kpca", metric = "norm_mi", iIter = iIter )


#=========================================================
# Cake Cutting 
#=========================================================
"""
Think about the differences between pdf and cdf 
"""

def identity_cut( data_length, iCuts ):
	cake = range(data_length)
	return [[i] for i in cake]

def uniform_cut( pArray, iCuts, iAxis = 1):
	"""
	Uniform cuts of the data 

	Parameters
	-------------

	pArray : numpy array, array-like 
		Input array 
	iCuts : int 
		Number of cuts 
	iAxis : int 

	Returns 
	----------

	C : list  
		Divided array 

	Note
	------

	Code still observes sub-optimal behavior; fix. 

	"""

	def _uniform_cut( iData, iCuts ):
		pData = range( iData )
		if iCuts > iData:
			sys.stderr.write("Number of cuts exceed the length of the data\n")
			return [[i] for i in pData]
		else:		
			iMod = iData % iCuts 
			iStep = iData/iCuts 
			aOut = [pData[i*iStep:(i+1)*iStep] for i in range(iCuts)]
			pRemain = pData[iCuts*iStep:] 
			assert( iMod == len(pRemain) )
			for j,x in enumerate( pRemain ):
				aOut[j].append(x)
		return aOut 

	if not iCuts: 
		iCuts = math.floor( math.log( len(pArray), 2 ) )

	pArray = array( pArray )

	aIndex = map( array, _uniform_cut( len(pArray), iCuts = iCuts ) ) #Make sure each subset is an array so we can map with numpy 
	return [pArray[x] for x in aIndex]

def cumulative_uniform_cut( cake_length, iCuts ):
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

	return [ numpy.hstack( aTmp[:i] ) for i in range(1,iLength) ]

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

def step_function( ):
	pass 

### This is a very simple linear cutting method, with \sqrt{N} bins 
### To be tested with other estimators, like kernel density estimators for improved performance 
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
# Classification and Validation 
#=========================================================

def m( pArray, pFunc, axis = 0 ):
	""" 
	Maps pFunc over the array pArray 
	"""

	if bool(axis): 
		pArray = pArray.T
		# Set the axis as per numpy convention 
	if isinstance( pFunc , numpy.ndarray ):
		return pArray[pFunc]
	else: #generic function type
		return array( [pFunc(item) for item in pArray] ) 

#@staticmethod 
def mp( pArray, pFunc, axis = 0 ):
	"""
	Map _by pairs_ ; i.e. apply pFunc over all possible pairs in pArray 
	"""

	if bool(axis): 
		pArray = pArray.T

	pIndices = itertools.combinations( range(pArray.shape[0]), 2 )

	return array([pFunc(pArray[i],pArray[j]) for i,j in pIndices])

def md( pArray1, pArray2, pFunc, axis = 0 ):
	"""
	Map _by dot product_
	"""

	if bool(axis): 
		pArray1, pArray2 = pArray1.T, pArray2.T

	iRow1 = len(pArray1)
	iRow2 = len(pArray2)

	assert( iRow1 == iRow2 )
	aOut = [] 
	for i,item in enumerate(pArray1):
		aOut.append( pFunc(item, pArray2[i]) )
	return aOut 

#@staticmethod 
def mc( pArray1, pArray2, pFunc, axis = 0, bExpand = False ):
	"""
	Map _by cross product_ for ; i.e. apply pFunc over all possible pairs in pArray1 X pArray2 
	
	If not bExpand, gives a flattened array; else give full expanded array 
	"""

	if bool(axis): 
		pArray1, pArray2 = pArray1.T, pArray2.T

	#iRow1, iCol1 = pArray1.shape
	#iRow2, iCol2 = pArray2.shape 

	iRow1 = len(pArray1)
	iRow2 = len(pArray2)

	pIndices = itertools.product( range(iRow1), range(iRow2) )

	aOut = array([pFunc(pArray1[i],pArray2[j]) for i,j in pIndices])
	return ( aOut if not bExpand else numpy.reshape( aOut, (iRow1, iRow2) ) )

def threshold( self, pArray, fValue ):
	return m( pArray, lambda x: int(x <= fValue) )

def accuracy( true_labels, emp_labels ):
	assert( len(true_labels) == len(emp_labels) )
	iLen = len(true_labels)
	return sum( md( true_labels, emp_labels, lambda x,y: int(x==y) ) )*(1/float(iLen))

def accuracy_with_threshold( true_labels, prob_vec, fThreshold = 0.05 ):
	if not fThreshold:
		fThreshold = self.q 
	return accuracy( true_labels, threshold( prob_vec, fThreshold ) )


def bag2association( aaBag, A ):
	"""
	Parameters 
	---------------

		aaBag: list 
		A: array 


	Returns 
	----------

		A_conditional_flattened: list
		A_emp_conditional_flattened: list
		
	"""

	A_emp_conditional_flattened = [] 
	A_conditional_flattened = [] 

	##aBag is in order 
	for aBag in aaBag:
		aPair, fAssoc = aBag 
		i,j = aPair 
		A_emp_conditional_flattened.append( 1.0 - fAssoc )
		A_conditional_flattened.append( A[i][j] )

	assert( len(A_conditional_flattened ) == len(A_emp_conditional_flattened) )

	return A_conditional_flattened, A_emp_conditional_flattened

