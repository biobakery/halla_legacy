#!/usr/bin/env python 
"""
unified statistics module 
"""

# native python 
from itertools import compress
from itertools import product
import itertools
import math
from numpy import array 
import numpy 
from numpy.random import shuffle, binomial, normal, multinomial 
import scipy
import sys
from scipy.stats import scoreatpercentile, pearsonr, rankdata, percentileofscore, spearmanr

import sklearn 
from sklearn.metrics import roc_curve, auc 
from sklearn import manifold
from . import distance


# External dependencies 
# from scipy.stats import percentileofscore
# ML plug-in 
# Internal dependencies 
def fpr_tpr(condition=None, outcome=None):
	condition_negative = numpy.where(condition < .45, 1, 0).sum();  # FP + TN
	conditon_positive = numpy.where(condition >= .45, 1, 0).sum();  # TP + FN
	tp = 0
	fp = 0
	print(condition_negative, conditon_positive)
	for i in range(len(condition)):
		for j in range(len(condition)):
			if condition[i][j] <= .45 and outcome[i][j] == 1:
				fp = fp + 1
			if condition[i][j] >= .45 and outcome[i][j] == 1:
				tp = tp + 1
	print(fp , tp)
	if condition_negative == 0:
		return 0.0, 0.0
	fpr = float(fp) / condition_negative
	if conditon_positive == 0:
		return 0.0, 0.0
	tpr = float(tp) / conditon_positive
	
	return fpr , tpr
	

#=========================================================
# Feature Selection 
#=========================================================

def alpha_threshold(pArray, alpha=0.05, func=distance.nmi):
	"""
	*Within Covariance* estimation 
	Threshold association values in X and Y based on alpha cutoff. 
	This determines the line where the features are indistinguishable. 
	Uses normalized mutual information by default. 
	"""
	fPercentile = 100.0 * (1.0 - float(alpha))

	X = pArray
	XP = array([numpy.random.permutation(x) for x in X])
	D = discretize(XP)
	A = numpy.array([func(D[i], D[j]) for i, j in itertools.combinations(range(len(XP)), 2)])
	fScore = scoreatpercentile(A, fPercentile)
	# print "alpha_threshold: ", fScore
	# print "Distance Function:", str(func)
	return fScore 
def pca_explained_variance_ratio_(pArray, iComponents=1):
	
	"""
	 Input: N x D matrix 
	 Output: D x N matrix 

	 """
	from sklearn.decomposition import PCA

 	iRow, iCol = pArray.shape 
 	pPCA = PCA(n_components=iComponents)
	# # doing this matrix inversion twice doesn't seem to be a good idea 
	pPCA.fit(pArray.T)
	# PCA(copy=True, n_components=1, whiten=False)
	#print "PCA variance", pPCA.explained_variance_ratio_ 
	return pPCA.explained_variance_ratio_ 

	
def pca(pArray, iComponents=1):
	 """
	 Input: N x D matrix 
	 Output: D x N matrix 

	 """
	 from sklearn.decomposition import PCA
	 # print "pArray:", pArray
	 try:
	 	iRow, iCol = pArray.shape 
	 	pPCA = PCA(n_components=iComponents)
		# # doing this matrix inversion twice doesn't seem to be a good idea 
		# print"PCA:",   pPCA.fit_transform( pArray.T ).T 
		# print "End PCA"
		return pPCA.fit_transform(pArray.T).T

	 except ValueError:
	 	iRow = pArray.shape
	 	iCol = None 

	 	return pArray
def mds(pArray, iComponents=1):
	 """
	 Input: N x D matrix 
	 Output: D x N matrix 

	 """
	 from sklearn.decomposition import MDS
	 # print "pArray:", pArray
	 try:
	 	iRow, iCol = pArray.shape
	 	mds = manifold.MDS(n_components=iComponents, max_iter=3000, eps=1e-9, random_state=seed,
                   dissimilarity="precomputed", n_jobs=1)
		pos = mds.fit_transform(similarities)
		

		print "End MDS", pos.T
		return pos.T
		
	 except ValueError:
	 	iRow = pArray.shape
	 	iCol = None 

	 	return pArray
def first_rep(pArray, decomposition, iComponents=1 ):
	
	from sklearn.decomposition import PCA
	pPCA = PCA(n_components=iComponents)
	pPCA.fit(pArray.T)
	return pPCA.explained_variance_ratio_[0]

def kpca(pArray, iComponents=1):
	from sklearn.decomposition import KernelPCA
	
	if pArray.ndim == 1:
		pArray = array([pArray])

	kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
	
	return kpca.fit_transform(pArray.T).T 
def ica(pArray, iComponents=1):
	 """
	 Input: N x D matrix 
	 Output: D x N matrix 

	 """
	 from sklearn.decomposition import FastICA
	 
	 try:
	 	iRow, iCol = pArray.shape 
	 	pICA = FastICA(n_components=iComponents)
		# # doing this matrix inversion twice doesn't seem to be a good idea 
		return pICA.fit_transform(pArray.T).T 

	 except ValueError:
	 	iRow = pArray.shape
	 	iCol = None 

	 	return pArray

def cca(pArray1, pArray2, StrMetric, iComponents=1):
	"""
	Input N X D matrix 
	Output: D x N matrix 
	"""
	from sklearn.cross_decomposition import CCA

	pArray1, pArray2 = array(pArray1).T, array(pArray2).T

	pCCA = CCA(n_components=iComponents)
	pCCA.fit(pArray1, pArray2)
	X_c, Y_c = pCCA.transform(pArray1, pArray2)
	X_cout, Y_cout = X_c.T, Y_c.T
	
	# print X_cout 
	if X_cout.ndim > 1:
		X_cout = X_cout[0]
	if Y_cout.ndim > 1:
		Y_cout = Y_cout[0]
	# print X_cout 

	# return array(X_cout), array(Y_cout)
	return X_cout, Y_cout 


def pls(pArray1, pArray2, iComponents=1):

	import sklearn.cross_decomposition	
	X, Y = pArray1, pArray2 

	pls = sklearn.cross_decomposition.PLSRegression(n_components=1)

	pls.fit(X.T, Y.T)
	X_pls, Y_pls = pls.transform(X.T, Y.T)

	X_plsout, Y_plsout = X_pls.T, Y_pls.T
	
	# print X_cout 
	if X_plsout.ndim > 1:
		X_plsout = X_plsout[0]
	if Y_plsout.ndim > 1:
		Y_plsout = Y_plsout[0]
	#print X_plsout, Y_plsout 

	# return array(X_cout), array(Y_cout)
	return X_plsout, Y_plsout 

def similarity_score(X, Y, strMetric="nmi", bPval=1, bParam=False):
	# from sklearn.cross_decomposition import CCA
	#import scipy.stats 
	X_c = discretize (X)
	Y_c = discretize (Y)
	hash_metric = distance.c_hash_metric
	pMethd = hash_metric[strMetric]
	return pMethd(X_c, Y_c)


def plsc():
	pass 

def kernel_cca():
	pass

def kernel_cca_score():
	pass 

def get_medoid(pArray, iAxis=0, pMetric=distance.l2):
	"""
	Input: numpy array 
	Output: float
	
	For lack of better way, compute centroid, then compute medoid 
	by looking at an element that is closest to the centroid. 

	Can define arbitrary metric passed in as a function to pMetric 

	"""

	d = pMetric 

	pArray = (pArray.T if iAxis == 1 else pArray) 

	mean_vec = numpy.mean(pArray, 0) 
	
	pArrayCenter = pArray - (mean_vec * numpy.ones(pArray.shape))

	return pArray[numpy.argsort(map(numpy.linalg.norm, pArrayCenter))[0], :]


def get_representative(pArray, pMethod=None):
	hash_method = {None: get_medoid, "pca": pca, "ica": ica }
	return hash_method[pMethod](pArray)


#=========================================================
# Multiple comparison adjustment 
#=========================================================

def bh(afPVAL, q):
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

	# afPVAL_reduced = list(set(afPVAL)) ##duplicate elements removed 
	# iLenReduced = len(afPVAL_reduced)
	# pRank = scipy.stats.rankdata( afPVAL) ##the "dense" method ranks ties as if the list did not contain any redundancies 
	# # source: http://docs.scipy.org/doc/scipy-dev/reference/generated/scipy.stats.rankdata.html
	pRank = rankdata(afPVAL, method='ordinal')

	aOut = [] 
	iLen = len(afPVAL)
	for i, fP in enumerate(afPVAL):
		# fAdjusted = fP*1.0*pRank[i]/iLen#iLenReduced
		fAdjusted = q * 1.0 * pRank[i] / iLen  # iLenReduced
		aOut.append(fAdjusted)
	# print aOut
	# assert( all(map(lambda x: x <= 1.0, aOut)) ) ##sanity check 

	return aOut, pRank 

def p_adjust(pval, q, method="BH"):
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
	except (TypeError, IndexError):
		pval = [pval]

	return bh(pval, q) 

#=========================================================
# Statistical test 
#=========================================================

def association_by_representative(pArray1, pArray2, metric="nmi", decomposition="pca", iIter=1000):
	"""
	Returns the inverse of the strength of the association (smaller scores for better association)
	"""

	pass 

# # BUGBUG: why am I getting a p-value of this 1.0099009901? 

def parametric_test_by_representative(pArray1, pArray2):

	U1 = pca(pArray1, 1)
	U2 = pca(pArray2, 1)
	
	fAssoc, fP = pearsonr(U1[0], U2[0])

	return fP 
def parametric_test_by_representative_ica(pArray1, pArray2):

	U1 = ica(pArray1, 1)
	U2 = ica(pArray2, 1)
	
	fAssoc, fP = pearsonr(U1[0], U2[0])

	return fP

def permutation_test_by_medoid(pArray1, pArray2, metric="nmi", iIter=1000):
	"""
	Input: 
	pArray1, pArray2, metric = "mi", decomposition = "pca", iIter = 100

	metric = {pca": pca} 
	"""

	# numpy.random.seed(0)

	strMetric = metric 
	pHashDecomposition = {"pca": pca, "kpca": kpca, "ica":ica }
	pHashMetric = distance.c_hash_metric 
	
	def _permutation(pVec):
		return numpy.random.permutation(pVec)

	pDe = get_medoid
	pMe = pHashMetric[strMetric] 

	# # implicit assumption is that the arrays do not need to be discretized prior to input to the function
	
	pRep1 = get_medoid(discretize(pArray1), 0, pMe)
	pRep2 = get_medoid(discretize(pArray1), 0, pMe)

	# pRep1, pRep2 = [ discretize( pDe( pA ) )[0] for pA in [pArray1,pArray2] ] 

	fAssociation = pMe(pRep1, pRep2) 

	aDist = numpy.array([ pMe(_permutation(pRep1), pRep2) for _ in xrange(iIter) ])
	# WLOG, permute pArray1 instead of 2, or both. Can fix later with added theory. 
	# # BUGBUG: currently this permutes the discretized values; we may be using information, but better than doing iIter iterations of PCA

	fPercentile = percentileofscore(aDist, fAssociation, kind="strict")  # #source: Good 2000 
	# ## \frac{ \sharp\{\rho(\hat{X},Y) \geq \rho(X,Y) \} +1  }{ k + 1 }
	# ## k number of iterations, \hat{X} is randomized version of X 
	# ## PercentileofScore function ('strict') is essentially calculating the additive inverse (1-x) of the wanted quantity above 
	# ## consult scipy documentation at: http://docs.scipy.org/doc/scipy-0.7.x/reference/generated/scipy.stats.percentileofscore.html

	fP = ((1.0 - fPercentile / 100.0) * iIter + 1) / (iIter + 1)

	assert(fP <= 1.0)

	return fP
	
def permutation_test_by_representative(pArray1, pArray2, metric="nmi", decomposition="pca", iIter=1000):
	"""
	Input: 
	pArray1, pArray2, metric = "mi", decomposition = "pca", iIter = 1000

	metric = {pca": pca} 
	"""
	# numpy.random.seed(0)
	# return g_test_by_representative( pArray1, pArray2, metric = "nmi", decomposition = "pca", iIter = 1000 )
	X, Y = pArray1, pArray2 

	strMetric = metric 
	# step 5 in a case of new decomposition method
	pHashDecomposition = {'pls':pls, 'cca':cca, "pca": pca, "kpca": kpca, "ica":ica, "mds":mds }
	pHashMetric = distance.c_hash_metric 
	
	def _permutation(pVec):
		return numpy.random.permutation(pVec)

	pDe = pHashDecomposition[decomposition]
	pMe = pHashMetric[strMetric] 
	# # implicit assumption is that the arrays do not need to be discretized prior to input to the function
	
	aDist = [] 
	left_rep = 1.0
	right_rep = 1.0
	#### Calculate Point estimate 
	if decomposition in ['pls', 'cca']:
		pRep1, pRep2 = pDe(pArray1, pArray2, metric)
	else:
		pRep1, pRep2 = [ discretize(pDe(pA))[0] for pA in [pArray1, pArray2] ] if bool(distance.c_hash_association_method_discretize[strMetric]) else [pDe(pA) for pA in [pArray1, pArray2]]
		left_rep = first_rep(pArray1, decomposition)
		right_rep = first_rep(pArray2, decomposition)
	fAssociation = pMe(pRep1, pRep2) 
	# print left_rep, right_rep, fAssociation
	#### Perform Permutation 
	for _ in xrange(iIter):

		#XP = array([numpy.random.permutation(x) for x in X])
		#YP = array([numpy.random.permutation(y) for y in Y])

		#pRep1_, pRep2_ = [ discretize(pDe(pA))[0] for pA in [XP, YP] ] if bool(distance.c_hash_association_method_discretize[strMetric]) else [pDe(pA) for pA in [pArray1, pArray2]]
		permuted_pRep2 = numpy.random.permutation(pRep2)
		# Similarity score between representatives  
		#fAssociation_ = pMe(pRep1_, pRep2_)  # NMI
		fAssociation_ = pMe(pRep1, permuted_pRep2)  # NMI
		aDist.append(fAssociation_)

		# aDist = numpy.array( [ pMe( _permutation( pRep1 ), pRep2 ) for _ in xrange( iIter ) ] )
	
	fPercentile = percentileofscore(aDist, fAssociation, kind="mean")  # #source: Good 2000  
	# ## \frac{ \sharp\{\rho(\hat{X},Y) \geq \rho(X,Y) \} +1  }{ k + 1 }
	# ## k number of iterations, \hat{X} is randomized version of X 
	# ## PercentileofScore function ('strict') is essentially calculating the additive inverse (1-x) of the wanted quantity above 
	# ## consult scipy documentation at: http://docs.scipy.org/doc/scipy-0.7.x/reference/generated/scipy.stats.percentileofscore.html

	fP = ((1.0 - fPercentile / 100.0) * iIter + 1) / (iIter + 1)
	
	assert(fP <= 1.0)
	# print fP
	return fP, fAssociation, left_rep, right_rep

def g_test_by_representative(pArray1, pArray2, metric="nmi", decomposition="pca", iIter=1000):
	"""
	Input: 
	pArray1, pArray2, metric = "mi", decomposition = "pca", iIter = 1000

	metric = {pca": pca} 
	"""
	X, Y = pArray1, pArray2 
	
	strMetric = metric 
	# step 5 in a case of new decomposition method
	pHashDecomposition = {"pca": pca, "kpca": kpca, "ica":ica }
	pHashMetric = distance.c_hash_metric 
	
	pDe = pHashDecomposition[decomposition]
	pMe = pHashMetric[strMetric] 
	# # implicit assumption is that the arrays do not need to be discretized prior to input to the function
	
	#### Caclulate Point estimate 
	pRep1, pRep2 = [ discretize(pDe(pA))[0] for pA in [pArray1, pArray2] ] if bool(distance.c_hash_association_method_discretize[strMetric]) else [pDe(pA) for pA in [pArray1, pArray2]]
	print "pRep1:", pRep1
	print "pRep2:", pRep2
	left_rep = first_rep(pArray1)
	right_rep = first_rep(pArray2)
	fAssociation = pMe(pRep1, pRep2) 
	
	filtered_pRep1 = []
	filtered_pRep2 = []
	for i in range(len(pRep1)):
		if pRep1[i] or pRep2[i]:
			filtered_pRep1.append(pRep1[i])
			filtered_pRep2.append(pRep2[i])
	# compute degrees of freedom for table
	xdf = len(set(filtered_pRep1)) - 1
	ydf = len(set(filtered_pRep2)) - 1
	df = xdf * ydf
	print "degrees of freedom... =", df
	n = len (filtered_pRep1)
	import os, sys, re, glob, argparse
	from random import choice, random, shuffle
	from collections import Counter
	from scipy.stats import chi2
	from math import log, exp, sqrt
	mi = distance.NormalizedMutualInformation(pRep1, pRep2).get_distance()
	# mutual information
	def _mutinfo( x, y ):
	    px, py, pxy = [Counter() for i in range( 3 )]
	    assert len( x ) == len( y ), "unequal lengths"
	    delta = 1 / float( len( x ) )
	    for xchar, ychar in zip( x, y ):
	        px[xchar] += delta 
	        py[ychar] += delta
	        pxy[( xchar, ychar )] += delta
	    S = 0
	    for ( xchar, ychar ), value in pxy.items():
	        S += value * ( log( value, 2 ) - log( px[xchar], 2 ) - log( py[ychar], 2 ) )
	    return S
	
	
	# actual value
	#mi = _mutinfo( filtered_pRep1, filtered_pRep1 )
	print "nmi", mi
	mi_natural_log = mi / log(exp(1.0), 2.0)
	print "nmi_natural_log"
	fP = 1 - chi2.cdf(2.0 * n * mi_natural_log, df)
	print "G-test P-value: ", fP
				
	# g, fP, dof, expctd = scipy.stats.chi2_contingency([filtered_pRep1, filtered_pRep2], lambda_="log-likelihood")
	
	# print "G-test, Pvalue:", fP 
	return fP, fAssociation, left_rep, right_rep
def parametric_test_by_max_pca(pArray1, pArray2, k=2, metric="spearman", iIter=1000):

	aOut = [] 

	iMinDim = min(pArray1.ndim, pArray2.ndim)
	
	if iMinDim < k:
		k = iMinDim 

	pRep1 = pca(pArray1, k)
	pRep2 = pca(pArray2, k)

	assert(len(pRep1) == k)
	assert(len(pRep2) == k)

	for x, y in itertools.product(pRep1, pRep2):
		
		aOut.append(spearmanr(x, y)[1])

	return aOut 

def permutation_test_by_max_pca(pArray1, pArray2, k=2, metric="nmi", iIter=1000):

	aOut = [] 

	iMinDim = min(pArray1.ndim, pArray2.ndim)
	
	if iMinDim < k:
		k = iMinDim 

	strMetric = metric 
	
	pHashMetric = distance.c_hash_metric 
	
	def _permutation(pVec):
		return numpy.random.permutation(pVec)

	pMe = pHashMetric[strMetric] 

	pRep1 = pca(pArray1, k)
	pRep2 = pca(pArray2, k)

	assert(len(pRep1) == k)
	assert(len(pRep2) == k)

	for x, y in itertools.product(pRep1, pRep2):
		pOne = discretize(x) if "mi" in metric else x
		pTwo = discretize(y) if "mi" in metric else y 

		fAssociation = pMe(pOne, pTwo) 

		aDist = numpy.array([ pMe(_permutation(pOne), pTwo) for _ in xrange(iIter) ])
		# WLOG, permute pArray1 instead of 2, or both. Can fix later with added theory. 
		# # BUGBUG: currently this permutes the discretized values; we may be using information, but better than doing iIter iterations of PCA

		fPercentile = percentileofscore(aDist, fAssociation, kind="strict")  # #source: Good 2000 
		# ## \frac{ \sharp\{\rho(\hat{X},Y) \geq \rho(X,Y) \} +1  }{ k + 1 }
		# ## k number of iterations, \hat{X} is randomized version of X 
		# ## PercentileofScore function ('strict') is essentially calculating the additive inverse (1-x) of the wanted quantity above 
		# ## consult scipy documentation at: http://docs.scipy.org/doc/scipy-0.7.x/reference/generated/scipy.stats.percentileofscore.html

		fP = ((1.0 - fPercentile / 100.0) * iIter + 1) / (iIter + 1)

		assert(fP <= 1.0)

		aOut.append(fP)

	return aOut 	

def permutation_test_by_multiple_representative(pArray1, pArray2, k=2, metric="nmi", iIter=1000):

	return min(permutation_test_by_max_pca(pArray1, pArray2, k=k, metric=metric, iIter=iIter))

def parametric_test_by_multiple_representative(pArray1, pArray2, k=2, metric="spearman"):

	return min(parametric_test_by_max_pca(pArray1, pArray2, k=k, metric=metric))

def permutation_test_by_cca(pArray1, pArray2, metric="nmi", iIter=1000):

	# numpy.random.seed(0)

	strMetric = metric 
	pHashDecomposition = {"pca": pca, "kpca": kpca, "ica":ica }
	pHashMetric = distance.c_hash_metric 
	
	# # implicit assumption is that the arrays do not need to be discretized prior to input to the function	

	pArray1 = array(pArray1)
	pArray2 = array(pArray2)

	X, Y = pArray1, pArray2 

	if pArray1.ndim == 1:
		pArray1 = array([pArray1])
	if pArray2.ndim == 1:
		pArray2 = array([pArray2])

	def _permutation(pVec):
		return numpy.random.permutation(pVec)
	def _permute_matrix(X):
		return array([numpy.random.permutation(x) for x in X])

	# pDe = pHashDecomposition[decomposition]
	pMe = pHashMetric[strMetric] 

	# # implicit assumption is that the arrays do not need to be discretized prior to input to the function
	# pRep1, pRep2 = [ discretize( pDe( pA ) )[0] for pA in [pArray1,pArray2] ] if bool(py.distance.c_hash_association_method_discretize[strMetric]) else [pDe( pA ) for pA in [pArray1, pArray2]]

	#### Calculate Point Estimate 
	pRep1, pRep2 = cca(pArray1, pArray2)
	
	aDist = [] 
	fAssociation = similarity_score(pRep1, pRep2, strMetric= strMetric)

	#### Perform Permutaiton 
	for _ in xrange(iIter):

		XP = array([numpy.random.permutation(x) for x in X])
		YP = array([numpy.random.permutation(y) for y in Y])

		# pRep1_, pRep2_ = [ discretize( pDe( pA ) )[0] for pA in [XP,YP] ] if bool(py.distance.c_hash_association_method_discretize[strMetric]) else [pDe( pA ) for pA in [pArray1, pArray2]]

		pRep1_, pRep2_ = cca(XP, YP)

		pRep1_, pRep2_ = discretize(pRep1_), discretize(pRep2_)

		fAssociation_ = pMe(pRep1_, pRep2_) 

		aDist.append(fAssociation_)

	# if metric == "nmi":
	# 	fAssociation = cca_score_nmi( pArray1, pArray2 ) 
	# 	pMetric = cca_score_nmi 
	# elif metric == "pearson":
	# 	fAssociation = cca_score( pArray1, pArray2 ) 
	# 	pMetric = cca_score
	
	# aDist = [pMetric( _permute_matrix(pArray1), pArray2 ) for _ in xrange(iIter)]
	
	# aDist = numpy.array([ py.distance.nmi( _permutation( X_c ), Y_c ) for _ in xrange(iIter) ])
	# aDist = numpy.array( [ pMe( _permutation( pRep1 ), pRep2 ) for _ in xrange( iIter ) ] )
	# WLOG, permute pArray1 instead of 2, or both. Can fix later with added theory. 
	# # BUGBUG: currently this permutes the discretized values; we may be using information, but better than doing iIter iterations of PCA

	fPercentile = percentileofscore(aDist, fAssociation, kind="strict")  # #source: Good 2000 
	# ## \frac{ \sharp\{\rho(\hat{X},Y) \geq \rho(X,Y) \} +1  }{ k + 1 }
	# ## k number of iterations, \hat{X} is randomized version of X 
	# ## PercentileofScore function ('strict') is essentially calculating the additive inverse (1-x) of the wanted quantity above 
	# ## consult scipy documentation at: http://docs.scipy.org/doc/scipy-0.7.x/reference/generated/scipy.stats.percentileofscore.html

	fP = ((1.0 - fPercentile / 100.0) * iIter + 1) / (iIter + 1)

	return fP, fAssociation, pRep1[0], pRep2[0]

def permutation_test_by_pls(pArray1, pArray2, metric="nmi", iIter=1000):

	# numpy.random.seed(0)

	strMetric = metric 
	pHashMetric = distance.c_hash_metric 
	
	# # implicit assumption is that the arrays do not need to be discretized prior to input to the function	

	pArray1 = array(pArray1)
	pArray2 = array(pArray2)

	X, Y = pArray1, pArray2 

	if pArray1.ndim == 1:
		pArray1 = array([pArray1])
	if pArray2.ndim == 1:
		pArray2 = array([pArray2])

	def _permutation(pVec):
		return numpy.random.permutation(pVec)
	def _permute_matrix(X):
		return array([numpy.random.permutation(x) for x in X])

	# pDe = pHashDecomposition[decomposition]
	pMe = pHashMetric[strMetric] 

	# # implicit assumption is that the arrays do not need to be discretized prior to input to the function
	# pRep1, pRep2 = [ discretize( pDe( pA ) )[0] for pA in [pArray1,pArray2] ] if bool(py.distance.c_hash_association_method_discretize[strMetric]) else [pDe( pA ) for pA in [pArray1, pArray2]]

	#### Calculate Point Estimate 
	pRep1, pRep2 = pls(pArray1, pArray2, metric)
	
	aDist = [] 
	fAssociation = similarity_score(pRep1, pRep2, strMetric = metric)

	#### Perform Permutaiton 
	for _ in xrange(iIter):

		XP = array([numpy.random.permutation(x) for x in X])
		YP = array([numpy.random.permutation(y) for y in Y])

		# pRep1_, pRep2_ = [ discretize( pDe( pA ) )[0] for pA in [XP,YP] ] if bool(py.distance.c_hash_association_method_discretize[strMetric]) else [pDe( pA ) for pA in [pArray1, pArray2]]

		pRep1_, pRep2_ = pls(XP, YP)

		pRep1_, pRep2_ = discretize(pRep1_), discretize(pRep2_)

		fAssociation_ = pMe(pRep1_, pRep2_) 

		aDist.append(fAssociation_)

	# if metric == "nmi":
	# 	fAssociation = cca_score_nmi( pArray1, pArray2 ) 
	# 	pMetric = cca_score_nmi 
	# elif metric == "pearson":
	# 	fAssociation = cca_score( pArray1, pArray2 ) 
	# 	pMetric = cca_score
	
	# aDist = [pMetric( _permute_matrix(pArray1), pArray2 ) for _ in xrange(iIter)]
	
	# aDist = numpy.array([ py.distance.nmi( _permutation( X_c ), Y_c ) for _ in xrange(iIter) ])
	# aDist = numpy.array( [ pMe( _permutation( pRep1 ), pRep2 ) for _ in xrange( iIter ) ] )
	# WLOG, permute pArray1 instead of 2, or both. Can fix later with added theory. 
	# # BUGBUG: currently this permutes the discretized values; we may be using information, but better than doing iIter iterations of PCA

	fPercentile = percentileofscore(aDist, fAssociation, kind="strict")  # #source: Good 2000 
	# ## \frac{ \sharp\{\rho(\hat{X},Y) \geq \rho(X,Y) \} +1  }{ k + 1 }
	# ## k number of iterations, \hat{X} is randomized version of X 
	# ## PercentileofScore function ('strict') is essentially calculating the additive inverse (1-x) of the wanted quantity above 
	# ## consult scipy documentation at: http://docs.scipy.org/doc/scipy-0.7.x/reference/generated/scipy.stats.percentileofscore.html

	fP = ((1.0 - fPercentile / 100.0) * iIter + 1) / (iIter + 1)

	return fP, fAssociation, pRep1[0], pRep2[0]

def permutation_test_by_copula():
	pass 

def permutation_test_by_average(pArray1, pArray2, metric= "nmi", iIter=1000):

	# numpy.random.seed(0)

	pHashDecomposition = {"pca": pca, "ica":ica}
	
	pHashMetric = distance.c_hash_metric

	def _permutation(pVec):
		return numpy.random.permutation(pVec)

	strMetric = metric 
	pMe = pHashMetric[strMetric] 
	
	pFun = lambda x, y: numpy.average([pMe(i, j) for i, j in itertools.product(x, y)])

	dVal = pFun(pArray1, pArray2)

	# WLOG, permute pArray1 instead of 2, or both. Can fix later with added theory. 
	pArrayPerm = numpy.array([ pFun(array([_permutation(x) for x in pArray1]), pArray2) for i in xrange(iIter) ])

	dPPerm = percentileofscore(pArrayPerm, dVal) / 100.0 	

	return dPPerm

def permutation_test(pArray1, pArray2, metric, decomposition, iIter):
	if decomposition in ['cca', 'pls',"pca", "ica", "kpca"]:
		return permutation_test_by_representative(pArray1, pArray2, metric=metric, decomposition= decomposition, iIter=iIter)
	
	if decomposition in ["average"]:
		return permutation_test_by_average(pArray1, pArray2, metric=metric, iIter=iIter)


def g_test(pArray1, pArray2, metric, decomposition, iIter):
	if decomposition in ["pca", "ica", "kpca"]:
		return g_test_by_representative(pArray1, pArray2, metric=metric, decomposition= decomposition, iIter=iIter)
	'''
	if decomposition in ["average"]:
		return g_test_by_average(pArray1, pArray2, metric=metric, iIter=iIter)
	
	if decomposition in ["cca"]:
		return g_test_by_cca(pArray1, pArray2, metric=metric, iIter=iIter)
	'''
def parametric_test(pArray1, pArray2):
	
	# numpy.random.seed(0)

	pMe1 = lambda x, y:  cor(x, y, method="pearson", pval=True)
	pMe2 = lambda x, y:  cor(x, y, method="spearman", pval=True)

	pVal1 = [pMe1(i, j)[1] for i, j in itertools.product(pArray1, pArray2)]
 	pVal2 = [pMe2(i, j)[1] for i, j in itertools.product(pArray1, pArray2)]

	return numpy.average(pVal1), numpy.average(pVal2)


def parametric_test_by_cca(pArray1, pArray2, iIter=1000):
	
	# numpy.random.seed(0)

	pArray1 = array(pArray1)
	pArray2 = array(pArray2)

	if pArray1.ndim == 1:
		pArray1 = array([pArray1])
	if pArray2.ndim == 1:
		pArray2 = array([pArray2])

	def _permutation(pVec):
		return numpy.random.permutation(pVec)
	def _permute_matrix(X):
		return array([numpy.random.permutation(x) for x in X])

	X_c, Y_c = cca(pArray1, pArray2)
	
	fAssociation = cca_score(pArray1, pArray2) 
	pMetric = cca_score

	fP = pearsonr(X_c, Y_c)[1]

	return fP

def parametric_test_by_pls_pearson(pArray1, pArray2, iIter=1000):
	
	# numpy.random.seed(0)

	pArray1 = array(pArray1)
	pArray2 = array(pArray2)

	if pArray1.ndim == 1:
		pArray1 = array([pArray1])
	if pArray2.ndim == 1:
		pArray2 = array([pArray2])

	def _permutation(pVec):
		return numpy.random.permutation(pVec)
	def _permute_matrix(X):
		return array([numpy.random.permutation(x) for x in X])

	X_pls, Y_pls = pls(pArray1, pArray2)
	
	fP = pearsonr(X_pls, Y_pls)[1]

	return fP

#=========================================================
# Cake Cutting 
#=========================================================
"""
Think about the differences between pdf and cdf 
"""

def identity_cut(data_length, iCuts):
	cake = range(data_length)
	return [[i] for i in cake]

def uniform_cut(pArray, iCuts, iAxis=1):
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

	def _uniform_cut(iData, iCuts):
		pData = range(iData)
		if iCuts > iData:
			sys.stderr.write("Number of cuts exceed the length of the data\n")
			return [[i] for i in pData]
		else:		
			iMod = iData % iCuts 
			iStep = iData / iCuts 
			aOut = [pData[i * iStep:(i + 1) * iStep] for i in range(iCuts)]
			pRemain = pData[iCuts * iStep:] 
			assert(iMod == len(pRemain))
			for j, x in enumerate(pRemain):
				aOut[j].append(x)
		return aOut 

	if not iCuts: 
		iCuts = math.floor(math.log(len(pArray), 2))

	pArray = array(pArray)

	aIndex = map(array, _uniform_cut(len(pArray), iCuts=iCuts))  # Make sure each subset is an array so we can map with numpy 
	return [pArray[x] for x in aIndex]

def cumulative_uniform_cut(cake_length, iCuts):
	assert(cake_length > iCuts)

	aOut = [] 

	iSize = int(math.floor(float(cake_length) / iCuts))

	for iStep in range(1, iSize + 1):
		if iStep != iSize:
			aOut.append(range(cake_length)[:iStep * iCuts])
		else:
			aOut.append(range(cake_length)[:])

	return aOut 

def log_cut(cake_length, iBase=2):
	"""
	Input: cake_length <- length of array, iBase <- base of logarithm 

	Output: array of indices corresponding to the slice 

	Note: Probably don't want size-1 cake slices, but for proof-of-concept, this should be okay. 
	Avoid the "all" case 

	"""

	aOut = [] 

	iLength = cake_length 

	iSize = int(math.floor(math.log(iLength , iBase)))
	aSize = [2 ** i for i in range(iSize)] 

	iStart = 0 
	for item in aSize:
		iStop = iStart + item 
		if iStop == iLength - 1:
			iStop += 1 
			# ensure that the rest of the cake gets included in the tail case  
		aOut.append(array(range(iStart, iStop))) 
		iStart = iStop 

	aOut.reverse()  # bigger ones first 
	return aOut 

def cumulative_log_cut(cake_length, iBase=2):
	"""
	Input: cake_length <- length of array, iBase <- base of logarithm 

	Output: array of indices corresponding to the slice 

	Note: Probably don't want size-1 cake slices, but for proof-of-concept, this should be okay. 
	Avoid the "all" case 

	"""

	aOut = [] 

	iLength = cake_length 

	iSize = int(math.floor(math.log(iLength , iBase)))
	aSize = [2 ** i for i in range(iSize + 1)] 

	aOut = [ range(cake_length)[:x] for x in aSize]
	aOut.reverse()
	return map(array, aOut)

def tables_to_probability(aaTable):
	if not aaTable:
		raise Exception("Empty table.")
	
	aOut = [] 
	iN = sum([len(x) for x in aaTable])
	# iN = reduce( lambda x,y: len(x)+y, aaTable, 0 ) #number of elements 

	for aTable in aaTable:
		iB = len(aTable)
		aOut.append(float(iB) / (iN + 1))

	# probability of creating new table 
	aOut.append(1.0 / (iN + 1))

	return aOut 

def CRP_cut(cake_length):
	
	iLength = cake_length 

	aOut = None 

	for i in range(iLength):
		if not aOut:
			aOut = [[i]] 
			continue 
		else:
			pBool = multinomial(1, tables_to_probability(aOut)) 
		
			iK = [x for x in compress(range(len(pBool)), pBool)][0]
			if iK + 1 > len(aOut):  # create new table 
				aOut.append([i])
			else:
				aOut[iK].append(i)  # seat the customer on an existing table 

	return map(array, aOut)  # return as numpy array, so we can vectorize 

def cumulative_CRP_cut(cake_length):
	aTmp = sorted(CRP_cut(cake_length), key=lambda x:-1 * len(x))
	iLength = len(aTmp)

	return [ numpy.hstack(aTmp[:i]) for i in range(1, iLength) ]

def PY_cut(cake_length):
	""" 
	random cut generated by pitman-yor process prior
	"""

	pass

def IBP_cut(cake_length):
	"""
	random cut generated by Indian Buffet Process prior
	"""

	pass 
	

def p_val_plot(pArray1, pArray2, pCut=log_cut, iIter=1000):
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

		len1, len2 = len(D1), len(D2)
		cut1, cut2 = sorted(pCut(len1), key=lambda x:-1.0 * len(x)), sorted(pCut(len2), key=lambda x:-1.0 * len(x))
		lencut1, lencut2 = len(cut1), len(cut2)
		iMin = min(lencut1, lencut2)
		if not aOut:
			aOut = [[] for _ in range(iMin)] 

		print "cut1"
		print cut1
		print "cut2"
		print cut2

		for j in range(iMin):
			dP = permutation_test_by_representative(pArray1[cut1[j]], pArray2[cut2[j]])

			print "pval"
			print dP 

			# be careful when using CRP/PYP, don't know the cluster size in advance 
			try: 
				aOut[j].append(dP)
			except IndexError: 
				aOut += [[] for _ in range(j - len(aOut) + 1)]
				aOut[j].append(dP)

	return aOut 


#=========================================================
# Density estimation 
#=========================================================

def step_function():
	pass 

# ## This is a very simple linear cutting method, with \sqrt{N} bins 
# ## To be tested with other estimators, like kernel density estimators for improved performance 
def discretize(pArray, iN=None, method=None, aiSkip=[]):
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


	def _discretize_continuous(astrValues, iN=iN): 
		
		if iN == None:
			# Default to rounded sqrt(n) if no bin count requested
			iN = round(math.sqrt(len(astrValues)))  # **0.5 + 0.5)
		elif iN == 0:
			iN = len(astrValues)
		else:
			iN = min(iN, len(set(astrValues)))
		#print iN	
		# This is still a bit buggy since ( [0, 0, 0, 1, 2, 2, 2, 2], 3 ) will exhibit suboptimal behavior
		aiIndices = sorted(range(len(astrValues)), cmp=lambda i, j: cmp(astrValues[i], astrValues[j]))
		astrRet = [None] * len(astrValues)
		iPrev = 0
		for i, iIndex in enumerate(aiIndices):
			# If you're on a tie, you can't increase the bin number
			# Otherwise, increase by at most one
			iPrev = astrRet[iIndex] = iPrev if (i and (astrValues[iIndex] == astrValues[aiIndices[i - 1]])) else \
				min(iPrev + 1, int(iN * i / float(len(astrValues))))
		
		return astrRet

	try:
		# iRow1, iCol = pArray.shape 

		aOut = [] 
		# iN= len(pArray)
		# print iN
		for i, line in enumerate(pArray):
			if i in aiSkip:
				aOut.append(line)
			else:
				aOut.append(_discretize_continuous(line))

		return array(aOut)

	except Exception:
		iN = len(pArray)
		return _discretize_continuous(pArray)

def discretize2d(pX, pY, method=None):
	pass 


#=========================================================
# Classification and Validation 
#=========================================================

def m(pArray, pFunc, axis=0):
	""" 
	Maps pFunc over the array pArray 
	"""

	if bool(axis): 
		pArray = pArray.T
		# Set the axis as per numpy convention 
	if isinstance(pFunc , numpy.ndarray):
		return pArray[pFunc]
	else:  # generic function type
		return array([pFunc(item) for item in pArray]) 

# @staticmethod 
def mp(pArray, pFunc, axis=0):
	"""
	Map _by pairs_ ; i.e. apply pFunc over all possible pairs in pArray 
	"""

	if bool(axis): 
		pArray = pArray.T

	pIndices = itertools.combinations(range(pArray.shape[0]), 2)

	return array([pFunc(pArray[i], pArray[j]) for i, j in pIndices])

def md(pArray1, pArray2, pFunc, axis=0):
	"""
	Map _by dot product_
	"""

	if bool(axis): 
		pArray1, pArray2 = pArray1.T, pArray2.T

	iRow1 = len(pArray1)
	iRow2 = len(pArray2)

	assert(iRow1 == iRow2)
	aOut = [] 
	for i, item in enumerate(pArray1):
		aOut.append(pFunc(item, pArray2[i]))
	return aOut 

# @staticmethod 
def mc(pArray1, pArray2, pFunc, axis=0, bExpand=False):
	"""
	Map _by cross product_ for ; i.e. apply pFunc over all possible pairs in pArray1 X pArray2 
	
	If not bExpand, gives a flattened array; else give full expanded array 
	"""

	if bool(axis): 
		pArray1, pArray2 = pArray1.T, pArray2.T

	# iRow1, iCol1 = pArray1.shape
	# iRow2, iCol2 = pArray2.shape 

	iRow1 = len(pArray1)
	iRow2 = len(pArray2)

	pIndices = itertools.product(range(iRow1), range(iRow2))

	aOut = array([pFunc(pArray1[i], pArray2[j]) for i, j in pIndices])
	return (aOut if not bExpand else numpy.reshape(aOut, (iRow1, iRow2)))

def threshold(pArray, fValue):
	return m(pArray, lambda x: int(x <= fValue))

def accuracy(true_labels, emp_labels):
	assert(len(true_labels) == len(emp_labels))
	iLen = len(true_labels)
	return sum(md(true_labels, emp_labels, lambda x, y: int(x == y))) * (1 / float(iLen))




def bag2association(aaBag, A):
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

	# #aBag is in order 
	for aBag in aaBag:
		aPair, fAssoc = aBag 
		i, j = aPair 
		A_emp_conditional_flattened.append(1.0 - fAssoc)
		A_conditional_flattened.append(A[i][j])

	assert(len(A_conditional_flattened) == len(A_emp_conditional_flattened))

	return A_conditional_flattened, A_emp_conditional_flattened

c_hash_decomposition = {"pca"    : pca,
                        "ica"    : ica,
                        "cca"	 : cca,
                        "pls"	 : pls,
                        "average": None }
