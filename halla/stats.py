#!/usr/bin/env python 
"""
unified statistics module 
"""

# native python 
import exceptions 
from exceptions import ArithmeticError, ValueError 	
from itertools import compress
from itertools import product
import itertools
import math
from numpy import array , std, log2
import numpy 
from numpy.random import shuffle, binomial, normal, multinomial 
import scipy
import sys
import random
from scipy.stats import scoreatpercentile, pearsonr, rankdata, percentileofscore, spearmanr

import sklearn 
from sklearn.metrics import roc_curve, auc 
from sklearn import manifold
from . import distance, config
from scipy.spatial.distance import pdist, squareform
import pandas as pd
from sklearn.metrics import explained_variance_score
'''try:
	from mca import mca
except ImportError:
	sys.exit("Please install mca properly")
'''
# External dependencies 
# from scipy.stats import percentileofscore
# ML plug-in 
# Internal dependencies
def get_enropy(x):
	#print x
	d = x-1
	d = [float(val) for val in d]
	#print d
	P = numpy.bincount(d)/float(len(d))
	observed_entropy = -sum([p * numpy.log2(p) for p in P])
	max_entropy = numpy.log2(len(P))
	return observed_entropy/max_entropy 
def pvalues2qvalues ( pvalues, adjusted=False ):
    n = len( pvalues )
    # after sorting, index[i] is the original index of the ith-ranked value
    index = range( n )
    index = sorted( index, key=lambda i: pvalues[i] )
    pvalues = sorted( pvalues )
    qvalues = [pvalues[i-1] * n / i for i in range( 1, n+1 )]
    # adjust qvalues to enforce monotonic behavior?
    # q( i ) = min( q( i..n ) )
    if adjusted:
        qvalues.reverse()
        for i in range( 1, n ):
            if qvalues[i] > qvalues[i-1]:
                qvalues[i] = qvalues[i-1]
        qvalues.reverse()
    # rebuild qvalues in the original order
    ordered_qvalues = [None for q in qvalues]
    for i, q in enumerate( qvalues ):
        ordered_qvalues[index[i]] = q
    return ordered_qvalues 
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

 	#iRow, iCol = pArray.shape 
 	pPCA = PCA(n_components=iComponents)
	# # doing this matrix inversion twice doesn't seem to be a good idea 
	pPCA.fit(pArray.T)
	# PCA(copy=True, n_components=1, whiten=False)
	#print "PCA variance", pPCA.explained_variance_ratio_ 
	return pPCA.explained_variance_ratio_ 

	
def mca_method(pArray, discretize_style, iComponents=1):
	"""
	Input: N x D matrix 
	Output: D x N matrix 
	"""
	if len(pArray) < 2:
		return (pArray[0], 1.0, [1.0] )
	
	from rpy2 import robjects as ro
	from rpy2.robjects import r
	from rpy2.robjects.packages import importr
	#import rpy2.robjects as ro
	import pandas.rpy.common as com
	import rpy2.robjects.numpy2ri
	rpy2.robjects.numpy2ri.activate()
	dataFrame1 = pd.DataFrame(pArray.T, dtype= str)
	ro.r('library(FactoMineR)')
	ro.globalenv['r_dataframe'] = com.convert_to_r_dataframe(dataFrame1)
	#ro.globalenv['number_sub'] = dataFrame1.shape[1] 
	#print dataFrame1.shape[1] 
	ro.r('data(r_dataframe)')
	
	# To do it for all names
	ro.r('col_names <- names(r_dataframe)') 
	#ro.r('print (col_names[1])')
	# do it for some names in a vector named 'col_names' 
	ro.r('r_dataframe[,col_names] <- lapply(r_dataframe[,col_names] , factor)')
	#ro.r('r_dataframe = r_dataframe[,  c("1")]')
	ro.globalenv['mca1'] = ro.r('MCA(r_dataframe, ncp =1, method = "Burt", graph = FALSE)')
	rep = ro.r('mca1$ind$coord[,1]')
	loading =  ro.r('mca1$var$eta2[,1]')
	#print ro.r('mca1$var$eta2[,1]')
	#print ro.r('mca1$var$eta2[,2]')
	explained_variance_1 = ro.r('mca1$eig[1,2]')
	#print ro.r('mca1$var$contrib')
	#print ro.r('mca1$var$eta2[,1]')
	#print ro.r('mca1$eig[1,2]')
	#ro.r('plot.MCA(mca1,  cex=.7)')
	#print list(rep)
	#print rep
	#print (discretize(list(rep)), explained_variance_1, loading)   #[float("{0:.1f}".format(a)) for a in list(rep)]
	return (discretize(list(rep)), list(explained_variance_1)[0], list(loading))
	'''
	if len(pArray) < 2:
		print "len A:", len(pArray)
		return pArray[0,:]
	dataFrame = pd.DataFrame(pArray.T)
	try:	
		#print len(dataFrame.columns)
		#print dataFrame.shape
		mca_counts = mca(dataFrame, benzecri=True)#, cols=None, ncols=None, benzecri=True, TOL=1e-4)
		#print mca_counts.fs_r(1)
		#print "mcacounts shape:", mca_counts.fs_r().shape
		#print(mca_counts.L)
		#print "Explained variance:", mca_counts.expl_var(greenacre=False, N=1)
		#print(mca_counts.inertia, mca_counts.L.sum())
		#print mca_counts.fs_r()
		print "len A in MCA:", dataFrame.shape
		return discretize(mca_counts.fs_r(N=1)[:,0].T)
	except:
		print "len A in except:", dataFrame.shape
		return  medoid(pArray)#pArray[len(pArray)-1, :]
		#sys.exit("Error with mca")
	'''

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
		
		pcs = pPCA.fit_transform(pArray.T).T
		#print "Loading:", pPCA.components_
		loadings = pPCA.components_
		return (pcs[0], pPCA.explained_variance_ratio_[0], loadings[0])

	 except ValueError:
	 	iRow = pArray.shape
	 	iCol = None 

	 	return pArray, 1.0, [1.0]
def nlpca(pArray, iComponents=1):
	 """
	 Input: N x D matrix 
	 Output: D x N matrix 

	 """
	 from sklearn.decomposition import PCA
	 #print "pArray:", pArray
	 _, number_sample = pArray.shape
	 t = int(number_sample/math.sqrt(number_sample))
	 s = int(number_sample/t)
	 #print t
	 first_pc = []	
	 try:
	 	for i in range (t):
		 	#print pArray.shape 
		 	pPCA = PCA(n_components=iComponents)
			# # doing this matrix inversion twice doesn't seem to be a good idea 
			# print"PCA:",   pPCA.fit_transform( pArray.T ).T 
			# print "End PCA"
			sub_pArray = pArray[:, s*i:s*(i+1)-1]
			#print sub_pArray.shape
			sub_pc = pPCA.fit_transform(sub_pArray.T).T
			
			#print "sub_pc", sub_pc
			first_pc.extend(sub_pc[0])#map(math.fabs, sub_pc[0]))
			#print "first PC", i,": ",len(first_pc)
			
			#print "Loading:", pPCA.components_
		#print "first PC shape", i,": ",len(first_pc)
		return array([first_pc])

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
	
	return kpca.fit_transform(pArray.T).T [0]
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

def mean(X):
	rep = map(round,numpy.mean(X, axis=0))
	return rep

def middle(X):
	return X[len(X)/2]

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

def get_medoid_centroid(pArray, iAxis=0, pMetric=distance.l2):
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
	#print pArray[numpy.argsort(map(numpy.linalg.norm, pArrayCenter))[0], :]
	return pArray[numpy.argsort(map(numpy.linalg.norm, pArrayCenter))[0], :]
def medoid(pArray, iAxis=0, pMetric=distance.nmi):
	"""
	Input: numpy array 
	Output: float
	"""
	X = pArray
	#return X[len(X)/2]
	return pArray[len(pArray) -1, :]
	d = pMetric 
	def pDistance(x, y):
		return  1.0 - pMetric(x, y)
	D = squareform(pdist(pArray, metric=pDistance))
	#print D
	mean_index = 0
	med = 1.0 
	i = 0
	for i in range(len(D)):
		temp_mean = numpy.mean(D[i])
		if med >= temp_mean:
			med = temp_mean
			mean_index = i
	print "medoid index :", mean_index, len(pArray)
	return pArray[mean_index, :]
def concat(pArray, iAxis=0, pMetric=distance.nmi):
	"""
	Input: numpy array 
	Output: float
	"""
	return pArray.flatten()
def get_representative(pArray, pMethod=None):
	hash_method = c_hash_decomposition
	return hash_method[pMethod](pArray)

c_hash_decomposition = {"none":	"none",
						"pca"    : pca,"dpca"    : pca,
						"mca"    : mca_method,
						"nlpca"    : nlpca,
                        "ica"    : ica,
                        "cca"	 : cca,
                        "pls"	 : pls,
                        "kpca"   : kpca,
                        "medoid" : medoid,
                        "mean"   : mean,
                        "centroid-medoid" : get_medoid_centroid,
                        "average": None }
#=========================================================
# Multiple comparison adjustment 
#=========================================================
def bhy(afPVAL, q):
	"""
	Implement the benjamini-Yekutieli hierarchical hypothesis testing criterion 
	In practice, used for implementing Yekutieli criterion *per layer*.  


	Parameters
	-------------

		afPVAL : list 
		 	list of p-values 


	Returns 
	--------------

		abOUT : list 
			boolean vector corresponding to which hypothesis test rejected, corresponding to p-value 

	"""
	from fractions import Fraction
	harmonic_number = lambda n: sum(Fraction(1, d) for d in xrange(1, n+1))
	
	pRank = rankdata(afPVAL, method= 'ordinal')

	aAjusted = [] 
	#aQvalue = []
	iLen = len(afPVAL)
	q_bar = q/math.log(iLen)#q / harmonic_number(iLen)
	for i, fP in enumerate(afPVAL):
		# fAdjusted = fP*1.0*pRank[i]/iLen#iLenReduced
		
		fAdjusted = q_bar * 1.0 * pRank[i] / iLen  # iLenReduced
		#qvalue = fP * iLen / pRank[i] 
		aAjusted.append(fAdjusted)
		#aQvalue.append(qvalue)
	# print aOut
	# assert( all(map(lambda x: x <= 1.0, aOut)) ) ##sanity check 

	return aAjusted, pRank

def bh(afPVAL, q, cluster_size =None):
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
	#weighted_p = [afPVAL[i]*math.log(cluster_size[i],2) for i in range(len(cluster_size)) ]
	total_cluster_size = numpy.sum(cluster_size)
	pRank = rankdata(afPVAL, method= 'ordinal')
	pRank = [int(i) for i in pRank]
	aAjusted = [] 
	aQvalue = []
	m = len(afPVAL)
	q_bar=q#0.0
	iLen = len(afPVAL)
	
	q_bar = q#/math.log(total_cluster_size/m) #q*2/math.log1p(size_effect+1)
	#print q_bar
	for i, fP in enumerate(afPVAL):
		fAdjusted = q_bar * pRank[i] / iLen  # iLenReduce
		aAjusted.append(fAdjusted)
	'''zipped =  zip(pRank, cluster_size)
	zipped = sorted(zipped, key=lambda x: x[0])
	weight = [0 for i in range(len(afPVAL))]
	for i in range(len(afPVAL)):
		weight[i] = sum([zipped[i][1] for i in range(pRank[i]-1)])
	for i, fP in enumerate(afPVAL):
		fAdjusted = q_bar * (weight[pRank[i]-1])/(total_cluster_size)#pRank[i] / iLen#q_bar * ((1.0 + ((weight[pRank[i]-1])-1.0)/(total_cluster_size-1.0)*(m-1.0))  /  m)#q_bar * pRank[i] / iLen  # iLenReduce
		#(weight[pRank[i]-1])/(total_cluster_size)
		aAjusted.append(fAdjusted)'''
	# print aOut
	# assert( all(map(lambda x: x <= 1.0, aOut)) ) ##sanity check 
	#print aAjusted
	return aAjusted, pRank
def bonferroni(afPVAL, q):
	"""
	Implement the Bonferroni for FDR correction


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

	"""
	pRank = rankdata(afPVAL, method= 'ordinal')

	aAjusted = [] 
	aQvalue = []
	iLen = len(afPVAL)
	for i, fP in enumerate(afPVAL):
		# fAdjusted = fP*1.0*pRank[i]/iLen#iLenReduced
		fAdjusted = q / iLen  # iLenReduced
		aAjusted.append(fAdjusted)
	return aAjusted, pRank
def simple_no_adusting(afPVAL, q):
	"""
	No adusting
	Notes
	---------

	"""
	pRank = rankdata(afPVAL, method= 'ordinal')
	return afPVAL, pRank
def p_adjust(pval, q, cluster_size, method="BH"):
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
	if config.p_adjust_method == "bhy":
			return bhy(pval, q) 
			#fAdjusted = q * 1.0 * pRank[i] / (iLen*math.log(iLen))  # iLenReduced
	elif config.p_adjust_method == "bh":
		return bh(pval, q, cluster_size) 
	elif config.p_adjust_method == "bonferroni":
		return bonferroni(pval, q)
	elif config.p_adjust_method == "no_adjusting":
		return simple_no_adusting(pval, q)

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
	pHashDecomposition = c_hash_decomposition
	pHashMetric = distance.c_hash_metric 
	
	def _permutation(pVec):
		return numpy.random.permutation(pVec)

	pDe = get_medoid_centroid
	pMe = pHashMetric[strMetric] 

	# # implicit assumption is that the arrays do not need to be discretized prior to input to the function
	
	pRep1 = get_medoid_centroid(discretize(pArray1), 0, pMe)
	pRep2 = get_medoid_centroid(discretize(pArray1), 0, pMe)

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




def estimate_p_value(observed_value, random_distribution):
    """Estimate the p-value for a permutation test.

    The estimated p-value is simply `M / N`, where `M` is the number of
    exceedances, and `M > 10`, and `N` is the number of permutations
    performed to this point (rather than the total number of
    permutations to be performed, or possible). For further details, see
    Knijnenburg et al. [1]

    :Parameters:
    - `num_exceedances`: the number of values from the random
      distribution that exceeded the observed value
    - `num_permutations`: the number of permutations performed prior to
      estimation

    """
    # The most precise a p-value we can predict is not 0, but 1 / N
	# where N is the number of permutations.
    num_permutations = len(random_distribution)
    M = num_exceedances = len([score for score in random_distribution
            if score >= observed_value]) 
    if M > 10:
        return float(num_exceedances) / num_permutations
    else:
    	#from scipy.stats import genpareto
    	#genpareto.cdf(random_distribution, observed_value)
        return float(num_exceedances) / num_permutations
def null_fun(X, Y):
	strMetric = config.distance
	pHashDecomposition = c_hash_decomposition
	pHashMetric = distance.c_hash_metric 
	pMe = pHashMetric[strMetric]
	return math.fabs(pMe(X, numpy.random.permutation(Y)))
def permutation_test_pvalue(X, Y):
	 
	strMetric = config.distance 
	seed = config.seed
	iIter = config.iterations
	# step 5 in a case of new decomposition method
	pHashDecomposition = c_hash_decomposition
	pHashMetric = distance.c_hash_metric 
	
	def _permutation(pVec):
		return numpy.random.permutation(pVec)

	pMe = pHashMetric[strMetric] 
	# # implicit assumption is that the arrays do not need to be discretized prior to input to the function
	aDist = [] 
	sim_score= pMe(X, Y)
	fAssociation = math.fabs(sim_score)
	fP = 1.0 
	# print left_rep_variance, right_rep_variance, fAssociation
	#### Perform Permutation
	
	def _calculate_num_exceedances(observed_value, random_distribution):
	    """Determines the number of values from a random distribution which
	    exceed or equal the observed value.
	
	    :Parameters:
	    - `observed_value`: the value that was calculated from the original
	      data
	    - `random_distribution`: an iterable of values computed from
	      randomized data
	
	    """
	    num_exceedances = len([score for score in random_distribution
	            if score >= observed_value])
	    return num_exceedances


	        	
	def _calculate_pvalue(iter):
		fPercentile = percentileofscore(aDist, fAssociation, kind = 'strict')#, kind="mean")  # #source: Good 2000  
	# ## \frac{ \sharp\{\rho(\hat{X},Y) \geq \rho(X,Y) \} +1  }{ k + 1 }
	# ## k number of iterations, \hat{X} is randomized version of X 
	# ## PercentileofScore function ('strict') is essentially calculating the additive inverse (1-x) of the wanted quantity above 
	# ## consult scipy documentation at: http://docs.scipy.org/doc/scipy-0.7.x/reference/generated/scipy.stats.percentileofscore.html

		pval = ((1.0 - fPercentile / 100.0) * iter + 1) / (iter + 1)
		return pval
	#new_fP2 = 0.0
	#new_fP =0.0
	
	few_permutation = True
	if not few_permutation:
		iter = iIter
		if config.use_one_null_dist:
			if len(config.null_dist) == 0:
				generate_null_dist(X,Y)
			aDist = config.null_dist
		else:
			for i in xrange(iIter):
				numpy.random.seed(i+seed)
		
				#XP = array([numpy.random.permutation(x) for x in X])
				#YP = array([numpy.random.permutation(y) for y in Y])
				#pRep1_, _, _ = mca_method(XP) #mean(pArray1)#[len(pArray1)/2]
				#pRep2_, _, _ = mca_method(YP)#
				#pRep1_, pRep2_ = [ discretize(pDe(pA))[0] for pA in [XP, YP] ] if bool(distance.c_hash_association_method_discretize[strMetric]) else [pDe(pA) for pA in [pArray1, pArray2]]
				iter = i
				permuted_Y = numpy.random.permutation(Y)
				# Similarity score between representatives  
				#fAssociation_permuted = pMe(pRep1_, pRep2_)  
				fAssociation_permuted = math.fabs(pMe(X, permuted_Y))  
				aDist.append(fAssociation_permuted)
				if i % 50 == 0:
					new_fP2 = _calculate_pvalue(i) #estimate_pvalue(sim_score, aDist) #
					#num_exceedances = _calculate_num_exceedances(fAssociation_permuted, aDist)
					#new_fP = _estimate_p_value(num_exceedances, len(aDist))
					
					if new_fP2 > fP:
						
						#print "Break before the end of permutation iterations"
						break
					else: 
						fP = new_fP2
				
				# aDist = numpy.array( [ pMe( _permutation( pRep1 ), pRep2 ) for _ in xrange( iIter ) ] )
		fP = _calculate_pvalue(iter)
	
	else:
		fP = nonparametric_test_pvalue(fAssociation, X, Y)
	#print "Estimated P-value:",fP
	'''import matplotlib.pyplot as plt
	print sim_score, fP 
	fig, ax = plt.subplots(1, 1)
	ax.hist(aDist, normed=True, histtype='stepfilled', alpha=0.2)
	ax.legend(loc='best', frameon=False)
	plt.savefig("permut.pdf")
	if fP < 0.001:
		exit()
	'''
	return fP
def generate_null_dist(X, Y):
	#generate a null distrbution
	pHashMetric = distance.c_hash_metric 
	
	def _permutation(pVec):
		return numpy.random.permutation(pVec)

	pMe = pHashMetric[config.distance]
	aDist = []
	for i in xrange(config.iterations):
		numpy.random.seed(i+config.seed)
		iter = i
		permuted_Y = numpy.random.permutation(Y)
		fAssociation_permuted = math.fabs(pMe(X, permuted_Y))  
		aDist.append(fAssociation_permuted)	
	config.null_dist = aDist[:]
def permutation_test_by_representative(pArray1, pArray2):
	"""
	Input: 
	pArray1, pArray2, metric = "mi", decomposition = "pca", iIter = 1000

	"""
	metric = config.distance
	decomposition = config.decomposition
	iIter=config.iterations
	seed = config.seed
	discretize_style = config.strDiscretizing
	X, Y = pArray1, pArray2 
	strMetric = metric 
	# step 5 in a case of new decomposition method
	pHashDecomposition = c_hash_decomposition
	pHashMetric = distance.c_hash_metric 
	

	pDe = pHashDecomposition[config.decomposition]
	pMe = pHashMetric[config.distance] 
	# # implicit assumption is that the arrays do not need to be discretized prior to input to the function
	#aDist = [] 
	left_rep_variance = 1.0
	right_rep_variance = 1.0
	left_loading = []
	right_loading = []
	#print pArray1[0]
	#### Calculate Point estimate
	if (len(pArray1) == 1 and len(pArray2) == 1) or decomposition =="none":
		#if decomposition == "pca":
		#	pRep1 = discretize(pArray1[0, :])
		#	pRep2 = discretize(pArray2[0, :])
		#else:
		pRep1 = pArray1[0, :]
		pRep2 = pArray2[0, :]
		#left_rep_variance = 1.0
		#right_rep_variance = 1.0
		left_loading = [1.0]
		right_loading = [1.0]
		
	elif decomposition == 'mca':
		pRep1, left_rep_variance, left_loading = mca_method(pArray1, discretize_style = discretize_style) #mean(pArray1)#[len(pArray1)/2]
		pRep2, right_rep_variance, right_loading = mca_method(pArray2, discretize_style = discretize_style)#mean(pArray2)#[len(pArray2)/2]	
		#print len(pArray1)," Rep 1: ", pRep1,
		#print len(pArray2)," Rep 2: ", pRep2, 
		#print "Sim: ", pMe(pRep1, pRep2)
	elif decomposition == 'medoid':
		pRep1 = medoid(pArray1)
		pRep2 = medoid(pArray2)
	elif decomposition == "pca":
		[(pRep1, left_rep_variance, left_loading) , (pRep2, right_rep_variance, right_loading)] = [pDe(pA) for pA in [pArray1, pArray2]]
		if bool(distance.c_hash_association_method_discretize[strMetric]):
			[pRep1, pRep2] = [discretize(aRep) for aRep in [pRep1, pRep2] ]
	elif decomposition == "ica":
		[pRep1, pRep2] = [discretize(pDe(pA))[0] for pA in [pArray1, pArray2] ] if bool(distance.c_hash_association_method_discretize[strMetric]) else [pDe(pA)[0] for pA in [pArray1, pArray2]]
	elif decomposition in ['pls', 'cca']:
		[pRep1, pRep2] = discretize(pDe(pArray1, pArray2, metric)) if bool(distance.c_hash_association_method_discretize[strMetric]) else pDe(pArray1, pArray2, metric)
		#print "1:", pRep1
		#print "2:", pRep2
	else:
		[pRep1, pRep2] = [discretize(pDe(pA))[0] for pA in [pArray1, pArray2] ] if bool(distance.c_hash_association_method_discretize[strMetric]) else [pDe(pA) for pA in [pArray1, pArray2]]
		#print "1:", pRep1
		#print "2:", pRep2
		#if decomposition == 'nlpca':
		#	left_rep_variance = first_rep(pArray1, decomposition)
		#	right_rep_variance = first_rep(pArray2, decomposition)
	#print left_rep_variance
	#print "left loading: ", left_loading
	#print "right loading: ", right_loading
	#if decomposition != "pca":
	#	fAssociation = numpy.mean(numpy.array([pMe(pArray1[i], pArray2[j]) for i, j in itertools.product(range(len(pArray1)), range(len(pArray2)))]))
	#else:
	sim_score= pMe(pRep1, pRep2)
	fP = permutation_test_pvalue(X=pRep1, Y=pRep2)
	assert(fP <= 1.0)
	#print fP
	return fP, sim_score, left_rep_variance, right_rep_variance, left_loading, right_loading, pRep1, pRep2 

def g_test_by_representative(pArray1, pArray2, metric="nmi", decomposition="pca", iIter=1000):
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
	pHashDecomposition = c_hash_decomposition
	pHashMetric = distance.c_hash_metric 
	
	def _permutation(pVec):
		return numpy.random.permutation(pVec)

	pDe = pHashDecomposition[decomposition]
	pMe = pHashMetric[strMetric] 
	# # implicit assumption is that the arrays do not need to be discretized prior to input to the function
	
	aDist = [] 
	left_rep_variance = 1.0
	right_rep_variance = 1.0
	#### Calculate Point estimate 
	if decomposition in ['pls', 'cca']:
		[pRep1, pRep2] = discretize(pDe(pArray1, pArray2, metric)) if bool(distance.c_hash_association_method_discretize[strMetric]) else pDe(pArray1, pArray2, metric)
	else:
		[pRep1, pRep2] = [ discretize(pDe(pA))[0] for pA in [pArray1, pArray2] ] if bool(distance.c_hash_association_method_discretize[strMetric]) else [pDe(pA) for pA in [pArray1, pArray2]]
		#print "1:", pRep1
		#print "2:", pRep2
		left_rep_variance = first_rep(pArray1, decomposition)
		right_rep_variance = first_rep(pArray2, decomposition)
	fAssociation = pMe(pRep1, pRep2) 
	import os, sys, re, glob, argparse
	from random import choice, random, shuffle
	from collections import Counter
	from scipy.stats import chi2
	from math import log, exp, sqrt
	'''
	# constants
	trials = 10000
	chars = "ABC"
	n = 100
	k = 0.2 # degree of coupling
	
	# generate random data
	x = [choice( chars ) for i in range( n )]
	y = [x[i] if random() < k else choice( chars ) for i in range( n )]
	'''
	# mutual information
	def mutinfo( x, y ):
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
	mi = distance.mi( pRep1, pRep2 )
	print "mutual information... =", mi
	
	# compute degrees of freedom for table
	xdf = len( set( pRep1 ) ) - 1
	ydf = len( set( pRep2 ) ) - 1
	df = xdf * ydf
	print "degrees of freedom... =", df
	
	# calculate a pvalue from permutation
	pvalue = 0
	delta = 1 / float( iIter )
	for t in range( iIter ):
	    y2 = pRep2[:]
	    y2 = numpy.random.permutation( y2 )
	    if distance.mi( pRep1, y2 ) >= mi:
	        pvalue += delta
	print "permutation P-value.. =", pvalue
	
	# calculate a pvalue based on a G test
	# G = 2 * N * MI, with MI measured in nats (not bits)
	# behaves as a contingency chi^2, with df=(rows-1)(cols-1)
	mi_natural_log = mi / log( exp( 1 ), 2 )
	fP = 1 - chi2.cdf( 2 * len(pRep1) * mi_natural_log, df )
	print "G-test P-value....... =", fP
	
	#permerror = 2 * sqrt( pvalue * ( 1 - pvalue ) / float( trials ) )
	#print "permutation error.... =", permerror
	#print "within perm error.... =", abs( pvalue - pvalue2 ) < permerror

	return fP, fAssociation, left_rep_variance, right_rep_variance
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
	pHashDecomposition = c_hash_decomposition
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

	pHashDecomposition = c_hash_decomposition
	
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

def permutation_test(pArray1, pArray2):
	
	if config.decomposition in ['none','cca', 'pls',"pca", "dpca", "nlpca", "ica", "kpca","centroid-medoid","medoid","mean", "mca"]:
		return permutation_test_by_representative(pArray1, pArray2)
	
	if config.decomposition in ["average"]:
		return permutation_test_by_average(pArray1, pArray2, metric=metric, iIter=iIter)


def g_test(pArray1, pArray2, metric, decomposition, iIter):
	if decomposition in ['cca', 'pls',"pca", "nlpca", "ica", "kpca"]:
		return g_test_by_representative(pArray1, pArray2, metric=metric, decomposition= decomposition, iIter=iIter)
	
	if decomposition in ["average"]:
		return g_test_by_average(pArray1, pArray2, metric=metric, iIter=iIter)
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
def discretize(pArray, style = "equal-area", iN=None, method=None, aiSkip=[]):
	if iN == None:
		iN= config.NBIN
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
	
	>>> discretize( [0.001, 0.002, 0.003, 0.004, 1, 2, 3, 4, 100] )
	[0, 0, 0, 1, 1, 1, 2, 2, 2]
	
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
	#from sklearn.cluster.spectral import discretize
	#y_pred = discretize(y_true_noisy)
	if style in ['jenks', 'kmeans', 'hclust']:
		try:
			from rpy2 import robjects as ro
			from rpy2.robjects import r
			from rpy2.robjects.packages import importr
			#import rpy2.robjects as ro
			import pandas.rpy.common as com
			import rpy2.robjects.numpy2ri
			rpy2.robjects.numpy2ri.activate()
			ro.r('library(classInt)')
			ro.globalenv['style'] =  style
		except ImportError:
			sys.exit("Please install R package classInt")
		
	def _discretize_continuous(astrValues, iN=iN):
		if iN == None:
			# Default to rounded sqrt(n) if no bin count requested
			iN = min(len(set(astrValues)), round(math.sqrt(len(set(astrValues))))) #max(round(math.sqrt(len(astrValues))), round(math.log(len(astrValues), 2)))#round(len(astrValues)/math.log(len(astrValues), 2)))#math.sqrt(len(astrValues)))  # **0.5 + 0.5)
		elif iN == 0:
			iN = len(set(astrValues))
		else:
			iN = min(iN, len(set(astrValues)))
			
		if len(set(astrValues)) <= iN:
			try:
				return rankdata(astrValues, method= 'dense')
			except:
				setastrValues = list(set(astrValues))
				dictA ={}
				for i, item in enumerate(setastrValues):
					dictA[item] = i
				order = []
				for i, item in enumerate(astrValues):
					order.append(dictA[item])
				return order
				#print "Discretizing categorical data!!!"
		else:							
			if style in ['jenks', 'kmeans', 'hclust']:
				try:
					dataFrame1 = pd.DataFrame(astrValues, dtype= float)
					#print dataFrame1[0]
					ro.globalenv['number_of_bins'] = iN 
					ro.globalenv['v'] =  com.convert_to_r_dataframe(dataFrame1)[0]
					ro.r('clI <- classIntervals(v, n = number_of_bins, style = style)')
					ro.r(' descretized_v <- findCols(clI)')
					astrRet = ro.globalenv['descretized_v']
					return astrRet
				except Exception, err:
					print(traceback.format_exc())
					
					print "Discretizing as exeception in ClassInt happend!!!"
					try:
						order = rankdata(astrValues, method= 'min')
					except:
						return astrValues
			else:
				try:
					order = rankdata(astrValues, method= 'min')# ordinal
				except: 
					#print "Categorical data descritizing 2!"
					setastrValues = list(set(astrValues))
					dictA ={}
					for i, item in enumerate(setastrValues):
						dictA[item] = i
					order = []
					for i, item in enumerate(astrValues):
						order.append(dictA[item])
					return order
					'''
					temp = numpy.array(astrValues).argsort()
					order = numpy.arange(len(astrValues))[temp.argsort()]#array(astrValues).argsort().argsort()
					order = rankdata(order, method= 'min') # ordinal #array([order[i]+1.0 for i in range(len(order))])
					'''
			#elif type(astrValues[0]) == float or type(astrValues[0]) == int:
	
			#print "prank",order
			#aiIndices = sorted(range(len(astrValues)), cmp=lambda i, j: cmp(astrValues[i], astrValues[j]))
			#print "aiIndices", aiIndices
		astrRet = [None] * len(astrValues)
		for i in range(len(astrValues)):
			astrRet[i] = int(numpy.ceil(order[i]/iN))
		astrRet = rankdata(astrRet, method= 'dense')
		return astrRet

	def _discretize_continuous_orginal(astrValues, iN=iN): 
		
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
		#print "aiIndices:", aiIndices
		#print "astrRet:", astrRet
		iPrev = 0
		for i, iIndex in enumerate(aiIndices):
			# If you're on a tie, you can't increase the bin number
			# Otherwise, increase by at most one
			iPrev = astrRet[iIndex] = iPrev if (i and (astrValues[iIndex] == astrValues[aiIndices[i - 1]])) else \
				min(iPrev + 1, int(iN * i / float(len(astrValues))))
			#print "astrRet:", astrRet
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
				aOut.append(_discretize_continuous(line, iN))
		#print aOut
		return array(aOut)

	except Exception:
		iN = len(pArray)
		#print "in discritizing exception!!!!"
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



##P-value with less permutation

import scipy.optimize
import scipy.stats
import numpy
import math

def estimate_gpd_params_ML(samples):
	"""
	Maximum likelihood estimate of the GPD from the given samples
	Assumes that the location parameter is 0
	"""

	# Implements gpfit.m from Matlab
	
	# Get an initial guess from the Method of Moments
	n = len(samples)
	xmean = numpy.mean(samples)
	xvar = numpy.var(samples)
	xmax = max(samples)
	xsnr = xmean * xmean / xvar
	shape0 = -.5 * (xsnr - 1)
	scale0 = .5 * xmean * (xsnr + 1)
	if shape0 < 0 and xmax >= -scale0/shape0:
		# MOM failed - start with exponential
		shape0 = 0
		scale0 = xmean
		
	# Negative log-like function
	EPS = 7./3. - 4./3. - 1. # Machine epsilon
	def negloglike(parms, data):
		shape = parms[0]
		lnscale = parms[1]
		scale = math.exp(lnscale)

		n = len(data)
		#print scale
		
		Z = [x / scale for x in data]
		
		if abs(shape) > EPS:
			# Non-exponential
			if shape > 0 or max(Z) < -1/shape:
				try:
					sumln1pkz = sum([math.log1p(shape*z) for z in Z])
				except ValueError:
					return None
				return n * lnscale + (1 + 1/shape) * sumln1pkz
			else:
				return float('inf')#math.inf
		else:
			# Limiting exponential distribution as shape -> 0
			return n * lnscale + sum(Z)
	
	# Find the ML estimate numerically
	result = scipy.optimize.minimize(negloglike, (shape0, math.log(scale0)),
		method='Nelder-Mead', args=(samples), options={'disp': False})
	shapehat = result.x[0]
	scalehat = math.exp(result.x[1])
	return (shapehat, scalehat)

def gpd_goodness_of_fit(shape, samples):
	"""
	Goodness-of-fit test of the samples to a generalize pareto distribution
	with the given shape
	Returns the p-value
	"""
	
	# KS-test for now, until I get access to the Choulakian and Stephens (2001) paper
	gpd = scipy.stats.genpareto(shape)
	(D,p) = scipy.stats.kstest(samples, gpd.cdf)
	return p

def estimate_tail_gpd(samples):
	"""
	Fits a GPD to the tail of the distribution from the given samples
	Returns a frozen GPD object and the number of samples used to estimate it
	"""
	
	# Algorithm proposed in Knijnenburg2009
	
	# Sort the samples so that the tail samples are easily accessible
	sorted_samples = samples
	sorted_samples.sort()
	
	# Minimum number of samples exceeding the threshold
	minNexc = 10
	# Initial number of samples to fit the tail with
	Nexc = 250
	# Amount of samples to drop Nexc by if the goodness of fit test fails
	dNexc = 10
	
	# Try to make Nexc at most half of the samples, but at least minNexc,
	# but definitely no more than N-1
	# This is only problematic if N is < Nexc*2, which should never be the case
	Nexc = min(len(samples)-1, max(minNexc, min(Nexc, int(math.floor(len(samples)/2)))))
	
	while True:
		# Fit the GPD to the samples
		subsamples = sorted_samples[-Nexc:]
		t = (sorted_samples[-Nexc] + subsamples[1]) / 2
		(shape, scale) = estimate_gpd_params_ML([x - t for x in subsamples])
		
		if Nexc <= minNexc:
			# Too few samples - have to go with what we have
			# TODO: Emit warning?
			break
		
		# Does the GPD fit well with the samples?
		gof = gpd_goodness_of_fit(shape, [(x - t) / scale for x in subsamples])
		if gof < 0.05:
			# The GPD doesn't fit this tail.. reduce the number of samples
			Nexc = max(minNexc, Nexc - dNexc)
		else:
			# GPD is a good fit
			break

	return (scipy.stats.genpareto(shape, loc=t, scale=scale), Nexc)

def estimate_pvalue(x, null_samples,X, Y, regenrate_GPD_flag = False):
	"""
	Estimates the p-value, given the observed test statistic x and a set of
	samples from the null distribution.
	"""
	
	# Algorithm proposed in Knijnenburg2009
	
	if x == 0:
		return 1.0
	# Get M, the number of null samples greater than x
	M = len([1 for v in null_samples if v >= x])  # or v >= x 
	N = len(null_samples)
	
	# Use the ECDF to approximate p-values if M > 10
	if M >= 10 or N < 250:
		return float(M)/float(N)

	# Estimate the generalized pareto distribtion from tail samples
	if not config.use_one_null_dist or config.gp == None or regenrate_GPD_flag:
		
		try:
			#null_samples = list(set(null_samples))
			(gp, Nexc) = estimate_tail_gpd(null_samples)
			config.gp  = gp
			config.Nexc = Nexc
		except ArithmeticError, ValueError:
			return float(M)/float(N)
	else:
		(gp, Nexc) = (config.gp, config.Nexc)
	# GPD estimate of the actual p-value
	
	# Check if the result of the survival function is na then genrate more null samples
	sf_result =  gp.sf(x)
	if sf_result == 1:
		print sf_result, N, Nexc
	if math.isnan(float(sf_result)) or sf_result == 1.0:
		print "WARNING: the number of permutation for null samples wasn't enough and it's doubled!"
		print "This could happen when you have features with low variation or zero variation!"
		#sample_increments = 50
		if regenrate_GPD_flag:
			return float(M)/float(N)
		# Double the number of null samples for statistic if the intintazted number of samples wasn't enough.
		config.nullsamples = list(set([null_fun(X, Y) for val in range(0,len(config.nullsamples))] + config.nullsamples))
		try:
			return estimate_pvalue(x, config.nullsamples, X=X, Y=Y, regenrate_GPD_flag = True)
		except ArithmeticError, ValueError:
			return float(M)/float(N)
	else:	
		#print "final pvalue", (Nexc*1.0 / N) * sf_result
		return (float(Nexc) / float(N)) * sf_result

def prob_pvalue_lt(alpha, nexc, ntotal):
	"""
	Probability that the p-value is less than alpha, if there are nexc
	exceedances out of ntotal permutations
	"""
	
	# Use a prior with approx. 50% probability of p being < alpha
	# and the same certainty (a+b) as Bayes' prior
	a = 1/3 + alpha * 3/4
	b = 2 - a
	
	# Bayesian estimate of the p-value's actual value
	pvalue = scipy.stats.beta(a + nexc, b + ntotal - nexc)
	
	# Probability of the pvalue being < alpha
	return pvalue.cdf(alpha)

def prob_pvalue_lt_samples(alpha, x, null_samples):
	"""
	Probability that the p-value is less than alpha, given the test
	statistic x and a set of samples from the null distribution
	"""
	return prob_pvalue_lt(alpha, len([1 for v in null_samples if v > x]), len(null_samples))

def nonparametric_test_pvalue(x, X, Y, alpha_cutoff = 0.05):
	"""
	Performs a permutation test of the significance of x, given the function
	to sample the null distribution null_fun.
	This function will exit early if it becomes clear that the p-value will
	be greater than alpha_cutoff. In this case, the current approximation
	of the p-value is returned.
	"""
	# The number of null samples to start with
	start_samples = 100
	# Number of null samples to gather in each round
	sample_increments = 50
	# Maximum number of null samples, at which point the GPD approximation
	# is used
	max_samples = config.iterations
	
	# Sample the null distribution until we've got enough to estimate the tail
	# or if we're sure that the actual p-value is greater than the alpha cutoff
	if not config.use_one_null_dist or len(config.nullsamples) == 0:
		nullsamples = [null_fun(X, Y) for val in range(0,start_samples)]
		while len(nullsamples) < max_samples and prob_pvalue_lt_samples(config.q, x, nullsamples) > .01:
			#print("Gathering more.. N = %d; P(p<%f) = %.2f" % (len(nullsamples), config.q, prob_pvalue_lt_samples(config.q, x, nullsamples)))
			nullsamples = [null_fun(X, Y) for val in range(0,sample_increments)] + nullsamples
			config.nullsamples = nullsamples
	else:
		nullsamples = config.nullsamples
		
		
	#print("Finished gathering: N = %d; P(p<%f) = %f" % (len(nullsamples), alpha_cutoff, prob_pvalue_lt_samples(alpha_cutoff, x, nullsamples)))
	# Estimate the p-value from the current set of samples
	return estimate_pvalue(x, nullsamples, X=X, Y=Y)


	
### Testing functions
	
def test_permtest(p_true):
	"""
	Test permutation_test_pvalue
	"""
	
	rv = scipy.stats.norm(0, 1)
	return permutation_test_pvalue(rv.isf(p_true), rv.rvs)

def verbose_test_permtest(p_true):
	for i in range(1,10):
		p_est = test_permtest(p_true)
		print("P = %f; P_est = %f; ratio = %.2f" % (p_true, p_est, p_est/p_true))

def test_gof():
	rshape = 0.3
	rscale = 5
	rv = scipy.stats.genpareto(rshape, loc=10, scale=rscale)
	
	subsamples = rv.rvs(size=1000)
	t = 10
	(shape, scale) = estimate_gpd_params_ML([x - t for x in subsamples])
	print("est shape, scale = %f, %f (real = %f, %f)" % (shape, scale, rshape, rscale))
	gof = gpd_goodness_of_fit(shape, [(x - t) / scale for x in subsamples])
	print("gof = %f" % gof)

## End P-vlue less permutation
