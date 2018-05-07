#!/usr/bin/env python 
"""
unified statistics module 
"""

# native python 
#import exceptions 
#from exceptions import ArithmeticError, ValueError 	
from itertools import compress
from itertools import product
import itertools
import math
from numpy import array , std, log2, dtype
import numpy 
from numpy.random import shuffle, binomial, normal, multinomial 
import scipy
import scipy.stats
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
from scipy.cluster.hierarchy import fcluster
#from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
try:
	import jenkspy
except:
	pass

# External dependencies 
# from scipy.stats import percentileofscore
# ML plug-in 
# Internal dependencies
def get_enropy(x):
	try:
		'''
		if len(labels) == 0:
			return 1.0
		label_idx = unique(labels, return_inverse=True)[1]
		pi = np.bincount(label_idx).astype(np.float)
		pi = pi[pi > 0]
		pi_sum = np.sum(pi)
		# log(a / b) should be calculated as log(a) - log(b) for
		# possible loss of precision
		return -np.sum((pi / pi_sum) * (np.log(pi) - log(pi_sum)))
		'''
		if min(x)==1:
			d = x-1
		elif min(x) == 0:
			d = x
		d = [float(val) for val in d]
		#print d
		P = numpy.bincount(d)/float(len(d))
		observed_entropy = -sum([p * numpy.log2(p) for p in P])
	except:
		#sys.exit("entropy error")
		P = scipy.stats.itemfreq(x)[:,1]
		P = list(map(float, P))
		P = [ val/len(x) for val in P]
		observed_entropy = -sum([p * numpy.log2(p) for p in P])
	#max_entropy = numpy.log2(len(P))
	return observed_entropy#/max_entropy
def scale_data(X, scale = 'log'):
	if scale == 'sqrt':
		y = numpy.sqrt(numpy.abs(X)) * numpy.sign(X)
	elif scale =='log': 
		y = numpy.abs(numpy.log(numpy.abs(X))) * numpy.sign(X) 
	elif scale =='arcsin': 
		y = numpy.arcsin(X) 
	elif scale =='arcsinh': 
		y = numpy.arcsinh(X) 
	elif scale == '':
		y= X
	return y 
def pvalues2qvalues ( pvalues, adjusted=False ):
    n = len( pvalues )
    # after sorting, index[i] is the original index of the ith-ranked value
    index = list(range(n))
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
	A = numpy.array([func(D[i], D[j]) for i, j in itertools.combinations(list(range(len(XP))), 2)])
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
	if not distance.c_hash_association_method_discretize[config.similarity_method]:
		pArray = discretize(pArray)
	from rpy2 import robjects as ro
	from rpy2.robjects import r
	from rpy2.robjects.packages import importr
	#import pandas.rpy.common as com
	import rpy2.robjects.numpy2ri
	from rpy2.robjects import pandas2ri
	pandas2ri.activate()
	rpy2.robjects.numpy2ri.activate()
	dataFrame1 = pd.DataFrame(pArray.T, dtype= str)
	ro.r('library(FactoMineR)')
	ro.globalenv['r_dataframe'] =  pandas2ri.py2ri(dataFrame1)
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
	return (rep, list(explained_variance_1)[0], loading)
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
	pPCA = PCA(n_components=iComponents)
	pcs = pPCA.fit_transform(pArray.T).T
	#print "Loading:", pPCA.components_
	loadings = pPCA.components_
	return (pcs[0], pPCA.explained_variance_ratio_[0], loadings[0])

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
	 		pPCA = PCA(n_components=iComponents)
	 		sub_pArray = pArray[:, s*i:s*(i+1)-1]
	 		sub_pc = pPCA.fit_transform(sub_pArray.T).T
	 		first_pc.extend(sub_pc[0])
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
		print ("End MDS", pos.T)
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
	rep = list(map(round,numpy.mean(X, axis=0)))
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
	return pArray[numpy.argsort(list(map(numpy.linalg.norm, pArrayCenter)))[0], :]
def medoid(pArray, iAxis=0, pMetric=distance.nmi):
	"""
	Input: numpy array 
	Output: float
	"""
	#X = pArray
    #return X[len(X)/2]
	#return pArray[len(pArray) -1, :]
	D = squareform(pdist(pArray, metric=distance.pDistance))
	#print D
	medoid_index = 0
	med = 1.0 
	#i = 0
	for i in range(len(D)):
		temp_mean = numpy.mean(D[i])
		if temp_mean <= med:
			med = temp_mean
			medoid_index = i
	#print "medoid index :", medoid_index, len(pArray)-1
	return pArray[medoid_index, :]
def farthest (pArray1, pArray2, similarity_method):
	pMe = distance.c_hash_metric [similarity_method] 
	best_rep_1 = pArray1[0, :]
	best_rep_2 = pArray2[0, :]
	worst_rep_1 = pArray1[0, :]
	worst_rep_2 = pArray2[0, :]
	best_similarity = 0.0
	worst_similarity = 1.0
	for i in range(len(pArray1)):
		m1 = pArray1[i, :]
		for j in range(len(pArray2)):
			m2 = pArray2[j, :] 
			sim_score_temp = math.fabs(pMe(m1, m2)) #= 1.0 - permutation_test_pvalue(m1, m2)#
			if sim_score_temp <= worst_similarity:
				worst_similarity = sim_score_temp
				worst_rep_1, worst_rep_2 =  m1, m2
			if  sim_score_temp >= best_similarity:
				best_similarity = sim_score_temp
				best_rep_1, best_rep_2 = m1 , m2 
	#print best_similarity, worst_similarity
	#pRep1 = worst_rep_1
	#pRep2 = worst_rep_2
	return worst_rep_1, worst_rep_2, best_rep_1, best_rep_2
	
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
						"pca"    : pca,
						"dpca"    : pca,
						"mca"    : mca_method,
						"nlpca"    : nlpca,
                        "ica"    : ica,
                        "cca"	 : cca,
                        "pls"	 : pls,
                        "kpca"   : kpca,
                        "medoid" : medoid,
                        "farthest" : farthest,
                        "mean"   : mean,
                        "centroid-medoid" : get_medoid_centroid,
                        "average": None }
#=========================================================
# Multiple comparison adjustment 
#=========================================================
def by(afPVAL, q):
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
	#from fractions import Fraction
	#harmonic_number = lambda n: sum(Fraction(1, d) for d in xrange(1, n+1))
	
	pRank = rankdata(afPVAL, method= 'ordinal')

	aAjusted = [] 
	#aQvalue = []
	iLen = len(afPVAL)
	q_bar = q/math.log(iLen)#q / harmonic_number(iLen)
	for i, fP in enumerate(afPVAL):
		# fAdjusted = fP*1.0*pRank[i]/iLen#iLenReduced
		
		fAdjusted = float(q_bar *  pRank[i]) / float(iLen)  # iLenReduced
		#qvalue = fP * iLen / pRank[i] 
		aAjusted.append(fAdjusted)
		#aQvalue.append(qvalue)
	# print aOut
	# assert( all(map(lambda x: x <= 1.0, aOut)) ) ##sanity check 

	return aAjusted, pRank

def is_leaf(node):
    return bool(not(node.m_pData and node.m_arrayChildren))

def get_children(node): 
    return node.m_arrayChildren


def halla_meinshausen(current_level_tests):
	
	c_si = []
	for test in current_level_tests:
		num_leaf_child = 0
		if math.log(len(test.m_pData[0]), 2) < 2 and math.log(len(test.m_pData[1]), 2) < 2:
			num_leaf_child = len(test.m_pData[0]) * len(test.m_pData[1])
		#print len(test.m_pData[0]), len(test.m_pData[1]), num_leaf_child
		c_si.append(num_leaf_child + test.c)
	
	m = sum([len(test.m_pData[0])* len(test.m_pData[1]) for test in current_level_tests])
		
	p_adjusted = [ val * config.q /m for  val in c_si]
	return p_adjusted
def halla_bh(current_level_tests):
	worst_rank= rankdata([test.worst_pvalue  for test in current_level_tests], method= 'ordinal')
	for i in range(len(current_level_tests)):
		current_level_tests[i].worst_rank = worst_rank[i]
	for i in range(len(current_level_tests)):
		#current_level_tests[i].worst_rank = 1
		for j in range(len(current_level_tests)):
			if i != j:
				if current_level_tests[i].worst_pvalue >= current_level_tests[j].worst_pvalue:
				    #if current_level_tests[j].significance != None:
					num_sub_h = 1# len(current_level_tests[j].m_pData[0]) * len(current_level_tests[j].m_pData[1]) - 1
					current_level_tests[i].worst_rank += num_sub_h
			else:
				current_level_tests[i].worst_rank += 1
				#current_level_tests[i].worst_rank += len(current_level_tests[j].m_pData[0]) * len(current_level_tests[j].m_pData[1]) - 1
				
		continue
	worst_rank = [test.worst_rank  for test in current_level_tests]
	m = max(worst_rank)
	#print m
	p_adjusted_worst = [test.worst_rank * config.q / m for test in current_level_tests]
	
	'''for i in range(len(current_level_tests)):
	    current_level_tests[i].rank = 1
	    for j in range(len(current_level_tests)):
	        if i != j:
	            if current_level_tests[i].pvalue >= current_level_tests[j].pvalue:
	                if current_level_tests[j].significance != None:
	                    num_sub_h = len(current_level_tests[j].m_pData[0]) * len(current_level_tests[j].m_pData[1])
	                    current_level_tests[i].rank += num_sub_h
	                else:
	                    current_level_tests[i].rank += 1 
	rep_rank = [test.rank  for test in current_level_tests]
	m = max(rep_rank)
	p_adjusted = [test.rank * config.q / m for test in current_level_tests]'''
	return p_adjusted_worst #, p_adjusted  
def halla_y(pvalues, q, level):
	worst_rank= rankdata(pvalues , method= 'ordinal')
	m = len(pvalues)
	q  = q/(2.0*1.44)
	q_bar =   (q * 1.0) / (sum([1.0/i for i in range(1,m+1)]))#q#/(2.0*1.44) # (m + 1)/(4* math.log( m)) * q * 1.0 /sum([1.0/i for i in range(1,m+1)])
	#print q, q_bar, m, level
	p_adjusted_worst = [worst_rank[i] * q_bar / m for i in range(m)]
	return p_adjusted_worst, worst_rank 
 
def bh(afPVAL, q, add_exra_order =0 , minus_extra_order = 0, cluster_size =None):
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
	pRank = [int(i) + add_exra_order for i in pRank]
	aAjusted = [] 
	aQvalue = []
	iLen = len(afPVAL) + add_exra_order + minus_extra_order
	
	q_bar = q #+ (1-q)*add_exra_order*(1-q)/(iLen + minus_extra_order)#/math.log(total_cluster_size/m) #q*2/math.log1p(size_effect+1)
	#print q_bar, iLen
	aAjusted = [q_bar * pRank[i] / iLen for i in range(len(afPVAL))]
	'''for i, fP in enumerate(afPVAL):
		fAdjusted = q_bar * pRank[i] / iLen  # iLenReduce
		aAjusted.append(fAdjusted)'''
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
	aAjusted = [q/iLen for val in afPVAL]
	return aAjusted, pRank
def simple_no_adusting(afPVAL, q):
	"""
	No adusting
	Notes
	---------

	"""
	pRank = rankdata(afPVAL, method= 'ordinal')
	return afPVAL, pRank
def p_adjust(pval, q, add_exra_order =0 , minus_extra_order = 0, cluster_size = None, method="BH"):
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
	if config.p_adjust_method == "by":
			return by(pval,  q) 
			#fAdjusted = q * 1.0 * pRank[i] / (iLen*math.log(iLen))  # iLenReduced
	elif config.p_adjust_method == "bh" or config.p_adjust_method == "y" :
		return bh(pval, q, add_exra_order, minus_extra_order, cluster_size) 
	elif config.p_adjust_method == "bonferroni":
		return bonferroni(pval, q)
	elif config.p_adjust_method == "no_adjusting":
		return simple_no_adusting(pval, q)

#=========================================================
# Statistical test 
#=========================================================

def association_by_representative(pArray1, pArray2, metric="nmi", decomposition="pca", iterations=1000):
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

def permutation_test_by_medoid(pArray1, pArray2, metric="nmi", iterations=1000):
	"""
	Input: 
	pArray1, pArray2, metric = "mi", decomposition = "pca", iterations = 100

	metric = {pca": pca} 
	"""

	# numpy.random.seed(0)

	strMetric = metric 
	pHashDecomposition = c_hash_decomposition
	pHashMetric = distance.c_hash_metric 
	
	def _permutation(pVec):
		return numpy.random.permutation(pVec)

	pDe = medoid
	pMe = pHashMetric[strMetric] 

	# # implicit assumption is that the arrays do not need to be discretized prior to input to the function
	
	pRep1 = medoid(discretize(pArray1), 0, pMe)
	pRep2 = medoid(discretize(pArray1), 0, pMe)

	# pRep1, pRep2 = [ discretize( pDe( pA ) )[0] for pA in [pArray1,pArray2] ] 

	fAssociation = pMe(pRep1, pRep2) 

	aDist = numpy.array([ pMe(_permutation(pRep1), pRep2) for _ in range(iterations) ])
	# WLOG, permute pArray1 instead of 2, or both. Can fix later with added theory. 
	# # BUGBUG: currently this permutes the discretized values; we may be using information, but better than doing iterations iterations of PCA

	fPercentile = percentileofscore(aDist, fAssociation, kind="strict")  # #source: Good 2000 
	# ## \frac{ \sharp\{\rho(\hat{X},Y) \geq \rho(X,Y) \} +1  }{ k + 1 }
	# ## k number of iterations, \hat{X} is randomized version of X 
	# ## PercentileofScore function ('strict') is essentially calculating the additive inverse (1-x) of the wanted quantity above 
	# ## consult scipy documentation at: http://docs.scipy.org/doc/scipy-0.7.x/reference/generated/scipy.stats.percentileofscore.html

	fP = ((1.0 - fPercentile / 100.0) * iterations + 1) / (iterations + 1)

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
	strMetric = config.similarity_method
	pHashDecomposition = c_hash_decomposition
	pHashMetric = distance.c_hash_metric 
	pMe = pHashMetric[strMetric]
	return math.fabs(pMe(X, numpy.random.permutation(Y)))
def permutation_test_pvalue(X, Y, iterations = None, permutation_func= None, similarity_method = None, seed = 0 ):
	 
	if not similarity_method:
		similarity_method = config.similarity_method 
	else:
		config.similarity_method = similarity_method
	if not seed:
		seed = config.seed
	if not iterations:
		iterations = config.iterations
	if not permutation_func:
		permutation_func = config.permutation_func
	#pHashDecomposition = c_hash_decomposition
	pHashMetric = distance.c_hash_metric 
	def _permutation(pVec):
		return numpy.random.permutation(pVec)
	pMe = pHashMetric[similarity_method] 
	aDist = [] 
	sim_score= pMe(X, Y)
	fAssociation = math.fabs(sim_score)
	fP = 1.0 
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
		pval = ((1.0 - fPercentile / 100.0) * iter + 1) / (iter + 1)
		return pval

	few_permutation = False
	if permutation_func == 'ecdf':
		iter = iterations
		if config.use_one_null_dist:
			if len(config.nullsamples) == 0:
				config.nullsamples = generate_null_dist(X,Y)
			aDist = config.nullsamples
		else:
			for i in range(iterations):
				iter = i
				permuted_Y = numpy.random.permutation(Y)
				fAssociation_permuted = math.fabs(pMe(X, permuted_Y))  
				aDist.append(fAssociation_permuted)
				if i % 50 == 0:
					new_fP2 = _calculate_pvalue(i) #estimate_pvalue(sim_score, aDist) #
					if new_fP2 > fP:
						#print "Break before the end of permutation iterations"
						break
					else: 
						fP = new_fP2
		fP = _calculate_pvalue(iter)
	elif permutation_func == 'gpd':
		fP = nonparametric_test_pvalue(X, Y)
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

	pMe = pHashMetric[config.similarity_method]
	n_samples = []
	for i in range(config.iterations):
		numpy.random.seed(i+config.seed+1)
		iter = i
		permuted_Y = numpy.random.permutation(Y)
		fAssociation_permuted = math.fabs(pMe(X, permuted_Y))  
		n_samples.append(fAssociation_permuted)	
	return n_samples
def permutation_test_by_representative(pArray1, pArray2):
	"""
	Input: 
	pArray1, pArray2, metric = "mi", decomposition = "pca", iterations = 1000

	"""
	metric = config.similarity_method
	decomposition = config.decomposition
	iterations=config.iterations
	#seed = config.seed
	discretize_style = config.strDiscretizing
	#X, Y = pArray1, pArray2 
	strMetric = metric 
	# step 5 in a case of new decomposition method
	pHashDecomposition = c_hash_decomposition
	pHashMetric = distance.c_hash_metric 
	

	pDe = pHashDecomposition[config.decomposition]
	pMe = pHashMetric[config.similarity_method] 
	# # implicit assumption is that the arrays do not need to be discretized prior to input to the function
	#aDist = [] 
	left_rep_variance = 1.0
	right_rep_variance = 1.0
	left_loading = []
	right_loading = []
	#### Calculate Point estimate
	worst_rep_1, worst_rep_2, best_rep_1, best_rep_2 = farthest(pArray1, pArray2, config.similarity_method)
	pRep1 = worst_rep_1# medoid(pArray1)
	pRep2 = worst_rep_2#medoid(pArray2)
	if config.similarity_method == 'spearman' and config.permutation_func == "none":# and randomization_method != "permutation" :
		best_sim_score, best_pvalue = scipy.stats.spearmanr(best_rep_1, best_rep_2, nan_policy='omit')
		worst_sim_score, worst_pvalue = scipy.stats.spearmanr(worst_rep_1, worst_rep_2, nan_policy='omit')
		#medoid_pvalue = scipy.stats.spearmanr(pRep1, pRep2, nan_policy='omit')
		
	elif  config.similarity_method == 'pearson' and config.permutation_func == "none":# and randomization_method != "permutation" :
		best_sim_score, best_pvalue = scipy.stats.pearsonr(best_rep_1, best_rep_2)
		worst_sim_score, worst_pvalue = scipy.stats.pearsonr(worst_rep_1, worst_rep_2)
	else:
		best_sim_score = pMe(best_rep_1, best_rep_2)
		best_pvalue = permutation_test_pvalue(X = best_rep_1, Y = best_rep_2)
		worst_sim_score = pMe(best_rep_1, best_rep_2)
		worst_pvalue = permutation_test_pvalue(X = worst_rep_1, Y = worst_rep_2)
		#medoid_pvalue = permutation_test_pvalue(X = pRep1, Y = pRep2)
	if (len(pArray1) == 1 and len(pArray2) == 1) or decomposition =="none":
		#if decomposition == "pca":
		#	pRep1 = discretize(pArray1[0, :])
		#	pRep2 = discretize(pArray2[0, :])
		#else:
		worst_rep_1 = best_rep_1 = pRep1 = pArray1[0, :]
		worst_rep_2 = best_rep_2 = pRep2 = pArray2[0, :]
		#left_rep_variance = 1.0
		#right_rep_variance = 1.0
		left_loading = [1.0]
		right_loading = [1.0]
		
	elif decomposition == 'mca':
		pRep1, left_rep_variance, left_loading = mca_method(pArray1, discretize_style = discretize_style) #mean(pArray1)#[len(pArray1)/2]
		pRep2, right_rep_variance, right_loading = mca_method(pArray2, discretize_style = discretize_style)#mean(pArray2)#[len(pArray2)/2]	
		if bool(distance.c_hash_association_method_discretize[strMetric]):
			[pRep1, pRep2] = [discretize(aRep) for aRep in [pRep1, pRep2] ]
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
	else:
		#[pRep1, pRep2] = [discretize(pDe(pA))[0] for pA in [pArray1, pArray2] ] if bool(distance.c_hash_association_method_discretize[strMetric]) else [pDe(pA) for pA in [pArray1, pArray2]]
		sys.exit('Decomposition method is not defined!')
	'''if config.similarity_method == 'spearman' and config.permutation_func == "none":# and randomization_method != "permutation" :
		rep_sim_score, rep_pvalue = scipy.stats.spearmanr(pRep1, pRep2, nan_policy='omit')
		
	elif  config.similarity_method == 'pearson' and config.permutation_func == "none":# and randomization_method != "permutation" :
		rep_sim_score, medoid_pvalue = scipy.stats.pearsonr(pRep1, pRep2,)
	else:
		rep_sim_score = pMe(pRep1, pRep2)
		rep_pvalue = permutation_test_pvalue(X = pRep1, Y = pRep2)'''
	rep_pvalue = None
	return worst_pvalue, best_pvalue, rep_pvalue,  worst_sim_score, best_sim_score, left_rep_variance, right_rep_variance, left_loading, right_loading, pRep1, pRep2 

def g_test_by_representative(pArray1, pArray2, metric="nmi", decomposition="pca", iterations=1000):
	"""
	Input: 
	pArray1, pArray2, metric = "mi", decomposition = "pca", iterations = 1000

	metric = {pca": pca} 
	"""
	# numpy.random.seed(0)
	# return g_test_by_representative( pArray1, pArray2, metric = "nmi", decomposition = "pca", iterations = 1000 )
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
	    for ( xchar, ychar ), value in list(pxy.items()):
	        S += value * ( log( value, 2 ) - log( px[xchar], 2 ) - log( py[ychar], 2 ) )
	    return S
	
	# actual value
	mi = distance.mi( pRep1, pRep2 )
	print ("mutual information... =", mi)
	
	# compute degrees of freedom for table
	xdf = len( set( pRep1 ) ) - 1
	ydf = len( set( pRep2 ) ) - 1
	df = xdf * ydf
	print ("degrees of freedom... =", df)
	
	# calculate a pvalue from permutation
	pvalue = 0
	delta = 1 / float( iterations )
	for t in range( iterations ):
	    y2 = pRep2[:]
	    y2 = numpy.random.permutation( y2 )
	    if distance.mi( pRep1, y2 ) >= mi:
	        pvalue += delta
	print ("permutation P-value.. =", pvalue)
	
	# calculate a pvalue based on a G test
	# G = 2 * N * MI, with MI measured in nats (not bits)
	# behaves as a contingency chi^2, with df=(rows-1)(cols-1)
	mi_natural_log = mi / log( exp( 1 ), 2 )
	fP = 1 - chi2.cdf( 2 * len(pRep1) * mi_natural_log, df )
	print ("G-test P-value....... =", fP)
	
	#permerror = 2 * sqrt( pvalue * ( 1 - pvalue ) / float( trials ) )
	#print "permutation error.... =", permerror
	#print "within perm error.... =", abs( pvalue - pvalue2 ) < permerror

	return fP, fAssociation, left_rep_variance, right_rep_variance
def parametric_test_by_max_pca(pArray1, pArray2, k=2, metric="spearman", iterations=1000):

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

def permutation_test_by_max_pca(pArray1, pArray2, k=2, metric="nmi", iterations=1000):

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

		aDist = numpy.array([ pMe(_permutation(pOne), pTwo) for _ in range(iterations) ])
		# WLOG, permute pArray1 instead of 2, or both. Can fix later with added theory. 
		# # BUGBUG: currently this permutes the discretized values; we may be using information, but better than doing iterations iterations of PCA

		fPercentile = percentileofscore(aDist, fAssociation, kind="strict")  # #source: Good 2000 
		# ## \frac{ \sharp\{\rho(\hat{X},Y) \geq \rho(X,Y) \} +1  }{ k + 1 }
		# ## k number of iterations, \hat{X} is randomized version of X 
		# ## PercentileofScore function ('strict') is essentially calculating the additive inverse (1-x) of the wanted quantity above 
		# ## consult scipy documentation at: http://docs.scipy.org/doc/scipy-0.7.x/reference/generated/scipy.stats.percentileofscore.html

		fP = ((1.0 - fPercentile / 100.0) * iterations + 1) / (iterations + 1)

		assert(fP <= 1.0)

		aOut.append(fP)

	return aOut 	

def permutation_test_by_multiple_representative(pArray1, pArray2, k=2, metric="nmi", iterations=1000):

	return min(permutation_test_by_max_pca(pArray1, pArray2, k=k, metric=metric, iterations=iterations))

def parametric_test_by_multiple_representative(pArray1, pArray2, k=2, metric="spearman"):

	return min(parametric_test_by_max_pca(pArray1, pArray2, k=k, metric=metric))

def permutation_test_by_cca(pArray1, pArray2, metric="nmi", iterations=1000):

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
	for _ in range(iterations):

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
	
	# aDist = [pMetric( _permute_matrix(pArray1), pArray2 ) for _ in xrange(iterations)]
	
	# aDist = numpy.array([ py.distance.nmi( _permutation( X_c ), Y_c ) for _ in xrange(iterations) ])
	# aDist = numpy.array( [ pMe( _permutation( pRep1 ), pRep2 ) for _ in xrange( iterations ) ] )
	# WLOG, permute pArray1 instead of 2, or both. Can fix later with added theory. 
	# # BUGBUG: currently this permutes the discretized values; we may be using information, but better than doing iterations iterations of PCA

	fPercentile = percentileofscore(aDist, fAssociation, kind="strict")  # #source: Good 2000 
	# ## \frac{ \sharp\{\rho(\hat{X},Y) \geq \rho(X,Y) \} +1  }{ k + 1 }
	# ## k number of iterations, \hat{X} is randomized version of X 
	# ## PercentileofScore function ('strict') is essentially calculating the additive inverse (1-x) of the wanted quantity above 
	# ## consult scipy documentation at: http://docs.scipy.org/doc/scipy-0.7.x/reference/generated/scipy.stats.percentileofscore.html

	fP = ((1.0 - fPercentile / 100.0) * iterations + 1) / (iterations + 1)

	return fP, fAssociation, pRep1[0], pRep2[0]

def permutation_test_by_pls(pArray1, pArray2, metric="nmi", iterations=1000):

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
	for _ in range(iterations):

		XP = array([numpy.random.permutation(x) for x in X])
		YP = array([numpy.random.permutation(y) for y in Y])

		# pRep1_, pRep2_ = [ discretize( pDe( pA ) )[0] for pA in [XP,YP] ] if bool(py.distance.c_hash_association_method_discretize[strMetric]) else [pDe( pA ) for pA in [pArray1, pArray2]]

		pRep1_, pRep2_ = pls(XP, YP)

		pRep1_, pRep2_ = discretize(pRep1_), discretize(pRep2_)

		fAssociation_ = pMe(pRep1_, pRep2_) 

		aDist.append(fAssociation_)

	fPercentile = percentileofscore(aDist, fAssociation, kind="strict")  # #source: Good 2000 
	# ## \frac{ \sharp\{\rho(\hat{X},Y) \geq \rho(X,Y) \} +1  }{ k + 1 }
	# ## k number of iterations, \hat{X} is randomized version of X 
	# ## PercentileofScore function ('strict') is essentially calculating the additive inverse (1-x) of the wanted quantity above 
	# ## consult scipy documentation at: http://docs.scipy.org/doc/scipy-0.7.x/reference/generated/scipy.stats.percentileofscore.html

	fP = ((1.0 - fPercentile / 100.0) * iterations + 1) / (iterations + 1)

	return fP, fAssociation, pRep1[0], pRep2[0]

def permutation_test_by_average(pArray1, pArray2, metric= "nmi", iterations=1000):

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
	pArrayPerm = numpy.array([ pFun(array([_permutation(x) for x in pArray1]), pArray2) for i in range(iterations) ])

	dPPerm = percentileofscore(pArrayPerm, dVal) / 100.0 	

	return dPPerm

def permutation_test(pArray1, pArray2):
	
	if config.decomposition in ['none','cca', 'pls',"pca", "dpca", "nlpca", "ica", "kpca","centroid-medoid","medoid","mean", "mca", "farthest"]:
		return permutation_test_by_representative(pArray1, pArray2)
	
	if config.decomposition in ["average"]:
		return permutation_test_by_average(pArray1, pArray2, metric=metric, iterations=iterations)


def g_test(pArray1, pArray2, metric, decomposition, iterations):
	if decomposition in ['cca', 'pls',"pca", "nlpca", "ica", "kpca"]:
		return g_test_by_representative(pArray1, pArray2, metric=metric, decomposition= decomposition, iterations=iterations)
	
	if decomposition in ["average"]:
		return g_test_by_average(pArray1, pArray2, metric=metric, iterations=iterations)
def parametric_test(pArray1, pArray2):
	pMe1 = lambda x, y:  cor(x, y, method="pearson", pval=True)
	pMe2 = lambda x, y:  cor(x, y, method="spearman", pval=True)
	pVal1 = [pMe1(i, j)[1] for i, j in itertools.product(pArray1, pArray2)]
	pVal2 = [pMe2(i, j)[1] for i, j in itertools.product(pArray1, pArray2)]
	return numpy.average(pVal1), numpy.average(pVal2)

def parametric_test_by_cca(pArray1, pArray2, iterations=1000):
	
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

def parametric_test_by_pls_pearson(pArray1, pArray2, iterations=1000):
	
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

def classify(value, breaks):
    for i in range(1, len(breaks)):
        if value < breaks[i]:
            return i
    return len(breaks) - 1
def jenks_discretize(values, n):
	breaks = jenkspy.jenks_breaks(values, int(n))
	values_in_bins = [ classify(value, breaks) for value in values]
	return values_in_bins
	 
def discretize(pArray, style = "equal-freq", data_type = None, number_of_bins=None, method=None, aiSkip=[]):
	
	"""
	This functio discretizes data and has two approach one for continuse data
	and one for categorical data and categories names start with 1 and 0 is uses for
	missind data(nans).
	
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
	>>> dx = discretize( x, number_of_bins = None, method = None, aiSkip = [1,3] )
	>>> dx
	array([[ 0.,  0.,  1.,  1.],
	       [ 1.,  1.,  1.,  0.],
	       [ 0.,  0.,  1.,  1.],
	       [ 0.,  0.,  0.,  1.]])
	>>> dy = discretize( y, number_of_bins = None, method = None, aiSkip = [1] )
	>>> dy 
	array([[ 1.,  1.,  0.,  0.],
	       [ 1.,  1.,  0.,  0.],
	       [ 0.,  0.,  1.,  1.],
	       [ 0.,  0.,  1.,  1.]])

	"""
	def _discretize_categorical(astrValues, number_of_bins=number_of_bins):
		if config.similarity_method in ['mic', 'dmic']:
					sys.exit('No categorical data is allowed with mic or dmic!')
		setastrValues = list(set(astrValues))
		dictA ={}
		for i, item in enumerate(setastrValues):
			if str(astrValues[i]) != 'NaN':
				dictA[item] = i+1
			else:
				#if str(astrValues[i]) == 'nan':
				#print  astrValues[i] 
				dictA[item]=  0
		#dictA[numpy.nan] = numpy.nan 
		result_discretized_data = []
		for i, item in enumerate(astrValues):
			result_discretized_data.append(dictA[item])
		return result_discretized_data	
	def _discretize_continuous(astrValues, number_of_bins=number_of_bins):
		#decide about the number of bins
		if number_of_bins == None:
			# Default to rounded sqrt(n) if no bin count requested
			number_of_bins = min(len(set(astrValues)), round(math.sqrt(len(astrValues)))) 
			if config.similarity_method == 'dmic':
				number_of_bins = number_of_bins*2
		elif number_of_bins == 0:
			number_of_bins = len(set(astrValues))
		else:
			number_of_bins = min(number_of_bins, len(set(astrValues)))

		# descritize the vector
		if len(set(astrValues)) <= number_of_bins:
			try:
				return rankdata(astrValues, method= 'dense')
			except:
				print ("An exception happend with discretizing continuose data!!!")
				#return _discretize_categorical(astrValues, number_of_bins=number_of_bins)
		else:							
			#try:
			if config.strDiscretizing == 'equal-freq':
				order = rankdata(astrValues, method= 'min')# ordinal
			elif config.strDiscretizing == 'hclust':
				#astrValues = distance.remove_pairs_with_a_missing(astrValues, astrValues)
				distanceMatrix = abs(numpy.array([astrValues],  dtype= float).T-numpy.array([astrValues], dtype= float))
				order = fcluster(Z=linkage(distanceMatrix, method=config.linkage_method), t=number_of_bins,criterion='maxclust')
				#print order, number_of_bins
				return order
			elif config.strDiscretizing == 'jenks':
				order = jenks_discretize(astrValues, number_of_bins)
				return order
			'''except:
				print ("An exception happend with discretizing continuose data!!!")
			'''	#return _discretize_categorical(astrValues, number_of_bins=number_of_bins)

		discretized_result = [None] * len(astrValues)
		bins_size = numpy.ceil(len(astrValues)/float(number_of_bins))
		#print "bin size: ", bins_size, "len of the array", len(astrValues)
		#print (astrValues)
		for i in range(len(astrValues)):
			discretized_result[i] = int((order[i]-1) / bins_size)
		discretized_result = rankdata(discretized_result, method= 'dense')
		for i in range(len(astrValues)):
			if str(astrValues[i]) == 'NaN':
				discretized_result[i]= 0
		#print astrRet
		return discretized_result

	# iRow1, iCol = pArray.shape
	if number_of_bins == None:
		number_of_bins= config.NBIN
	discretized_data = [] 
	if isinstance(pArray[0], list) or (hasattr(pArray[0], "__len__") and (not isinstance(pArray[0], str))):
		for i, line in enumerate(pArray):
			if i in aiSkip:
				#print "SKIPE LINE!"
				discretized_data.append(line)
			elif data_type!= None and data_type[i] == 'LEX':
				discretized_data.append(array(_discretize_categorical(line, number_of_bins)))
			else:
				discretized_data.append(_discretize_continuous(line, number_of_bins))
	else:
		try:
			discretized_data = _discretize_continuous(pArray, number_of_bins)
		except:
			discretized_data = _discretize_categorical(pArray, number_of_bins)

	return array(discretized_data)

def _discretize_continuous_old_R(astrValues, number_of_bins=None, style =None):
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
	if style in ['jenks', 'kmeans', 'hclust']:
		try:
			dataFrame1 = pd.DataFrame(astrValues, dtype= float)
			#print dataFrame1[0]
			ro.globalenv['number_of_bins'] = number_of_bins 
			ro.globalenv['v'] =  com.convert_to_r_dataframe(dataFrame1)[0]
			ro.r('clI <- classIntervals(v, n = number_of_bins, style = style)')
			ro.r(' descretized_v <- findCols(clI)')
			astrRet = ro.globalenv['descretized_v']
			return astrRet
		except Exception as err:
			print(traceback.format_exc())
			
			print ("Discretizing as exeception in ClassInt happend!!!")
			try:
				order = rankdata(astrValues, method= 'min')
			except:
				return astrValues

def _discretize_continuous_old(astrValues, number_of_bins=None): 
		
	if number_of_bins == None:
		# Default to rounded sqrt(n) if no bin count requested
		number_of_bins = round(math.sqrt(len(astrValues)))  # **0.5 + 0.5)
	elif number_of_bins == 0:
		number_of_bins = len(astrValues)
	else:
		number_of_bins = min(number_of_bins, len(set(astrValues)))
	#print number_of_bins	
	# This is still a bit buggy since ( [0, 0, 0, 1, 2, 2, 2, 2], 3 ) will exhibit suboptimal behavior
	aiIndices = sorted(list(range(len(astrValues))), cmp=lambda i, j: cmp(astrValues[i], astrValues[j]))
	astrRet = [None] * len(astrValues)
	#print "aiIndices:", aiIndices
	#print "astrRet:", astrRet
	iPrev = 0
	for i, iIndex in enumerate(aiIndices):
		# If you're on a tie, you can't increase the bin number
		# Otherwise, increase by at most one
		iPrev = astrRet[iIndex] = iPrev if (i and (astrValues[iIndex] == astrValues[aiIndices[i - 1]])) else \
			min(iPrev + 1, int(number_of_bins * i / float(len(astrValues))))
	return astrRet	

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

	pIndices = itertools.combinations(list(range(pArray.shape[0])), 2)

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

	pIndices = itertools.product(list(range(iRow1)), list(range(iRow2)))

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

def estimate_pvalue(x, null_samples, regenrate_GPD_flag = False):
	"""
	Estimates the p-value, given the observed test statistic x and a set of
	samples from the null distribution.
	"""
	
	# Algorithm proposed in Knijnenburg2009
	
	if x == 0:
		return 1.0
	# Get M, the number of null samples greater than x
	M = len([1 for v in null_samples if v > x])  # or v >= x 
	N = len(null_samples)
	# Use the ECDF to approximate p-values if M > 10
	if M >= 10:# or N < 100 or M >= .1 * N:
		return float(M)/float(N)

	# Estimate the generalized pareto distribtion from tail samples
	if not config.use_one_null_dist or config.gp == None or regenrate_GPD_flag:
		try:
			#null_samples = list(set(null_samples))
			(gp, Nexc) = estimate_tail_gpd(null_samples)
			config.gp  = gp
			config.Nexc = Nexc
		except ArithmeticError as ValueError:
			return float(M)/float(N)
	else:
		(gp, Nexc) = (config.gp, config.Nexc)
	# GPD estimate of the actual p-value
	
	# Check if the result of the survival function is na then genrate more null samples
	sf_result =  gp.sf(x)
	#if sf_result == 1:
		#print sf_result, N, Nexc
	if math.isnan(float(sf_result)) or sf_result == 1.0:
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

def nonparametric_test_pvalue(X, Y, similarity_method = None,  alpha_cutoff = 0.05):
	"""
	Performs a permutation test of the significance of x, given the function
	to sample the null distribution null_fun.
	This function will exit early if it becomes clear that the p-value will
	be greater than alpha_cutoff. In this case, the current approximation
	of the p-value is returned.
	input:
	 X: first vector
	 Y: second vector
	 similarity_method: the similarity method to be used
	"""
	
	# calculate imilarity between to orginal features
	if similarity_method == None:
		similarity_method = config.similarity_method
		
	pMe = distance.c_hash_metric[similarity_method]
	sim_score = pMe(X, Y)
	sim_score = math.fabs(sim_score)
	# The number of null samples to start with
	start_samples =  100
	# Number of null samples to gather in each round
	sample_increments = 50
	# Maximum number of null samples, at which point the GPD approximation
	# is used
	max_samples =  config.iterations
	
	# Sample the null distribution until we've got enough to estimate the tail
	# or if we're sure that the actual p-value is greater than the alpha cutoff
	if config.use_one_null_dist and len(config.nullsamples) == 0:
		nullsamples = [null_fun(X, Y) for val in range(0, max_samples)]
		config.nullsamples = nullsamples
	elif not config.use_one_null_dist: #or not config.use_one_null_dist:
		nullsamples = [null_fun(X, Y) for val in range(0, start_samples)]
		while len(nullsamples) < max_samples and prob_pvalue_lt_samples(config.q, sim_score, nullsamples) > .05 * 1.0/len((X)* len(Y)):
			#print("Gathering more.. N = %d; P(p<%f) = %.2f" % (len(nullsamples), config.q, prob_pvalue_lt_samples(config.q, x, nullsamples)))
			nullsamples = [null_fun(X, Y) for val in range(0,sample_increments)] + nullsamples 
		#nullsamples = [null_fun(X, Y) for val in range(0,max_samples)]

		config.nullsamples = nullsamples
	#print("Finished gathering: N = %d; P(p<%f) = %f" % (len(nullsamples), alpha_cutoff, prob_pvalue_lt_samples(alpha_cutoff, x, nullsamples)))
	# Estimate the p-value from the current set of samples
	return estimate_pvalue(x = sim_score, null_samples = config.nullsamples)


	
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
