#!/usr/bin/env python 
'''
Abstract distance module providing different notions of distance
'''
import sys
from abc import ABCMeta
import abc
import itertools
import math
from numpy import array, mean
import numpy
import scipy
import scipy.cluster
from scipy.spatial.distance import cdist
import scipy.stats

from sklearn.metrics import mutual_info_score, normalized_mutual_info_score, \
    adjusted_mutual_info_score, make_scorer
from scipy.spatial.distance import pdist, squareform
import numpy as np

#from numbapro import jit, float32

# from minepy import MINE

# mi-based distances from scikit-learn; (log e)-based (i.e. returns nats instead of bits)
#==========================================================================#
# CONSTANTS 
#==========================================================================#
c_hash_association_method_discretize = {"pearson": False,
										"spearman": False,
										"kw": False,
										"anova": False,
										"x2": False,
										"fisher": False,
                                        "mic": False,
                                        "dcor":False,
										"nmi": True,
										"mi": True,
                                        "dmic":True,
                                        "ami": True,
										}


class Distance:
	''' 
	abstract distance, handles numpy arrays (probably should support lists for compatibility issues)
	'''
	__metaclass__ = ABCMeta
		
	c_hashInvertFunctions = {"logistic": lambda x:1.0 / (1 + numpy.exp(-1.0 * x)), "flip": lambda x:-1.0 * x, "1mflip": lambda x: 1 - 1.0 * x }

	class EMetricType:
		NONMETRIC = 0
		METRIC = 1
	def __init__(self, c_array1, c_array2): 
		self.m_data1 = c_array1
		self.m_data2 = c_array2 

	def get_inverted_distance(self, strFunc=None):
		pFunc = Distance.c_hashInvertFunctions[strFunc or "flip"] 
		return pFunc(self.get_distance()) 
						
	@abc.abstractmethod
	def get_distance(self): pass 

	@abc.abstractmethod 
	def get_distance_type(self): pass
	
	
class EuclideanDistance(Distance):

	__metaclass__ = ABCMeta 

	def __init__(self, c_array1, c_array2):
		self.m_data1 = c_array1
		self.m_data2 = c_array2
		self.c_distance_type = Distance.EMetricType.METRIC  # CDistance.EMetricType.METRIC 

	def get_distance(self):
		return numpy.linalg.norm(self.m_data2 - self.m_data1) 

	def get_distance_type(self):
		return self.c_distance_type 	
	
class MutualInformation(Distance):
	"""
	Scikit-learn uses the convention log = ln
	Adjust multiplicative factor of log(e,2) 
	"""	

	__metaclass__ = ABCMeta 

	def __init__(self, c_array1, c_array2, bSym=False):
		self.m_data1 = c_array1 
		self.m_data2 = c_array2 
		self.bSym = bSym
		self.c_distance_type = Distance.EMetricType.NONMETRIC 
	
	def get_distance(self):
		# assert( numpy.shape(self.m_data1) == numpy.shape(self.m_data2) )
		return math.log(math.e, 2) * mutual_info_score(self.m_data1, self.m_data2) 	
	def get_distance_type(self):
		return self.c_distance_type 	

class NormalizedMutualInformation(Distance):
	"""
	normalized by sqrt(H1*H2) so the range is [0,1]
	"""	
	__metaclass__ = ABCMeta 

	def __init__(self, c_array1, c_array2):
		self.m_data1 = c_array1 
		self.m_data2 = c_array2 
		self.c_distance_type = Distance.EMetricType.NONMETRIC 
	
	def get_distance(self):
		# assert( numpy.shape(self.m_data1) == numpy.shape(self.m_data2) )
		return normalized_mutual_info_score(self.m_data1, self.m_data2)

	def get_distance_type(self):
		return self.c_distance_type 	
	
class AdjustedMutualInformation(Distance):
	"""
	adjusted for chance
	""" 
	
	__metaclass__ = ABCMeta 

	def __init__(self, c_array1, c_array2):
		self.m_data1 = c_array1 
		self.m_data2 = c_array2 
		self.c_distance_type = Distance.EMetricType.NONMETRIC 
	
	def get_distance(self):
		# assert( numpy.shape(self.m_data1) == numpy.shape(self.m_data2) )
		return adjusted_mutual_info_score(self.m_data1, self.m_data2)

	def get_distance_type(self):
		return self.c_distance_type 	

#==========================================================================#
# DISTANCE FUNCTIONS  
#==========================================================================#

def l2(pData1, pData2):
	"""
	Returns the l2 distance

	>>> x = numpy.array([1,2,3]); y = numpy.array([4,5,6])
	>>> l2(x,y)
	5.196152422706632
	"""
	return numpy.linalg.norm(pData1 - pData2)

def absl2(pData1, pData2):
	return numpy.abs(l2(pData1, pData2))

def mi(pData1, pData2):
	"""
	Static implementation of mutual information, returns bits 

	Parameters
	--------------
	pData1, pData2 : Numpy arrays

	Returns
	---------------
	mi : float 

	Examples
	--------------

	>>> x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])
	>>> y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0],[0.015625,0.125,0.421875,1.0]])
	>>> dx = halla.stats.discretize( x, iN = None, method = None, aiSkip = [1,3] )
	>>> dy = halla.stats.discretize( y, iN = None, method = None, aiSkip = [1] )
	>>> p = itertools.product( range(len(x)), range(len(y)) )
	>>> for item in p: i,j = item; print (i,j), mi( dx[i], dy[j] )
	(0, 0) 1.0
	(0, 1) 1.0
	(0, 2) 1.0
	(0, 3) 1.0
	(1, 0) 0.311278124459
	(1, 1) 0.311278124459
	(1, 2) 0.311278124459
	(1, 3) 0.311278124459
	(2, 0) 1.0
	(2, 1) 1.0
	(2, 2) 1.0
	(2, 3) 1.0
	(3, 0) 0.311278124459
	(3, 1) 0.311278124459
	(3, 2) 0.311278124459
	(3, 3) 0.311278124459
	"""

	return math.log(math.e, 2) * mutual_info_score(pData1, pData2)#return MutualInformation(pData1, pData2).get_distance()

def nmi(pData1, pData2):
	"""
	Static implementation of normalized mutual information 

	Examples
	---------------

	>>> x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])
	>>> y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0],[0.015625,0.125,0.421875,1.0]])
	>>> dx = halla.stats.discretize( x, iN = None, method = None, aiSkip = [1,3] )
	>>> dy = halla.stats.discretize( y, iN = None, method = None, aiSkip = [1] )
	>>> p = itertools.product( range(len(x)), range(len(y)) )
	>>> for item in p: i,j = item; print (i,j), nmi( dx[i], dy[j] )
	(0, 0) 1.0
	(0, 1) 1.0
	(0, 2) 1.0
	(0, 3) 1.0
	(1, 0) 0.345592029944
	(1, 1) 0.345592029944
	(1, 2) 0.345592029944
	(1, 3) 0.345592029944
	(2, 0) 1.0
	(2, 1) 1.0
	(2, 2) 1.0
	(2, 3) 1.0
	(3, 0) 0.345592029944
	(3, 1) 0.345592029944
	(3, 2) 0.345592029944
	(3, 3) 0.345592029944

	"""

	return normalized_mutual_info_score(pData1, pData2) #return NormalizedMutualInformation(pData1, pData2).get_distance() 

def ami(pData1, pData2):
	""" 
	Static implementation of adjusted distance 

	Examples
	-----------

	>>> x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])
	>>> y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0],[0.015625,0.125,0.421875,1.0]])
	>>> dx = halla.stats.discretize( x, iN = None, method = None, aiSkip = [1,3] )
	>>> dy = halla.stats.discretize( y, iN = None, method = None, aiSkip = [1] )
	>>> p = itertools.product( range(len(x)), range(len(y)) )
	>>> for item in p: i,j = item; print (i,j), ami( dx[i], dy[j] )
	(0, 0) 1.0
	(0, 1) 1.0
	(0, 2) 1.0
	(0, 3) 1.0
	(1, 0) 2.51758394487e-08
	(1, 1) 2.51758394487e-08
	(1, 2) 2.51758394487e-08
	(1, 3) 2.51758394487e-08
	(2, 0) 1.0
	(2, 1) 1.0
	(2, 2) 1.0
	(2, 3) 1.0
	(3, 0) -3.72523550982e-08
	(3, 1) -3.72523550982e-08
	(3, 2) -3.72523550982e-08
	(3, 3) -3.72523550982e-08

	"""

	return adjusted_mutual_info_score(pData1, pData2) #return AdjustedMutualInformation(pData1, pData2).get_distance()

# ## Changeset March 11, 2014
# ## NB: As a general rule, always use notion of "strength" of association; i.e. 0 for non-associated and 1 for strongly associated 
# ## This will alleviate confusion and enforce an invariance principle 
# ## For most association measures you can take 1-measure as a "distance" measure, but this should never be proscribed to a variable 
# ## The only place I can see use for this is in hierarchical clustering; otherwise, not relevant 

def pearson(X, Y):
    X = array(X)
    Y = array(Y)
    if X.ndim > 1: 
    	X = X[0]
    if Y.ndim > 1:
    	Y = Y[0]
    #X = [float(x) for x in X]
    #Y = [float(y) for y in Y]
    #print "pearson:", scipy.stats.pearsonr(X, Y)[0]
    return scipy.stats.pearsonr(X, Y)[0]
def spearman(X, Y):
    X = array(X)
    Y = array(Y)
    if X.ndim > 1: 
        X = X[0]
    if Y.ndim > 1:
        Y = Y[0]
    #X = [float(x) for x in X]
    #Y = [float(y) for y in Y]
    #print "pearson:", scipy.stats.pearsonr(X, Y)[0]
    return scipy.stats.spearmanr(X, Y)[0]
def mic (X, Y):
    try:
        import minepy
        from minepy import MINE
    except (ImportError):
        sys.exit("CRITICAL ERROR:2 Unable to import minepy package." + 
            " Please check your install.") 
    '''if X.ndim > 1: 
        X = X[0]
        #print X
    if Y.ndim > 1:
        Y = Y[0]
        '''
    mine = MINE(alpha=0.6, c=15)
    mine.compute_score(X, Y)
    # print "MIC:" , mine.mic()
    return mine.mic()

def distcorr(X, Y):
    """ Compute the distance correlation function
    
    >>> a = [1,2,3,4,5]
    >>> b = np.array([1,2,9,4,4])
    >>> distcorr(a, b)
    0.762676242417
    """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    
    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor

    
c_hash_metric = {"nmi": nmi,
				"mi": mi,
				"l2": l2,
                "ami":ami,
				"pearson": pearson,
                "spearman": spearman,
                "mic": mic,
                "dmic":mic,
                "dcor":distcorr
				}

# ## Visible and shareable to the outside world 

#==========================================================================#
# STRUCTURAL FUNCTIONS   
#==========================================================================#

def squareform(pArray):
	"""
	Switches back and forth between square and flat distance matrices 
	"""
	return scipy.cluster.hierarchy.distance.squareform(pArray)

def pdist(pArray, metric="euclidean"):
	"""
	Performs pairwise distance computation 

	Parameters
	------------

	pArray : numpy array 
	metric : str 

	Returns
	---------
	D : redundancy-checked distance matrix (flat)
	
	Examples
	-----------

	>>> x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])
	>>> y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0],[0.015625,0.125,0.421875,1.0]])
	>>> dx = halla.stats.discretize( x, iN = None, method = None, aiSkip = [1,3] )
	>>> dy = halla.stats.discretize( y, iN = None, method = None, aiSkip = [1] )
	>>> list( halla.distance.pdist( x, halla.distance.cord ) )
	[0.22540333075851648, 0.015625961302302871, 0.22540333075851648, 0.1358414347819068, 0.0, 0.1358414347819068]
	>>> list( halla.distance.pdist( y, halla.distance.cord ) )
	[0.10557280900008403, 0.0, 0.048630144207595816, 0.10557280900008414, 0.16134197652754312, 0.048630144207595594]
	>>> list( halla.distance.pdist( x, lambda u,v: halla.distance.cord(u,v, method="spearman") ) )
	[0.2254033307585166, 0.0, 0.2254033307585166, 0.2254033307585166, 0.0, 0.2254033307585166]
	>>> list( halla.distance.pdist( y, lambda u,v: halla.distance.cord(u,v, method="spearman") ) )
	[0.10557280900008414, 0.0, 0.0, 0.10557280900008414, 0.10557280900008414, 0.0]
	>>> list( halla.distance.pdist( dx, halla.distance.nmid ) )
	[0.65440797005578877, 0.0, 0.65440797005578877, 0.65440797005578877, 0.0, 0.65440797005578877]
	>>> list( halla.distance.pdist( dy, halla.distance.nmid ) )
	[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
	""" 
	pMetric = metric 
	return scipy.cluster.hierarchy.distance.pdist(pArray, pMetric)

