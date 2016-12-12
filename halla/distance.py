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
from . import config


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

def nmi(X, Y):
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
    if  not config.missing_char_category:
        test = [0 in [a, b] for a,b in zip(X,Y)]
        new_X= [a for a,b in zip (X,test) if ~b]
        new_Y= [a for a,b in zip (Y,test) if ~b]
        #print test
        #print new_X, new_Y
    else:
        new_X = X
        new_Y = Y 
    return normalized_mutual_info_score(new_X, new_Y) #return NormalizedMutualInformation(pData1, pData2).get_distance() 

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
    result = adjusted_mutual_info_score(pData1, pData2)
    return result 
 
def pearson(X, Y):
    X = array(X)
    Y = array(Y)
    if X.ndim > 1: 
    	X = X[0]
    if Y.ndim > 1:
    	Y = Y[0]
    return scipy.stats.pearsonr(X, Y)[0]
def spearman(X, Y):
    X = array(X)
    Y = array(Y)
    if X.ndim > 1: 
        X = X[0]
    if Y.ndim > 1:
        Y = Y[0]
    return scipy.stats.spearmanr(X, Y, nan_policy='omit')[0]
def mic (X, Y):
    try:
        import minepy
        from minepy import MINE
    except (ImportError):
        sys.exit("CRITICAL ERROR:2 Unable to import minepy package." + 
            " Please check your install.") 
    mine = MINE(alpha=0.6, c=15)
    mine.compute_score(X, Y)
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

	""" 
	pMetric = metric 
	return scipy.cluster.hierarchy.distance.pdist(pArray, pMetric)
def pDistance(x, y):
    pMetric = c_hash_metric[config.similarity_method]
    dist = math.fabs(1.0 - math.fabs(pMetric(x, y)))
    return  dist
