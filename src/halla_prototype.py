#!/usr/bin/env python 

"""
An object-oriented halla prototype 
Aim to be as self-containied as possible 
"""

## native python packages 

import itertools 

## structural packages 

import sys 
import re 

## statistics packages 

from datum import discretize 
import numpy as np
import scipy as sp
from numpy import array 
import sklearn.decomposition
from sklearn.decomposition import PCA #remember that the matrix is X = (n_samples,n_features)
import csv 
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree, leaves_list
import pylab as pl 
import random  
from distance import mi, sym_mi 
from numpy.random import normal 
from scipy.misc import * 
from scipy.stats import kruskal, ttest_ind, ttest_lsamp 
import pandas as pd 


class HAllA():

	hashMethods = {"id":			lambda x: x.id 
					"l2":			lambda x,y: np.linalg.norm(x-y)
					"ttest": 		lambda x,y: ttest_ind(x,y) 
					"kruskal":		lambda x,y: kruskal(x,y)

					}  

	def __init__( self, *ta ): 
		self.meta_array = array( ta )

	def _discretize( self ):
		pass

	def _distance_matrix( self ):
		pass 


	
	def _medoid( pArray, iAxis = 0, pMetric = "l2" ):
		"""
		Input: numpy array 
		Output: float
		
		For lack of better way, compute centroid, then compute medoid 
		by looking at an element that is closest to the centroid. 

		Can define arbitrary metric passed in as a function to pMetric 

		"""

		d = hashMethods[pMetric]

		pArray = ( pArray.T if bool(iAxis) else pArray  ) 

		print pArray.shape 

		mean_vec = np.mean(pArray, 0) 
		
		pArrayCenter = pArray - ( mean_vec * np.ones(pArray.shape) )

		return pArray[np.argsort( map( np.linalg.norm, pArrayCenter) )[0],:]

	def _representative( pArray, pMethod = None ):
		hash_method = {None: get_medoid}
		return hash_method[pMethod]( pArray )

	def _reduce_tree( pClusterNode, pFunction = lambda x: x.id, aOut = [] ):
		func = pFunction

		if pClusterNode.is_leaf():
			return ( aOut + [func(pClusterNode)] )
		else:
			return _reduce_tree( pClusterNode.left, func, aOut ) + \
				_reduce_tree( pClusterNode.right, func, aOut ) 


	def _plot_dendrogram():
 

		#aargs = sys.argv[1:]
		
		## plot stuff 
		pl.figure(1)

		pl.subplot(211)
		dendrogram( Z1 )
		pl.title("sym_mi 1")

		pl.subplot(212)
		dendrogram( Z2 )
		pl.title("sym_mi 2")

		pl.figure(2)

		pl.subplot(211)
		dendrogram( Z11 )
		pl.title("euc 1")

		pl.subplot(212)
		dendrogram( Z22 )
		pl.title("euc 2")

		pl.show()  

	def run():
		pass 
