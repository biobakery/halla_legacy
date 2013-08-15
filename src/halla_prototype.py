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
import os 

## statistics packages 

from datum import discretize 
import numpy as np
import scipy as sp
from numpy import array 
import sklearn.decomposition
import matplotlib 
matplotlib.use("Agg") #disable X-windows display backend 
from sklearn.decomposition import PCA #remember that the matrix is X = (n_samples,n_features)
import csv 
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree, leaves_list
import pylab as pl 
import random  
from distance import mi, sym_mi 
from numpy.random import normal 
from scipy.misc import * 
from scipy.stats import kruskal, ttest_ind, ttest_1samp 
import pandas as pd 

class HAllA():

	hashMethods = {"id":			lambda x: x.id, 
			"l2":			lambda x,y: np.linalg.norm(x-y),
			"ttest": 		lambda x,y: ttest_ind(x,y), 
			"kruskal":		lambda x,y: kruskal(x,y),

					}  

	def __init__( self, *ta ): 
		self.htest = ttest_ind
		self.distance = sym_mi 
		self.rep = None 
		self.meta_array = array( ta )
		self.meta_discretize = None 
		self.meta_linkage = None 
		self.meta_distance = None 
		self.directory = None 

	def set_directory( self, strDir ):
		self.directory = strDir 
		#return self.directory 

	@staticmethod 
	def m( pArray, pFunc, axis = 0 ):
		""" 
		Maps pFunc over the array pArray 
		"""

		if bool(axis): 
			pArray = pArray.T
			# Set the axis as per numpy convention 

		if type(pFunc) == np.ndarray:
			return pArray[pFunc]
		else: #generic function type
			return array( [pFunc(item) for item in pArray] ) 

	def _discretize( self ):
		self.meta_discretize = self.m( self.meta_array, discretize )
		return self.meta_discretize 

	def _distance_matrix( self ):
		self.meta_distance =  self.m( self.meta_array, lambda x: pdist( x, metric=self.distance ) )
		return self.meta_distance 

	def _linkage( self ):
		self.meta_linkage = self.m( self.meta_array, linkage ) 
		return self.meta_linkage 

	def _medoid( self, pArray, iAxis = 0, pMetric = "l2" ):
		"""
		Input: numpy array 
		Output: float
		
		For lack of better way, compute centroid, then compute medoid 
		by looking at an element that is closest to the centroid. 

		Can define arbitrary metric passed in as a function to pMetric 

		"""

		d = hashMethods[pMetric]

		pArray = ( pArray.T if bool(iAxis) else pArray  ) 

		mean_vec = np.mean(pArray, 0) 
		
		pArrayCenter = pArray - ( mean_vec * np.ones(pArray.shape) )

		return pArray[np.argsort( map( np.linalg.norm, pArrayCenter) )[0],:]

	def _representative( self, pArray, pMethod = None ):
		hash_method = {None: get_medoid}
		return hash_method[pMethod]( pArray )

	def _reduce_tree( self, pClusterNode, pFunction = lambda x: x.id, aOut = [] ):
		func = pFunction

		if pClusterNode.is_leaf():
			return ( aOut + [func(pClusterNode)] )
		else:
			return _reduce_tree( pClusterNode.left, func, aOut ) + \
				_reduce_tree( pClusterNode.right, func, aOut ) 


	def _plot_dendrogram( self ):
		for i, pArray in enumerate( self.meta_linkage ):
			
			pl.clf()
			pl.figure(i)
			dendrogram( pArray ) 
			pl.title( str( self.distance ) + " " + str(i) )
			pl.savefig( self.directory + str(i) + ".pdf" ) 
				
 

		"""	
		pl.figure(1)

		pl.subplot(211)
		dendrogram( Z1 )
		pl.title("sym_mi 1")

		pl.subplot(212)
		dendrogram( Z2 )
		pl.title("sym_mi 2")

		pl.figure(2)

		pl.subplot(211)
		dendrog ram( Z11 )
		pl.title("euc 1")

		pl.subplot(212)
		dendrogram( Z22 )
		pl.title("euc 2")

		pl.show()  
		"""
	def run( self ):
		print self._discretize()
		self._distance_matrix()
		self._linkage()
		self._plot_dendrogram()
		
if __name__ == "__main__":
	c_strOutputPath = "/home/ysupmoon/Dropbox/halla/output/" 
	
	c_DataArray1 = np.array([[normal() for x in range(100)] for y in range(20)])
	c_DataArray2 = np.array([[normal() for x in range(100)] for y in range(20)]) 

	CH = HAllA( c_DataArray1, c_DataArray2 )

	CH.set_directory( c_strOutputPath )
	
	#CH._discretize() 
	#CH._distance_matrix()
	#CH._linkage()
	#CH._plot_dendrogram() 

	CH.run() 

