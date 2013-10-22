#!/usr/bin/env python 
'''
Hiearchy module, used to build trees and other data structures.
Handles clustering and other organization schemes. 
'''
 
## structural packages 

import sys 
import re 
import math 
import itertools
from itertools import product 

## halla-specific modules 

from distance import mi, l2 
from stats import discretize,pca, mca 

## statistics packages 

import numpy as np
from numpy import array 
import scipy as sp
import sklearn.decomposition
from sklearn.decomposition import PCA #remember that the matrix is X = (n_samples,n_features)
import csv 
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree, leaves_list
import pylab as pl 
import random  
from numpy.random import normal 
from scipy.misc import * 
import pandas as pd 



## this should be in dimensionality reduction script 
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

	print pArray.shape 

	mean_vec = np.mean(pArray, 0) 
	
	pArrayCenter = pArray - ( mean_vec * np.ones(pArray.shape) )

	return pArray[np.argsort( map( np.linalg.norm, pArrayCenter) )[0],:]


## this should be in dimensionality reduction script 
def get_representative( pArray, pMethod = None ):
	hash_method = {None: get_medoid, "pca": pca, "mca": mca}
	return hash_method[pMethod]( pArray )

def hclust( pArray, pdist_metric = mi, cluster_metric = l2, bTree = False ):
	#returns linkage matrix 
	pdist_data = pdist( pArray, metric= pdist_metric )  
	linkage_data = linkage( pdist_data, metric=l2 ) 
	return to_tree( linkage_data ) if bTree else linkage_data 

## this is the most useful function 
def reduce_tree( pClusterNode, pFunction = lambda x: x.id, aOut = [] ):
	func = pFunction

	if pClusterNode.is_leaf():
		return ( aOut + [func(pClusterNode)] )
	else:
		return reduce_tree( pClusterNode.left, func, aOut ) + \
			reduce_tree( pClusterNode.right, func, aOut ) 

def traverse_by_layer( pClusterNode1, pClusterNode2, pArray1, pArray2, pFunction ):
	"""
	Depends: reduce_tree 

	Useful function for doing all-against-all comparison between nodes in each layer 

	traverse two trees at once, applying function `pFunction` to each layer pair 

	latex: $pFunction: data1 \times data2 \rightarrow \mathbb{R}$
	"""

	def _traverse_helper( apParents, iLevel = 0, iStop = None ):
		
		return [(iLevel, reduce_tree(p)) for p in apParents ] + _traverse_helper( [ q.left for q in apParents ] + [ r.right for r in apParents ], iLevel = iLevel+1 ) 

	def _get_min( tData ):
		
		return np.min([i[0] for i in tData)

	def _get_layer( tData, iLayer ):
		"""
		Get output from `_traverse_helper` and parse 
		"""

		dummyOut = [] 

		for couple in tData:
			if couple[0] < iLayer:
				continue 
			elif couple[0] == iLayer:
				dummyOut.append(couple[1])
				tData = tData[1:]
			else:
				break
		return dummyOut, tData 

	tData1, tData2 = [ _transverse_helper( [pC] ) for pC in [pClusterNode1, pClusterNode2] ]

	iMin = np.min( [_get_min(tData1), _get_min(tData2)] ) 

	for iLevel in range(iMin):
		pLayer1, pLayer2 = _get_layer( tData1, iLevel )[0], _get_layer( tData2, iLevel )[0]
		iLayer1, iLayer2 = len(pLayer1), len(pLayer2)

		for i,j in product( range(iLayer1), range(iLayer2) ):
			yield (i,j), pFunction( pArray1[:,i], pArray2[:,j] ) 


def htest():
	pass 

#==========================================================================#
# DATA STRUCTURES 
#==========================================================================#

class Tree():
	''' 
	A hierarchically nested structure containing nodes as
	a basic core unit	
	'''	

	def __init__(self):
		self.m_pData = None 
		self.m_arrayChildren = []
		self.m_iLayer = 0 
    
	def pop(self):
		# pop one of the children, else return none, since this amounts to killing the singleton 
		if self.m_arrayChildren:
			return self.m_arrayChildren.pop()
		
	def is_leaf(self):
		return bool(not(self.m_pData and self.m_arrayChildren))

	def is_degenerate(self):
		return ( not(self.m_pData) and not(self.m_arrayChildren) )			

	def add_child(self, node_object):
		self.m_arrayChildren.append(node_object)
		
	def get_children(self): 
		return self.m_arrayChildren
	
	def get_child(self,iIndex=None):
		return self.m_arrayChildren[iIndex or 0]
	
	def add_data(self, pDatum):
		self.m_pData = pDatum 
		return self 
	
	def get_data(self):
		return self.m_pData 


#==========================================================================#
# METHODS  
#==========================================================================#


#==========================================================================#
# META
#==========================================================================#

class Gardener():
	"""
	A gardener object is a handler for the different types of hierarchical data structures ("trees")
	Can collapse and manipulate data structures and wrap them in different objects, depending on the 
	context. 
	"""

	@staticmethod 
	def PlantTree():
		"""
		Input: halla.Dataset object 
		Output: halla.hierarchy.Tree object 
		"""

		return None 

	def __init__(self):
		pass 

	def next(self):
		'''
		return the data of the tree, layer by layer
		input: None 
		output: a list of data pointers  
		'''
		
		if self.is_leaf():
			return Exception("Empty Tree")

		elif self.m_pData:
			pTmp = self.m_pData 
			self.m_queue.extend(self.m_arrayChildren)
			self.m_arrayChildren = None 
			self = self.m_queue 
			assert( self.is_degenerate() )
			return pTmp 	
		
		else:
			assert( self.is_degenerate() )
			aOut = [] 
	
			for pTree in self.m_queue:
				aOut.append( pTree.get_data() )

	
		if self.m_queue:
			self = self.m_queue.pop()
		elif self.m_arrayChildren:
			pSelf = self.m_arrayChildren.pop() 
			self.m_queue = self.m_arrayChildren
			self = pSelf 
		return pTmp 


#==========================================================================#
# OBJECT WRAPPERS
#==========================================================================#

