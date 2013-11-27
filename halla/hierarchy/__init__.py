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

from halla.distance import mi, l2, absl2, norm_mid
from halla.stats import discretize,pca, mca, bh, permutation_test_by_representative 

## statistics packages 

import numpy as np
from numpy import array 
import scipy 
import scipy.stats 
import matplotlib 
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


def hclust( pArray, pdist_metric = norm_mid, cluster_method = "single", bTree = False ):
	"""
	Performs hierarchical clustering on an numpy array 

	Parameters
	------------
		pArray : numpy array 
		pdist_metric : str 
		cluster_method : str
		bTree : boolean 

	Returns
	-------------

		Z : numpy array or ClusterNode object 

	Examples
	---------------

	* Pearson correlation 1::

		from numpy import array 
		from scipy.cluster.hierarchy import dendrogram, linkage
		import scipy.cluster.hierarchy 
		import halla

		x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])

		lxpearson = halla.hierarchy.hclust( x, pdist_metric = halla.distance.cord )

		dendrogram(lxpearson)	

	.. plot::

		from numpy import array 
		from scipy.cluster.hierarchy import dendrogram, linkage
		import scipy.cluster.hierarchy 
		import halla

		x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])

		lxpearson = halla.hierarchy.hclust( x, pdist_metric = halla.distance.cord )

		dendrogram(lxpearson)	

	* Pearson correlation 2::

		from numpy import array 
		from scipy.cluster.hierarchy import dendrogram, linkage
		import scipy.cluster.hierarchy 
		import halla

		y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0], [0.015625,0.125,0.421875,1.0]])

		lypearson = halla.hierarchy.hclust( y, pdist_metric = halla.distance.cord )

		dendrogram(lypearson)
	
	.. plot::

		from numpy import array 
		from scipy.cluster.hierarchy import dendrogram, linkage
		import scipy.cluster.hierarchy 
		import halla

		y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0], [0.015625,0.125,0.421875,1.0]])

		lypearson = halla.hierarchy.hclust( y, pdist_metric = halla.distance.cord )

		dendrogram(lypearson)

	* Spearman correlation 1::

		from numpy import array 
		from scipy.cluster.hierarchy import dendrogram, linkage
		import scipy.cluster.hierarchy 
		import halla

		x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])

		lxspearman = halla.hierarchy.hclust( x, pdist_metric = lambda u,v: halla.distance.cord(u,v,method="spearman") )

		dendrogram(lxspearman)

	.. plot::

		from numpy import array 
		from scipy.cluster.hierarchy import dendrogram, linkage
		import scipy.cluster.hierarchy 
		import halla

		x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])

		lxspearman = halla.hierarchy.hclust( x, pdist_metric = lambda u,v: halla.distance.cord(u,v,method="spearman") )

		dendrogram(lxspearman)

	* Spearman correlation 2::

		from numpy import array 
		from scipy.cluster.hierarchy import dendrogram, linkage
		import scipy.cluster.hierarchy 
		import halla

		y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0], [0.015625,0.125,0.421875,1.0]])

		lyspearman = halla.hierarchy.hclust( y, pdist_metric = lambda u,v: halla.distance.cord(u,v,method="spearman") )

		dendrogram(lyspearman)

	.. plot::
		
		from numpy import array 
		from scipy.cluster.hierarchy import dendrogram, linkage
		import scipy.cluster.hierarchy 
		import halla

		y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0], [0.015625,0.125,0.421875,1.0]])

		lyspearman = halla.hierarchy.hclust( y, pdist_metric = lambda u,v: halla.distance.cord(u,v,method="spearman") )

		dendrogram(lyspearman)
	
	* Mutual Information 1::

		from numpy import array 
		from scipy.cluster.hierarchy import dendrogram, linkage
		import scipy.cluster.hierarchy 
		import halla

		x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])
		dx = halla.stats.discretize( x, iN = None, method = None, aiSkip = [1,3] )

		lxmi = halla.hierarchy.hclust( dx, pdist_metric = halla.distance.norm_mid )

		dendrogram(lxmi)

	.. plot::

		from numpy import array 
		from scipy.cluster.hierarchy import dendrogram, linkage
		import scipy.cluster.hierarchy 
		import halla

		x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])
		dx = halla.stats.discretize( x, iN = None, method = None, aiSkip = [1,3] )

		lxmi = halla.hierarchy.hclust( dx, pdist_metric = halla.distance.norm_mid )

		dendrogram(lxmi)

	* Mutual Information 2::

		from numpy import array 
		from scipy.cluster.hierarchy import dendrogram 
		import halla

		y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0], [0.015625,0.125,0.421875,1.0]])
		dy = halla.stats.discretize( y, iN = None, method = None, aiSkip = [1] )		

		lymi = halla.hierarchy.hclust( dy, pdist_metric = halla.distance.norm_mid )

		dendrogram( lymi )	

	.. plot::

		from numpy import array 
		from scipy.cluster.hierarchy import dendrogram 
		import halla

		y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0], [0.015625,0.125,0.421875,1.0]])
		dy = halla.stats.discretize( y, iN = None, method = None, aiSkip = [1] )		

		lymi = halla.hierarchy.hclust( dy, pdist_metric = halla.distance.norm_mid )

		dendrogram( lymi )	

	Notes 
	-----------

		This hclust function is not quite right for the MI case. Need a generic MI function that can take in clusters of RV's, not just single ones 
		Use the "grouping property" as discussed by Kraskov paper. 
	"""

	D = pdist( pArray, metric= pdist_metric )   
	Z = linkage( D ) 
	return to_tree( Z ) if bTree else Z 
	
def dendrogram( Z ):
	return scipy.cluster.hierarchy.dendrogram( Z )

def couple_tree( pClusterNode1, pClusterNode2, method = "unif" ):
	"""
	Couples two data trees to produce a hypothesis tree 

	Parameters:



	Returns:

	Examples: ::



	"""

def truncate_tree( apClusterNode, iSkip, iLevel = 0 ):
	"""
	Chop tree from root, returning smaller tree towards the leaves 

	Input: pClusterNode, iLevel 

	Output: list of ClusterNodes 

	"""

	if iLevel < iSkip:
		return truncate_tree( [p.right for p in apClusterNode] + [q.left for q in apClusterNode], iSkip, iLevel = iLevel+1 ) 

	elif iSkip == iLevel:
		if any(apClusterNode):
			return filter( lambda x: bool(x), apClusterNode )
	
		else:
			raise Exception("truncated tree is malformed--empty!")

def reduce_tree( pClusterNode, pFunction = lambda x: x.id, aOut = [] ):
	"""
	Recursive

	Input: pClusterNode, pFunction = lambda x: x.id, aOut = []

	Output: a list of pFunction calls (node ids by default)
	"""

	func = pFunction

	if pClusterNode.is_leaf():
		return ( aOut + [func(pClusterNode)] )
	else:
		return reduce_tree( pClusterNode.left, func, aOut ) + reduce_tree( pClusterNode.right, func, aOut ) 

def reduce_tree_by_layer( apParents, iLevel = 0, iStop = None ):
	"""

	Traverse one tree. 

	Input: apParents, iLevel = 0, iStop = None

	Output: a list of (iLevel, list_of_nodes_at_iLevel)
	"""
	
	if (iStop and (iLevel > iStop)) or not(apParents):
		return [] 
	else:	
		return [(iLevel, reduce_tree(p)) for p in apParents ] + reduce_tree_by_layer( [ q.left for q in filter( lambda x: not(x.is_leaf()) , apParents ) ] + 
			[ r.right for r in filter( lambda x: not(x.is_leaf()) , apParents ) ], iLevel = iLevel+1 ) 

def depth_tree( pClusterNode ):
	"""
	Get the depth of a tree 
	"""
	
	aOut = reduce_tree_by_layer( pClusterNode )
	return max(aOut)+1

def get_layer( atData, iLayer ):
	"""
	Get output from `reduce_tree_by_layer` and parse 

	Input: atData = a list of (iLevel, list_of_nodes_at_iLevel), iLayer = zero-indexed layer number 
	"""

	dummyOut = [] 

	for couple in atData:
		if couple[0] < iLayer:
			continue 
		elif couple[0] == iLayer:
			dummyOut.append(couple[1])
			atData = atData[1:]
		else:
			break
	return dummyOut, atData 


def one_against_one( pClusterNode1, pClusterNode2, pArray1, pArray2 ):
	"""

	one_against_one hypothesis testing for a particular layer 
	
	Input: pClusterNode1, pClusterNode2, pArray1, pArray2

	Output: aiIndex1, aiIndex2, pVal
	
	"""

	aiIndex1, aiIndex2 = reduce_tree( pClusterNode1 ) , reduce_tree( pClusterNode2 )

	pData1, pData2 = pArray1[array(aiIndex1)], pArray2[array(aiIndex2)]

	return aiIndex1, aiIndex2, permutation_test_by_representative( pData1, pData2 )


def all_against_all( apClusterNode1, apClusterNode2, pArray1, pArray2 ):
	""" 
	Perform all-against-all per layer 

	Input: apClusterNode1, apClusterNode2, pArray1, pArray2

	Output: a list of ( (i,j), (aiIndex1, aiIndex2, pVal) )
	"""

	dummyOut = [] 

	iC1, iC2 = map( len, [apClusterNode1, apClusterNode2] )

	for i,j in product(range(iC1), range(iC2)):
		dummyOut.append( ( (i,j), one_against_one( apClusterNode1[i], apClusterNode2[j], pArray1, pArray2 ) ) )

	return dummyOut 

def recursive_all_against_all( apClusterNode1, apClusterNode2, pArray1, pArray2, pOut = [], pFDR = bh ):
	"""

	Performs recursive all-against-all (the default HAllA routine) with fdr correction

	Input: apClusterNode1, apClusterNode2, pArray1, pArray2, pFDR

	Output: a list of ( (aiIndex1, pBag1), (aiIndex2, pBag2) )

	"""

	pOutNew = pOut 

	atAll = all_against_all( apClusterNode1, apClusterNode2, pArray1, pArray2 )

	print "This is all against all atAll"
	print atAll 

	atIJ, atOAO = zip(*atAll)
	aaN, aaM, aPVAL = zip(*atOAO)

	aBool = pFDR(aPVAL)
	print aPVAL

	if not any(aBool):
		print "END!"
		return pOutNew  
	else:
		"CONTINUE!"
		apC1, apC2, = [],[] 
		for k, couple in enumerate( atIJ ):
			i,j = couple  
			if aBool[k]: #if hypothesis was rejected, then go down to lower layers 

				#pNode1Left, pNode1Right = apClusterNode1[i].left, apClusterNode1[i].right
				#pNode2Left, pNode2Right = apClusterNode2[j].left, apClusterNode2[j].right

				#if pNode1Left.is_leaf or pNode1Right.is_leaf or pNode2Left.is_leaf or pNode2Right.is_leaf: 
				#	continue  
				#else: 
				#	print "Down!"
				
				for item in [ apClusterNode1[i].left, apClusterNode1[i].right ]:
					if not item.is_leaf():
						apC1.append( item )

				for item in [ apClusterNode2[j].left, apClusterNode2[j].right ]:
					if not item.is_leaf():
						apC2.append( item )
				
				#apC1.append( apClusterNode1[i].left ) ; apC1.append( apClusterNode1[i].right )
				#apC2.append( apClusterNode2[j].left ); apC2.append( apClusterNode2[j].right )

				#pOutNew.append( ( (aaN[k], apClusterNode1[i] ), (aaM[k], apClusterNode2[j] ) ) )
				pOutNew.append( (aaN[k],aaM[k]) )

		return recursive_all_against_all( apC1, apC2, pArray1, pArray2, pOut = pOutNew , pFDR = pFDR )


## I probably don't need this anymore? 

def traverse_by_layer( pClusterNode1, pClusterNode2, pArray1, pArray2, pFunction ):
	"""

	Useful function for doing all-against-all comparison between nodes in each layer 

	traverse two trees at once, applying function `pFunction` to each layer pair 

	latex: $pFunction: data1 \times data2 \rightarrow \mathbb{R}^k, $ for $k$ the size of the cross-product set per layer 

	Input: pClusterNode1, pClusterNode2, pArray1, pArray2, pFunction
	Output: (i,j), pFunction( pArray[:,i], pArray2[:,j])


	"""

	dummyOut = [] 

	def _get_min( tData ):
		
		return np.min([i[0] for i in tData])

	tData1, tData2 = [ reduce_tree_by_layer( [pC] ) for pC in [pClusterNode1, pClusterNode2] ]

	iMin = np.min( [_get_min(tData1), _get_min(tData2)] ) 

	for iLevel in range(iMin):
		pLayer1, pLayer2 = get_layer( tData1, iLevel )[0], get_layer( tData2, iLevel )[0]
		iLayer1, iLayer2 = len(pLayer1), len(pLayer2)

		for i,j in product( range(iLayer1), range(iLayer2) ):
			dummyOut.append( ( (i,j), pFunction( pArray1[:,i], pArray2[:,j] ) ) )

#==========================================================================#
# DATA STRUCTURES 
#==========================================================================#

## Probably don't need this anymore -- keep for now. 

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

