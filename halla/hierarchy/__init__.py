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

## halla-specific modules 

import halla 
import halla.stats 

from halla.distance import mi, l2, absl2, norm_mid
from halla.stats import discretize,pca, mca, bh, permutation_test_by_representative 

## statistics packages 

import numpy 
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


#==========================================================================#
# DATA STRUCTURES 
#==========================================================================#

class Tree():
	''' 
	A hierarchically nested structure containing nodes as
	a basic core unit	

	A general object, tree need not be 2-tree 
	'''	

	def __init__(self, data = None):
		self.m_pData = data 
		self.m_arrayChildren = []
    
	def pop(self):
		# pop one of the children, else return none, since this amounts to killing the singleton 
		if self.m_arrayChildren:
			return self.m_arrayChildren.pop()

	def l(self):
		return self.left()

	def r(self):
		return self.right()

	def left(self):
		#assert( len(self.m_arrayChildren) == 2 )
		return self.get_child( iIndex = 0)
	
	def right(self):
		#assert( len(self.m_arrayChildren) == 2 )
		return self.get_child( iIndex = 1)

	def is_leaf(self):
		return bool(not(self.m_pData and self.m_arrayChildren))

	def is_degenerate(self):
		return ( not(self.m_pData) and not(self.m_arrayChildren) )			

	def add_child(self, data):
		if not isinstance( data, Tree ):
			pChild = Tree( data )
		else:
			pChild = data 
		self.m_arrayChildren.append( pChild )
		
	def add_children(self, aData):
		for item in aData:
			self.add_child( item )

	def get_children(self): 
		return self.m_arrayChildren
	
	def get_child(self,iIndex=None):
		return self.m_arrayChildren[iIndex or 0] if self.m_arrayChildren else None 
	
	def add_data(self, pDatum):
		self.m_pData = pDatum 
		return self 
	
	def get_data(self):
		return self.m_pData 


#==========================================================================#
# METHODS  
#==========================================================================#

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
	
	if not isinstance( apParents, list ):
		apParents = [apParents]

	if (iStop and (iLevel > iStop)) or not(apParents):
		return [] 
	else:	
		return [(iLevel, reduce_tree(p)) for p in apParents ] + reduce_tree_by_layer( [ q.left for q in filter( lambda x: not(x.is_leaf()) , apParents ) ] + 
			[ r.right for r in filter( lambda x: not(x.is_leaf()) , apParents ) ], iLevel = iLevel+1 ) 

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

		for i,j in itertools.product( range(iLayer1), range(iLayer2) ):
			dummyOut.append( [ (i,j), pFunction( pArray1[:,i], pArray2[:,j] ) ] )


def depth_tree( pClusterNode, bLayerform = False ):
	"""
	Get the depth of a tree 
	"""
	
	aOut = reduce_tree_by_layer( [pClusterNode] ) if not bLayerform else pClusterNode 
	return max(zip(*aOut)[0])+1

def get_children( pClusterNode ):
	if not pClusterNode.is_leaf( ):
		return [pClusterNode.left, pClusterNode.right]

def get_layer( atData, iLayer = None, bTuple = False, bIndex = False ):
	"""
	Get output from `reduce_tree_by_layer` and parse 

	Input: atData = a list of (iLevel, list_of_nodes_at_iLevel), iLayer = zero-indexed layer number 
	"""

	if not atData:
		return None 

	dummyOut = [] 

	if not isinstance( atData, list ): 
		atData = reduce_tree_by_layer( atData )

	if not iLayer:
		iLayer = atData[0][0]

	for couple in atData:
		if couple[0] < iLayer:
			continue 
		elif couple[0] == iLayer:
			dummyOut.append(couple[1])
			atData = atData[1:]
		else:
			break

	if bIndex:
		dummyOut = (iLayer, dummyOut)
	return ( dummyOut, atData ) if bTuple else dummyOut 

def cross_section_tree( pClusterNode, method = "uniform", cuts = "complete" ):
	"""
	Returns cross sections of the tree depths in layer_form
	"""
	aOut = [] 

	layer_form = reduce_tree_by_layer( pClusterNode )
	iDepth = depth_tree( layer_form, bLayerform = True )
	pCuts = halla.stats.uniform_cut( range(iDepth), iDepth if cuts == "complete" else cuts )
	aCuts = [x[0] for x in pCuts]

	for item in layer_form:
		iDepth, pBag = item
		if iDepth in aCuts:
			aOut.append( (iDepth,pBag) )

	return aOut 

def couple_tree( pClusterNode1, pClusterNode2, method = "uniform", linkage = "min" ):
	"""
	Couples two trees to make a hypothesis tree 

	Parameters
	------------
	pClusterNode1, pClusterNode2 : ClusterNode objects
	method : str 
		{"uniform", "2-uniform", "log-uniform"}
	linkage : str 
		{"max", "min"}

	Returns
	--------------
	lf : layer_form 
	"""

	#implement min version first 

	
	aOut = [] 

	layer_form1, layer_form2 = pClusterNode1, pClusterNode2 
	depth1, depth2 = depth_tree( layer_form1 ), depth_tree( pClusterNode2 )

	for i in range(depth1):
		pBags1 = get_layer( layer_form1, i )
		pBags2 = get_layer( layer_form2, i )

		if not pBags1 or not pBags2:
			break 
		else:

			pP = itertools.product( pBags1, pBags2 )
			aOut.append( [(item[0],item[1]) for item in pP] )

	return aOut 


def all_against_all( pClusterNode1, pClusterNode2, pArray1, pArray2, method = "permutation_test_by_representative", metric = "norm_mi"):
	"""
	Get output from couple_tree and perform all_against_all 
	"""

	aOut = [] 

	phashMethods = {"permutation_test_by_representative" : halla.stats.permutation_test_by_representative, 
					"permutation_test_by_average" : halla.stats.permutation_test_by_average,
					"parametric_test" : halla.stats.parametric_test}
	strMethod = method 

	pMethod = phashMethods[strMethod]

	pCouple = couple_tree( pClusterNode1, pClusterNode2 )
	for aLayer in pCouple:
		for pPair in aLayer:
			pOne, pTwo = map( array, pPair )
			aOut.append( [pPair, pMethod( pArray1[pOne], pArray2[pTwo] )] )

	return aOut 

def naive_all_against_all( pArray, metric = "norm_mi" ):

	iRow = len(pArray)

	X = numpy.zeros((iRow,iRow))

	for i,j in itertools.product( range(iRow), range(iRow) ):
		fVal = halla.distance.norm_mi( pArray[i], pArray[j] )
		X[i][j] = fVal ; X[j][i] = fVal 

	return X 


#=======================================#
#### Old code; clean later 
#=======================================#


def old_couple_tree( apClusterNode1, apClusterNode2, method = "uniform", linkage = "min", pTree = Tree() ):
	"""
	Couples two data trees to produce a hypothesis tree 

	Parameters
	------------
	pClusterNode1, pClusterNode2 : ClusterNode objects
	method : str 
		{"uniform", "2-uniform", "log-uniform"}
	linkage : str 
		{"max", "min"}

	Returns
	-----------
	tH : halla.Tree object 

	Examples
	----------------
	"""

	#pTreeNew = pTree 
	pMethod = method 
	pLinkage = linkage 

	def _decider( pClusterNode1, pClusterNode2, linkage = "min" ):
		if linkage == "max": #max linkage 
			if pClusterNode1.is_leaf() and pClusterNode2.is_leaf():
				pOut1, pOut2 = [], [] 
			elif pClusterNode1.is_leaf() and not pClusterNode2.is_leaf():
				pOut1, pOut2 = [pClusterNode1], get_children( pClusterNode2 ) 
			elif not pClusterNode1.is_leaf() and pClusterNode2.is_leaf():
				pOut1, pOut2 = get_children( pClusterNode1 ), [pClusterNode2]
			else:
				pOut1, pOut2 = get_children( pClusterNode1 ), get_children( pClusterNode2 )
			return pOut1, pOut2 
		else: # minimum linkage by default 
			if pClusterNode1.is_leaf() or pClusterNode2.is_leaf(): #if one of them is a leaf, then stop the process
				return [],[] 
			else:
				return get_children( pClusterNode1 ), get_children( pClusterNode2 )

	if not isinstance( apClusterNode1, list ):
		apClusterNode1 = [apClusterNode1]
	if not isinstance( apClusterNode2, list ):
		apClusterNode2 = [apClusterNode2]

	if not apClusterNode1 and not apClusterNode2:
		return pTree
	else:
		aChildren = [] 
		pP = itertools.product( apClusterNode1, apClusterNode2 )
		for i,j in pP:
			#the first has only has one instance
			aIndices1, aIndices2 = get_layer( i ), get_layer( j )
			#pTreeNew.add_data( (aIndices1, aIndices2) )
			pTreeNew = Tree()
			pTreeNew.add_data( (aIndices1, aIndices2) )
			apClusterNodeNew1, apClusterNodeNew2 = _decider( i,j, linkage = pLinkage )
			
			aChildren += [old_couple_tree( apClusterNodeNew1, apClusterNodeNew2, method = pMethod, linkage = pLinkage )]


		return aChildren 
		#pTree.add_children( aChildren )
		
		#pTreeNew.add_child( couple_tree( apClusterNodeNew1, apClusterNodeNew2, method= pMethod, linkage= pLinkage, pTree = pTreeNew ) ) 
		

def old_one_against_one( pClusterNode1, pClusterNode2, pArray1, pArray2 ):
	"""

	one_against_one hypothesis testing for a particular layer 
	
	Input: pClusterNode1, pClusterNode2, pArray1, pArray2

	Output: aiIndex1, aiIndex2, pVal
	
	"""

	aiIndex1, aiIndex2 = reduce_tree( pClusterNode1 ) , reduce_tree( pClusterNode2 )

	pData1, pData2 = pArray1[array(aiIndex1)], pArray2[array(aiIndex2)]

	return aiIndex1, aiIndex2, permutation_test_by_representative( pData1, pData2 )


def old_all_against_all( apClusterNode1, apClusterNode2, pArray1, pArray2 ):
	""" 
	Perform all-against-all per layer 

	Input: apClusterNode1, apClusterNode2, pArray1, pArray2

	Output: a list of ( (i,j), (aiIndex1, aiIndex2, pVal) )
	"""

	dummyOut = [] 

	iC1, iC2 = map( len, [apClusterNode1, apClusterNode2] )

	for i,j in itertools.product(range(iC1), range(iC2)):
		dummyOut.append( ( (i,j), one_against_one( apClusterNode1[i], apClusterNode2[j], pArray1, pArray2 ) ) )

	return dummyOut 

def old_recursive_all_against_all( apClusterNode1, apClusterNode2, pArray1, pArray2, pOut = [], pFDR = bh ):
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

"""
	def _all_against_all( iLayer1, iLayer2, pNodes1, pNodes2 ):
		
		pT = Tree() 
		pP = itertools.product( pNodes1, pNodes2 )
		
		

		for tNodes in pP:
			pT.add_child( ((iLayer1,iLayer2), tNodes)  )

		return pT 



	aIndices = [] 
	aBags = [] 

	hashMethods = {"uniform": halla.stats.uniform_cut }

	layer_form1 = cross_section_tree( pClusterNode1, method=method )
	layer_form2 = cross_section_tree( pClusterNode2, method=method )

	lf1, lf2 = layer_form1, layer_form2 

	#depth1, depth2 = depth_tree( layer_form1, bLayerform = True ), depth_tree( pClusterNode2, bLayerform = True )

	bOne = ( depth1 >= depth2 ) #is the first layer_form greater in depth?
	#lf_big = layer_form1 if bOne else layer_form2
	#lf_small = layer_form1 if not bOne else layer_form2 

	#minDepth, maxDepth = min( [depth1, depth2] ), max( [depth1, depth2] )
	
	## First perform the min formulation 

	bStop = False 

	pNew1, pNew2 = lf1, lf2 

	while not bStop:
		pLayerOut1 = get_layer( pNew1, bLayerform = True, bTuple = True, bIndex = True )
		pLayerOut2 = get_layer( pNew2, bLayerform = True, bTuple = True, bIndex = True )
		if not pLayerOut1 or not pLayerOut2:
			bStop = True
			break 

		else:
			pOut1, pNew1 = pLayerOut1
			iIndex1, pData1 = pOut1
			
			pOut2, pNew2 = pLayerOut2 
			iIndex2, pData2 = pOut2 

			pT = _all_against_all( iIndex1, iIndex2, pData1, pData2 )



	iX, iY = depth_tree( pClusterNode1 ), depth_tree( pClusterNode2 )
	"""