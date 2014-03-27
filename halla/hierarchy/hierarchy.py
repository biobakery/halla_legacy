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

from halla.distance import mi, l2, absl2, norm_mi
from halla.stats import discretize,pca, bh, permutation_test_by_representative, p_adjust
from halla.stats import permutation_test_by_kpca_norm_mi, permutation_test_by_kpca_pearson, permutation_test_by_cca_pearson, permutation_test_by_cca_norm_mi

# permutation_test_by_representative  
# permutation_test_by_kpca_norm_mi
# permutation_test_by_kpca_pearson
# permutation_test_by_cca_pearson 
# permutation_test_by_cca_norm_mi


## statistics packages 

import numpy 
import numpy as np
from numpy import array 
import scipy 
import scipy.stats 
import scipy.cluster 
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


#==========================================================================#
# DATA STRUCTURES 
#==========================================================================#

# Use decorators 
# @ClusterNode 

#class ClusterNode #scipy.cluster.hierarchy.ClusterNode

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
		return self 
		
	def add_children(self, aData):
		for item in aData:
			self.add_child( item )
		return self 

	def get_children(self): 
		return self.m_arrayChildren
	
	def get_child(self,iIndex=None):
		return self.m_arrayChildren[iIndex or 0] if self.m_arrayChildren else None 
	
	def add_data(self, pDatum):
		self.m_pData = pDatum 
		return self 
	
	def get_data(self):
		return self.m_pData 

class Gardener():
	"""
	A gardener object is a handler for the different types of hierarchical data structures ("trees")
	
	Always return a copied version of the tree being modified. 

	"""

	import copy 

	def __init__( self, apTree = None ):

		self.delta = 1.0 ##step parameter 
		self.sigma = 0.5 ##start parameter 

		self.apTree = [copy.deepcopy(ap) for ap in apTree] ## the list of tree objects that is going to be modified 
		## this is a list instead of a single tree object because it needs to handle any cross section of a given tree 

		assert(0.0 <= self.delta <= 1.0)
		assert(0.0 <= self.sigma <= 1.0)

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

	def prune( self, ):
		"""
		Return a pruned version of the tree, with new starting node(s) 

		"""
		pass

	def slice( self, ):
		"""
		Slice tree, giving arise to thinner cross sections of the tree, still put together by hierarchy 
		"""
		pass 


	###NOTE: I think doing the modification at run-time makes a lot more sense, and is a lot less complicated 

#==========================================================================#
# FUNCTORS   
#==========================================================================#

def lf2tree( lf ):
	"""
	Functor converting from a layerform object to halla Tree object 

	Parameters
	------------
		lf : layerform 

	Returns 
	-------------
		t : halla Tree object 
	""" 

	pass 

def tree2clust( pTree, exact = True ):
	"""
	Functor converting from a halla Tree object to a scipy ClusterNode object.
	When exact is True, gives error when the map Tree() -> ClusterNode is not injective (more than 2 children per node)
	"""

	pass 

def clust2tree( pTree ):
	"""
	Functor converting from a scipy ClusterNode to a halla Tree object; 
	can always be done 
	"""

	pass 

#==========================================================================#
# METHODS  
#==========================================================================#

def is_tree( pObj ):
	"""
	duck type implementation for checking if
	object is ClusterNode or Tree, more or less
	"""

	try:
		pObj.get_data 
		return True 
	except Exception:
		return False 


def hclust( pArray, strMetric = "norm_mi", cluster_method = "single", bTree = False ):
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

	pMetric = halla.distance.c_hash_metric[strMetric] 
	## Remember, pMetric is a notion of _strength_, not _distance_ 

	def pDistance( x,y ):
		return 1.0 - pMetric(x,y)

	D = pdist( pArray, metric = pDistance )
	#print D    
	Z = linkage( D ) 
	return to_tree( Z ) if bTree else Z 


def dendrogram( Z ):
	return scipy.cluster.hierarchy.dendrogram( Z )

def truncate_tree( apClusterNode, level = 0, skip = 0 ):
	"""
	Chop tree from root, returning smaller tree towards the leaves 

	Parameters
	---------------
		
		list_clusternode : list of ClusterNode objects 
		level : int 
		skip : int 

	Output 
	----------

		lC = list of ClusterNode objects 

	"""

	#if isinstance( list_clusternode, scipy.cluster.hierarchy.ClusterNode ):
	#	apClusterNode = [list_clusternode]
	#else:
	#	apClusterNode = list_clusternode

	iSkip = skip 
	iLevel = level 

	if iLevel < iSkip:
		return truncate_tree( filter( lambda x: bool(x), [(p.right if p.right else None) for p in apClusterNode]) \
			+ filter( lambda x: bool(x), [(q.left if p.left else None) for q in apClusterNode] ), level = iLevel+1, skip = iSkip ) 

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

	Should be designed to handle both ClusterNode and Tree types 
	""" 

	bTree = is_tree( pClusterNode )

	func = pFunction if not bTree else lambda x: x.get_data() 

	if pClusterNode:

		if not bTree:
			if pClusterNode.is_leaf():
				return ( aOut + [func(pClusterNode)] )
			else:
				return reduce_tree( pClusterNode.left, func, aOut ) + reduce_tree( pClusterNode.right, func, aOut ) 
		elif bTree:
			if pClusterNode.is_leaf():
				return ( aOut + [func(pClusterNode)] )
			else:
				pChildren = pClusterNode.get_children()
				iChildren = len( pChildren )
				return reduce( lambda x,y: x+y, [reduce_tree( pClusterNode.get_child(i), func, aOut) for i in range(iChildren)], [] )
	else:
		return [] 

def reduce_tree_by_layer( apParents, iLevel = 0, iStop = None ):
	"""

	Traverse one tree. 

	Input: apParents, iLevel = 0, iStop = None

	Output: a list of (iLevel, list_of_nodes_at_iLevel)

		Ex. 

		[(0, [0, 2, 6, 7, 4, 8, 9, 5, 1, 3]),
		 (1, [0, 2, 6, 7]),
		 (1, [4, 8, 9, 5, 1, 3]),
		 (2, [0]),
		 (2, [2, 6, 7]),
		 (2, [4]),
		 (2, [8, 9, 5, 1, 3]),
		 (3, [2]),
		 (3, [6, 7]),
		 (3, [8, 9]),
		 (3, [5, 1, 3]),
		 (4, [6]),
		 (4, [7]),
		 (4, [8]),
		 (4, [9]),
		 (4, [5]),
		 (4, [1, 3]),
		 (5, [1]),
		 (5, [3])]

	"""

	apParents = list(apParents)
	apParents = filter(bool, apParents)

	bTree = False 
	
	if not isinstance( apParents, list ):
		bTree = is_tree( apParents )
		apParents = [apParents]
	else:
		try:
			bTree = is_tree( apParents[0] )
		except IndexError:
			pass 

	if (iStop and (iLevel > iStop)) or not(apParents):
		return [] 
	else:
		filtered_apParents = filter( lambda x: not(x.is_leaf()) , apParents )
		new_apParents = [] 
		for q in filtered_apParents:
			if not bTree:
				new_apParents.append( q.left ); new_apParents.append( q.right )
			else:
				for item in q.get_children():
					new_apParents.append(item)
		if not bTree:
			return [(iLevel, reduce_tree(p)) for p in apParents ] + reduce_tree_by_layer( new_apParents, iLevel = iLevel+1 )
		else:
			return [(iLevel, p.get_data()) for p in apParents ] + reduce_tree_by_layer( new_apParents, iLevel = iLevel+1 )

def tree2lf( apParents, iLevel = 0, iStop = None ):
	"""
	An alias of reduce_tree_by_layer, for consistency with functor definitions 
	"""
	return reduce_tree_by_layer( apParents ) 

def fix_layerform( lf, iExtend = 0 ):
	""" 
	
	iExtend is when you want to extend layerform beyond normal global depth 

	There is undesired behavior when descending down singletons
	Fix this behavior 

		Example 

		[(0, [0, 7, 4, 6, 8, 2, 9, 1, 3, 5]),
		 (1, [0]),
		 (1, [7, 4, 6, 8, 2, 9, 1, 3, 5]),
		 (2, [7, 4, 6, 8, 2, 9]),
		 (2, [1, 3, 5]),
		 (3, [7]),
		 (3, [4, 6, 8, 2, 9]),
		 (3, [1]),
		 (3, [3, 5]),
		 (4, [4]),
		 (4, [6, 8, 2, 9]),
		 (4, [3]),
		 (4, [5]),
		 (5, [6]),
		 (5, [8, 2, 9]),
		 (6, [8]),
		 (6, [2, 9]),
		 (7, [2]),
		 (7, [9])]

	"""

	aOut = [] 

	iDepth = depth_tree( lf, bLayerform = True ) ## how many layers? 
	iLevel = iDepth - 1 ##layer level 

	for tD in lf: ##tuple data  
		iCurrent, aiIndices = tD[:2] 
		if len(aiIndices)==1: ##if singleton 
			aOut += [(i,aiIndices) for i in range(iCurrent+1,iLevel+1)]

	lf += aOut 

	## Need to sort to ensure correct layerform  
	## source: http://docs.scipy.org/doc/numpy/reference/generated/numpy.sort.html
	
	dtype = [('layer', int), ('indices', list)]
	return filter(bool,list(numpy.sort( array(lf, dtype=dtype ), order = 'layer' )))

def fix_clusternode( pClusterNode, iExtend = 0 ):
	"""
	Same as fix_layerform, but for ClusterNode objects 

	Note: should NOT alter original ClusterNode object; make a deep copy of it instead 
	"""

	import copy 

	def _fix_clusternode( pChild ):
		#pChildUpdate = copy.deepcopy( pChild )
		iChildDepth = get_depth( pChild )
		iDiff = iGlobalDepth - iChildDepth 
		if iChildDepth == 1:
			#print "singleton"
			#print "original", reduce_tree_by_layer( [pChild] ) 
			#print "difference", iDiff 
			assert( pChild.id == reduce_tree(pChild)[0] )
			pChild = spawn_clusternode( pData = pChild.id, iCopy = iDiff ) 
			#print "fixed", reduce_tree_by_layer( [pChild])
			#pChild = pChildUpdate 
			return pChild
		else:
			#print "non-singleton"
			#print reduce_tree_by_layer( [pChild] )
			pChild = fix_clusternode( pChild, iExtend = iExtend )
			return pChild 
			
	pClusterNode = copy.deepcopy( pClusterNode ) ##make a fresh instance 
	iGlobalDepth = get_depth( pClusterNode ) + iExtend 
	if iGlobalDepth == 1:
		return pClusterNode
	else:
		pClusterNode.left = _fix_clusternode( pClusterNode.left )
		pClusterNode.right = _fix_clusternode( pClusterNode.right )
			
		return pClusterNode 

def get_depth( pClusterNode, bLayerform = False ):
	"""
	Get the depth of a tree 

	Parameters
	--------------

		pClusterNode: clusternode or layerform object 
		bLayerform: bool 


	Returns
	-------------

		iDepth: int 
	"""

	aOut = reduce_tree_by_layer( [pClusterNode] ) if not bLayerform else pClusterNode 
	aZip = zip(*aOut)[0]
	return max(aZip)-min(aZip) +1

def depth_tree( pClusterNode, bLayerform = False ):
	"""
	alias for get_depth
	"""

	return get_depth( pClusterNode, bLayerform = bLayerform )

def depth_min( pClusterNode, bLayerform = False ):
	"""
	Get the index for the minimnum layer 
	"""
	aOut = reduce_tree_by_layer( [pClusterNode] ) if not bLayerform else pClusterNode 
	aZip = zip(*aOut)[0]
	return min(aZip)

def depth_max( pClusterNode, bLayerform = False ):
	"""
	Get the index for the maximum layer
	"""
	aOut = reduce_tree_by_layer( [pClusterNode] ) if not bLayerform else pClusterNode 
	aZip = zip(*aOut)[0]
	return max(aZip)

def get_layer( atData, iLayer = None, bTuple = False, bIndex = False ):
	"""
	Get output from `reduce_tree_by_layer` and parse 

	Input: atData = a list of (iLevel, list_of_nodes_at_iLevel), iLayer = zero-indexed layer number 

	BUGBUG: Need get_layer to work with ClusterNode and Tree objects as well! 
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

def spawn_clusternode( pData, iCopy = 1, iDecider = -1 ):
	"""
	Parameters
	-----------------

		pData: any data in node

		iCopy: int
			When iCopy > 0, makes a copy of pData and spawns children `iCopy` number of times 

		iDecider: int
			When adding copies of nodes, need to know if 
			-1-> only add to left 
			0 -> add both to left and right 
			1 -> only add to right 

	Returns 
	------------------

		C: scipy.cluster.hierarchy.ClusterNode instance 

	"""

	assert( iCopy >= 1 ), "iCopy must be a positive integer!"

	def _spawn_clusternode( pData, iDecider = -1 ):
		"""
		spawns a "clusterstump" 
		"""
		return scipy.cluster.hierarchy.ClusterNode( pData ) ##should return a new instance **each time**

	if iCopy == 1:
		return _spawn_clusternode( pData )

	else: 
		pOut = _spawn_clusternode( pData )
		pLeft = spawn_clusternode( pData, iCopy = iCopy - 1, iDecider = iDecider )
		pOut.left = pLeft 
		return pOut 

def spawn_tree( pData, iCopy = 0, iDecider = -1 ):
	"""
	Extends `spawn_clusternode` to the halla.hierarchy.Tree object 
	"""
	return None 

def couple_tree( apClusterNode1, apClusterNode2, strMethod = "uniform", strLinkage = "min" ):
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

	import copy 

	#-------------------------------------#
	# Decider Functions                   #
	#-------------------------------------#

	def _filter_true( x ):
		return filter( lambda y: bool(y), x )

	def _decider_min( node1, node2 ):
		return ( not(_filter_true( [node1.left, node1.right] ) ) or not(_filter_true( [node2.left, node2.right] ) ) )

	def _decider_max( node1, node2 ):
		pass

	def _next( ):
		"""
		gives the next node on the chain of linkages 
		"""
		pass	

	## hash containing string mappings for deciders 

	hashMethod = {"min": _decider_min, 
				"max": _decider_max, 
				}

	pMethod = hashMethod[strLinkage] ##returns 'aNode x aNode -> bool' object 


	#-------------------------------------#
	# Parsing Steps                       #
	#-------------------------------------#

	aiGlobalDepth1 = [get_depth( ap ) for ap in apClusterNode1]
	aiGlobalDepth2 = [get_depth( ap ) for ap in apClusterNode2]

	iMaxDepth = max(max(aiGlobalDepth1),max(aiGlobalDepth2))
	iMinDepth = min(min(aiGlobalDepth1),min(aiGlobalDepth2))

	## Unalias data structure so this does not alter the original data type
	## Fix for depth 
	apClusterNode1 = [fix_clusternode(apClusterNode1[i], iExtend = iMaxDepth - aiGlobalDepth1[i]) for i in range(len(apClusterNode1))]
	apClusterNode2 = [fix_clusternode(apClusterNode2[i], iExtend = iMaxDepth - aiGlobalDepth2[i]) for i in range(len(apClusterNode2))]

	def _couple_tree( apClusterNode1, apClusterNode2, strMethod = strMethod, strLinkage = strLinkage ):
		"""
		recursive function 
		"""
		
		aOut = []

		for a,b in itertools.product( apClusterNode1, apClusterNode2 ):
			
			data1 = reduce_tree( a )
			data2 = reduce_tree( b )

			pStump = Tree([data1,data2])

			apChildren1, apChildren2 = _filter_true([a.left, a.right]), _filter_true([b.left,b.right])
			

			##Children should _already be_ adjusted for depth 
			if not(any(apChildren1)) or not(any(apChildren2)):
				aOut.append( pStump )

			else: 
				aOut.append( pStump.add_children( _couple_tree( apChildren1, apChildren2, strMethod = strMethod, strLinkage = strLinkage ) ) )

		return aOut 

	return _couple_tree( apClusterNode1, apClusterNode2, strMethod, strLinkage )

def naive_all_against_all( pArray1, pArray2, strMethod = "permutation_test_by_representative" ):

	phashMethods = {"permutation_test_by_representative" : halla.stats.permutation_test_by_representative, 
						"permutation_test_by_average" : halla.stats.permutation_test_by_average,
						"parametric_test" : halla.stats.parametric_test}

	iRow = len(pArray1)
	iCol = len(pArray2)

	aOut = [] 

	for i,j in itertools.product( range(iRow), range(iCol) ):

		pDist = phashMethods[strMethod]
		fVal = pDist( array([pArray1[i]]), array([pArray2[j]]) )
		aOut.append(fVal)

	return numpy.reshape( aOut, (iRow,iCol) )


def traverse_by_layer( pClusterNode1, pClusterNode2, pArray1, pArray2, pFunction = None ):
	"""

	Useful function for doing all-against-all comparison between nodes in each layer 

	traverse two trees at once, applying function `pFunction` to each layer pair 

	latex: $pFunction: index1 \times index2 \times data1 \times data2 \rightarrow \mathbb{R}^k, $ for $k$ the size of the cross-product set per layer 

	Parameters
	----------------
		pClusterNode1, pClusterNode2, pArray1, pArray2, pFunction
	
	Returns 
	---------
		All-against-all per layer 

		Ex. 

		[[([0, 2, 6, 7, 4, 8, 9, 5, 1, 3], [0, 2, 6, 7, 4, 8, 9, 5, 1, 3])],
		 [([0, 2, 6, 7], [0, 2, 6, 7]),
		  ([0, 2, 6, 7], [4, 8, 9, 5, 1, 3]),
		  ([4, 8, 9, 5, 1, 3], [0, 2, 6, 7]),
		  ([4, 8, 9, 5, 1, 3], [4, 8, 9, 5, 1, 3])],
		 [([0], [0]),
		  ([0], [2, 6, 7]),
		  ([0], [4]),
		  ([0], [8, 9, 5, 1, 3]),
		  ([2, 6, 7], [0]),
		  ([2, 6, 7], [2, 6, 7]),
		  ([2, 6, 7], [4]),
		  ([2, 6, 7], [8, 9, 5, 1, 3]),
		  ([4], [0]),
		  ([4], [2, 6, 7]),
		  ([4], [4]),
		  ([4], [8, 9, 5, 1, 3]),
		  ([8, 9, 5, 1, 3], [0]),
		  ([8, 9, 5, 1, 3], [2, 6, 7]),
		  ([8, 9, 5, 1, 3], [4]),
		  ([8, 9, 5, 1, 3], [8, 9, 5, 1, 3])],
		 [([2], [2]),
		  ([2], [6, 7]),
		  ([2], [8, 9]),
		  ([2], [5, 1, 3]),
		  ([6, 7], [2]),
		  ([6, 7], [6, 7]),
		  ([6, 7], [8, 9]),
		  ([6, 7], [5, 1, 3]),
		  ([8, 9], [2]),
		  ([8, 9], [6, 7]),
		  ([8, 9], [8, 9]),
		  ([8, 9], [5, 1, 3]),
		  ([5, 1, 3], [2]),
		  ([5, 1, 3], [6, 7]),
		  ([5, 1, 3], [8, 9]),
		  ([5, 1, 3], [5, 1, 3])],
		 [([6], [6]),
		  ([6], [7]),
		  ([6], [8]),
		  ([6], [9]),
		  ([6], [5]),
		  ([6], [1, 3]),
		  ([7], [6]),
		  ([7], [7]),
		  ([7], [8]),
		  ([7], [9]),
		  ([7], [5]),
		  ([7], [1, 3]),
		  ([8], [6]),
		  ([8], [7]),
		  ([8], [8]),
		  ([8], [9]),
		  ([8], [5]),
		  ([8], [1, 3]),
		  ([9], [6]),
		  ([9], [7]),
		  ([9], [8]),
		  ([9], [9]),
		  ([9], [5]),
		  ([9], [1, 3]),
		  ([5], [6]),
		  ([5], [7]),
		  ([5], [8]),
		  ([5], [9]),
		  ([5], [5]),
		  ([5], [1, 3]),
		  ([1, 3], [6]),
		  ([1, 3], [7]),
		  ([1, 3], [8]),
		  ([1, 3], [9]),
		  ([1, 3], [5]),
		  ([1, 3], [1, 3])],
		 [([1], [1]), ([1], [3]), ([3], [1]), ([3], [3])]]

	"""

	aOut = [] 

	def _link( i1, i2, a1, a2 ):
		return (i1,i2)

	if not pFunction:
		pFunction = _link 

	tData1, tData2 = [ fix_layerform( tree2lf( [pT] ) ) for pT in [pClusterNode1, pClusterNode2] ] ## adjusted layerforms 

	iMin = np.min( [depth_tree(tData1, bLayerform = True), depth_tree(tData2, bLayerform = True)] ) 

	for iLevel in range(iMin+1): ## min formulation 
		
		aLayerOut = [] 

		aLayer1, aLayer2 = get_layer( tData1, iLevel ), get_layer( tData2, iLevel )
		iLayer1, iLayer2 = len(aLayer1), len(aLayer2)

		for i,j in itertools.product( range(iLayer1), range(iLayer2) ):
			aLayerOut.append( pFunction( aLayer1[i], aLayer2[j], pArray1, pArray2 ) )

		aOut.append(aLayerOut)

	return aOut 

#### Perform all-against-all per layer, without adherence to hierarchical structure at first
def layerwise_all_against_all( pClusterNode1, pClusterNode2, pArray1, pArray2, adjust_method = "BH" ):
	"""
	Perform layer-wise all against all 

	Notes
	---------

		New behavior for coupling trees 

		CALCULATE iMaxLayer
		IF SingletonNode:
			CALCULATE iLayer 
			EXTEND (iMaxLayer - iLayer) times 
		ELSE:
			Compare bags 
	"""
	aOut = [] 

	pPTBR = lambda ai, aj, X,Y : halla.stats.permutation_test_by_representative( X[array(ai)], Y[array(aj)] ) 

	traverse_out = traverse_by_layer( pClusterNode1, pClusterNode2, pArray1, pArray2 ) ##just gives me the coupled indices 

	for layer in traverse_out:
		aLayerOut = [] 
		aPval = [] 
		for item in layer:
			fPval = halla.stats.permutation_test_by_representative( pArray1[array(item[0])], pArray2[array(item[1])] )
			aPval.append(fPval)
		
		adjusted_pval = halla.stats.p_adjust( aPval )
		if not isinstance( adjusted_pval, list ):
			## keep type consistency 
			adjusted_pval = [adjusted_pval]
	
		for i,item in enumerate(layer):
			aLayerOut.append( ([item[0], item[1]], adjusted_pval[i]) )
		
		aOut.append(aLayerOut)

	return aOut

#method
# permutation_test_by_representative  
# permutation_test_by_kpca_norm_mi
# permutation_test_by_kpca_pearson
# permutation_test_by_cca_pearson 
# permutation_test_by_cca_norm_mi 


def all_against_all( pTree, pArray1, pArray2, method = "permutation_test_by_representative", metric = "norm_mi", p_adjust = "BH", q = 0.1, 
	pursuer_method = "nonparameteric", step_parameter = 1.0, start_parameter = 0.0, bVerbose = False ):
	"""
	Perform all-against-all on a hypothesis tree.

	Notes:
		Right now, return aFinal, aOut 


	Parameters
	---------------

		pTree 
		pArray1
		pArray2
		method 
		metric
		p_adjust
		pursuer_method 
		verbose 

	Returns 
	----------

		Z_final, Z_all: numpy.ndarray
			Bags of associations of _final_ associations, and _all_ associations respectively. 


	Notes 
	----------

		Descension schemes: 
		
		* Weak negative: if q_hat <= 1.0 - q, proceed until q_hat > 1.0 - q. Report findings. 
		* Strong negative: if q_hat <= 1.0 - q, proceed until q_hat > 1.0 -q. If Leaf, report findings, else None.
		* Weak positive: if q_hat <= q, then continue down the tree until q_hat > q. Report findings.
		* Strong positive: if q_hat <= q, then continue down the tree until q_hat > q. If Leaf, report findings, else None.

		Overwhelming conclusion is that these are all variants of the same Yekutieli criterion -- all remains to pick is the $q$ cutoff, and 
		the starting point 

	"""

	def _start_parameter_to_iskip( start_parameter ):
		"""
		takes start_parameter, determines how many to skip
		"""

		assert( type( start_parameter ) == float )

		iDepth = get_depth( pTree )
		iSkip = int(start_parameter * (iDepth-1))

		return iSkip 

	def _step_parameter_to_aislice( step_parameter ):
		"""
		takes in step_parameter, returns a list of indices for which all-against-all will take place 
		"""

		pass 

	aOut = [] ## Full log 
	aFinal = [] ## Only the final reported values 

	iGlobalDepth = depth_tree( pTree )
	iSkip = _start_parameter_to_iskip( start_parameter )

	pHashMethods = {"permutation_test_by_representative" : halla.stats.permutation_test_by_representative, 
						"permutation_test_by_average" : halla.stats.permutation_test_by_average,
						"parametric_test" : halla.stats.parametric_test,
						"permutation_test_by_kpca_norm_mi" :halla.stats.permutation_test_by_kpca_norm_mi, 
						"permutation_test_by_kpca_pearson" :halla.stats.permutation_test_by_kpca_pearson,
						"permutation_test_by_cca_pearson" :halla.stats.permutation_test_by_cca_pearson,
						"permutation_test_by_cca_norm_mi" :halla.stats.permutation_test_by_cca_norm_mi,
						}

	# permutation_test_by_representative  
	# permutation_test_by_kpca_norm_mi
	# permutation_test_by_kpca_pearson
	# permutation_test_by_cca_pearson 
	# permutation_test_by_cca_norm_mi

	strMethod = method 
	pMethod = pHashMethods[strMethod]

	def _actor( pNode ):
		"""
		Performs a certain action at the node

			* E.g. compares two bags, reports distance and p-values 
		"""

		aIndicies = pNode.get_data( ) 
		aIndiciesMapped = map( array, aIndicies ) ## So we can vectorize over numpy arrays 
		dP = pMethod( pArray1[aIndiciesMapped[0]], pArray2[aIndiciesMapped[1]] )

		aOut.append( [aIndicies, dP] )

		return dP 

	def _pursuer( apChildren, pParent, aP_adjusted, fQ, fQParent ):
		"""
		Decides if you want to continue pursuing down recursively

		Parameters
		===============

			apChildren: list 
			aP: list 
			bP: bool 
			fQ: float 

		Returns
		===============

			aBool: list 
				list of bool values to see which nodes to continue on 

		Notes
		===============

			Greedy:

				If q_hat <= q, reject null. Go as deep as possible. 


			There are three cases for the pursuer 

			True, False -> stop, record value 
			True, True -> keep on going, unless no more children. Then record value 
		"""
		
		aBool = [] 

		try:
			iLen = len( aP_adjusted )
		except Exception:
			aP_adjusted = [aP_adjusted]
			iLen = 1 

		# See if children pass test 
		for i, p in enumerate( aP_adjusted ): 
			if p <= fQ:
				if bVerbose:
					print p, fQ 
				### met significance criteria 
				### if not the terminal node, continue on 
				aBool.append( True )

			else:
				### did not meet significance criteria
				aBool.append( False )
					
		### (*) Note the following situation:
		### A child has 3 siblings. That child passes the significance threshold, so its children can be added to the final list. 
		### A sibling doesn't quite cut the threshold, so the parent has to added instead. 
		### But, the child has overlapping data with the sibling that was just added. 
		### To combat this situation. Make sure that the lower nodes (even leaves) are always added AFTER the parent node 

		### This makes sure that (*) is taken care of. 
		### Notice that the parental addition happens prior to the continuation of the child 

		if not any(aBool):
			### if have to stop, then add to list of final association pairs 
			if fQParent <= fQ:
				aFinal.append( [pParent.get_data(), fQParent] )

		for j,bB in enumerate(aBool):
			if (bB == True) and (apChildren[j].get_children() == []):
				### if this is the terminal node, then do not traverse down any further
				### furthermore, append the final p-value to the list of final association pairs 
				aBool[j] = False 
				aFinal.append( [apChildren[j].get_data(), aP_adjusted[j]] ) 

		return aBool 

	def _fw_operator( pNode, fQParent = None ):
		"""
		
		Parameters
		=================

			pNode: halla.hierarchy.Tree 

			bP: bool

		Returns 
		=================

		Family-wise operator

			* Gets fed in a node, perform function to children 
		"""

		if not fQParent:
			aiI,aiJ = map( array, pNode.get_data() )
			fQParent = pMethod( pArray1[aiI], pArray2[aiJ] )

		apChildren = pNode.get_children( )
		
		if apChildren: 
			aP = [ _actor( c ) for c in apChildren ]
			aP_adjusted = halla.stats.p_adjust( aP )
			aPursuer = _pursuer( apChildren, pNode, aP_adjusted = aP_adjusted, fQ = q, fQParent = fQParent )

			for j, bP in enumerate( aPursuer ):
				_fw_operator( apChildren[j], fQParent = fQParent ) 

	_fw_operator( pTree ) 

	return aFinal, aOut 

#==========================================================================#
# TEST DATA  
#==========================================================================#

def randtree( n = 10, sparsity = 0.5, obj = True, disc = True ):
	""" 
	generate random trees 
	if obj is True, return ClusterNode object, else return in matrix form 
	"""
	
	iSamples = n 
	fSpar = sparsity 
	bObj = obj 
	bDisc = disc 

	#s = strudel.Strudel() 

	#X,A = s.generate_synthetic_data( iSamples, fSpar )

	X = halla.randmat( ) 
	X = discretize( X ) if bDisc else X 

	T = hclust( X, bTree = obj )

	return T 

### garbage 


#### Exploration: 

#	* if not bP and no current passes test, continue on 
#	* elif not bP and passes test, change bP to True and continue 
#	* elif bP and passes test, go on
#	* else bP and does not pass test, STOP 

#def _explore():
#
#			bPPrior = bP
#
#			try:
#				iLen = len( aP )
#			except Exception:
#				aP = [aP]
#				iLen = 1 
#
#			iMin = np.argmin( aP )
#
#			aBool = [] 
#
#			for i in range(iLen):
#				aBool.append( [1] if i == iMin else [0] )
#
#			bTest = 0 ## By default, do not know if q-val has been met 
#
#			aP_adjusted = halla.stats.p_adjust( aP ) 
#			#aP_adjusted = aP
#		
#			try:
#				aP_adjusted[0]
#			except Exception:
#				aP_adjusted = [aP_adjusted]
#
#			# See if children pass test 
#			for i, p in enumerate( aP_adjusted ): 
#				if p <= fQ:
#					aBool[i].append( 1 )
#					aFinal.append( [apChildren[i].get_data(), aP_adjusted[i]] )
#				else:
#					#if bPPrior: 
#						### Stop criterion; previous p-value cutoff passed, but now failed 
#						
#					aBool[i].append( 0 )
#
#			return aBool 




