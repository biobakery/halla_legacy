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
from halla.stats import discretize,pca, mca, bh, permutation_test_by_representative, p_adjust

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


## hutlab tools 

import strudel 

#==========================================================================#
# DATA STRUCTURES 
#==========================================================================#

# Use decorators 
# @ClusterNode 

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
# FUNCTORS   
#==========================================================================#

"""
New behavior for coupling trees 

CALCULATE iMaxLayer
IF SingletonNode:
	CALCULATE iLayer 
	EXTEND (iMaxLayer - iLayer) times 
ELSE:
	Compare bags 
"""


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
	print D    
	Z = linkage( D ) 
	return to_tree( Z ) if bTree else Z 


def dendrogram( Z ):
	return scipy.cluster.hierarchy.dendrogram( Z )

def truncate_tree( list_clusternode, level = 0, skip = 0 ):
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

	apClusterNode = None 

	if isinstance( list_clusternode, scipy.cluster.hierarchy.ClusterNode ):
		apClusterNode = [list_clusternode]
	else:
		apClusterNode = list_clusternode

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

def reduce_tree_by_layer( apParents, iLevel = 0, iStop = None ):
	"""

	Traverse one tree. 

	Input: apParents, iLevel = 0, iStop = None

	Output: a list of (iLevel, list_of_nodes_at_iLevel)
	"""

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
	return reduce_tree_by_layer 

def depth_tree( pClusterNode, bLayerform = False ):
	"""
	Get the depth of a tree 
	"""
	
	aOut = reduce_tree_by_layer( [pClusterNode] ) if not bLayerform else pClusterNode 
	return max(zip(*aOut)[0])+1

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

	Note
	--------
		The way it is written now, the function performs all-against-all per layer without regard to the inheritance pattern. 

	"""

	aOut = [] 

	def _link( i1, i2, a1, a2 ):
		return (i1,i2)

	if not pFunction:
		pFunction = _link 

	def _get_max( tData ):
		
		return np.max([i[0] for i in tData])

	tData1, tData2 = [ reduce_tree_by_layer( [pC] ) for pC in [pClusterNode1, pClusterNode2] ]

	iMin = np.min( [_get_max(tData1), _get_max(tData2)] ) 

	for iLevel in range(iMin+1):
		
		aLayerOut = [] 

		aLayer1, aLayer2 = get_layer( tData1, iLevel ), get_layer( tData2, iLevel )
		iLayer1, iLayer2 = len(aLayer1), len(aLayer2)

		## See if this lines up 
		assert( iLayer1 == iLayer2 )

		for i,j in itertools.product( range(iLayer1), range(iLayer2) ):
			aLayerOut.append( pFunction( aLayer1[i], aLayer2[j], pArray1, pArray2 ) )

		aOut.append(aLayerOut)

	return aOut 


def layerwise_all_against_all( pClusterNode1, pClusterNode2, pArray1, pArray2, adjust_method = "BH" ):
	"""
	Perform layer-wise all against all 
	"""
	aOut = [] 

	traverse_out = traverse_by_layer( pClusterNode1, pClusterNode2, pArray1, pArray2 )
	for layer in traverse_out:
		aLayerOut = [] 
		aPval = [] 
		for item in layer:
			fPval = halla.stats.permutation_test_by_representative( pArray1[array(item[0])], pArray2[array(item[1])] )
			aPval.append(fPval)
		
		#print aPval 

		adjusted_pval = halla.stats.p_adjust( aPval )
		if not isinstance( adjusted_pval, list ):
			## keep type consistency 
			adjusted_pval = [adjusted_pval]

		#print adjusted_pval 

		for i,item in enumerate(layer):
			aLayerOut.append( (item[0], item[1], adjusted_pval[i]) )

		aOut.append(aLayerOut)

	return aOut 

def naive_couple_tree( pClusterNode1, pClusterNode2, method = "uniform", linkage = "min" ):
	"""
	Couples two trees to make a hypothesis tree in layerform. 
	Doesn't take dependencies into account  

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

def couple_tree( apClusterNode1, apClusterNode2, method = "uniform", linkage = "min" ):
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

	## initialize 

	strMethod = method 
	strLinkage = linkage

	## decider functions 

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

	pMethod = hashMethod[linkage] ##returns 'aNode x aNode -> bool' object 

	## main 

	aOut = [] 

	for a,b in itertools.product( apClusterNode1, apClusterNode2 ):
		
		data1, data2 = [ reduce_tree( x ) for x in [a,b] ]

		pStump = Tree([data1,data2])

		if pMethod( a,b ): # design choice: pMethod has priority; this is when algorithm determines that it can stop. 
			aOut.append( pStump )
		else: # can go on recursively 
			## in actuality, need linkage function to give the desired behavior for node pattern 
			aOut.append( pStump.add_children( couple_tree( [a.left, a.right], [b.left, b.right], method = strMethod, linkage = strLinkage ) ) )

	return aOut 

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
		aOut.append([(i,j), fVal])

	return aOut 

def all_against_all( pTree, pArray1, pArray2, method = "permutation_test_by_representative", metric = "norm_mid", correction = "BH", q = 0.1, 
	pursuer_method = "nonparameteric", verbose = True ):
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
		correction
		pursuer_method 
		verbose 

	Returns 
	----------

		Z_final, Z_all: numpy.ndarray
			Bags of associations of _final_ associations, and _all_ associations respectively. 


	"""

	aOut = [] ## Full log 

	aFinal = [] ## Only the ones that passes test 

	phashMethods = {"permutation_test_by_representative" : halla.stats.permutation_test_by_representative, 
						"permutation_test_by_average" : halla.stats.permutation_test_by_average,
						"parametric_test" : halla.stats.parametric_test}

	strMethod = method 

	pMethod = phashMethods[strMethod]

	def _actor( pNode ):
		"""
		Performs a certain action at the node

			* E.g. compares two bags, reports distance and p-values 
		"""

		aIndicies = pNode.get_data( ) 

		aIndiciesMapped = map( array, pNode.get_data( ) )

		dP = pMethod( pArray1[aIndiciesMapped[0]], pArray2[aIndiciesMapped[1]] )

		aOut.append( [aIndicies, dP] )

		return dP 

	def _pursuer( apChildren, aP, bP, fQ ):
		"""
		Decides if you want to continue pursuing down recursively

		Right now, do as follows: 

			* if not bP and no current passes test, continue on 
			* elif not bP and passes test, change bP to True and continue 
			* elif bP and passes test, go on
			* else bP and does not pass test, STOP 

		Returns boolean value: (go down?, has the q-val criterion been met?)

			* E.g. aOut = [(True, False), (True, True), (True, False), (True, False)]

		IMPORTANT:

			* Right now, just greedily go down, without any filtration. Maybe change this to the smallest p-value 
		"""
		
		bPPrior = bP

		iLen = len( aP )

		iMin = np.argmin( aP )

		aBool = [] 

		for i in range(iLen):
			aBool.append( [1] if i == iMin else [0] )

		bTest = 0 ## By default, do not know if q-val has been met 

		aP_adjusted = p_adjust( aP ) 

		# See if children pass test 
		for i, p in enumerate( aP_adjusted ): 
			if p <= fQ:
				aBool[i].append( 1 )
				aFinal.append( [apChildren[i].get_data(), aP_adjusted[i]] )
			else:
				#if bPPrior: 
					### Stop criterion; previous p-value cutoff passed, but now failed 
					
				aBool[i].append( 0 )

		return aBool 

	def _fw_operator( pNode, bP = 0 ):
		"""
		Family-wise operator

			* Gets fed in a node, do stuff with its children 
		"""

		pChildren = pNode.get_children( )
		
		if pChildren:

			aP = [ _actor( c ) for c in pChildren ]
			
			aP_adjusted = halla.stats.p_adjust( aP )

			aPursuer = _pursuer( pChildren, aP_adjusted, bP=bP, fQ = q )

			for j, tB in enumerate( aPursuer ):

				#if tB[0] == 1:
				#	_fw_operator( pChildren[j], tB[1] ) ##Why the hell is this not workign?  Stupid python bug
				
				_fw_operator( pChildren[j], tB[1] ) 

	### bP_old = True; if not bP_new or no more children, then STOP. Append to aFinal. This is the only way to be appended here. 
	### Note how in the current implementation the first node is automatically passed; this is probably desired behavior anyways; can add more functionality later. 

	_fw_operator( pTree ) 

	return aFinal, aOut

def all_all_against_all( pTree, pArray1, pArray2, method = "permutation_test_by_representative", metric = "adj_mid", verbose = True ):
	"""
	Perform all-against-all on a hypothesis tree all the way to the bottom.

	Notes:

		Assumes that pArray1, pArray2 have been properly discretized, if mi-based metric is being used

	"""

	aOut = [] 

	phashMethods = {"permutation_test_by_representative" : halla.stats.permutation_test_by_representative, 
						"permutation_test_by_average" : halla.stats.permutation_test_by_average,
						"parametric_test" : halla.stats.parametric_test}

	strMethod = method 

	pMethod = phashMethods[strMethod]

	aLayer = [t[1] for t in reduce_tree_by_layer( pTree )]

	for pPair in aLayer:
		pOne, pTwo = map( array, pPair )
		aOut.append( [pPair, pMethod( pArray1[pOne], pArray2[pTwo] )] )

	return aOut 


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

	s = strudel.Strudel() 

	X,A = s.generate_synthetic_data( iSamples, fSpar )
	X = discretize( X ) if bDisc else X 

	T = hclust( X, bTree = obj )

	return T 





