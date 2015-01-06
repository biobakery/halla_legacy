#!/usr/bin/env python 
'''
Hiearchy module, used to build trees and other data structures.
Handles clustering and other organization schemes. 
'''
## structural packages 
import halla 
import halla.stats 

from halla.distance import mi, l2, absl2, norm_mi
from halla.stats import discretize,pca, bh, permutation_test_by_representative,permutation_test_by_representative_mic, p_adjust
from halla.stats import permutation_test_by_kpca_norm_mi, permutation_test_by_kpca_pearson, permutation_test_by_cca_pearson, permutation_test_by_cca_norm_mi
import math 
import itertools

## statistics packages 
import numpy 
import numpy as np
from numpy import array 
import scipy.cluster 
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, to_tree

#==========================================================================#
# DATA STRUCTURES 
#==========================================================================#

# maximum distance in hierarchical  trees.
global max_dist
max_dist = 1.0 

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

	 

	def __init__( self, apTree = None ):
		import copy
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
	Functor converting from a layerform object to py Tree object 

	Parameters
	------------
		lf : layerform 

	Returns 
	-------------
		t : py Tree object 
	""" 

	pass 

def tree2clust( pTree, exact = True ):
	"""
	Functor converting from a py Tree object to a scipy ClusterNode object.
	When exact is True, gives error when the map Tree() -> ClusterNode is not injective (more than 2 children per node)
	"""

	pass 

def clust2tree( pTree ):
	"""
	Functor converting from a scipy ClusterNode to a py Tree object; 
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
		import py

		x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])

		lxpearson = py.hierarchy.hclust( x, pdist_metric = py.distance.cord )

		dendrogram(lxpearson)	

	.. plot::

		from numpy import array 
		from scipy.cluster.hierarchy import dendrogram, linkage
		import scipy.cluster.hierarchy 
		import py

		x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])

		lxpearson = py.hierarchy.hclust( x, pdist_metric = py.distance.cord )

		dendrogram(lxpearson)	

	* Pearson correlation 2::

		from numpy import array 
		from scipy.cluster.hierarchy import dendrogram, linkage
		import scipy.cluster.hierarchy 
		import py

		y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0], [0.015625,0.125,0.421875,1.0]])

		lypearson = py.hierarchy.hclust( y, pdist_metric = py.distance.cord )

		dendrogram(lypearson)
	
	.. plot::

		from numpy import array 
		from scipy.cluster.hierarchy import dendrogram, linkage
		import scipy.cluster.hierarchy 
		import py

		y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0], [0.015625,0.125,0.421875,1.0]])

		lypearson = py.hierarchy.hclust( y, pdist_metric = py.distance.cord )

		dendrogram(lypearson)

	* Spearman correlation 1::

		from numpy import array 
		from scipy.cluster.hierarchy import dendrogram, linkage
		import scipy.cluster.hierarchy 
		import py

		x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])

		lxspearman = py.hierarchy.hclust( x, pdist_metric = lambda u,v: py.distance.cord(u,v,method="spearman") )

		dendrogram(lxspearman)

	.. plot::

		from numpy import array 
		from scipy.cluster.hierarchy import dendrogram, linkage
		import scipy.cluster.hierarchy 
		import py

		x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])

		lxspearman = py.hierarchy.hclust( x, pdist_metric = lambda u,v: py.distance.cord(u,v,method="spearman") )

		dendrogram(lxspearman)

	* Spearman correlation 2::

		from numpy import array 
		from scipy.cluster.hierarchy import dendrogram, linkage
		import scipy.cluster.hierarchy 
		import py

		y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0], [0.015625,0.125,0.421875,1.0]])

		lyspearman = py.hierarchy.hclust( y, pdist_metric = lambda u,v: py.distance.cord(u,v,method="spearman") )

		dendrogram(lyspearman)

	.. plot::
		
		from numpy import array 
		from scipy.cluster.hierarchy import dendrogram, linkage
		import scipy.cluster.hierarchy 
		import py

		y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0], [0.015625,0.125,0.421875,1.0]])

		lyspearman = py.hierarchy.hclust( y, pdist_metric = lambda u,v: py.distance.cord(u,v,method="spearman") )

		dendrogram(lyspearman)
	
	* Mutual Information 1::

		from numpy import array 
		from scipy.cluster.hierarchy import dendrogram, linkage
		import scipy.cluster.hierarchy 
		import py

		x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])
		dx = halla.discretize( x, iN = None, method = None, aiSkip = [1,3] )

		lxmi = py.hierarchy.hclust( dx, pdist_metric = py.distance.norm_mid )

		dendrogram(lxmi)

	.. plot::

		from numpy import array 
		from scipy.cluster.hierarchy import dendrogram, linkage
		import scipy.cluster.hierarchy 
		import py

		x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])
		dx = py.discretize( x, iN = None, method = None, aiSkip = [1,3] )

		lxmi = py.hierarchy.hclust( dx, pdist_metric = py.distance.norm_mid )

		dendrogram(lxmi)

	* Mutual Information 2::

		from numpy import array 
		from scipy.cluster.hierarchy import dendrogram 
		import py

		y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0], [0.015625,0.125,0.421875,1.0]])
		dy = halla.discretize( y, iN = None, method = None, aiSkip = [1] )		

		lymi = py.hierarchy.hclust( dy, pdist_metric = py.distance.norm_mid )

		dendrogram( lymi )	

	.. plot::

		from numpy import array 
		from scipy.cluster.hierarchy import dendrogram 
		import py

		y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0], [0.015625,0.125,0.421875,1.0]])
		dy = py.discretize( y, iN = None, method = None, aiSkip = [1] )		

		lymi = py.hierarchy.hclust( dy, pdist_metric = py.distance.norm_mid )

		dendrogram( lymi )	

	Notes 
	-----------

		This hclust function is not quite right for the MI case. Need a generic MI function that can take in clusters of RV's, not just single ones 
		Use the "grouping property" as discussed by Kraskov paper. 
	"""
	import pylab
	pMetric = halla.distance.c_hash_metric[strMetric] 
	## Remember, pMetric is a notion of _strength_, not _distance_ 
	#print str(pMetric)
	def pDistance( x,y ):
		return  1.0 - pMetric(x,y)

	D = pdist( pArray, metric = pDistance )
	#print "Distance",D    
	Z = linkage( D, metric = pDistance  )
	#scipy.cluster.hierarchy.dendrogram(Z)
	#pylab.show() 
	#print "Linkage Matrix:", Z
	#print fcluster(Z, .75 )
	#print fcluster(Z, .9 )
	#print fcluster(Z, .3 )
	
	#cutted_Z = np.where(Z[:,2]<.7)
	#print  cutted_Z 
	#scipy.all( (Z[:,3] >= .4, Z[:,3] <= .6), axis=0 ).nonzero()
	#print pos.distance()
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
			return []
			#print "truncated tree is malformed--empty!"
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
		atData = reduce_tree_by_layer( [atData] )

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
	Extends `spawn_clusternode` to the py.hierarchy.Tree object 
	"""
	return None 
#-------------------------------------#
# Threshold Helpers                   #
#-------------------------------------# 

def _min_tau(X, func):
	X = numpy.array(X) 
	D = halla.discretize( X )
	A = numpy.array([func(D[i],D[j]) for i,j in itertools.combinations( range(len(X)), 2 )])

	#assert(numpy.any(A))

	if X.shape[0] < 2:
		return numpy.min([func(D[0],D[0])])

	else:
		return numpy.min( A )

def _max_tau(X, func):
	X = numpy.array(X) 
	D = halla.discretize( X )
	A = numpy.array([func(D[i],D[j]) for i,j in itertools.combinations( range(len(X)), 2 )])

	#assert(numpy.any(A))

	if X.shape[0] < 2:
		return numpy.max([func(D[0],D[0])])

	else:
		return numpy.max( A )

def _mean_tau( X, func ):
	X = numpy.array(X) 
	D = halla.discretize( X )
	A = numpy.array([func(D[i],D[j]) for i,j in itertools.combinations( range(len(X)), 2 )])

	if X.shape[0] < 2:
		return numpy.mean([func(D[0],D[0])])

	else:
		return numpy.mean( A )

	#print "X:"
	#print X

	#print "D:"
	#print D

	#print "Mean Tau:"
	#print A 
	
	#assert(numpy.any(A))



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
def _percentage(dist):
	global max_dist
	return float(dist)/float(max_dist)

def _is_start(ClusterNode,  X, func, distance ):
	#node_indeces = reduce_tree(ClusterNode)
	#print "Node: ",node_indeces
	#if halla.pca_explained_variance_ratio_(X[array(node_indeces)])[0] > .65 or len(node_indeces) ==1:# and _min_tau(X[array(node_indeces)], func) <= x_threshold:
	if _percentage(ClusterNode.dist) <= distance: #and py.stats.pca_explained_variance_ratio_(X[array(node_indeces)])[0] > .60  :#and _min_tau(X[array(node_indeces)], func) <= cluster_threshold :# and halla.pca_explained_variance_ratio_(X[array(node_indeces)])[0] > .60 :#and ClusterNode.get_count() >2 :
		return True
	else: 
		return False

def _is_stop(ClusterNode, X, func, distance):
		node_indeces = reduce_tree(ClusterNode)
		if ClusterNode.is_leaf():# or py.pca_explained_variance_ratio_(X[array(node_indeces)])[0] > .85:#or len(node_indeces) < 2 or ClusterNode.dist< distance:#or py.pca_explained_variance_ratio_(X[array(node_indeces)])[0] > .8 or _mean_tau(X[array(node_indeces)], func) >= .6:
			return True
		else:
			return False
		
def _cutree_to_log2 (apNode, X, func, distance, cluster_threshold ):
	temp_apChildren = []
	temp_sub_apChildren = []
	print "Length of ", len(apNode)
	for node in apNode:
		n = node.get_count()
		print "Number of feature in node: ",n
		sub_apChildren = truncate_tree( [node], level = 0, skip = 1 )
		if sub_apChildren == None:
			sub_apChildren = [node]
		else:
			while len(set(sub_apChildren)) < round(math.log(n)):
				temp_sub_apChildren = truncate_tree( sub_apChildren, level = 0, skip = 1 )
				for i in range(len(sub_apChildren)):
						if sub_apChildren[i].is_leaf():
							if temp_sub_apChildren:
								temp_sub_apChildren.append(sub_apChildren[i])
							else:
								temp_sub_apChildren = [sub_apChildren[i]]
				
				if temp_sub_apChildren == None:
					temp_sub_apChildren = sub_apChildren
				sub_apChildren = temp_sub_apChildren
				temp_sub_apChildren = []
		temp_apChildren += sub_apChildren
	#print "Number of sub-clusters: ", len(set(temp_apChildren))
	return set(temp_apChildren)
def _cutree (clusterNodelist, X, func, distance):
	clusterNode = clusterNodelist
	n = clusterNode[0].get_count()
	sub_clusters = []
	while clusterNode :
		temp_apChildren = []
		sub_clusterNode = truncate_tree( clusterNode, level = 0, skip = 1 )
		for node in clusterNode:
			if node.is_leaf():
				sub_clusterNode += [node]
		if sub_clusterNode:
			for node in sub_clusterNode:
				if _is_start(node ,X, func, distance):
					sub_clusters.append(node)
				else:
					if not node.is_leaf():
						truncated_result = truncate_tree( [node], level = 0, skip = 1 )	
					if truncated_result:
						temp_apChildren += truncated_result
	
		clusterNode = temp_apChildren
	next_dist = distance - 0.1 
	aDist = []
	for i in range(len(sub_clusters)):
		if sub_clusters[i].dist > 0.0:
			aDist += [sub_clusters[i].dist]
	while len(sub_clusters) < round(math.sqrt(n)):
		aDist = []
		max_dist_node = sub_clusters[0]
		for i in range(len(sub_clusters)):
			if sub_clusters[i].dist > 0.0:
				aDist += [sub_clusters[i].dist]
			if max_dist_node.dist < sub_clusters[i].dist:
				max_dist_node = sub_clusters[i]
		#print "Max Distance in this level", _percentage(max_dist_node.dist)
		if not max_dist_node.is_leaf():
			sub_clusters += truncate_tree( [max_dist_node], level = 0, skip = 1 )
			sub_clusters.remove(max_dist_node)
		else:
			break
	if 	aDist:
		next_dist = _percentage(numpy.min(aDist))
	#print len(sub_clusters), n			
	return sub_clusters , next_dist

	
def couple_tree( apClusterNode1, apClusterNode2, pArray1, pArray2, afThreshold = None, strMethod = "uniform", strLinkage = "min", func ="norm_mi", exploration = "couple_tree_iterative"):
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
	tH : py.Tree object 

	Examples
	----------------
	"""
	
	X,Y = pArray1, pArray2
	global max_dist 
	max_dist = max (node.dist for node in apClusterNode1+apClusterNode2)
	#print "Max distance:", max_dist
	#if not afThreshold:	
	#	afThreshold = [halla.stats.alpha_threshold(a, fAlpha, func ) for a in [pArray1,pArray2]]
	
	#x_threshold, y_threshold = afThreshold[0], afThreshold[1]
	#print "x_threshold, y_threshold:", x_threshold, y_threshold
	#aTau = [] ### Did the child meet the intra-dataset confidence cut-off? If not, its children will continue to be itself. 
		#### tau_hat <= tau 
		#### e.g.[(True,False),(True,True),]

	## hash containing string mappings for deciders 

	hashMethod = {"min": _decider_min, 
				"max": _decider_max, 
				}

	pMethod = hashMethod[strLinkage] ##returns 'aNode x aNode -> bool' object 

	#-------------------------------------#
	# Parsing Steps                       #
	#-------------------------------------#
	
	## Unalias data structure so this does not alter the original data type
	## Fix for depth 
	aiGlobalDepth1 = [get_depth( ap ) for ap in apClusterNode1]
	aiGlobalDepth2 = [get_depth( ap ) for ap in apClusterNode2]
	
	iMaxDepth = max(max(aiGlobalDepth1),max(aiGlobalDepth2))
	iMinDepth = min(min(aiGlobalDepth1),min(aiGlobalDepth2))
	
	#apClusterNode1 = [fix_clusternode(apClusterNode1[i], iExtend = iMaxDepth - aiGlobalDepth1[i]) for i in range(len(apClusterNode1))]
	#apClusterNode2 = [fix_clusternode(apClusterNode2[i], iExtend = iMaxDepth - aiGlobalDepth2[i]) for i in range(len(apClusterNode2))]

	#print "Hierarchical TREE 1 ", reduce_tree_by_layer(apClusterNode1)
	#print "Hierarchical TREE 2 ", reduce_tree_by_layer(apClusterNode2)
	
	#print "TREE1 ", reduce_tree_by_layer(apClusterNode1)
	#print "TREE2 ", reduce_tree_by_layer(apClusterNode2)
	aOut = []
	'''for a,b in itertools.product( apClusterNode1, apClusterNode2 ):
		data1 = reduce_tree( a )
		data2 = reduce_tree( b )

	pStump = Tree([data1,data2])
	aOut.append(pStump)
	L = [(pStump, (a,b))]
	'''
	distance = 1.0
	#print "apClusterNode1", apClusterNode1
	apChildren1, distance1 = _cutree (apClusterNode1, X, func, distance = distance)
	apChildren2, distance2 = _cutree (apClusterNode2,  Y, func, distance = distance)
	if distance1 > 0.0:
		if distance2 > 0.0:
			new_distance = numpy.min([distance1, distance2])
		else:
			new_distance = distance1 - 0.1
	elif distance2 > 0.0:
		new_distance = distance2 - 0.1
	else: 
		new_distance = 0.0
	#print "new_distance",new_distance
	#print "Start Nodes 1: ", [reduce_tree( node) for node in apChildren1]
	#print "Start Nodes 2: ", [reduce_tree( node) for node in apChildren2]
	'''
	Establish the root of the coupling tree based on based matched clusters in
	the first layer of relevent depth
	'''	
	max_nmi = 0.0
	for node1 in apChildren1:
		for node2 in apChildren2:
			data1 = reduce_tree( node1 )
			data2 = reduce_tree( node2 )
			temp_nmi = func (pca(X[array(data1)])[0] , pca(Y[array(data2)])[0])
			if max_nmi < temp_nmi:
				max_nmi = temp_nmi
				pStump = Tree([data1,data2])
	aOut.append(pStump)
	'''
	End establishing the root of the coupling tree
	'''
	L = []
	childList= []	
	for a,b in itertools.product( apChildren1, apChildren2 ):
		data1 = reduce_tree( a )
		data2 = reduce_tree( b )
		tempTree = Tree([data1,data2])
		childList.append(tempTree)
		L.append((tempTree, (a,b)))
	pStump.add_children( childList )
	
	distance = new_distance
	next_L = []
	while L:
		
		(pStump, (a,b)) = L.pop(0)
		
		data1 = reduce_tree( a )
		data2 = reduce_tree( b )
				
		bTauX = _is_stop(a, X, func, distance = distance)# ( _min_tau(X[array(data1)], func) >= x_threshold ) ### parametrize by mean, min, or max
		bTauY = _is_stop(b, Y, func, distance = distance)#( _min_tau(Y[array(data2)], func) >= y_threshold ) ### parametrize by mean, min, or max
		if (bTauX == True) and (bTauY == True):
			continue

		apChildren1 = [a]
		apChildren2 = [b]
		if not bTauX:
			apChildren1, distance1 = _cutree([a],  X, func, distance = distance)#_filter_true([a.left,a.right])
			#print "Children 1: "#, apChildren1
			#for node in apChildren1:
				#print reduce_tree(node)
		if not bTauY:
			apChildren2, distance2 = _cutree([b],  Y, func, distance = distance)

		new_distance = numpy.min([distance1, distance2])
		LChild = [(a,b) for a,b in itertools.product( apChildren1, apChildren2 )] 
		#print "After appending", L 
		childList = []
		while LChild:
			(a1,b1) = LChild.pop(0)
			
			#for a,b in itertools.product( apClusterNode1, apClusterNode2 ):
	
			data1 = reduce_tree( a1 )
			data2 = reduce_tree( b1 )
			tempTree = Tree([data1,data2])
			childList.append(tempTree)
			#print childList
			#if len(data1) > 1 or len(data2) > 1:
			next_L.append((tempTree, (a1,b1)))
			#print L					
		pStump.add_children( childList )
		if not L:
			#print "Next level of Coupling"
			L = next_L
			distance = new_distance
			#L.extend(childList)
	#print "Coupled Tree", reduce_tree_by_layer(aOut)
	#return
	return aOut

def couple_tree_all_clusters( apClusterNode1, apClusterNode2, pArray1, pArray2, afThreshold = None, strMethod = "uniform", strLinkage = "min", fAlpha = 0.05, func ="norm_mi"):
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
	tH : py.Tree object 

	Examples
	----------------
	"""
	#Increase recursive size to avoid limit reduce_tree recursion 

#	import copy 

	X,Y = pArray1, pArray2 

	if not afThreshold:	
		afThreshold = [halla.stats.alpha_threshold(a, fAlpha, func ) for a in [pArray1,pArray2]]
	
	x_threshold, y_threshold = afThreshold[0], afThreshold[1]
	#print "x_threshold, y_threshold:", x_threshold, y_threshold
	aTau = [] ### Did the child meet the intra-dataset confidence cut-off? If not, its children will continue to be itself. 
		#### tau_hat <= tau 
		#### e.g.[(True,False),(True,True),]


	## hash containing string mappings for deciders 

	hashMethod = {"min": _decider_min, 
				"max": _decider_max, 
				}

	pMethod = hashMethod[strLinkage] ##returns 'aNode x aNode -> bool' object 


	#==========================================#	
	# See if children pass intradataset test 
	# This should be done at the tree coupling level. 
	#==========================================#
				
	#for i, p in enumerate( aP_children ): 
		 
	#	ai,aj = map(array, aP_children.get_data())

	#	bX = (_mean_tau(X[ai]) <= fAlpha)
	#	bY = (_mean_tau(Y[aj]) <= fAlpha)

	#	aTau.append([bX,bY])

	#-------------------------------------#
	# Parsing Steps                       #
	#-------------------------------------#
	
	## Unalias data structure so this does not alter the original data type
	## Fix for depth 
	aiGlobalDepth1 = [get_depth( ap ) for ap in apClusterNode1]
	aiGlobalDepth2 = [get_depth( ap ) for ap in apClusterNode2]
	
	iMaxDepth = max(max(aiGlobalDepth1),max(aiGlobalDepth2))
	iMinDepth = min(min(aiGlobalDepth1),min(aiGlobalDepth2))
	
	#apClusterNode1 = [fix_clusternode(apClusterNode1[i], iExtend = iMaxDepth - aiGlobalDepth1[i]) for i in range(len(apClusterNode1))]
	#apClusterNode2 = [fix_clusternode(apClusterNode2[i], iExtend = iMaxDepth - aiGlobalDepth2[i]) for i in range(len(apClusterNode2))]
	
	skip = max(aiGlobalDepth1)/max(aiGlobalDepth2)


	print "Hierarchical TREE 1 ", reduce_tree_by_layer(apClusterNode1)
	print "Hierarchical TREE 2 ", reduce_tree_by_layer(apClusterNode2)

	aOut = []
	distance = .98
	#print "apClusterNode1", apClusterNode1
	apChildren1 = _cutree (apClusterNode1, X, func, distance = distance, cluster_threshold =x_threshold)
	#print "apClusterNode1", apClusterNode1

	#print "Qualified Nodes:"
	#for node in apChildren1:
	#	print reduce_tree(node)
	apChildren2 = _cutree (apClusterNode2,  Y, func, distance = distance, cluster_threshold =y_threshold)
	#print "Cluster 1"	
	#print "Qualified Nodes:"
	#for node in apChildren2:
	#	print reduce_tree(node)
	#print "Cluster 2"	
	
	print "Start Nodes 1: ", [reduce_tree( node) for node in apChildren1]
	print "Start Nodes 2: ", [reduce_tree( node) for node in apChildren2]
	'''
	Establish the root of the coupling tree based on based matched clusters in
	the first layer of relevent depth
	'''	
	min_nmi = 0.0
	for node1 in apChildren1:
		for node2 in apChildren2:
			data1 = reduce_tree( node1 )
			data2 = reduce_tree( node2 )
			temp_nmi = func (pca(X[array(data1)])[0] , pca(Y[array(data2)])[0])
			if min_nmi < temp_nmi:
				min_nmi = temp_nmi
				pStump = Tree([data1,data2])
	aOut.append(pStump)
	'''
	End establishing the root of the coupling tree
	'''
	L = []
	childList= []	
	for a,b in itertools.product( apChildren1, apChildren2 ):
		data1 = reduce_tree( a )
		data2 = reduce_tree( b )
		tempTree = Tree([data1,data2])
		childList.append(tempTree)
		L.append((tempTree, (a,b)))
	pStump.add_children( childList )
	L1 = [(pStump, (a,b))]
	#b = apClusterNode2
	#print "satrt", L1
	#root = True
	while L1:
		#print L
		(pStump, (a1,b1)) = L1.pop(0)		
		data1 = reduce_tree( a1 )
		#data = reduce_tree( b1 )
		
		
		L2 = [(pStump, (a1,b1))]
		while L2:
			(pStump, (_,b2)) = L2.pop(0)
			cdata2 = reduce_tree( b2 )
			bTauY = _is_stop(b2, Y, func, distance = distance)#( _min_tau(Y[array(cdata2)], func) >= y_threshold ) ### parametrize by mean, min, or max
			if bTauY:
				continue
			else:
				apChildren2 = _filter_true([b2.left,b2.right])
			if any(apChildren2):
				childList = []
				for b3 in apChildren2:
					#for a,b in itertools.product( apClusterNode1, apClusterNode2 ):
			
					#data1 = reduce_tree( a1 )
					cdata2 = reduce_tree( b3 )
					tempTree = Tree([data1,cdata2])
					childList.append(tempTree)
					L2.append((tempTree, (a1,b3)))				
				pStump.add_children( childList )
				#print "After L2:", reduce_tree( pStump )
		#print "End of while L2"		
		bTauX = _is_stop(a1, X, func, distance = distance)
		if bTauX:
			continue
		else:
			apChildren1 = _filter_true([a1.left,a1.right])
		if any(apChildren1):
			childlist1 = [] 
			for child in apChildren1:
				cdata1 = reduce_tree(child)
				tempTree = Tree([cdata1,data2])
				L1.append((tempTree, (child,b)))
				childlist1.append(tempTree)
			pStump.add_children( childlist1 )
			#print "After L1: ",reduce_tree( pStump )
	#print "End of while L1"
	#print reduce_tree_by_layer(aOut)
	return aOut

def couple_tree_recursive( apClusterNode1, apClusterNode2, pArray1, pArray2, afThreshold = None, strMethod = "uniform", strLinkage = "min", fAlpha = 0.05, func ="norm_mi"):
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
	tH : py.Tree object 

	Examples
	----------------
	"""
	#Increase recursive size to avoid limit reduce_tree recursion 
	#sys.setrecursionlimit(10000)

#	import copy 

	X,Y = pArray1, pArray2 

	if not afThreshold:	
		afThreshold = [halla.stats.alpha_threshold(a, fAlpha, func ) for a in [pArray1,pArray2]]
	
	x_threshold, y_threshold = afThreshold[0], afThreshold[1]
	#print "x_threshold, y_threshold:", x_threshold, y_threshold
	aTau = [] ### Did the child meet the intra-dataset confidence cut-off? If not, its children will continue to be itself. 
		#### tau_hat <= tau 
		#### e.g.[(True,False),(True,True),]

	## hash containing string mappings for deciders 

	hashMethod = {"min": _decider_min, 
				"max": _decider_max, 
				}

	pMethod = hashMethod[strLinkage] ##returns 'aNode x aNode -> bool' object 


	#==========================================#	
	# See if children pass intradataset test 
	# This should be done at the tree coupling level. 
	#==========================================#
				
	#for i, p in enumerate( aP_children ): 
		 
	#	ai,aj = map(array, aP_children.get_data())

	#	bX = (_mean_tau(X[ai]) <= fAlpha)
	#	bY = (_mean_tau(Y[aj]) <= fAlpha)

	#	aTau.append([bX,bY])

	#-------------------------------------#
	# Parsing Steps                       #
	#-------------------------------------#
	
	## Unalias data structure so this does not alter the original data type
	## Fix for depth 
	aiGlobalDepth1 = [get_depth( ap ) for ap in apClusterNode1]
	aiGlobalDepth2 = [get_depth( ap ) for ap in apClusterNode2]
	
	iMaxDepth = max(max(aiGlobalDepth1),max(aiGlobalDepth2))
	iMinDepth = min(min(aiGlobalDepth1),min(aiGlobalDepth2))
	
	apClusterNode1 = [fix_clusternode(apClusterNode1[i], iExtend = iMaxDepth - aiGlobalDepth1[i]) for i in range(len(apClusterNode1))]
	apClusterNode2 = [fix_clusternode(apClusterNode2[i], iExtend = iMaxDepth - aiGlobalDepth2[i]) for i in range(len(apClusterNode2))]
	
	skip = max(aiGlobalDepth1)/max(aiGlobalDepth2)

	#print "General2: ", max(aiGlobalDepth1),max(aiGlobalDepth2)
	'''
	aiGlobalDepth1 = [get_depth( ap ) for ap in apClusterNode1]
	aiGlobalDepth2 = [get_depth( ap ) for ap in apClusterNode2]
	if aiGlobalDepth1 > 2*aiGlobalDepth2:
		apClusterNode1 = truncate_tree( apClusterNode1, level = 0, skip = max(aiGlobalDepth1)/max(aiGlobalDepth2) ) 
	if aiGlobalDepth1*2 < aiGlobalDepth2:
		apClusterNode2 = truncate_tree( apClusterNode2, level = 0, skip = max(aiGlobalDepth2)/max(aiGlobalDepth1) )
	# End Truncate
	'''
	print "Hierarchical TREE 1 ", reduce_tree_by_layer(apClusterNode1)
	print "Hierarchical TREE 2 ", reduce_tree_by_layer(apClusterNode2)
	def _couple_tree_recursive( apClusterNode1, apClusterNode2, strMethod = strMethod, strLinkage = strLinkage ):
		"""
		recursive function 
		"""
		
		aOut = []
		'''Depth1 = [get_depth( ap ) for ap in apClusterNode1]
		Depth2 = [get_depth( ap ) for ap in apClusterNode2]
				
		print max(Depth1),max(Depth2)
		'''
		for a,b in itertools.product( apClusterNode1, apClusterNode2 ):

			data1 = reduce_tree( a )
			data2 = reduce_tree( b )

			pStump = Tree([data1,data2])

			bTauX = ( _min_tau(X[array(data1)], func) >= x_threshold ) ### parametrize by mean, min, or max
			bTauY = ( _min_tau(Y[array(data2)], func) >= y_threshold ) ### parametrize by mean, min, or max

			if bTauX:
				apChildren1 = _filter_true([a])
			else:
				if skip > 1:
					# Starte Truncate larger tree to have Hierarchical trees in the samle scale in terms of depth 
					apChildren1 = truncate_tree( a, level = 0, skip = skip )
				else:
					apChildren1 = _filter_true([a.left,a.right])

			if bTauY:
				apChildren2 = _filter_true([b])
			else:
				# Starte Truncate larger tree to have Hierarchical trees in the samle scale in terms of depth 
				if skip > 1:
					apChildren2 = truncate_tree( b, level = 0, skip = skip )
				else:
					apChildren2 = _filter_true([b.left,b.right])

			##Children should _already be_ adjusted for depth 
			if not(any(apChildren1)) or not(any(apChildren2)):
				aOut.append( pStump )

			elif (bTauX == True) and (bTauY == True):
				aOut.append( pStump ) ### Do not continue on, since alpha threshold has been met.

			else: 		
				aOut.append( pStump.add_children( _couple_tree_recursive( apChildren1, apChildren2, strMethod = strMethod, strLinkage = strLinkage ) ) )

		return aOut 
	result = _couple_tree_recursive( apClusterNode1, apClusterNode2, strMethod, strLinkage )
	return result
def couple_tree_iterative_by_actual_depth( apClusterNode1, apClusterNode2, pArray1, pArray2, afThreshold = None, strMethod = "uniform", strLinkage = "min", fAlpha = 0.05, func ="norm_mi"):
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
	tH : py.Tree object 

	Examples
	----------------
	"""
	#Increase recursive size to avoid limit reduce_tree recursion 
	#sys.setrecursionlimit(10000)

#	import copy 

	X,Y = pArray1, pArray2 

	if not afThreshold:	
		afThreshold = [halla.stats.alpha_threshold(a, fAlpha, func ) for a in [pArray1,pArray2]]
	
	x_threshold, y_threshold = afThreshold[0], afThreshold[1]
	#print "x_threshold, y_threshold:", x_threshold, y_threshold
	aTau = [] ### Did the child meet the intra-dataset confidence cut-off? If not, its children will continue to be itself. 
		#### tau_hat <= tau 
		#### e.g.[(True,False),(True,True),]

	## hash containing string mappings for deciders 

	hashMethod = {"min": _decider_min, 
				"max": _decider_max, 
				}

	pMethod = hashMethod[strLinkage] ##returns 'aNode x aNode -> bool' object 


	#==========================================#	
	# See if children pass intradataset test 
	# This should be done at the tree coupling level. 
	#==========================================#
				
	#for i, p in enumerate( aP_children ): 
		 
	#	ai,aj = map(array, aP_children.get_data())

	#	bX = (_mean_tau(X[ai]) <= fAlpha)
	#	bY = (_mean_tau(Y[aj]) <= fAlpha)

	#	aTau.append([bX,bY])

	#-------------------------------------#
	# Parsing Steps                       #
	#-------------------------------------#
	
	## Unalias data structure so this does not alter the original data type
	## Fix for depth 
	aiGlobalDepth1 = [get_depth( ap ) for ap in apClusterNode1]
	aiGlobalDepth2 = [get_depth( ap ) for ap in apClusterNode2]
	
	iMaxDepth = max(max(aiGlobalDepth1),max(aiGlobalDepth2))
	iMinDepth = min(min(aiGlobalDepth1),min(aiGlobalDepth2))
	
	apClusterNode1 = [fix_clusternode(apClusterNode1[i], iExtend = iMaxDepth - aiGlobalDepth1[i]) for i in range(len(apClusterNode1))]
	apClusterNode2 = [fix_clusternode(apClusterNode2[i], iExtend = iMaxDepth - aiGlobalDepth2[i]) for i in range(len(apClusterNode2))]
	
	skip = max(aiGlobalDepth1)/max(aiGlobalDepth2)

	#print "General2: ", max(aiGlobalDepth1),max(aiGlobalDepth2)
	'''
	aiGlobalDepth1 = [get_depth( ap ) for ap in apClusterNode1]
	aiGlobalDepth2 = [get_depth( ap ) for ap in apClusterNode2]
	if aiGlobalDepth1 > 2*aiGlobalDepth2:
		apClusterNode1 = truncate_tree( apClusterNode1, level = 0, skip = max(aiGlobalDepth1)/max(aiGlobalDepth2) ) 
	if aiGlobalDepth1*2 < aiGlobalDepth2:
		apClusterNode2 = truncate_tree( apClusterNode2, level = 0, skip = max(aiGlobalDepth2)/max(aiGlobalDepth1) )
	# End Truncate
	'''
	print "Hierarchical TREE 1 ", reduce_tree_by_layer(apClusterNode1)
	print "Hierarchical TREE 2 ", reduce_tree_by_layer(apClusterNode2)
	def _couple_tree_itrative( apClusterNode1, apClusterNode2, strMethod = strMethod, strLinkage = strLinkage ):
		#Nonrecursive _couple_tree
		'''
		nonrecursive function 
		'''
		#print "TREE1 ", reduce_tree_by_layer(apClusterNode1)
		#print "TREE2 ", reduce_tree_by_layer(apClusterNode2)
		aOut = []
		for a,b in itertools.product( apClusterNode1, apClusterNode2 ):
			data1 = reduce_tree( a )
			data2 = reduce_tree( b )
	
		pStump = Tree([data1,data2])
		aOut.append(pStump)
		L = [(pStump, (a,b))]
		#print "satrt", L
		while L:
			#print L
			(pStump, (a,b)) = L.pop(0)
			#print "after pop", L 
			#for a,b in itertools.product( apClusterNode1, apClusterNode2 ):
			
			data1 = reduce_tree( a )
			data2 = reduce_tree( b )
			
			#pStump = Tree([data1,data2])
					
			bTauX = ( _min_tau(X[array(data1)], func) >= x_threshold ) ### parametrize by mean, min, or max
			bTauY = ( _min_tau(Y[array(data2)], func) >= y_threshold ) ### parametrize by mean, min, or max
	
			if bTauX:
				apChildren1 = _filter_true([a])
			else:
				if skip > 1:
					# Starte Truncate larger tree to have Hierarchical trees in the samle scale in terms of depth 
					apChildren1 = truncate_tree( a, level = 0, skip = skip )
				else:
					apChildren1 = _filter_true([a.left,a.right])
	
			if bTauY:
				apChildren2 = _filter_true([b])
			else:
				# Starte Truncate larger tree to have Hierarchical trees in the samle scale in terms of depth 
				if skip > 1:
					apChildren2 = truncate_tree( b, level = 0, skip = skip )
				else:
					apChildren2 = _filter_true([b.left,b.right])
	
			##Children should _already be_ adjusted for depth 
			if not(any(apChildren1)) and not(any(apChildren2)):
				#aOut.append( pStump )
				continue
	
			elif (bTauX == True) and (bTauY == True):
				#aOut.append( pStump ) ### Do not continue on, since alpha threshold has been met.
				continue
	
			else:
				LChild = [(a,b) for a,b in itertools.product( apChildren1, apChildren2 )] 
				
				#print "After appending", L 
				childList = []
				while LChild:
					(a1,b1) = LChild.pop(0)
					
					#for a,b in itertools.product( apClusterNode1, apClusterNode2 ):
			
					data1 = reduce_tree( a1 )
					data2 = reduce_tree( b1 )
					tempTree = Tree([data1,data2])
					childList.append(tempTree)
					#print childList
					L.append((tempTree, (a1,b1)))
					#print L					
				pStump.add_children( childList )
				#L.extend(childList)
		#print reduce_tree_by_layer(aOut)
		return aOut
	
	#start_time = time.time()
	result = _couple_tree_itrative( apClusterNode1, apClusterNode2, strMethod, strLinkage )
	#print("--- %s seconds ---" % (time.time() - start_time))
	#print "Coupled Hypothesis Tree I", reduce_tree_by_layer(result)
	
	return result
		
def naive_all_against_all( pArray1, pArray2, strMethod = "permutation_test_by_representative", iIter = 100 ):

	phashMethods = {"permutation_test_by_representative" : halla.stats.permutation_test_by_representative, 
					"permutation_test_by_representative_mic" : halla.stats.permutation_test_by_representative_mic,
						"permutation_test_by_average" : halla.stats.permutation_test_by_average,
						"parametric_test" : halla.stats.parametric_test}

	iRow = len(pArray1)
	iCol = len(pArray2)

	aOut = [] 

	for i,j in itertools.product( range(iRow), range(iCol) ):

		pDist = phashMethods[strMethod]
		fVal = pDist( array([pArray1[i]]), array([pArray2[j]]), iIter = iIter )
		aOut.append([[i,j],fVal])

	aOut_header = zip(*aOut)[0]
	aOut_adjusted = halla.stats.p_adjust( zip(*aOut)[1] )

	return zip(aOut_header,aOut_adjusted)
	#return numpy.reshape( aOut, (iRow,iCol) )


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

#### BUGBUG: When q = 1.0, results should be _exactly the same_ as naive all_against_all, but something is going on that messes this up
#### Need to figure out what -- it's probably in the p-value consolidation stage 
#### Need to reverse sort by the sum of the two sizes of the bags; the problem should be fixed afterwards 

def all_against_all( pTree, pArray1, pArray2, method = "permutation_test_by_representative", metric = "norm_mi", p_adjust = "BH", fQ = 0.1, 
	iIter = 1000, pursuer_method = "nonparameteric", step_parameter = 1.0, start_parameter = 0.0, bVerbose = False, afThreshold = None, fAlpha = 0.05):
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

		
	"""
	global number_performed_test 
	number_performed_test = 0
	X,Y = pArray1, pArray2 

	if bVerbose:
		print reduce_tree_by_layer( [pTree] )

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

	iSkip = _start_parameter_to_iskip( start_parameter )

	#print "layers to skip:", iSkip 

	aOut = [] ## Full log 
	aFinal = [] ## Only the final reported values 

	iGlobalDepth = depth_tree( pTree )
	#iSkip = _start_parameter_to_iskip( start_parameter )
    # step 3: to add a new method to HAllA (extension step)
    # for examplewe add "permutation_test_by_ica_norm_mi": py.stats.permutation_test_by_ica_norm_mi
	pHashMethods = {"permutation_test_by_representative" : halla.stats.permutation_test_by_representative, 
						"permutation_test_by_average" : halla.stats.permutation_test_by_average,
						"parametric_test" : halla.stats.parametric_test,
						"permutation_test_by_kpca_norm_mi" :halla.stats.permutation_test_by_kpca_norm_mi, 
						"permutation_test_by_kpca_pearson" :halla.stats.permutation_test_by_kpca_pearson,
						"permutation_test_by_cca_pearson" :halla.stats.permutation_test_by_cca_pearson,
						"parametric_test_by_pls_pearson": halla.stats.parametric_test_by_pls_pearson,
						"permutation_test_by_cca_norm_mi" :halla.stats.permutation_test_by_cca_norm_mi,
						"permutation_test_by_multiple_representative" : halla.stats.permutation_test_by_multiple_representative,
						"parametric_test_by_representative": halla.stats.parametric_test_by_representative, 
						"permutation_test_by_medoid": halla.stats.permutation_test_by_medoid,
						"permutation_test_by_pls_norm_mi": halla.stats.permutation_test_by_pls_norm_mi,
						"permutation_test_by_representative_mic" : halla.stats.permutation_test_by_representative_mic,
						"permutation_test_by_representative_adj_mi" : halla.stats.permutation_test_by_representative_adj_mi,
						 "permutation_test_by_ica_norm_mi": halla.stats.permutation_test_by_ica_norm_mi,
						 "permutation_test_by_ica_mic": halla.stats.permutation_test_by_ica_mic
						}

	strMethod = method
	pMethod = pHashMethods[strMethod]
	
	def _simple_descending():
		L = pTree.get_children()
		global number_performed_test
		number_performed_test += len(L)
		while L:
			currentNode = L.pop(0)
			print currentNode.get_data()
			aiI,aiJ = map( array, currentNode.get_data() )
			p_value = pMethod( pArray1[aiI], pArray2[aiJ])
			aOut.append( [currentNode.get_data(), float(p_value), float(p_value)] )
			if p_value <= fQ:
				print "************Pass with p-value:", p_value
				aFinal.append( [currentNode.get_data(), float(p_value), float(p_value)] )
			elif p_value >fQ and p_value <= 1.0 - fQ:
				L += currentNode.get_children()
				#global number_performed_test
				number_performed_test += len(currentNode.get_children())
				print "Conitinue, gray area with p-value:", p_value
			else:
				print "Stop: no chance of association by descending", p_value
		'''if p_value <= fQ or iSkip >= 1:
			aFinal.append( [pTree.get_data(), float(p_value)] )
			_fw_operator( pTree, p_value = p_value )
		elif p_value >fQ or p_value <= 1.0- fQ and iSkip >=1:
			_fw_operator( pTree, p_value = p_value )
		'''
		if bVerbose:
			print aFinal 
			print "length is", len(aFinal)
			
		return aFinal, aOut
	
	def _ybh_descending():
		apChildren = [pTree]
		level = 1
		global number_performed_test
		
		next_level_apChildren = []
		while apChildren:
			Current_Family_Children = apChildren.pop(0).get_children()
			number_performed_test += len(Current_Family_Children)
			aP = [ _actor( c ) for c in Current_Family_Children ]
			aP_adjusted, pRank = halla.stats.p_adjust( aP, fQ )
			max_r_t = 0
			#print "aP", aP
			#print "aP_adjusted: ", aP_adjusted  
			for i in range(len(aP)):
				if aP[i] <= aP_adjusted[i] and max_r_t <= pRank[i]:
					max_r_t =  pRank[i]
					#print "max_r_t", max_r_t
			for i in range(len(aP)):
				if pRank[i] <= max_r_t:
					print "************Pass with p-value:", aP[i], Current_Family_Children[i].get_data()[0], Current_Family_Children[i].get_data()[1]
					aOut.append( [Current_Family_Children[i].get_data(), float(aP[i]), aP_adjusted[i]] )
					aFinal.append( [Current_Family_Children[i].get_data(), float(aP[i]), aP_adjusted[i]] )
				else :
					aOut.append( [Current_Family_Children[i].get_data(), float(aP[i]), aP_adjusted[i]] )
					if not Current_Family_Children[i].is_leaf() and aP[i]* level/(len(Current_Family_Children[i].get_data()[0]) * len(Current_Family_Children[i].get_data()[1])) <= fQ:
						#print "Gray area with p-value:", aP[i]
						next_level_apChildren.append(Current_Family_Children[i])
					elif Current_Family_Children[i].is_leaf():
						#print "End of branch, leaf!"
						continue
					else:
						print "Bypass, no hope to find an association in the branch with p-value: ",aP[i]," and ",len(Current_Family_Children[i].get_children()), " sub-hepotheses."
			#global number_performed_test		
			#print "No association in:", apChildren
			if not apChildren:
				print "Hypotheses testing level ",level," is finished."
				#number_performed_test += len(next_level_apChildren)
				apChildren = next_level_apChildren
				level += 1
				next_level_apChildren =[]
												
		return aFinal, aOut
	
	def _actor( pNode ):
		"""
		Performs a certain action at the node

			* E.g. compares two bags, reports distance and p-values 
		"""

		aIndicies = pNode.get_data( ) 
		aIndiciesMapped = map( array, aIndicies ) ## So we can vectorize over numpy arrays 
		dP = pMethod( pArray1[aIndiciesMapped[0]], pArray2[aIndiciesMapped[1]], iIter = iIter )

		#aOut.append( [aIndicies, dP] ) #### dP needs to appended AFTER multiple hypothesis correction

		return dP 

	
	#======================================#
	# Execute 
	#======================================#
	#aFinal, aOut = _simple_descending()
	aFinal, aOut = _ybh_descending()
	print "____Number of performed test:", number_performed_test
	return aFinal, aOut 