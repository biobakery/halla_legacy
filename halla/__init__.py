#!/usr/bin/env python 

"""
An object-oriented halla prototype 
Aim to be as self-containied as possible 

Global namespace conventions: 

	m() <- map for arrays 
	r() <- reduce for arrays 
	rd() <- generic reduce-dimension method 
	
"""

## native python packages 

import itertools 

## structural packages 

import sys 
import re 
import os 
import pprint 

## halla modules 
from stats import discretize, p_val_plot
from distance import mi, adj_mi, l2, mid, adj_mid 
import hierarchy 
from test import * 


## statistics packages 

import numpy as np
import scipy as sp
from numpy import array 
import sklearn.decomposition
import matplotlib 
#matplotlib.use("Agg") #disable X-windows display backend 
from sklearn.decomposition import PCA #remember that the matrix is X = (n_samples,n_features)
import csv 
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree, leaves_list
import pylab as pl 
import random  
from numpy.random import normal 
from scipy.misc import * 
from scipy.stats import kruskal, ttest_ind, ttest_1samp, percentileofscore, pearsonr
import pandas as pd 

class HAllA():
	
	def __init__( self, *ta ): 
		
		self.hashMethods = {
					"id":			lambda x: x.id, 
					"l2":			lambda x,y: np.linalg.norm(x-y),
					"ttest": 		lambda x,y: ttest_ind(x,y), 
					"kruskal":		lambda x,y: kruskal(x,y),

					}  

		## Think about lazy implementation to save time during run-time;
		## Don't have to keep everything in memory 


		self.htest = ttest_ind
		self.distance = adj_mid 
		
		self.rep = None 
		self.meta_array = array( ta )

		self.meta_discretize = None 
		self.meta_linkage = None 
		self.meta_distance = None 
		self.directory = None 
		self.outhash = {} 
		self.outtable = None

		self.header = ["Var", "MID", "pBoot", "pPerm"]

		## this is not so efficient for huge arrays, fix later 

		self.m_iIter = 100
	
	def _issingle( self ):
		
		return ( self.meta_array[0] == self.meta_array[1] ).all()

	def set_directory( self, strDir ):
		self.directory = strDir 
		#return self.directory 

	#@staticmethod 
	#def reduce_tree( pClusterNode ):
	#	return reduce_tree( pClusterNode ) 

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

	 
	def permute_by_column( self, pArray ):
		return array( [np.random.permutation( pArray[:,i] ) for i in range(pArray.shape[1])] ).T 

	 
	def permute_by_row( self, pArray ):
		return array( [np.random.permutation( sub ) for sub in pArray] )

	 
	def _bootstrap( self, pArray ):
		""" 
		returns one instance of bootstrap 
		"""
		return array( [random.sample( pSub, 1 ) for pSub in pArray] )
		 
	 
	def bootstrap_by_column( self, pArray, pFilter = None, iIter = 100):
		"""
		return iIter copies of the bootstrap 
		"""

		iRow, iCol = pArray.shape 

		pFilter = pFilter or np.arange( iRow )

		if iIter == 1:
			return self._bootstrap( pArray[pFilter] )
		else:
			return np.hstack( [ self._bootstrap( pArray[pFilter] ) ] * iIter )


	#BUGBUG: bootstrap test is garbage now ... 
	def _bootstrap_test( self, pArray1, pArray2, pMedoid1, pMedoid2, iX, iY ): 

		iIter = self.m_iIter 
		
		dMID = self.outhash[(iX,iY)]["MID"]
		pArrayAll = [ self.distance( self._get_medoid( self.bootstrap_by_column( pArray1 ) ), pMedoid2 ) ] * iIter
		pArrayDist12 = [ self.distance( self.bootstrap_by_column( pArray1 )[iX], pMedoid2 ) ] * iIter

		dU, dPBoot = ttest_ind( pArrayAll, pArrayDist12 )
		dPBoot /= 2 
		
		if np.average( pArrayDist12 ) > np.average( pArrayAll ):
			dPBoot = 1- dPBoot 

		self.outhash[(iX,iY)]["pBoot"] = dPBoot

		print "dPBoot is " + str(dPBoot)
	
		return dPBoot


	#Permutation test / statistical testing by random matrix theory is the way to go ... 
	def _permute_test( self, pArray1, pArray2, pMedoid1, pMedoid2, iX, iY ):

		iIter = self.m_iIter 

		dMID = self.outhash[(iX,iY)]["MID"]
		#pArrayPerm = [ self.distance( self.permute_by_column( pArray1 )[iX], pMedoid2 ) for i in xrange( iIter )]  

		#print "Perm is "
		#print self.permute_by_row( pArray1 )[iX]
		
		#print np.sort( self.permute_by_row( pArray1 )[iX] ) 
		#print np.sort( pMedoid1 ) 

		#print "Permuted distance"
		#print self.distance( self.permute_by_row( pArray1 )[iX], pMedoid2 )	
		#print "sorted distance"
		#print self.distance( np.sort( self.permute_by_row( pArray1 )[iX] ) , np.sort( pMedoid1 ) )
		#print "normal"
		#print self.distance( pMedoid1, pMedoid2 ) 

		#assert( self.distance( self.permute_by_row( pArray1 )[iX], pMedoid2 ) == self.distance( pMedoid1, pMedoid2 ) )
		
		pArrayPerm = [ self.distance( self.permute_by_row( pArray1 )[iX], pMedoid2 ) for i in xrange( iIter )]  

		#print "ArrayPerm is " 
		#print pArrayPerm
		
		#dPPerm = 1- ( percentileofscore( [dMID-1] + pArrayPerm, dMID ) / 100 )

		## smaller distance means closer 

		dPPerm = percentileofscore( pArrayPerm, dMID ) / 100

		self.outhash[(iX,iY)]["pPerm"] = dPPerm

		#print "dPPerm is " + str( dPPerm )

		return dPPerm

	def _discretize( self ):
		self.meta_discretize = self.m( self.meta_array, discretize )
		## Should do a better job at detecting whether dataset is 
		return self.meta_discretize 

	def _distance_matrix( self ):
		self.meta_distance =  self.m( self.meta_array, lambda x: pdist( x, metric=self.distance ) )
		return self.meta_distance 

	def _linkage( self ):
		self.meta_linkage = self.m( self.meta_array, linkage ) 
		return self.meta_linkage 

	def _get_medoid( self, pArray, iAxis = 0, pMetric = "l2" ):
		"""
		Input: numpy array 
		Output: float
		
		For lack of better way, compute centroid, then compute medoid 
		by looking at an element that is closest to the centroid. 

		Can define arbitrary metric passed in as a function to pMetric 

		"""

		if len(pArray) == 1:
			""" 
			Sanity check case
			"""
			return pArray 

		else: 
			d = self.hashMethods[pMetric]

			pArray = ( pArray.T if bool(iAxis) else pArray  ) 

			mean_vec = np.mean(pArray, 0) 
			
			pArrayCenter = pArray - ( mean_vec * np.ones(pArray.shape) )

			return pArray[np.argsort( map( np.linalg.norm, pArrayCenter) )[0],:]

	def _representative( self, pArray, pMethod = None ):
		hash_method = {None: self._get_medoid}
		return hash_method[pMethod]( pArray )

	def _reduce_tree( self, pClusterNode, pFunction = lambda x: x.id, aOut = [] ):
		func = pFunction

		if pClusterNode.is_leaf():
			return ( aOut + [func(pClusterNode)] )
		else:
			return _reduce_tree( pClusterNode.left, func, aOut ) + _reduce_tree( pClusterNode.right, func, aOut ) 
	
	def _reduce_children( pClusterNode ):
		
		node = pClusterNode 
		node_left, node_right = node.left, node.right 
		if node_left.get_id() and node_right.get_id():
			return (array( self._reduce_tree( node_left ) ), 
				array( self._reduce_tree( node_right ) ) )  	

	def _htestpair( self ):
		assert( len( self.meta_array ) >= 2 )
		tree1, tree2 = map( to_tree, self.meta_linkage[:2] )	
		
		self.outhash[(0,0)] = self._ttest( self._representative( self.meta_array[0] ), 
			self._representative( self.meta_array[1] ) ) 			

		return self.outhash 

	def _cakecut( self ):
		"""
		Run tests for cake cutting procedure 
		"""

		rand_sample = rand( (100,1) ).flatten()  
		rand_mixture = array( uniformly_spaced_gaussian( 8 ) )

		pRaw1, pRaw2 = self.meta_array[0], self.meta_array[1]

		pOut = p_val_plot( pRaw1, pRaw2, pCut = lambda x: stats.log_cut(x), iIter = 100 )

		return pOut 

	def _htest_rev1( self ):
		"""
		Run htest for revision 1 
		Simple discretized hypothesis tree, 

		"""

		iSkip = 0

		pRaw1, pRaw2 = self.meta_array[0], self.meta_array[1]

		pData1, pData2 = self.meta_discretize[0], self.meta_discretize[1]

		pClusterNode1Tmp, pClusterNode2Tmp = hierarchy.hclust( pData1 , bTree = True ), hierarchy.hclust( pData2, bTree = True)

		apClusterNode1, apClusterNode2 = hierarchy.truncate_tree( [pClusterNode1Tmp], iSkip ), hierarchy.truncate_tree( [pClusterNode2Tmp], iSkip )

		#Z1, Z2 = hierarchy.hclust( pData1 ), hierarchy.hclust( pData2 ) 

		#Plot to see 
		#sp.cluster.hierarchy.dendrogram( Z1 )
		#sp.cluster.hierarchy.dendrogram( Z2 )

		pBags = hierarchy.recursive_all_against_all( apClusterNode1, apClusterNode2, pRaw1, pRaw2, pOut = [] )
		#pBags = hierarchy.recursive_all_against_all( [pClusterNode1], [pClusterNode2], pRaw1, pRaw2, pOut = [] )

		print "pBags"

		for item in pBags:
			print item 


	def _htest_pr( self ):
		""" 
		Run htest for the progress report 
		This is assuming that the standard suite of preprocessing functions have been run 
		"""

		pRaw1, pRaw2 = self.meta_array[0], self.meta_array[1]

		pData1, pData2 = self.meta_discretize[0], self.meta_discretize[1] 
		
		## BUGBUG: Uncomment this for the general case later 
		## pMedoid1, pMedoid2 = self._representative( pData1 ), self._representative( pData2 )
		iRow1, iCol1 = pData1.shape 
		iRow2, iCol2 = pData2.shape 

		#iterate through every single pair 

		gen_product = itertools.combinations( xrange(iRow1), 2 ) if self._issingle() else itertools.product( xrange(iRow1), xrange(iRow2) )

		for tPair in gen_product:
			iOne, iTwo = tPair
			pMedoid1, pMedoid2 = pData1[iOne], pData2[iTwo]

			sys.stderr.write( "iteration %s,%s \n" %(iOne, iTwo) )

			#print "first medoid:"
			#print pMedoid1 
			#print "second medoid"
			#print pMedoid2 

			#print "iteration %s,%s" % (iOne, iTwo )

			## convention: when bootstrapping and permuting, the left dataset is the one that the action is applied to. 
			
			dMID =  self.distance( pData1[iOne], pData2[iTwo] ) 

			#print "dMID is " + str(dMID)

			#print "raw1"
			#print pRaw1[iOne]
			#print "raw2"
			#print pRaw2[iTwo]

			#print "data1"
			#print pData1[iOne]
			#print "data2"
			#print pData2[iOne]

			#dPearsonr, dPearsonp = pearsonr( pData1[iOne], pData2[iTwo] )
			try:
				dPearsonr, dPearsonp = pearsonr( pRaw1[iOne], pRaw2[iTwo] )
				sys.stderr.write(str(dPearsonr) + ", " + str(dPearsonp) + "\n" )
			except ValueError:
				dPearsonr, dPearsonp = np.nan, np.nan 
			
			self.outhash[(iOne,iTwo)] = {"MID": dMID, "pPearson": dPearsonp, "rPearson": dPearsonr}

			#self._bootstrap_test( pData1, pData2, pMedoid1, pMedoid2, iOne, iTwo )
			self._permute_test( pData1, pData2, pMedoid1, pMedoid2, iOne, iTwo )

	def _htest_baseline( self ):
		pass 

	def _htest( self ):
		pass 

	def _save_table( self ):

		import csv 

		csvw = csv.writer(open( sys.stdout ), csv.excel_tab )

		csvw.writerow( self.header )

		for k,v in self.outhash.items():
			csvw.writerow( [k] + v )

	def _plot_dendrogram( self ):
		for i, pArray in enumerate( self.meta_linkage ):
			
			pl.clf()
			pl.figure(i)
			dendrogram( pArray ) 
			pl.title( str( self.distance ) + " " + str(i) )
			pl.savefig( self.directory + str(i) + ".pdf" ) 
				
	def run( self ):
		self._discretize()
		self._distance_matrix()
		self._linkage()
		self._plot_dendrogram()
		print self._htestpair()

	def run_pr_test( self ):
		self._discretize()
		self._htest_pr()

		return self.outhash 

	def run_rev1_test( self ):
		self._discretize()
		self._htest_rev1()

	def run_caketest( self ):
		from pylab import * 
		import numpy 

		print "running caketest ... "
		print "OUTPUT"
		pOut = self._cakecut() 

		print "length"
		print len(pOut)

		for item in pOut:
			print "average"
			print numpy.average(item)

		pOut.reverse()

		boxplot( pOut, 0, '', 0)
		show() 


		#return self._cakecut()



