"""
HAllA: Hiearchical All-against All 
==============================================

Description
 An object-oriented halla implementation 
 Aim to be as self-contained as possible 

Global namespace conventions: 

	* `m()` <- map for arrays 
	* `r()` <- reduce for arrays 
	* `rd()` <- generic reduce-dimension method 

Design direction: 

	* never ever import anything directly from an external module; 
	first wrap around interal module, so that abstraction even within 
	development is strictly enforced. 

"""

## native python packages 

import itertools 

## structural packages 

import sys 
import re 
import os 
import pprint 
import csv
import random 

## halla modules 
from test import * 
from stats import * 
from distance import * 
from hierarchy import * 

## statistics packages 

import numpy as np
import scipy as sp
from numpy import array 
import matplotlib 

## miscellaneous 
#matplotlib.use("Agg") #disable X-windows display backend; this is for batch mode ; remember! 

class HAllA():
	
	def __init__( self, *ta ): 

		## Think about lazy implementation to save time during run-time;
		## Don't have to keep everything in memory 
		## Write so that you can feed in a tuple of numpy.ndarrays; in practice the core unit of comparison is always
		## the pair of arrays

		## Parameters  
		self.q = 0.05 
		self.distance = adj_mid 
		self.iterations = 100		
		self.reduce_method = "pca" 
		self.step_function = "uniform"

		self.ebar_method = "permutation" #method to generate error bars 
		
		## Static Meta Objects 
		self.hash_reduce_method = {"pca"	: None, 
									"mca"	: None, }

		self.hash_metric 		= {"norm_mid" : norm_mid }

		# Presets set by the programmer which is determined to be useful for the user 
		self.hash_preset = 	{"default"		: None, 
								"time"		: None, 
								"accuracy"	: None, 
								"parallel"	: None, }

		## Mutable Meta Objects 
		self.meta_array = array( ta ) if ta else None 
		self.meta_discretize = None

		## Output 
		self.directory = None 
		self.hashOut = {} 
		self.tableOut = None

	#==========================================================#
	# Static Methods 
	#==========================================================# 

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

	@staticmethod 
	def r( ):
		pass 

	@staticmethod 
	def rd( ):
		pass 

	#==========================================================#
	# Helper Functions 
	#==========================================================# 

	def _discretize( self ):
		self.meta_discretize = self.m( self.meta_array, discretize )
		# Should do a better job at detecting whether dataset is categorical or continuous
		# Take information from the parser module 
		return self.meta_discretize 

	def _hclust( self ):
		pass 

	def _tcouple( self ):
		pass 

	def _all_against_all( self ):
		pass 

	def _compare( self ):
		pass 

	def _report( self ):
		"""
		helper function for reporting the output to the user 
		"""
		pass 

	def _load_preset( self ):
		pass 

	#==========================================================#
	# Set parameters 
	#==========================================================# 

	def set_q( self, fQ ):
		self.q = fQ
		return self.q 

	def set_metric( self, pMetric ):
		if isinstance( pMetric, str ):
			self.distance = self.hash_metric[pMetric]
		else:
			self.distance = pMetric 
		return self.distance 

	def set_iterations( self, iIterations ):
		self.m_iIter = iIterations
		return self.iterations 

	def set_step_function( self, strFun ):
		"""
		set step function used to couple tree to make hypothesis tree 
		"""
		pass 

	def set_preset( self ):
		pass 

	#==========================================================#
	# Presets  
	#==========================================================# 

	def __preset_default( self ):
		pass 

	def __preset_time( self ):
		pass 

	#==========================================================#
	# Main Pipeline 
	#==========================================================# 	

	def get_attribute( self ):
		"""
		returns current attributes and statistics about HAllA object implementation 
		"""
		pass 

	def run( self ):
		"""
		Main run module 
		"""
		X,Y = self.meta_array 
		dX, dY = self._discretize( )

		tX, tY = hclust( X, bTree = True ), hclust( Y, bTree = True )

		tH = couple_tree( [tX], [tY] )[0]

		aOut = all_against_all( tH, X, Y )

		return aOut 


####################################################################################

#=======================================================================#
# LEGACY CODE -- to be incoroporated into later or to be tossed away 
#=======================================================================#

def legacy():

	def _issingle( self ):
		bOut = False
		aTmp = ( self.meta_array[0] == self.meta_array[1] )
		try:
			bOut = aTmp.all()
		except Exception:
			pass  

		return bOut 
	 
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
		
		pArrayPerm = [ self.distance( self.permute_by_row( pArray1 )[iX], pMedoid2 ) for i in xrange( iIter )]  

		dPPerm = percentileofscore( pArrayPerm, dMID ) / 100

		self.outhash[(iX,iY)]["pPerm"] = dPPerm

		return dPPerm

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

		"""
		View

		#Z1, Z2 = hierarchy.hclust( pData1 ), hierarchy.hclust( pData2 ) 

		#Plot to see 
		#sp.cluster.hierarchy.dendrogram( Z1 )
		#sp.cluster.hierarchy.dendrogram( Z2 )
		"""

		pBags = hierarchy.recursive_all_against_all( apClusterNode1, apClusterNode2, pRaw1, pRaw2, pOut = [] )
		
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
