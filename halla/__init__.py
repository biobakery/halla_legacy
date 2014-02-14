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

	* Never ever import anything directly from an external module; 
	first wrap around interal module, so that abstraction even within 
	development is strictly enforced. 
	* Try to keep around string pointers as much as possible, as opposed to pointers to actual implementation, this string can then be passed 
	to hashes within specific functions in external submodules. 
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
from plot import * 

## statistics packages 

import numpy as np
import scipy as sp
from numpy import array 
import matplotlib 

## miscellaneous 
#matplotlib.use("Agg") #disable X-windows display backend; this is for batch mode ; remember! 

class HAllA():
	
	def __init__( self, *ta ): 
		"""
		Think about lazy implementation to save time during run-time;
		Don't have to keep everything in memory 
		Write so that you can feed in a tuple of numpy.ndarrays; in practice the core unit of comparison is always
		the pair of arrays
		"""

		## BEGIN INIT

		#==================================================================#
		# Parameters  
		#==================================================================#

		self.q = 0.1
		self.distance = adj_mid 
		self.iterations = 100		
		self.reduce_method = "pca" 
		self.step_function = "uniform"
		self.p_adjust_method = "BH"
		self.ebar_method = "permutation" #method to generate error bars 
		
		#------------------------------------------------------------------#
		# Discretization  
		#------------------------------------------------------------------#

		self.meta_disc_skip = None # which indices to skip when discretizing? 


		#------------------------------------------------------------------#
		# Feature Normalization   
		#------------------------------------------------------------------#

		## Beta warping, copulas? 

		#==================================================================#
		# Static Objects  
		#==================================================================#

		self.__description__ 	= """
		  _    _          _ _          
		 | |  | |   /\   | | |   /\    
		 | |__| |  /  \  | | |  /  \   
		 |  __  | / /\ \ | | | / /\ \  
		 | |  | |/ ____ \| | |/ ____ \ 
		 |_|  |_/_/    \_\_|_/_/    \_\
		                               

		HAllA Object for hierarchical all-against-all association testing 
		"""

		self.__doc__			= __doc__ 
		self.__version__ 		= "0.1.1"
		self.__author__			= ["YS Joseph Moon", "Curtis Huttenhower"]
		self.__contact__		= "moon.yosup@gmail.com"

		self.hash_reduce_method = {"pca"	: pca, 
									"mca"	: mca, }

		self.hash_metric 		= {"norm_mid" : norm_mid }

		self.keys_attribute = ["__description__", "__version__", "__author__", "__contact__", "q","distance","iterations", "reduce_method", "step_function", "p_adjust_method","ebar_method"]

		#==================================================================#
		# Presets
		#==================================================================#

		self.hash_preset = 	{"default"		: self.__preset_default, 
								"mid"		: self.__preset_mid,
								"time"		: self.__preset_time, 
								"accuracy"	: self.__preset_accuracy, 
								"parallel"	: self.__preset_parallel, 
								"flat"		: self.__preset_flat,
							}

		#==================================================================#
		# Global Defaults 
		#==================================================================#

		self.num_iter = 1000
		self.summary_method = "all" ## "final"

		#==================================================================#
		# Mutable Meta Objects  
		#==================================================================#
 
		self.meta_array = array( ta ) if ta else None 
		self.meta_feature = None
		self.meta_data_tree = None 
		self.meta_hypothesis_tree = None 
		self.meta_alla = None # results of all-against-all
		self.meta_out = None # final output array; some methods (e.g. all_against_all) have multiple outputs piped to both self.meta_alla and self.meta_out 
		self.meta_summary = None # summary statistics 

		## END INIT 

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
		if isinstance( pFunc ,np.ndarray ):
			return pArray[pFunc]
		else: #generic function type
			return array( [pFunc(item) for item in pArray] ) 

	@staticmethod 
	def bp( pArray, pFunc, axis = 0 ):
		"""
		Map _by pairs_ ; i.e. apply pFunc over all possible pairs in pArray 
		"""

		if bool(axis): 
			pArray = pArray.T

		pIndices = itertools.combinations( range(pArray.shape[0]), 2 )

		return array([pFunc(pArray[i],pArray[j]) for i,j in pIndices])

	@staticmethod 
	def bc( pArray1, pArray2, pFunc, axis = 0 ):
		"""
		Map _by cross product_ for ; i.e. apply pFunc over all possible pairs in pArray1 X pArray2 
		"""

		if bool(axis): 
			pArray1, pArray2 = pArray1.T, pArray2.T

		pIndices = itertools.product( range(pArray1.shape[0]), range(pArray2.shape[0]) )

		return array([pFunc(pArray1[i],pArray2[j]) for i,j in pIndices])

	@staticmethod 
	def r( pArray, pFunc, axis = 0 ):
		"""
		Reduce over array 

		pFunc is X x Y -> R 

		"""
		if bool(axis):
			pArray = pArray.T

		return reduce( pFunc, pArray )

	@staticmethod 
	def rd( ):
		"""
		General reduce-dimension method 
		"""
		pass 

	#==========================================================#
	# Helper Functions 
	#==========================================================# 

	def _discretize( self ):
		self.meta_feature = self.m( self.meta_array, discretize )
		# Should do a better job at detecting whether dataset is categorical or continuous
		# Take information from the parser module 
		return self.meta_feature

	def _featurize( self, strMethod = "_discretize" ):
		pMethod = None 
		try:
			pMethod = getattr( self, strMethod )
		except AttributeError:
			raise Exception("Invalid Method.")

		if pMethod:
			return pMethod( )

	def _hclust( self ):
		self.meta_data_tree = self.m( self.meta_feature, lambda x: hclust(x,bTree=True) )
		return self.meta_data_tree 

	def _couple( self ):
		self.meta_hypothesis_tree = self.m( self.bp( self.m(self.meta_data_tree, lambda x: [x]), couple_tree ), lambda y: y[0] ) 
		## remember, `couple_tree` returns object wrapped in list 
		return self.meta_hypothesis_tree 

	def _naive_all_against_all( self ):
		self.meta_alla = naive_all_against_all( self.meta_array[0], self.meta_array[1] )
		return self.meta_alla 

	def _all_against_all( self ):
		self.meta_alla = all_against_all( self.meta_hypothesis_tree[0], self.meta_array[0], self.meta_array[1] ) 
		## Choose to keep to 2 arrays for now -- change later to generalize 
		return self.meta_alla 

	def _summary_statistics( self, strMethod = None ): 
		"""
		provides summary statistics on the output given by _all_against_all 
		"""

		if not strMethod:
			strMethod = self.summary_method

		def __add_pval_product_wise( _x, _y, _fP ):
			S[_x][_y] = _fP ; S[_y][_x] = _fP 

		def __get_pval_from_bags( _Z, _strMethod = None ):
			"""
			
			_strMethod: str 
				{"default",}

			The default option does the following: go through the bags, treating the p-value for each bag pair as applying to all the variables inside the bag. 
			If new instance arises (e.g. [[3],[5]] following [[3,5,6],[3,5,6]] ), override the p-value to the one with more precision. 
			"""

			for aLine in _Z:
				print aLine 
				#break
				aaBag, fAssoc = aLine
				aBag1, aBag2 = aaBag 
				aBag1, aBag2 = array(aBag1), array(aBag2)
				self.bc( aBag1, aBag2, pFunc = lambda x,y: __add_pval_product_wise( _x = x, _y = y, _fP = fAssoc ) )

		X = self.meta_array[0]
		Y = self.meta_array[1] 
		iX, iY = X.shape[0], Y.shape[0]
		
		S = -1 * numpy.ones( (iX, iY) ) ## matrix of all associations; symmetric if using a symmetric measure of association  

		Z = self.meta_alla 
		Z_final, Z_all = Z ## Z_final is the final bags that passed criteria; Z_all is all the associations delineated throughout computational tree 
		Z_final, Z_all = array(Z_final),array(Z_all)
		assert( Z_all.any() ), "association bags empty." ## Technically, Z_final could be empty 

		#if Z_final.any():
		if strMethod == "final":
			print "Using only final p-values"

			__get_pval_from_bags( Z_final )
			assert( S.any() )
			self.meta_summary = [S]
			return self.meta_summary

		#elif Z_all.any():
		elif strMethod == "all":
			print "Using all p-values"
			__get_pval_from_bags( Z_all )
			assert( S.any() )
			self.meta_summary = [S]
			return self.meta_summary

	def _plot( self ):
		"""
		Wrapper for plotting facilities
		"""

	def _report( self ):
		"""
		helper function for reporting the output to the user,
		"""
		return self.meta_summary   

	def _run( self ):
		"""
		helper function: runs vanilla run of HAllA _as is_. 
		"""

		pass 

	#==========================================================#
	# Load and set data 
	#==========================================================# 

	def set_data( self, *ta ):
		if ta:
			self.meta_array = ta 
			return self.meta_array 
		else:
			raise Exception("Data empty")


	#==========================================================#
	# Set parameters 
	#==========================================================# 

	def set_q( self, fQ ):
		self.q = fQ
		return self.q 

	def set_p_adjust_method( self, strMethod ):
		"""
		Set multiple hypothesis test correction method 

			{"BH", "FDR", "Bonferroni", "BHY"}
		"""

		self.p_adjust_method = strMethod 
		return self.p_adjust_method 

	def set_metric( self, pMetric ):
		if isinstance( pMetric, str ):
			self.distance = self.hash_metric[pMetric]
		else:
			self.distance = pMetric 
		return self.distance 

	def set_reduce_method( self, strMethod ):
		if isinstance( strMethod, str ):
			self.reduce_method = self.hash_reduce_method[strMethod]
		else:
			self.reduce_method = strMethod 
		return self.reduce_method

	def set_iterations( self, iIterations ):
		self.m_iIter = iIterations
		return self.iterations 

	def set_ebar_method( self, strMethod ):
		self.ebar_method = strMethod 
		return self.ebar_method 

	def set_step_function( self, strFun ):
		"""
		set step function used to couple tree to make hypothesis tree 
		"""
		pass 

	def set_preset( self, strPreset ):
		try:
			pPreset = self.hash_preset[strPreset] 
			pPreset() ## run method 
		except KeyError:
			raise Exception("Preset not found. For the default preset, try set_preset('default')")

	#==========================================================#
	# Presets  
	#==========================================================# 
	"""
	These are hard-coded presets deemed useful for the user 
	"""

	def __preset_mi( self ):
		"""
		Mutual Information Distance Preset 
		"""

		## Constants for this preset 
		fQ = 0.1
		pDistance = adj_mid 
		iIter = 100
		strReduce = "pca"
		strStep = "uniform"
		strAdjust = "BH"
		strEbar = "permutation"

		## Set 
		self.set_q( fQ ) 
		self.set_metric( adj_mi )
		self.set_iterations( iIter )
		self.set_reduce_method( strReduce )
		self.set_step_function( strStep )
		self.set_p_adjust_method( strAdjust )
		self.set_ebar_method( strEbar )

		## Run 		
		self._featurize( )
		self._hclust( )
		self._couple( )
		self._all_against_all( )
		return self._report( )


	def __preset_mid( self ):
		"""
		Mutual Information Distance Preset 
		"""

		## Constants for this preset 
		fQ = 0.1
		pDistance = adj_mid 
		iIter = 100
		strReduce = "pca"
		strStep = "uniform"
		strAdjust = "BH"
		strEbar = "permutation"

		## Set 
		self.set_q( fQ ) 
		self.set_metric( adj_mid )
		self.set_iterations( iIter )
		self.set_reduce_method( strReduce )
		self.set_step_function( strStep )
		self.set_p_adjust_method( strAdjust )
		self.set_ebar_method( strEbar )

		## Run 		
		self._featurize( )
		self._hclust( )
		self._couple( )
		self._all_against_all( )
		self._summary_statistics( )
		return self._report( )


	def __preset_default( self ):
		return self.__preset_mid( )

	def __preset_time( self ):
		pass 

	def __preset_accuracy( self ):
		## Constants for this preset 
		fQ = 0.05
		pDistance = adj_mid 
		iIter = 1000
		strReduce = "pca"
		strStep = "uniform"
		strAdjust = "BH"
		strEbar = "permutation"

		## Set 
		self.set_q( fQ ) 
		self.set_metric( adj_mid )
		self.set_iterations( iIter )
		self.set_reduce_method( strReduce )
		self.set_step_function( strStep )
		self.set_p_adjust_method( strAdjust )
		self.set_ebar_method( strEbar )

		## Run 
		self._featurize( )
		self._hclust( )
		self._couple( )
		return self._all_against_all( )
 
	def __preset_parallel( self ):
		pass 

	def __preset_flat( self ):
		"""
		Regular all-against-all pairwise, without hierarchical clustering 
		"""

		self._featurize( )
		return self._naive_all_against_all( )

	#==========================================================#
	# Public Functions / Main Pipeline  
	#==========================================================# 	

	def load_data( self ):
		pass 

	def get_data( self ):
		return self.meta_array 

	def get_feature( self ):
		return self.meta_feature

	def get_tree( self ):
		return self.meta_data_tree

	def get_hypothesis( self ):
		return self.meta_hypothesis_tree

	def get_association( self ):
		return self.meta_alla 

	def get_attribute( self ):
		"""
		returns current attributes and statistics about HAllA object implementation 

			* Print parameters in a text-table style 
		"""
		
		for item in self.keys_attribute:
			sys.stderr.write( "\t".join( [item,str(getattr( self, item ))] ) + "\n" ) 

	def run( self, strMethod = "default" ):
		"""
		Main run module 

		Parameters
		------------

			method : str 
				Specifies what method to use; e.g. which preset to follow 
				{"default", "custom", "time", "accuracy", "parallel"}

				* Custom: 
				* Default:  

		Returns 
		-----------

			Z : HAllA output object 
		
		Notes 
		---------

		* Main steps

			+ Parse input and clean data 
			+ Feature selection (discretization for MI, beta warping, copula selection)
			+ Hierarchical clustering 
			+ Hypothesis generation (tree coupling via appropriate step function)
			+ Hypothesis testing and agglomeration of test statistics, with multiple hypothesis correction 
			+ Parse output 

		* Visually, looks much nicer and is much nicely wrapped if functions are entirely self-contained and we do not have to pass around pointers 

		"""

		try:
			pMethod = self.hash_preset[strMethod]
			return pMethod( )
		except KeyError:			
			raise Exception( "Invalid Method.")


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
