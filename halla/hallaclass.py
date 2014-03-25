"""
HAllA class  
"""

## structural packages 
import itertools 
import logging 
import argparse 
from numpy import array 
import csv 
import sys 
import os 

## internal dependencies 
import halla
from halla import stats
from halla import distance
import halla.parser  
from halla.parser import Input, Output 

from halla.test import *
from halla.stats import *
from halla.distance import *
import halla.distance
from halla.hierarchy import *
from halla.plot import *

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

		#----------------------------------#
		# Single and cross-decomposition 
		#----------------------------------#

		self.distance = adj_mi 
		self.reduce_method = "pca" 
		
		#----------------------------------#
		# Step and jump methods 
		#----------------------------------#
		
		self.exploration_function = "default"
			##{"layerwise", "greedy", "default"}

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
		# delta 
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

		self.step_function = "uniform" 
		self.step_parameter = 0.0 ## a value between 0.0 and 1.0; a fractional value of the layers to be tested 

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
		# sigma 
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

		self.start_parameter = 0.0 ## a value between 0.0 and 1.0; 0.0 performs the simplest comparison at the top of the tree; 
		## 1.0 starts in the bottom 

		#------------------------------------------------#
		# Randomization and multiple correction  methods 
		#------------------------------------------------#

		self.alpha = 0.05 
		self.q = 0.1
		self.iterations = 100
		self.p_adjust_method = "BH"
		self.randomization_method = "permutation" #method to generate error bars 
		
		#------------------------------------------------------------------#
		# Discretization  
		#------------------------------------------------------------------#

		self.meta_disc_skip = None # which indices to skip when discretizing? 

		#------------------------------------------------------------------#
		# Feature Normalization   
		#------------------------------------------------------------------#

		## Beta warping, copulas? 

		#------------------------------------------------------------------#
		# Output Parameters    
		#------------------------------------------------------------------#

		self.summary_method = "all" ## "final"
		self.verbose = False 

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
		self.__version__ 		= "0.1.0"
		self.__author__			= ["YS Joseph Moon", "Curtis Huttenhower"]
		self.__contact__		= "moon.yosup@gmail.com"

		self.hash_reduce_method = {"pca"	: pca, }

		self.hash_metric 		= halla.distance.c_hash_metric 

		self.keys_attribute = ["__description__", "__version__", "__author__", "__contact__", "q","distance","iterations", "reduce_method", "step_function", "p_adjust_method","randomization_method"]

		#==================================================================#
		# Presets
		#==================================================================#

		self.hash_preset = 	{"default"		: self.__preset_default, 
								"time"		: self.__preset_time, 
								"accuracy"	: self.__preset_accuracy, 
								"parallel"	: self.__preset_parallel, 
								"flat"		: self.__preset_flat,
								"layerwise" : self.__preset_layerwise, 
								"naive" 	: self.__preset_naive,
							}

		#==================================================================#
		# Global Defaults 
		#==================================================================#

		

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

	#==================================================================#
	# Type Checking
	#==================================================================#

	def _check( self, pObject, pType, pFun = isinstance, pClause = "or" ):
		"""
		Wrapper for type checking 
		"""

		if (isinstance(pType,list) or isinstance(pType,tuple) or isinstance(pType,numpy.ndarray)):
			aType = pType 
		else:
			aType = [pType]

		return reduce( lambda x,y: x or y, [isinstance( pObject, t ) for t in aType], False )

	def _cross_check( self, pX, pY, pFun = len ):
		"""
		Checks that pX and pY are consistent with each other, in terms of specified function pFun. 
		"""

	def _is_meta( self, pObject ):
		"""	
		Is pObject an iterable of iterable? 
		"""

		try: 
			pObject[0]
			return self._is_iter( pObject[0] )	
		except IndexError:
			return False 

	def _is_empty( self, pObject ):
		"""
		Wrapper for both numpy arrays and regular lists 
		"""
		
		aObject = array(pObject)

		return not aObject.any()

	### These functions are absolutely unncessary; get rid of them! 
	def _is_list( self, pObject ):
		return self._check( pObject, list )

	def _is_tuple( self, pObject ):
		return self._check( pObject, tuple )

	def _is_str( self, pObject ):
		return self._check( pObject, str )

	def _is_int( self, pObject ):
		return self._check( pObject, int )    

	def _is_array( self, pObject ):
		return self._check( pObject, numpy.ndarray )

	def _is_1d( self, pObject ):
		"""
		>>> import strudel 
		>>> s = strudel.Strudel( )
		>>> s._is_1d( [] )
		"""

		strErrorMessage = "Object empty; cannot determine type"
		bEmpty = self._is_empty( pObject )

		## Enforce non-empty invariance 
		if bEmpty:
			raise Exception(strErrorMessage)

		## Assume that pObject is non-empty 
		try:
			iRow, iCol = pObject.shape 
			return( iRow == 1 ) 
		except ValueError: ## actual arrays but are 1-dimensional
			return True
		except AttributeError: ## not actual arrays but python lists 
			return not self._is_iter( pObject[0] )

	def _is_iter( self, pObject ):
		"""
		Is the object a list or tuple? 
		Disqualify string as a true "iterable" in this sense 
		"""

		return self._check( pObject, [list, tuple, numpy.ndarray] )

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
		self.meta_alla = all_against_all( self.meta_hypothesis_tree[0], self.meta_array[0], self.meta_array[1], q = self.q ) 
		## Choose to keep to 2 arrays for now -- change later to generalize 
		return self.meta_alla 

	def _layerwise_all_against_all( self ):

		X, Y = self.meta_array[0], self.meta_array[1]
		dX, dY = self.meta_feature[0], self.meta_feature[1]
		tX, tY = self.meta_data_tree[0], self.meta_data_tree[1]
		iX, iY = X.shape[0], Y.shape[0]

		aOut = filter(bool,list(halla.hierarchy.layerwise_all_against_all( tX, tY, X, Y )))

		aMetaOut = [] 

		def _layer( Z ):

			S = -1 * numpy.ones( (iX, iY) ) ## matrix of all associations; symmetric if using a symmetric measure of association  

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
					if self.verbose:
						print aLine 
					#break
					aaBag, fAssoc = aLine
					aBag1, aBag2 = aaBag 
					aBag1, aBag2 = array(aBag1), array(aBag2)
					self.bc( aBag1, aBag2, pFunc = lambda x,y: __add_pval_product_wise( _x = x, _y = y, _fP = fAssoc ) )

			__get_pval_from_bags( Z )
			return S 		

		for Z in aOut:
			aMetaOut.append(_layer(Z))

		return aMetaOut 

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
				if self.verbose:
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
			if self.verbose:
				print "Using only final p-values"
			__get_pval_from_bags( Z_final )
			assert( S.any() )
			self.meta_summary = [S]
			return self.meta_summary

		#elif Z_all.any():
		elif strMethod == "all":
			if self.verbose:
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

	def set_randomization_method( self, strMethod ):
		self.randomization_method = strMethod 
		return self.randomization_method 

	def set_step_function( self, strFun ):
		"""
		set step function used to couple tree to make hypothesis tree 
		"""
		pass 

	def set_exploration_function( self, strFunction ):
			self.exploration_function = strFunction

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

	def __preset_layerwise( self ):
		"""
		Layerwise MI preset 
		"""

		## Constants for this preset 
		fQ = 0.1
		pDistance = norm_mi 
		iIter = 100
		strReduce = "pca"
		strStep = "uniform"
		strAdjust = "BH"
		strRandomization = "permutation"
		strExplorationFunction = "layerwise"

		## Set 
		self.set_q( fQ ) 
		self.set_metric( pDistance )
		self.set_iterations( iIter )
		self.set_reduce_method( strReduce )
		self.set_step_function( strStep )
		self.set_p_adjust_method( strAdjust )
		self.set_randomization_method( strRandomization )
		self.set_exploration_function( strExplorationFunction )

		## Run 		
		self._featurize( )
		self._hclust( )
		#self._couple( )
		#self._all_against_all( )
		return self._layerwise_all_against_all( )
		#self._summary_statistics( )
		#return self._report( )

	def __preset_norm_mi( self ):
		"""
		Mutual Information Preset 
		"""

		## Constants for this preset 
		fQ = 0.1
		pDistance = norm_mi 
		iIter = 100
		strReduce = "pca"
		strStep = "uniform"
		strAdjust = "BH"
		strRandomization = "permutation"

		## Set 
		self.set_q( fQ ) 
		self.set_metric( pDistance )
		self.set_iterations( iIter )
		self.set_reduce_method( strReduce )
		self.set_step_function( strStep )
		self.set_p_adjust_method( strAdjust )
		self.set_randomization_method( strRandomization )

		## Run 		
		self._featurize( )
		self._hclust( )
		self._couple( )
		self._all_against_all( ) 
		self._summary_statistics( ) 
		return self._report( )

	def __preset_default( self ):
		return self.__preset_norm_mi( )
		#return self.__preset_layerwise( )

	def __preset_time( self ):
		pass 

	def __preset_accuracy( self ):
		## Constants for this preset 
		fQ = 0.05
		pDistance = adj_mi 
		iIter = 1000
		strReduce = "pca"
		strStep = "uniform"
		strAdjust = "BH"
		strRandomization = "permutation"

		## Set 
		self.set_q( fQ ) 
		self.set_metric( adj_mi )
		self.set_iterations( iIter )
		self.set_reduce_method( strReduce )
		self.set_step_function( strStep )
		self.set_p_adjust_method( strAdjust )
		self.set_randomization_method( strRandomization )

		## Run 
		self._featurize( )
		self._hclust( )
		self._couple( )
		return self._all_against_all( )
 
	def __preset_parallel( self ):
		pass 

	def __preset_naive( self ):
		"""
		All against all 
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

	def view_singleton( self, pBags ):
		aOut = [] 
		for aIndices, fP in pBags:
			if len(aIndices[0]) == 1 and len(aIndices[1]) == 1:
				aOut.append( [aIndices, fP] )
		return aOut 



