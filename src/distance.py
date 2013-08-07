#!/usr/bin/env python 
'''
Abstract distance module providing different notions of distance
'''

import abc 
from abc import ABCMeta
import numpy 
from numpy import array 
import sklearn as sk 

#mi-based distances from scikit-learn 
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score, adjusted_mutual_info_score 

#==========================================================================#
# Measures of Divergence 
#==========================================================================#

class CDistance:
	''' 
	abstract distance, handles numpy arrays (probably should support lists for compatibility issues)
	'''
	__metaclass__ = ABCMeta
		
	c_hashInvertFunctions = {"logistic": lambda x:1.0/(1+numpy.exp(-1.0*x)), "flip": lambda x: -1.0*x, "1mflip": lambda x: 1-1.0*x }

	class EMetricType:
		NONMETRIC = 0
		METRIC = 1
	def __init__( self, c_array1, c_array2 ): 
		self.m_data1 = c_array1
		self.m_data2 = c_array2 
 
	def get_inverted_distance( self, strFunc = None ):
		pFunc = CDistance.c_hashInvertFunctions[strFunc or "flip"] 
		return pFunc( self.get_distance() ) 
						
	@abc.abstractmethod
	def get_distance( self ): pass 

	@abc.abstractmethod 
	def get_distance_type( self ): pass
	
	
class CEuclideanDistance( CDistance ):

	__metaclass__ = ABCMeta 

	def __init__( self, c_array1, c_array2 ):
		self.m_data1 = c_array1
		self.m_data2 = c_array2
		self.c_distance_type = CDistance.EMetricType.METRIC 

	def get_distance( self ):
		return numpy.linalg.norm( self.m_data2-self.m_data1 ) 

	def get_distance_type( self ):
		return self.c_distance_type 	
	
class CMutualInformation( CDistance ):
	
	__metaclass__ = ABCMeta 

	def __init__( self, c_array1, c_array2 ):
		self.m_data1 = c_array1 
		self.m_data2 = c_array2 
		self.c_distance_type = CDistance.EMetricType.NONMETRIC 
	
	def get_distance( self ):
		assert( numpy.shape(self.m_data1) == numpy.shape(self.m_data2) )
		return mutual_info_score( self.m_data1, self.m_data2 )
	
	def get_distance_type( self ):
		return self.c_distance_type 	
	
class CNormalizedMutualInformation( CDistance ):
	
	__metaclass__ = ABCMeta 

	def __init__( self, c_array1, c_array2 ):
		self.m_data1 = c_array1 
		self.m_data2 = c_array2 
		self.c_distance_type = CDistance.EMetricType.NONMETRIC 
	
	def get_distance( self ):
		assert( numpy.shape(self.m_data1) == numpy.shape(self.m_data2) )
		return normalized_mutual_info_score( self.m_data1, self.m_data2 )

	def get_distance_type( self ):
		return self.c_distance_type 	
	
class CAdjustedMutualInformation( CDistance ):
	
	__metaclass__ = ABCMeta 

	def __init__( self, c_array1, c_array2 ):
		self.m_data1 = c_array1 
		self.m_data2 = c_array2 
		self.c_distance_type = CDistance.EMetricType.NONMETRIC 
	
	def get_distance( self ):
		assert( numpy.shape(self.m_data1) == numpy.shape(self.m_data2) )
		return adjusted_mutual_info_score( self.m_data1, self.m_data2 )

	def get_distance_type( self ):
		return self.c_distance_type 	

#==========================================================================#
# Hypothesis Testing 
#==========================================================================#

