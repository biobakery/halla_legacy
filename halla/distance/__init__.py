#!/usr/bin/env python 
'''
Abstract distance module providing different notions of distance
'''

import abc 
from abc import ABCMeta
import numpy 
from numpy import array 
import sklearn as sk 
import math 

#mi-based distances from scikit-learn; (log e)-based.  
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score, adjusted_mutual_info_score 

#==========================================================================#
# DIVERGENCE CLASSES
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
	"""
	Scikit-learn uses the convention log = ln
	Adjust multiplicative factor of log(e,2) 
	"""	

	__metaclass__ = ABCMeta 

	def __init__( self, c_array1, c_array2, bSym = False ):
		self.m_data1 = c_array1 
		self.m_data2 = c_array2 
		self.bSym = bSym
		self.c_distance_type = CDistance.EMetricType.NONMETRIC 
	
	def get_distance( self ):
		#assert( numpy.shape(self.m_data1) == numpy.shape(self.m_data2) )
		return math.log(math.e,2) *  mutual_info_score( self.m_data1, self.m_data2 ) 	
	def get_distance_type( self ):
		return self.c_distance_type 	

class CNormalizedMutualInformation( CDistance ):
	"""
	normalized by sqrt(H1*H2) so the range is [0,1]
	"""	
	__metaclass__ = ABCMeta 

	def __init__( self, c_array1, c_array2 ):
		self.m_data1 = c_array1 
		self.m_data2 = c_array2 
		self.c_distance_type = CDistance.EMetricType.NONMETRIC 
	
	def get_distance( self ):
		#assert( numpy.shape(self.m_data1) == numpy.shape(self.m_data2) )
		return normalized_mutual_info_score( self.m_data1, self.m_data2 )

	def get_distance_type( self ):
		return self.c_distance_type 	
	
class CAdjustedMutualInformation( CDistance ):
	"""
	adjusted for chance
	""" 
	
	__metaclass__ = ABCMeta 

	def __init__( self, c_array1, c_array2 ):
		self.m_data1 = c_array1 
		self.m_data2 = c_array2 
		self.c_distance_type = CDistance.EMetricType.NONMETRIC 
	
	def get_distance( self ):
		#assert( numpy.shape(self.m_data1) == numpy.shape(self.m_data2) )
		return adjusted_mutual_info_score( self.m_data1, self.m_data2 )

	def get_distance_type( self ):
		return self.c_distance_type 	

#==========================================================================#
# HYPOTHESIS TESTING CLASSES 
#==========================================================================#


#==========================================================================#
# FUNCTIONS  
#==========================================================================#

def l2( pData1, pData2 ):
	return numpy.linalg.norm(pData1 - pData2)

def mi( pData1, pData2 ):
	"""
	static implementation of mutual information, 
	caveat: already normalized by CMutualInformation  
	"""

	return CMutualInformation( pData1, pData2 ).get_distance()

def norm_mi( pData1, pData2 ):
	"""
	static implementation of normalized mutual information 

	"""

	return CNormalizedMutualInformation( pData1, pData2 ).get_distance() 

def adj_mi( pData1, pData2 ):
	""" 
	static implementation of adjusted distance 
	"""

	return 1 - CAdjustedMutualInformation( pData1, pData2 ).get_distance()

def mid( pData1, pData2 ):
	"""
	static implementation of mutual information, 
	caveat: returns nats, not bits 
	"""

	return 1 - CMutualInformation( pData1, pData2 ).get_distance()

def norm_mid( pData1, pData2 ):
	"""
	static implementation of normalized mutual information 

	"""

	return 1 - CNormalizedMutualInformation( pData1, pData2 ).get_distance() 

def adj_mid( pData1, pData2 ):
	""" 
	static implementation of adjusted distance 
	"""

	return 1 - CAdjustedMutualInformation( pData1, pData2 ).get_distance()
