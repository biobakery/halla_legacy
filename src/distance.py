#!/usr/bin/env python 
'''
Abstract distance module providing different notions of distance
'''

import abc 
from abc import ABCMeta
import numpy 

class CDistance:
	''' 
	abstract distance, handles numpy arrays (probably should support lists for compatibility issues)
	'''
	__metaclass__ = ABCMeta
	
	class EDataType:
		DISCRETE = 0 
		CONTINUOUS = 1
	class EMetricType:
		NONMETRIC = 0
		METRIC = 1
	def __init__( self, c_array1, c_array2 ): 
		self.m_data1 = c_array1
		self.m_data2 = c_array2 

	@abc.abstractmethod
	def get_distance( self ): pass 

	@abc.abstractmethod 
	def get_distance_type( self ): pass

	def get_data_type( self ) : pass 
		
	
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
	
	@abc.abstractmethod 
	def get_data_type( self ): pass 

a1, a2 = numpy.array( [1,2,3,4,5] ), numpy.array( [6,7,8,9,10] ) 

CE = CEuclideanDistance( a1, a2 )
print CE.get_distance()
print CE.get_distance_type() 
print CE.get_data_type()  
