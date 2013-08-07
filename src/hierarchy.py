#!/usr/bin/env python 
'''
Hiearchy module, used to build trees and other data structures for HAllA
'''

import math 
import numpy as np  
from numpy import array 
import datum 
import distance 

#c_hashStructures = {"tree": lambda: Tree()}


#==========================================================================#
# DATA STRUCTURES 
#==========================================================================#

class Tree():
	''' 
	A hierarchically nested structure containing nodes as
	a basic core unit	
	'''	

	def __init__(self):
		self.m_pData = None 
		self.m_arrayChildren = []
		self.m_iLayer = 0 
    
	def pop(self):
		# pop one of the children, else return none, since this amounts to killing the singleton 
		if self.m_arrayChildren:
			return self.m_arrayChildren.pop()
		
	def is_leaf(self):
		return bool(not(self.m_pData and self.m_arrayChildren))

	def is_degenerate(self):
		return ( not(self.m_pData) and not(self.m_arrayChildren) )			

	def add_child(self, node_object):
		self.m_arrayChildren.append(node_object)
		
	def get_children(self): 
		return self.m_arrayChildren
	
	def get_child(self,iIndex=None):
		return self.m_arrayChildren[iIndex or 0]
	
	def add_data(self, pDatum):
		self.m_pData = pDatum 
		return self 
	
	def get_data(self):
		return self.m_pData 


#==========================================================================#
# METHODS  
#==========================================================================#


#==========================================================================#
# META
#==========================================================================#

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
# OBJECT WRAPPERS
#==========================================================================#

def HAC( pDataset ):
	"""
	Hierarchical Agglomerative Clustering
	"""
	
