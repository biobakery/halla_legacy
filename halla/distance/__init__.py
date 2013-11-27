#!/usr/bin/env python 
'''
Abstract distance module providing different notions of distance
'''

import itertools 
import abc 
import math

from abc import ABCMeta
import numpy 
from numpy import array 
import scipy 
import sklearn as sk 
 
import halla.stats
import scipy.stats 
from scipy.stats import pearsonr, spearmanr

import pylab 

#mi-based distances from scikit-learn; (log e)-based.  
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score, adjusted_mutual_info_score 

#==========================================================================#
# DIVERGENCE CLASSES
#==========================================================================#

class Distance:
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
		pFunc = Distance.c_hashInvertFunctions[strFunc or "flip"] 
		return pFunc( self.get_distance() ) 
						
	@abc.abstractmethod
	def get_distance( self ): pass 

	@abc.abstractmethod 
	def get_distance_type( self ): pass
	
	
class EuclideanDistance( Distance ):

	__metaclass__ = ABCMeta 

	def __init__( self, c_array1, c_array2 ):
		self.m_data1 = c_array1
		self.m_data2 = c_array2
		self.c_distance_type = CDistance.EMetricType.METRIC 

	def get_distance( self ):
		return numpy.linalg.norm( self.m_data2-self.m_data1 ) 

	def get_distance_type( self ):
		return self.c_distance_type 	
	
class MutualInformation( Distance ):
	"""
	Scikit-learn uses the convention log = ln
	Adjust multiplicative factor of log(e,2) 
	"""	

	__metaclass__ = ABCMeta 

	def __init__( self, c_array1, c_array2, bSym = False ):
		self.m_data1 = c_array1 
		self.m_data2 = c_array2 
		self.bSym = bSym
		self.c_distance_type = Distance.EMetricType.NONMETRIC 
	
	def get_distance( self ):
		#assert( numpy.shape(self.m_data1) == numpy.shape(self.m_data2) )
		return math.log(math.e,2) *  mutual_info_score( self.m_data1, self.m_data2 ) 	
	def get_distance_type( self ):
		return self.c_distance_type 	

class NormalizedMutualInformation( Distance ):
	"""
	normalized by sqrt(H1*H2) so the range is [0,1]
	"""	
	__metaclass__ = ABCMeta 

	def __init__( self, c_array1, c_array2 ):
		self.m_data1 = c_array1 
		self.m_data2 = c_array2 
		self.c_distance_type = Distance.EMetricType.NONMETRIC 
	
	def get_distance( self ):
		#assert( numpy.shape(self.m_data1) == numpy.shape(self.m_data2) )
		return normalized_mutual_info_score( self.m_data1, self.m_data2 )

	def get_distance_type( self ):
		return self.c_distance_type 	
	
class AdjustedMutualInformation( Distance ):
	"""
	adjusted for chance
	""" 
	
	__metaclass__ = ABCMeta 

	def __init__( self, c_array1, c_array2 ):
		self.m_data1 = c_array1 
		self.m_data2 = c_array2 
		self.c_distance_type = Distance.EMetricType.NONMETRIC 
	
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

def cor( pData1, pData2, method = "pearson", pval = False ):
	"""
	Get correlation coefficient and corresponding parametric p-value (t-test)

	Parameters
	------------
	pData1, pData2 : numpy arrays
		 data matrices 
	method : str 
		{"pearson", "spearman"}
		"abs"
	pval : bool
		True if parametric estimate of p-value requested 

	Returns
	-----------
	rho: float
		correlation coefficient 
	p: float
 		p-value  
	
	Examples
	---------
	View pairwise correlation measures (pearson,spearman) between two datasets `x` and `y`:

	>>> x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])
	>>> y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0],[0.015625,0.125,0.421875,1.0]])
	>>> p = [_ for _ in itertools.product( range(len(x)), range(len(y)) )]
	>>> for item in p: i,j = item; print (i,j),cor(x[i],y[j], method="pearson"),cor(x[i],y[j], method="spearman")
	(0, 0) -1.0 -1.0
	(0, 1) -0.894427191 -0.894427191
	(0, 2) 1.0 1.0 
	(0, 3) 0.951369855792 1.0
	(1, 0) 0.774596669241 0.774596669241
	(1, 1) 0.57735026919 0.57735026919
	(1, 2) -0.774596669241 -0.774596669241
	(1, 3) -0.921159901892 -0.774596669241
	(2, 0) -0.984374038698 -1.0
	(2, 1) -0.880450906326 -0.894427191
	(2, 2) 0.984374038698 1.0
	(2, 3) 0.99053285189 1.0
	(3, 0) -0.774596669241 -0.774596669241
	(3, 1) -0.57735026919 -0.57735026919
	(3, 2) 0.774596669241 0.774596669241
	(3, 3) 0.921159901892 0.774596669241

	Generate p-values for pearson correlation:

	>>> pval_pearson = sorted( [cor(x[i],y[j], method="pearson", pval=True)[1] for i,j in p] )
	>>> pval_pearson
	[0.0, 0.0, 0.0094671481098304033, 0.01562596130230276, 0.015625961302302867, 0.048630144207595816, 0.07884009810806647, 0.07884009810806647, 0.10557280900008403, 0.11954909367437615, 0.22540333075851643, 0.22540333075851643, 0.2254033307585166, 0.2254033307585166, 0.42264973081037405, 0.42264973081037405]

	View plot::

		plt.plot( pval_pearson )
		plt.show()

	.. plot::

		import matplotlib.pyplot as plt 
		plt.plot( [0.0, 0.0, 0.0094671481098304033, 0.01562596130230276, 0.015625961302302867, 0.048630144207595816, 
			0.07884009810806647, 0.07884009810806647, 0.10557280900008403, 0.11954909367437615, 0.22540333075851643, 
			0.22540333075851643, 0.2254033307585166, 0.2254033307585166, 0.42264973081037405, 0.42264973081037405], 
			linestyle='--', marker='o' )
		plt.show()

	Generate p-values for spearman correlation:

	>>> pval_spearman = sorted( [cor(x[i],y[j], method="spearman", pval=True)[1] for i,j in p] )
	>>> pval_spearman
	[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10557280900008413, 0.10557280900008413, 0.22540333075851657, 0.22540333075851657, 0.22540333075851657, 0.22540333075851657, 0.22540333075851657, 0.22540333075851657, 0.42264973081037427, 0.42264973081037427]

	View plot::
		
		plt.plot(pval_pearson)
		plt.show()

	.. plot::
		
		import matplotlib.pyplot as plt 
		plt.plot( [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10557280900008413, 0.10557280900008413, 0.22540333075851657, 
			0.22540333075851657, 0.22540333075851657, 0.22540333075851657, 0.22540333075851657, 0.22540333075851657, 
			0.42264973081037427, 0.42264973081037427], linestyle='--', marker='o')
		plt.show()			
	
		""" 
	try:
		str(method)
		if not(method == "pearson") and not(method=="spearman"):
			raise ValueError  
		pMethod = scipy.stats.spearmanr if method == "spearman" else scipy.stats.pearsonr
	except Exception:
		pMethod = method 

	return pMethod( pData1, pData2 )[0] if not pval else pMethod( pData1, pData2 )

def cord( pData, pData2, method = "pearson", inversion_method = "abs", pval=False ):
	"""
	Get correlation divergence 

	Parameters
	-----------
	pData1, pData2 : numpy arrays
		 data matrices 
	method : str 
		{"pearson", "spearman"}
	inversion_method : str
		"abs"
	pval : bool
		True if parametric estimate of p-value requested 

	Returns
	-----------
	rho: float
		correlation coefficient 
	p: float
 		p-value  

	Examples
	----------
	>>> x = [0.1,0.2,0.3,0.4]
	>>> y = [-0.1,-0.2,-0.3,-0.4]
	>>> cord( x,y )
	0.0
	"""

	pMethod, pPval = method, pval 
	pCor = cor( pData, pData2, method=pMethod, pval=pPval )
	return (1.0-abs(pCor[0]), pCor[1]) if pval else 1.0-abs(pCor)


def l2( pData1, pData2 ):
	"""
	Returns the l2 distance

	>>> x = numpy.array([1,2,3]); y = numpy.array([4,5,6])
	>>> l2(x,y)
	5.196152422706632
	"""
	return numpy.linalg.norm(pData1 - pData2)

def absl2( pData1, pData2 ):
	return numpy.abs( l2( pData1, pData2 ) )

def mi( pData1, pData2 ):
	"""
	Static implementation of mutual information, returns bits 

	Parameters
	--------------
	pData1, pData2 : Numpy arrays

	Returns
	---------------
	mi : float 

	Examples
	--------------

	>>> x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])
	>>> y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0],[0.015625,0.125,0.421875,1.0]])
	>>> dx = halla.stats.discretize( x, iN = None, method = None, aiSkip = [1,3] )
	>>> dy = halla.stats.discretize( y, iN = None, method = None, aiSkip = [1] )
	>>> p = itertools.product( range(len(x)), range(len(y)) )
	>>> for item in p: i,j = item; print (i,j), mi( dx[i], dy[j] )
	(0, 0) 1.0
	(0, 1) 1.0
	(0, 2) 1.0
	(0, 3) 1.0
	(1, 0) 0.311278124459
	(1, 1) 0.311278124459
	(1, 2) 0.311278124459
	(1, 3) 0.311278124459
	(2, 0) 1.0
	(2, 1) 1.0
	(2, 2) 1.0
	(2, 3) 1.0
	(3, 0) 0.311278124459
	(3, 1) 0.311278124459
	(3, 2) 0.311278124459
	(3, 3) 0.311278124459
	"""

	return MutualInformation( pData1, pData2 ).get_distance()

def norm_mi( pData1, pData2 ):
	"""
	Static implementation of normalized mutual information 

	Examples
	---------------

	>>> x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])
	>>> y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0],[0.015625,0.125,0.421875,1.0]])
	>>> dx = halla.stats.discretize( x, iN = None, method = None, aiSkip = [1,3] )
	>>> dy = halla.stats.discretize( y, iN = None, method = None, aiSkip = [1] )
	>>> p = itertools.product( range(len(x)), range(len(y)) )
	>>> for item in p: i,j = item; print (i,j), norm_mi( dx[i], dy[j] )
	(0, 0) 1.0
	(0, 1) 1.0
	(0, 2) 1.0
	(0, 3) 1.0
	(1, 0) 0.345592029944
	(1, 1) 0.345592029944
	(1, 2) 0.345592029944
	(1, 3) 0.345592029944
	(2, 0) 1.0
	(2, 1) 1.0
	(2, 2) 1.0
	(2, 3) 1.0
	(3, 0) 0.345592029944
	(3, 1) 0.345592029944
	(3, 2) 0.345592029944
	(3, 3) 0.345592029944

	"""

	return NormalizedMutualInformation( pData1, pData2 ).get_distance() 

def adj_mi( pData1, pData2 ):
	""" 
	Static implementation of adjusted distance 

	Examples
	-----------

	>>> x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])
	>>> y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0],[0.015625,0.125,0.421875,1.0]])
	>>> dx = halla.stats.discretize( x, iN = None, method = None, aiSkip = [1,3] )
	>>> dy = halla.stats.discretize( y, iN = None, method = None, aiSkip = [1] )
	>>> p = itertools.product( range(len(x)), range(len(y)) )
	>>> for item in p: i,j = item; print (i,j), adj_mi( dx[i], dy[j] )
	(0, 0) 1.0
	(0, 1) 1.0
	(0, 2) 1.0
	(0, 3) 1.0
	(1, 0) 2.51758394487e-08
	(1, 1) 2.51758394487e-08
	(1, 2) 2.51758394487e-08
	(1, 3) 2.51758394487e-08
	(2, 0) 1.0
	(2, 1) 1.0
	(2, 2) 1.0
	(2, 3) 1.0
	(3, 0) -3.72523550982e-08
	(3, 1) -3.72523550982e-08
	(3, 2) -3.72523550982e-08
	(3, 3) -3.72523550982e-08

	"""

	return AdjustedMutualInformation( pData1, pData2 ).get_distance()

def mid( pData1, pData2 ):
	"""
	Static implementation of mutual information, 



	Examples 
	-------------

	>>> x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])
	>>> y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0],[0.015625,0.125,0.421875,1.0]])
	>>> dx = halla.stats.discretize( x, iN = None, method = None, aiSkip = [1,3] )
	>>> dy = halla.stats.discretize( y, iN = None, method = None, aiSkip = [1] )
	>>> p = itertools.product( range(len(x)), range(len(y)) )
	>>> for item in p: i,j = item; print (i,j), mid( dx[i], dy[j] )
	(0, 0) 0.0
	(0, 1) 0.0
	(0, 2) 0.0
	(0, 3) 0.0
	(1, 0) 0.688721875541
	(1, 1) 0.688721875541
	(1, 2) 0.688721875541
	(1, 3) 0.688721875541
	(2, 0) 0.0
	(2, 1) 0.0
	(2, 2) 0.0
	(2, 3) 0.0
	(3, 0) 0.688721875541
	(3, 1) 0.688721875541
	(3, 2) 0.688721875541
	(3, 3) 0.688721875541

	"""

	return 1 - MutualInformation( pData1, pData2 ).get_distance()

def norm_mid( pData1, pData2 ):
	"""
	Static implementation of normalized mutual information 

	Examples
	-------------


	>>> x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])
	>>> y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0],[0.015625,0.125,0.421875,1.0]])
	>>> dx = halla.stats.discretize( x, iN = None, method = None, aiSkip = [1,3] )
	>>> dy = halla.stats.discretize( y, iN = None, method = None, aiSkip = [1] )
	>>> p = itertools.product( range(len(x)), range(len(y)) )
	>>> for item in p: i,j = item; print (i,j), norm_mid( dx[i], dy[j] )
	(0, 0) 0.0
	(0, 1) 0.0
	(0, 2) 0.0
	(0, 3) 0.0
	(1, 0) 0.654407970056
	(1, 1) 0.654407970056
	(1, 2) 0.654407970056
	(1, 3) 0.654407970056
	(2, 0) 0.0
	(2, 1) 0.0
	(2, 2) 0.0
	(2, 3) 0.0
	(3, 0) 0.654407970056
	(3, 1) 0.654407970056
	(3, 2) 0.654407970056
	(3, 3) 0.654407970056
	"""

	return 1 - NormalizedMutualInformation( pData1, pData2 ).get_distance() 

def adj_mid( pData1, pData2 ):
	""" 
	Static implementation of adjusted distance 


	Examples
	-----------

	>>> x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])
	>>> y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0],[0.015625,0.125,0.421875,1.0]])
	>>> dx = halla.stats.discretize( x, iN = None, method = None, aiSkip = [1,3] )
	>>> dy = halla.stats.discretize( y, iN = None, method = None, aiSkip = [1] )
	>>> p = itertools.product( range(len(x)), range(len(y)) )
	>>> for item in p: i,j = item; print (i,j), adj_mid( dx[i], dy[j] )
	(0, 0) 0.0
	(0, 1) 0.0
	(0, 2) 0.0
	(0, 3) 0.0
	(1, 0) 0.999999974824
	(1, 1) 0.999999974824
	(1, 2) 0.999999974824
	(1, 3) 0.999999974824
	(2, 0) 0.0
	(2, 1) 0.0
	(2, 2) 0.0
	(2, 3) 0.0
	(3, 0) 1.00000003725
	(3, 1) 1.00000003725
	(3, 2) 1.00000003725
	(3, 3) 1.00000003725
	"""

	return 1 - AdjustedMutualInformation( pData1, pData2 ).get_distance()

