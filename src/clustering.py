#!/usr/bin/env python 

"""
Scaffold script for HAllA development
"""

## native python packages 

import itertools 

## structural packages 

import sys 
import re 

## halla-specific modules 

from distance import mi, l2 
from datum import discretize 


## statistics packages 

import numpy as np
import scipy as sp
import sklearn.decomposition
from sklearn.decomposition import PCA #remember that the matrix is X = (n_samples,n_features)
import csv 
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree, leaves_list
import pylab as pl 
import random  
from numpy.random import normal 
from scipy.misc import * 
import pandas as pd 


#=================================================# 
# FUNCTIONS  
#=================================================# 



#=================================================# 
# RUNTIME  
#=================================================# 
"""
Hierarchically compare two datasets 



if len(sys.argv[1:]) < 2:
	raise Exception("Usage: python clustering.py [file1.csv] [file2.csv] .. <additional_files> .. ")

args = sys.argv[1:]
strFile1, strFile2 = args[:2]

## Parse 

## Load into dataframe 

df1 = pd.read_csv(strFile1)
df2 = pd.read_csv(strFile2)

"""

## Example Data 
#csvr = csv.reader(open("table.tsv"), csv.excel_tab)

## Continuous Data 

## Recall that the data matrix should be defined as in the "normal" fashion; 
## rows are features, columns are instances. This matrix should be transposed 
## for the scikit learn PCA function 

c_DataArray1 = np.array([[normal() for x in range(100)] for y in range(20)])
c_DataArray2 = np.array([[normal() for x in range(100)] for y in range(20)])

c_DataArray11 = np.array([[normal() for x in range(100)] for y in range(20)])
c_DataArray22 = np.array([[normal() for x in range(100)] for y in range(20)])


## Categorical Data 

c_DataArray3 = np.array([[random.randint(1,10) for x in range(100)] for y in range(20)])
c_DataArray4 = np.array([[random.randint(1,10) for x in range(100)] for y in range(20)])

## Discretize Continuous 

c_DataArrayDisc1 = discretize( c_DataArray1 )
c_DataArrayDisc2 = discretize( c_DataArray2 )

c_DataArrayDisc11 = discretize( c_DataArray11 )
c_DataArrayDisc22 = discretize( c_DataArray22 )

distance_matrix1 = pdist( c_DataArrayDisc1, metric=sym_mi )
distance_matrix2 = pdist( c_DataArrayDisc2, metric=sym_mi )

distance_matrix11 = pdist( c_DataArrayDisc1, metric='euclidean' )
distance_matrix22 = pdist( c_DataArrayDisc2, metric='euclidean' )

## Sanity Checks 

#print distance_matrix1
#print len(distance_matrix1)
#print comb(20,2)

#distance_matrix2 = pdist( c_DataArray2 )
#distance_matrix3 = pdist( c_DataArray3, adjusted_mi )

#print distance_matrix
#print distance_matrix.shape

#print distance_matrix2
#toy_distance = squareform( distance_matrix2 )

#print toy_distance 
#print squareform( toy_distance )
 
Z1 = linkage( distance_matrix1 )
Z2 = linkage( distance_matrix2 )

Z11 = linkage( distance_matrix11 )
Z22 = linkage( distance_matrix22 )


## this should be in dimensionality reduction script 
def get_medoid( pArray, iAxis = 0, pMetric = l2 ):
	"""
	Input: numpy array 
	Output: float
	
	For lack of better way, compute centroid, then compute medoid 
	by looking at an element that is closest to the centroid. 

	Can define arbitrary metric passed in as a function to pMetric 

	"""

	d = pMetric 

	pArray = ( pArray.T if bool(iAxis) else pArray  ) 

	print pArray.shape 

	mean_vec = np.mean(pArray, 0) 
	
	pArrayCenter = pArray - ( mean_vec * np.ones(pArray.shape) )

	return pArray[np.argsort( map( np.linalg.norm, pArrayCenter) )[0],:]


## this should be in dimensionality reduction script 
def get_representative( pArray, pMethod = None ):
	hash_method = {None: get_medoid}
	return hash_method[pMethod]( pArray )

def hclust( pArray, pdist_metric = mi, cluster_metric = l2 ):
	#returns linkage matrix 
	pdist_data = pdist( pArray, metric= pdist_metric )  
	linkage_data = linkage( pdist_data, metric=l2 ) 
	return linkage_data  


## this is the most useful function 
def reduce_tree( pClusterNode, pFunction = lambda x: x.id, aOut = [] ):
	func = pFunction

	if pClusterNode.is_leaf():
		return ( aOut + [func(pClusterNode)] )
	else:
		return reduce_tree( pClusterNode.left, func, aOut ) + \
			reduce_tree( pClusterNode.right, func, aOut ) 


def reduce_trees( pClusterNode1, pClusterNode2, pFunction = lambda x: x.id, aOut = [] ): 
	"""
	Meta version of reduce tree. 
	Can perform hierarchical all-against-all testing 
	""" 
	
	

## this is garbage 
def traverse( apClusterNode, pFunction = lambda x: x.id, aOut = [] ):

	def _compare( pClusterNode ):
		node = pClusterNode 
		get_representative( node.left ) 
		get_representative( node.right ) 
		 

	if len(apClusterNode) == 1:
		return reduce_tree( pClusterNode, pFunction )
	else:




	""" 
	Map/reduce like function for the scipy ClusterNode object.
	Perform all-against-all actions between clusters per layer on the tree

	"""

	ttest = lambda ttest_ind( )

	reduce_tree( pClusterNode, lambda ttest )




#tree = to_tree( Z11 )

if __name__ == "__main__":

	#aargs = sys.argv[1:]
	
	## plot stuff 
	pl.figure(1)

	pl.subplot(211)
	dendrogram( Z1 )
	pl.title("sym_mi 1")

	pl.subplot(212)
	dendrogram( Z2 )
	pl.title("sym_mi 2")

	pl.figure(2)

	pl.subplot(211)
	dendrogram( Z11 )
	pl.title("euc 1")

	pl.subplot(212)
	dendrogram( Z22 )
	pl.title("euc 2")

	pl.show()  

	print Z1 
