#!/usr/bin/env python 

"""
Scaffold script for HAllA development
"""

## structural packages 

import sys 
import re 

## statistics packages 

from datum import discretize 
import numpy as np
import scipy as sp
import sklearn.decomposition
from sklearn.decomposition import PCA #remember that the matrix is X = (n_samples,n_features)
import csv 
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree, leaves_list
import pylab as pl 
import random  
from distance import mi, sym_mi 
from numpy.random import normal 
from scipy.misc import * 
import pandas as pd 


#=================================================# 
# FUNCTIONS  
#=================================================# 

l2 = lambda x,y: np.linalg.norm(x-y)

#=================================================# 
# RUNTIME  
#=================================================# 
"""
Hierarchically compare two datasets 

"""

## Parse 

## Load into dataframe 

pd.DataFrame( )

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

def compute_medoid( pArray, pMetric = l2, iAxis = 0 ):
	"""
	Input: numpy array 
	Output: float
	
	For lack of better way, compute centroid, then compute medoid. 

	"""

	d = pMetric 

	mean_vec = np.mean(pArray,0) 
	
	pArrayCenter = pArray - ( mean_vec * np.ones(pArray.shape) )

	return pArray[np.argsort( map( np.linalg.norm, pArrayCenter) )[0],:]



def reduce_tree( pClusterNode, pFunction = lambda x: x.id, aOut = [] ):
	func = pFunction

	if pClusterNode.is_leaf():
		return ( aOut + [func(pClusterNode)] )
	else:
		return reduce_tree( pClusterNode.left, func, aOut ) + reduce_tree( pClusterNode.right, func, aOut ) 

tree = to_tree( Z11 )


if __name__ == "__main__":

	aargs = sys.argv[1:]




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


	#pl.show()  


	print Z1 