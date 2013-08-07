#!/usr/bin/env python 

import time as time
import numpy as np
import scipy as sp
from sklearn.cluster import Ward
import csv 
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram 
import pylab as pl 
from random import random 

csvr = csv.reader(open("table.tsv"), csv.excel_tab)

c_DataArray = np.array([x[1:] for x in csvr][1:])
c_DataArray2 = np.array([[0,0],[0,1],[1,0],[1,1]])
c_DataArray3 = np.array([[random() for x in range(2)] for y in range(10)])


distance_matrix = pdist( c_DataArray )
distance_matrix2 = pdist( c_DataArray2 )
distance_matrix3 = pdist( c_DataArray3 )

#print distance_matrix
#print distance_matrix.shape


#print distance_matrix2
#toy_distance = squareform( distance_matrix2 )

#print toy_distance 
#print squareform( toy_distance )

Z2 = linkage( distance_matrix2 )
Z = linkage( distance_matrix )
Z3 = linkage( distance_matrix3 )

#dendrogram( Z )  

dendrogram(Z3)
print Z3

pl.show()

