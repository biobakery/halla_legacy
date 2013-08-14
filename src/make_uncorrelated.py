#!/usr/bin/env python 

import numpy as np 
import pandas as pd 
import sys 
from pprint import pprint 
from sklearn.decomposition import PCA 

if len(sys.argv[1:]) < 3: 
	raise Exception("Too few arguments")

args = sys.argv[1:]
iRow, iCol, iComp = map(int, args[:3])

#The order of the rows and columns are reversed due to the scikit learn convention, so we don't have to invert the matrix twice. 
rand_data = np.random.random((iCol,iRow))

pPCA = PCA(n_components=iComp)

df = pd.DataFrame( pPCA.fit_transform(rand_data).T )

#print df 

df.to_csv(sys.stdout, sep="\t")

#print df.corr() 