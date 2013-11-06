#!/usr/bin/env python

"""
Module name: cakecut.py 
Description: Perform baseline test of cakecutting procedure 

Idea: I can define a deterministic log-cut of the cake, and feed in permutations of the cake; this achieves the same result 

Idea: Avoid memory intensive functions if possible. 

"""

from test import *  
import numpy as np
from numpy import array 
from numpy.random import shuffle 
import math 
from stats import pca, discretize, permutation_test_by_representative 
from test import rand, randmix, uniformly_spaced_gaussian
from distance import mi, norm_mi 
from pylab import * 

def cut( D, pZ ):
	"""
	Makes cut on D based on partition function pZ 
	Input: 
	Output: 
	"""
	pass 

def log_cut( cake_length, iBase = 2 ):
	"""
	Input: cake_length <- length of array, iBase <- base of logarithm 

	Output: array of indices corresponding to the slice 

	Note: Probably don't want size-1 cake slices, but for proof-of-concept, this should be okay. 
	Avoid the "all" case 

	Caveat: returns smaller slices first 
	"""

	aOut = [] 

	iLength = cake_length 

	iSize = int( math.floor( math.log( iLength , iBase ) ) )
	aSize = [2**i for i in range(iSize)] 

	iStart = 0 
	for item in aSize:
		iStop =  iStart + item 
		if iStop == iCol - 1:
			iStop += 1 
			# ensure that the rest of the cake gets included in the tail case  
		aOut.append( array( range(iStart, iStop) ) ) 
		iStart = iStop 

	aOut.reverse()
	return aOut 


def CP_cut( D ):
	pass 	


def p_val_plot( pArray1, pArray2, pCut = log_cut, iIter = 100 ):
	"""
	Returns p value plot of combinatorial cuts 

	In practice, works best when arrays are of similar size, since I implement the minimum ... 
	For future think about implementing the correct step function 

	"""
	aOut = None 

	for i in iIter:
		D1 = shuffle( pArray1 ) 
		D2 = shuffle( pArray2 )
		len1, len2 = len( D1 ), len( D2 )
		cut1, cut2 = pCut( len1 ), pCut( len2 )
		cut1.reverse(); cut2.reverse() #in practice, we want to test the bigger slices first 
		lencut1, lencut2 = len(cut1), len(cut2)
		iMin = min( lencut1, lencut2 )
		if not aOut:
			aOut = [[]]* iMin 

		for j in range(iMin):
			dP = permutation_test_by_representative( pArray1[cut1[j]], pArray2[cut2[j]] )
			aOut[j].append( dP )

	boxplot( aOut, '', 0)
	show() 

	return aOut 

if __name__ == "__main__":
	rand_sample = rand( (100,1) ).flatten()  
	rand_mixture = array( uniformly_spaced_gaussian( 100 ) )

	p_val_plot( rand_mixture, rand_mixture )





