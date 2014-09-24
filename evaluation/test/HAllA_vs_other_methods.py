#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
Wrappers for testing HAllA's procedures such as, 
random data generation, Multi-ROC curves plotting etc 
"""
import sys
sys.path.append('//Users/rah/Documents/Hutlab/halla')
sys.path.append('/Users/rah/Documents/Hutlab/strudel')
from itertools import compress
import matplotlib
from numpy import array
import numpy
from numpy.random import normal, multinomial
import scipy
import scipy.spatial.distance
from scipy.stats import percentileofscore
#import halla.plot as pl
#import halla.distance
#import halla.stats
import strudel, halla, pylab
from halla import data, stats
import itertools
from sklearn.metrics.metrics import roc_curve

def _main( ):
	
	#Different methods to run
	methods = { "HAllA-PCA-NMI", "HAllA-ICA-NMI", "AllA-NMI", "AllA-MIC","HAllA-MIC",  "HAllA-KPCA-NMI", "HAllA-KPCA-Pearson", "HAllA-CCA-Pearson", "HAllA-CCA-NMI", "HAllA-PLS-NMI", "HAllA-PLS-Pearson"}
	
	roc_info = [[]]
	fpr = dict()
	tpr = dict()
	
	#Generate simulated datasets
	s = strudel.Strudel()
	number_features = 32
	number_samples = 300
	number_blocks = 8
	print 'Synthetic Data Generation ...'
	#X = data.simulateData(number_features,number_samples,number_blocks , .95, .05)
	X,Y,A = s.double_cholesky_block( number_features, number_samples , number_blocks, fVal = 0.6, Beta = 3.0)# link = "line" )

	h = halla.HAllA( X,Y)
	
	# Setup alpha and q-cutoff and start parameter
	start_parameter = 0.0
	alpha = 0.2
	q = 0.1
	h.set_start_parameter (start_parameter)
	h.set_alpha (alpha)
	h.set_q(q)
	figure_name = str(number_features)+'_'+str(number_samples)+'_'+str(number_blocks)+'_'+str(alpha)+'_'+str(q)+'_'+str(start_parameter)

	for method in methods:
		print str(method) ,'is running ...'
		aOut = h.run(method)
		#fpr_temp_method, tpr_temp_method = stats.fpr_tpr( condition, h.outcome)
		#print 'report' ,h.meta_report
		print 'summary' ,h.meta_summary[0]
		#print 'output', aOut
		y_score = 1.0 - h.meta_summary[0].flatten()#[x for sublist in aOut[:][:][0] for x in sublist]
		
		#y_score = [1-y_score[i] for i in range(len(y_score)) if i % 2 == 1]
		#print 'yscore',y_score
		y_true = A.flatten() # [x for sublist in condition for x in sublist]
		#print 'ytrue',y_true
		fpr[method], tpr[method], _ = roc_curve(y_true, y_score, pos_label= 1)
		#s.roc(1.0 - A.flatten(), h.meta_summary[0].flatten())
		#del h
		
	#print(fpr_HAllA, tpr_HAllA)
	#print(fpr_AllA, tpr_AllA)
	for method in methods:
		method_info = [method, fpr[method], tpr[method]]
		#print(method_info)	
		if len(roc_info[0]) == 0:
			roc_info = [method_info]
		else:	
			roc_info.append(method_info)

	halla.plot.plot_roc(roc_info, figure_name)

if __name__ == '__main__':

	_main( )
#================================================================================
# Base cases 
#================================================================================
def randmat( tShape = (10,10), pDist = numpy.random.normal ):
	"""
	Returns a tShape-dimensional matrix given by base distribution pDist 
	Order: Row, Col 
	"""	
	H = pDist #base measure 
	
	iRow, iCol = tShape

	assert( iRow != 0 and iCol !=0 ) 
	
	return array( [[H() for i in range(iCol)] for j in range(iRow)] )

def randmix( N, pDist, atParam, tPi ):
	"""
	Returns N copies drawn from a mixture distribution $H$ 
	Input: N <- number of components
		pDist <- pointer to base distribution H 
		atParam <- length $k$ parameters to distribution pDist, $\theta$  
		tPi <- length $k$ tuple (vector) to categorical rv Z_n 

	Output: N copies from mixture distribution $\sum_{k=1}^{K} \pi_k H(.| \theta )$ 
	""" 
	
	assert( len( atParam ) == len( tPi ) )
	
	aOut = [] 

	K = len( atParam ) 
	H = pDist 
	for n in range(N):
		# multinomial returns boolean vector 
		aParam = [x for x in compress( atParam, multinomial( 1, tPi ) )][0]
		aOut.append( H( *aParam ) )
	return aOut

#================================================================================
# Special cases 
#================================================================================

def uniformly_spaced_gaussian( N, K = 4, fD = 2.0, tPi = (0.25,0.25,0.25,0.25) ):
	"""
	Generate uniformly spaced Gaussian, with spacing fD in the mean.
	Constant 1.0 variance 

	"""
	return randmix( N, pDist = normal, atParam = [(m,1.0) for m in [fD*i for i in range(K)]], tPi = tPi )


#================================================================================
# Pipelines  
#================================================================================

def nov_27_2013():
	x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])
	y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0],
		[0.015625,0.125,0.421875,1.0]])
	dx = halla.stats.discretize( x, iN = None, method = None, aiSkip = [1,3] )
	dy = halla.stats.discretize( y, iN = None, method = None, aiSkip = [1] )
	from scipy.stats import spearmanr
	spearman = lambda x,y: spearmanr(x,y)[0]

	#Pearson
	lxpearson	= halla.hierarchy.linkage(x, metric="correlation")
	lypearson	= halla.hierarchy.linkage(y, metric="correlation")

	#Spearman 
	lxspearman	= halla.hierarchy.linkage(x, metric= spearman)
	lyspearman	= halla.hierarchy.linkage(y, metric= spearman)

	#MI
	lxmi		= halla.hierarchy.linkage(x, metric= halla.distance.mi)
	lymi		= halla.hierarchy.linkage(y, metric= halla.distance.mi)

	aL			= [(lxpearson,lypearson),(lxspearman,lyspearman), (lxmi, lymi)]
	aNames		= ["pearson", "spearman", "mi"]


	#### Plotting 
	from scipy.cluster.hierarchy import dendrogram
	for k, item in enumerate(aL):
		i,j = item
		i,j = numpy.abs(i), numpy.abs(j)
		#matplotlib.pyplot.figure(aNames[k])
		#matplotlib.pyplot.subplot(1,2,1)
		dendrogram(i)
		#matplotlib.pyplot.subplot(1,2,2)
		dendrogram(j)







"""
		.. plot::

		x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])
		y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0],[0.015625,0.125,0.421875,1.0]])
		dx = halla.stats.discretize( x, iN = None, method = None, aiSkip = [1,3] )
		dy = halla.stats.discretize( y, iN = None, method = None, aiSkip = [1] )

		spearman = lambda x,y: scipy.stats.spearmanr(x,y)[0]

		#Pearson
		lxpearson	= linkage(x, metric="correlation")
		lypearson	= linkage(y, metric="correlation")

		#Spearman 
		lxspearman	= linkage(x, metric= spearman)
		lyspearman	= linkage(y, metric= spearman)

		#MI
		lxmi		= linkage(dx, metric= mi)
		lymi		= linkage(dy, metric= mi)

		aL			= [(lxpearson,lypearson),(lxspearman,lyspearman), (lxmi, lymi)]
		aNames		= ["pearson", "spearman", "mi"]


		#### Plotting 

		for k, item in enumerate(aL):
			i,j = item
			i,j = numpy.abs(i), numpy.abs(j)
			matplotlib.pyplot.figure(aNames[k])
			matplotlib.pyplot.subplot(1,2,1)
			dendrogram(i)
			matplotlib.pyplot.subplot(1,2,2)
			dendrogram(j)
"""