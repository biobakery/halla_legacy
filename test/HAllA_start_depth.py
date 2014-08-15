#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
Wrappers for testing HAllA's procedures such as, 
random data generation, Multi-ROC curves plotting etc 
"""

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
	methods = {"HAllA"}
	
	roc_info = [[]]
	fpr = dict()
	tpr = dict()
	
	#Generate simulated datasets
	s = strudel.Strudel()
	number_features = 16
	number_samples = 100
	number_blocks = 4
	print 'Synthetic Data Generation ...'
	'''X = data.simulateData(number_features,number_samples,number_blocks , .95, .05)
	Y,_ = s.spike( X, strMethod = "line" )
	'''
	X,Y,A = s.double_cholesky_block( number_features, number_samples , number_blocks, fVal = 2.0, Beta = 3.0)# link = "line" )


	#discretize the data prior to calculating the mutual information.
	print 'Discretize Data ...'
	dX = halla.discretize( X ) 
	dY = halla.discretize( Y )
	
	#The normalized mutual information function is captured in the distance module of halla
	NMI = lambda x,y: halla.distance.norm_mi(x,y)
	Pearson = lambda x,y: halla.distance.pearson(x, y)
	
	# Distance Matrix Generation
	Dx = scipy.spatial.distance.squareform( scipy.spatial.distance.pdist( dX, lambda u,v: 1.0 - NMI(u,v) ) )
	Dy = scipy.spatial.distance.squareform( 1.0 - scipy.spatial.distance.pdist( dY, NMI ) )

	l = len(Dx)
	condition = numpy.zeros((l,l))
	for i,j in itertools.product(range(l),range(l)):
		if abs(s.association(dX[i],dY[j]))>.5:
			condition[i][j] = 1 
	
	h = halla.HAllA( X,Y)
	new_methods = set()
	for alpha in {.05}:
		for q in {.1}:
			for start_parameter in {1.0, .9, .8, .5, .3, .2, .1}:
				# Setup alpha and q-cutoff and start parameter
				h.set_start_parameter (start_parameter)
				h.set_alpha (alpha)
				h.set_q(q)
				
				for method in methods:
					aOut = h.run(method)
					new_method = method+'_'+str(alpha)+'_'+str(q)+'_'+str(start_parameter)
					print new_method ,'is running ...'
					#new_method = new_method.replace(".", "")
					'''y_score =  [x for sublist in aOut[:][:][0] for x in sublist]
					y_score = [1-y_score[i] for i in range(len(y_score)) if i % 2 == 1]
					y_true = [x for sublist in condition for x in sublist]
					'''
					y_score = 1- h.meta_summary[0].flatten()
				#y_score= [1-y_score[i] for i in range(len(y_score)) if i % 2 == 1]
			
					y_true =  A.flatten()#[x for sublist in condition for x in sublist]
					fpr[new_method], tpr[new_method], _ = roc_curve(y_true, y_score, pos_label= 1)
					new_methods.add(new_method)
	figure_name = 'HAllA_start_parameter_'+str(number_features)+'_'+str(number_samples)+'_'+str(number_blocks)+'_'+str(alpha)+'_'+str(q)			
	for new_method in new_methods:
		method_info = [new_method, fpr[new_method], tpr[new_method]]
		if len(roc_info[0]) == 0:
			roc_info = [method_info]
		else:	
			roc_info.append(method_info)
	halla.plot.plot_roc(roc_info, figure_name)


if __name__ == '__main__':

	_main( )