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
import sys
sys.path.append('//Users/rah/Documents/Hutlab/halla')
sys.path.append('/Users/rah/Documents/Hutlab/strudel')
#sys.path.insert(1, '../../strudel')
import strudel, halla, pylab
from halla import stats
import itertools
from sklearn.metrics.metrics import roc_curve

def _main( ):
	
	#Different methods to run
	methods = {"HAllA"}
	
	roc_info = [[]]
	power = dict()
	typeI_error = dict()
	for method in methods:
		power[method] = []
		typeI_error[method] = []
		
	number_of_simulation = 40
	for i in range(number_of_simulation):
		
		#Generate simulated datasets
		s = strudel.Strudel()
		number_features = 8 + i 
		number_samples = 500 + i*5
		number_blocks = 2 + int(i/4)
		print 'Synthetic Data Generation ...'
		'''X = data.simulateData(number_features,number_samples,number_blocks , .95, .05)
		Y,_ = s.spike( X, strMethod = "line" )
		'''
		X,Y,A = s.double_cholesky_block( number_features, number_samples , number_blocks, fVal = .6 , Beta = 3.0 )# link = "line" )
			
		h = halla.HAllA( X,Y)
		new_methods = set()
		#start_parameter = .05
		#alpha = .3
		#h.set_start_parameter (start_parameter)
		#h.set_alpha (alpha)
		for q in {.05}:#, .25, .1, .05, .025, .01}:
			# Setup alpha and q-cutoff and start parameter
			#h.set_q(q)
			
			for method in methods:
				aOut = h.run(method)
				new_method = method#+'_'+str(alpha)+'_'+str(q)+'_'+str(start_parameter)
				print new_method ,'is running ...with q, cut-off, ',q
				#y_score = 1- h.meta_summary[0].flatten()
				#print 'h.meta_summary[0]', h.meta_summary
				#print 'A', A
				#print 'h.meta_alla[0]', h.meta_alla[0]
				score  = h.meta_summary[0].flatten()
				y_true =  A.flatten()#[x for sublist in condition for x in sublist]
				#print 'h.meta_summary[0]', zip (score, y_true)
				#print 'A', y_true
				#fpr[new_method], tpr[new_method], _ = roc_curve(y_true, y_score, pos_label= 1)
				all_positive_association = sum(1 for i in y_true if i==1.0)
				all_negative_association = len(y_true) - all_positive_association
				print all_positive_association
				print all_negative_association
				number_association_tp  = 0.0
				number_association_fp = 0.0
				for i in range(len(y_true)):
					#print score[i], '  ', y_true[i]
					if score[i] < q and y_true[i] == 1 :
						number_association_tp = number_association_tp + 1
					if score[i] < q and y_true[i] == 0:
						number_association_fp = number_association_fp + 1
				print 'number_association_tp', number_association_tp
				print 'number_association_fp:', number_association_fp
				power[method].append((number_association_tp/all_positive_association))
				typeI_error[method].append((number_association_fp/all_negative_association))
				print 'power:', power[method]
				print 'TypeI Error:', typeI_error[method] 
	#power[method] /= number_of_simulation
	#typeI_error[method] /= number_of_simulation	
	#print 'power:', power[method]
	#print 'TypeI Error:', typeI_error[method] 
	halla.plot.plot_box([power[method], typeI_error[method]], figure_name = 'HAllA_'+str(q), alpha = q)	
		
	return;
	'''figure_name = 'HAllA_start_parameter_'+str(number_features)+'_'+str(number_samples)+'_'+str(number_blocks)+'_'+str(alpha)+'_'+str(q)			
	for new_method in new_methods:
		method_info = [new_method, fpr[new_method], tpr[new_method]]
		if len(roc_info[0]) == 0:
			roc_info = [method_info]
		else:	
			roc_info.append(method_info)
	halla.plot.plot_roc(roc_info, figure_name)
	'''
if __name__ == '__main__':
	_main( ), 