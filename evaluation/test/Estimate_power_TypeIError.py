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
	methods = {'HAllA', 'AllA', 'HAllA-MIC', 'MIC'}
	tp_fp_counter = dict()
	roc_info = [[]]
	power = dict()
	power_data= []
	type_I_error_data = []
	labels = []
	typeI_error = dict()
	number_of_simulation = 2
	s = strudel.Strudel()
	number_features = 8
	number_samples = 10
	number_blocks = 3
	
	for q in {.05}:#, .05, .025, .01}:
		for method in methods:
			new_method = method+'_'+str(q)
			power[new_method] = []
			typeI_error[new_method] = []
			tp_fp_counter[new_method] = numpy.zeros((number_features,number_features))
			
	
	print 'Synthetic Data Generation ...'
	for iter_number in range(number_of_simulation):
		X,Y,A = s.double_cholesky_block( number_features, number_samples , number_blocks, fVal = 0.6 , Beta = 3.0 )# link = "line" )
		#A = numpy.zeros((number_features,number_features))
		#X = numpy.random.rand(number_features,number_features)	
		#Y = numpy.random.rand(number_features,number_features)
		h = halla.HAllA( X,Y)
		#start_parameter = .05
		#alpha = .3
		#h.set_start_parameter (start_parameter)
		#h.set_alpha (alpha)
		for q in { .05}:#, .05, .025, .01}:
			# Setup alpha and q-cutoff and start parameter
			h.set_q(q)	
			for method in methods:
				aOut = h.run(method)
				#print aOut
				new_method = method+'_'+str(q)
				print new_method ,'is running ...with q, cut-off, ',q
				#y_score = 1- h.meta_summary[0].flatten()
				#print 'h.meta_summary[0]', h.meta_alla
				#break;
				#print 'A', A
				for i,j in itertools.product(range(number_features),range(number_features)):
					if A[i][j] == 1 and h.meta_summary[0][i][j]<=q:
						tp_fp_counter[new_method][i][j] += 1 
					if A[i][j] == 0 and h.meta_summary[0][i][j]<q:
						tp_fp_counter[new_method][i][j] += 1		
				if iter_number == number_of_simulation-1:
					for i,j in itertools.product(range(number_features),range(number_features)):
						if A[i][j] ==1:
							power[new_method].append(tp_fp_counter[new_method][i][j]/number_of_simulation)
						else:
							typeI_error[new_method].append((tp_fp_counter[new_method][i][j]/number_of_simulation))

	#data =[] 
	for q in {.05}:#, .05, .025, .01}:
		for method in methods:
			labels.append(str(method))
			new_method = method+'_'+str(q)
			power_data.append(power[new_method])
			type_I_error_data.append(typeI_error[new_method])
			print 'power:', power[new_method]
			print 'TypeI Error:', typeI_error[new_method] 
		halla.plot.plot_box(power_data, figure_name = new_method+'_power_'+str(q), alpha = q, ylabel = 'Power', labels = labels)
		halla.plot.plot_box(type_I_error_data, figure_name = new_method+'_type_I_error_'+str(q), alpha = q, ylabel = 'type_I_error', labels = labels)
		labels = []
		power_data = []
		type_I_error_data = []
			#data.append(power[new_method])
			#data.append(typeI_error[new_method]) 
	return;
if __name__ == '__main__':
	_main( ) 
	