'''
Created on Aug 20, 2014

@author: rah
'''
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
    methods = { "HAllA-PCA-NMI", "HAllA-ICA-NMI", "AllA-NMI", "AllA-MIC","HAllA-MIC",  "HAllA-KPCA-NMI", "HAllA-KPCA-Pearson", "HAllA-CCA-Pearson", "HAllA-CCA-NMI", "HAllA-PLS-NMI", "HAllA-PLS-Pearson"}
    tp_fp_counter = dict()
    roc_info = [[]]
    power = dict()
    power_data= []
    type_I_error_data = []
    labels = []
    typeI_error = dict()
    number_of_simulation = 5
    s = strudel.Strudel()
    number_features = 30
    number_samples = 100
    number_blocks = 5 
    q_cutoff = {.1}
    for q in q_cutoff:#, .05, .025, .01}:
        for method in methods:
                new_method = method+'_'+str(q)
                power[new_method] = []
                typeI_error[new_method] = []
                tp_fp_counter[new_method] = numpy.zeros((number_features,number_features))
    
    s = strudel.Strudel()
    for i in range(number_of_simulation):
        
        #Generate simulated datasets
        
        number_features = 8 + i 
        number_samples = 100 + i*5
        number_blocks = 3 + int(i/4)
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
        for q in q_cutoff:#, .25, .1, .05, .025, .01}:
            # Setup alpha and q-cutoff and start parameter
            #h.set_q(q)
            
            for method in methods:
                print new_method ,'is running ...with q, cut-off, ',q
                aOut = h.run(method)
                new_method = method+'_'+str(q)#+'_'+str(alpha)+'_'+str(q)+'_'+str(start_parameter)
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
                print 'All positives', all_positive_association
                print 'All negetives', all_negative_association
                number_association_tp  = 0.0
                number_association_fp = 0.0
                for i in range(len(y_true)):
                    #print score[i], '  ', y_true[i]
                    if score[i] < q and y_true[i] == 1.0 :
                        number_association_tp = number_association_tp + 1.0
                    if score[i] < q and y_true[i] == 0:
                        number_association_fp = number_association_fp + 1.0
                print 'number_association_tp', number_association_tp
                print 'number_association_fp:', number_association_fp
                power[new_method].append((number_association_tp/all_positive_association))
                typeI_error[new_method].append((number_association_fp/all_negative_association))
                print 'power:', power[new_method]
                print 'TypeI Error:', typeI_error[new_method] 
    
    for q in q_cutoff: #, .05, .025, .01}:
        for method in methods:
            labels.append(str(method))
            new_method = method+'_'+str(q)
            power_data.append(power[new_method])
            type_I_error_data.append(typeI_error[new_method])
            #print 'power:', power[new_method]
            #print 'TypeI Error:', typeI_error[new_method] 
        halla.plot.plot_box(power_data, figure_name = 'Power_'+str(q), alpha = q, ylabel = 'Statistical Power', labels = labels)
        halla.plot.plot_box(type_I_error_data, figure_name = 'Type_I_error_'+str(q), alpha = q, ylabel = 'Type I Error', labels = labels)
        labels = []
        power_data = []
        type_I_error_data = []
        #data.append(power[new_method])
        #data.append(typeI_error[new_method]) 
    return;
if __name__ == '__main__':
    _main( )
    