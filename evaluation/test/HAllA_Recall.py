
#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
Author: Gholamali Rahnavard
Description: HAllA Evaluation.
"""

#####################################################################################
#Copyright (C) <2015>
#
#Permission is hereby granted, free of charge, to any person obtaining a copy of
#this software and associated documentation files (the "Software"), to deal in the
#Software without restriction, including without limitation the rights to use, copy,
#modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
#and to permit persons to whom the Software is furnished to do so, subject to
#the following conditions:
#
#The above copyright notice and this permission notice shall be included in all copies
#or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
#PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
#HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
#OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#####################################################################################

__author__ = "Gholamali Rahnavard"
__copyright__ = "Copyright 2015"
__credits__ = ["Gholamali Rahnavard"]
__license__ = "MIT"
__hallatainer__ = "Gholamali Rahnavard"
__email__ = "Gholamali.Rahnavard@gmail.com"
__status__ = "Development"

import numpy as np
import math
import sys
sys.path.append('/Users/rah/Documents/Hutlab/halla')
sys.path.append('/Users/rah/Documents/Hutlab/strudel')
import strudel, halla, pylab
from halla import stats, data, distance


def _main( ):
    #methods = { "HAllA-PCA-NMI","HAllA-PCA-AMI","HAllA-ICA-NMI", "HAllA-PCA-MIC", "HAllA-ICA-MIC", "HAllA-KPCA-NMI", "HAllA-KPCA-Pearson", "HAllA-CCA-Pearson", "HAllA-CCA-NMI", "HAllA-PLS-NMI", "HAllA-PLS-Pearson","AllA-NMI", "AllA-MIC"}
    # log
    #methods = { "HAllA-PCA-NMI","HAllA-ICA-NMI", "HAllA-PCA-MIC", "HAllA-ICA-MIC"}
    #methods = {  "HAllA-KPCA-Pearson", "HAllA-CCA-Pearson", "HAllA-PLS-NMI", "HAllA-PLS-Pearson"}
    #methods = {"HAllA-PCA-NMI", "HAllA-PCA-MIC"}
    #methods = { "HAllA-CCA-Pearson"}
    methods = {"HAllA-PCA-NMI"}
    #methods = {"HAllA-PCA-MIC"}
    #methods = {"AllA-NMI"}
    #methods = {"HAllA-KPCA-Pearson"}
    #methods = {"layerwise"}
    tp_fp_counter = dict()
    roc_info = [[]]
    recall = dict()
    recall_data= []
    fdr_data = []
    labels = []
    fdr = dict()
    mean_recall = []
    mean_fdr = []
    
    number_of_simulation = 1
    s = strudel.Strudel()
    q_cutoff = {.2}
    for q in q_cutoff:#, .05, .025, .01}:
        for method in methods:
                new_method = method+'_'+str(q)
                recall[new_method] = []
                fdr[new_method] = []
                #tp_fp_counter[new_method] = np.zeros((number_features,number_features))
    
    s = strudel.Strudel()
    for i in range(number_of_simulation):
        #Generate simulated datasets
        number_features = 4 + i
        number_samples = 50 + i*5
        number_blocks = 2 + int(i/2)
        print 'Synthetic Data Generation ...'
        
        X,Y,A = s.double_cholesky_block( number_features, number_samples , number_blocks, fVal = 2.6 , Beta = 3.0 )#, link = "line" )
#       
        halla.data.writeData(X,"X" )
        halla.data.writeData(Y,"Y")
        h = halla.HAllA( X,Y)
        
        for q in q_cutoff:#, .25, .1, .05, .025, .01}:
            # Setup alpha and q-cutoff and start parameter
            h.set_q(q)
            for method in methods:
                print method ,'is running ...with q, cut-off: ',q
                aOut = h.run(method)
                #print "aOut", h.meta_alla
                new_method = method+'_'+str(q)#+'_'+str(alpha)+'_'+str(q)+'_'+str(start_parameter)
                #y_score = 1- h.meta_summary[0].flatten()
                #print 'h.meta_summary[0]', h.meta_summary
                #print 'A', A
                print 'Discovered Associations:', h.meta_alla[0]
                print 'All Comparisons:', h.meta_alla[1]
                score  = h.meta_summary[:,:, 0].flatten()
                print 'score: ', score
                y_true =  A.flatten()#[x for sublist in condition for x in sublist]
               
                condition_positive = sum(1 for i in y_true if i==1.0)
                test_outcome_positive = sum(1 for i in score if math.fabs(i) <= q)
                
                number_association_tp  = 0.0
                number_association_fp = 0.0
                for i in range(len(y_true)):
                    #print score[i], '  ', y_true[i]
                    if math.fabs(score[i]) <= q and y_true[i] == 1.0 :
                        number_association_tp = number_association_tp + 1.0
                    if math.fabs(score[i]) <= q and y_true[i] == 0.0:
                        number_association_fp = number_association_fp + 1.0
                #print 'number_association_tp', number_association_tp
                #print 'number_association_fp:', number_association_fp
                if condition_positive > 0.0:
                    recall[new_method].append((number_association_tp/condition_positive))
                if test_outcome_positive > 0.0:
                    fdr[new_method].append((number_association_fp/test_outcome_positive))
                else:
                    fdr[new_method].append(0.0) 
                print str(new_method)
                print 'Recall:', recall[new_method]
                print 'FDR:', fdr[new_method] 
    
    for q in q_cutoff: #, .05, .025, .01}:
        for method in methods:
            labels.append(str(method))
            new_method = method+'_'+str(q)
            recall_data.append(recall[new_method])
            fdr_data.append(fdr[new_method])
            mean_recall.append(np.mean(recall[new_method]))
            mean_fdr.append(np.mean(fdr[new_method]))
            #print 'recall:', recall[new_method]
            #print 'TypeI Error:', fdr[new_method] 
        #halla.plot.plot_box(recall_data, figure_name = 'Figure2a', alpha = q, ylabel = 'Recall', labels = labels)
        #halla.plot.plot_box(fdr_data,figure_name = 'Figure2b', alpha = q, ylabel = 'FDR', labels = labels)
        halla.plot.scatter_plot( mean_recall, mean_fdr, alpha = q, labels = labels)
        f = open('power_fdr.txt', 'w')
        s = "Recall:" + str(recall_data) +" mean: "+ str(mean_recall) +"\n"
        f.write(s)
        s = "FDR:" + str(fdr_data) +" mean: "+ str(mean_fdr) +"\n"
        f.write(s)
        
        labels = []
        recall_data = []
        fdr_data = []
        #data.append(recall[new_method])
        #data.append(fdr[new_method]) 
    return
if __name__ == '__main__':
    _main( )
    