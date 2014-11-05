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
import numpy as np
from numpy.random import normal, multinomial
import scipy
import scipy.spatial.distance
from scipy.stats import percentileofscore
#import halla.plot as pl
#import halla.distance
#import halla.stats
import sys
#sys.path.append('//Users/rah/Documents/Hutlab/halla')
#sys.path.append('/Users/rah/Documents/Hutlab/strudel')
#sys.path.insert(1, '../../strudel')
import strudel, halla, pylab
from halla import stats, data, distance
import itertools
from sklearn.metrics.metrics import roc_curve

def _main( ):
    #methods = { "HAllA-PCA-NMI","HAllA-PCA-AMI","HAllA-ICA-NMI", "HAllA-PCA-MIC", "HAllA-ICA-MIC", "HAllA-KPCA-NMI", "HAllA-KPCA-Pearson", "HAllA-CCA-Pearson", "HAllA-CCA-NMI", "HAllA-PLS-NMI", "HAllA-PLS-Pearson","AllA-NMI", "AllA-MIC"}
    # log
    #methods = { "HAllA-PCA-NMI","HAllA-ICA-NMI", "HAllA-PCA-MIC", "HAllA-ICA-MIC"}
    #methods = {  "HAllA-KPCA-Pearson", "HAllA-CCA-Pearson", "HAllA-PLS-NMI", "HAllA-PLS-Pearson"}
    methods = {"HAllA-PCA-NMI", "HAllA-PCA-MIC"}
    #methods = { "HAllA-CCA-Pearson"}
    #methods = {"HAllA-PCA-NMI"}
    #methods = {"HAllA-PCA-MIC"}
    #methods = {"AllA-NMI"}
    #methods = {"HAllA-KPCA-Pearson"}
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
    #number_samples = 10
    #number_blocks = 2 
    #score1 = [0.03291897,-0.1777683,0.2899735,-0.110356,-0.02693113,0.02054444,-0.1809396,0.06206155,-0.1919755,-0.05025634,0.04550797,0.0610555,-0.03042005,0.07763993,0.2388963,-0.03225611,0.1364597,-0.1196715,0.07306166,-0.3518646,0.2563328,-0.1984173,0.1651143,0.1296622,-0.1538042,-0.2526885,-0.09633546,-0.04561689,-0.1851895,-0.123638,-0.1273299,-0.0003383346,0.2511064,0.2404161,-0.2041926,-0.07122211,0.1328209,0.1931385,-0.03912234,0.1926952,-0.5017717,-0.1065692,0.2602412,-0.01127374,0.06229825,0.07691513,-0.09128536,-0.03065815,-0.009960095,0.05464861,0.1398714,0.1741885,0.2058788,-0.08893114,-0.3944827,0.1926165,-0.1010819,-0.07875292,-0.009465269,-0.01591669,0.1953516,0.1620659,0.2494596,-0.08150731,0.0242429,0.2337929,0.0420111,-0.2709789,-0.04657647,0.2611743,-0.06291497,0.03527707,-0.7620585,0.2674127,0.2091163,-0.1033677,0.1370698,-0.03927038,-0.3626855,-0.1972406,0.1249753,-0.1731171,0.07606433,-0.05765939,-0.06256049,0.1429092,-0.1090209,0.1800892,-0.03595361,-0.1849367,-0.0532486,0.12641,-0.1057036,-0.1827086,0.09221026,0.02275268,0.08156469,0.008947097,0.2908267,0.2006068,0.3000042, 0.123867, -0.1856332, -0.06664027]
    #gentisate = [-2.803896939,0.848525004,2.337156759,1.109115819,1.149897344,0.13528992,0.158061311,0.714277918,1.328749628,1.119709746,-0.288184305,0.487029427,0.634337124,-0.098073725,0.097477949,0.071809458,0.454466981,-0.327865239,0.729327623,1.045276511,0.312906643,0.120882089,0.369824602,0.285065158,0.561143341,0.111381205,-0.293628799,-0.178924627,-0.112243296,1.237924395,2.227920252,1.222810016,2.537628451,1.220163919,0.093127075,0.501187279,1.030010539,1.437226516,-0.718826102,-0.171019997,-4.380434511,-0.247839406,-0.606073698,1.072803476,-1.496260884,-0.337527087,1.02548119,0.941821597,-0.902707592,2.012035388,-0.207272705,-0.575161602,2.713930255,0.216390163,0.543604771,-0.228780703,-0.257659499,-0.013931937,0.497006386,0.180530168,-0.964544835,-2.23245979,0.797249201,1.309643979,-0.062199528,0.542413658,0.103351616,-0.000857285,-0.397219976,0.707122363,-0.203863764,0.356717166,-0.642701928,0.124088691,0.546656896,-2.126731775,-0.861813436,-0.211929283,-1.235060927,0.111532633,0.180513843,-1.167946232,0.155406771,-0.450751692,-0.475081011,-0.801261868,0.384715843,0.047546605,-1.404889596,0.329788325,-0.377645309,-0.996422113,-2.261983185,-2.77811642,0.854787825,-0.573595364,-0.545498864,-0.834801286,-0.442397744,-0.175937405,-1.001478347,0.286864456,-1.5003544,-2.757857251]
    #print"NMI", distance.NormalizedMutualInformation( score1, gentisate ).get_distance()
    #return 
    q_cutoff = {.1}
    for q in q_cutoff:#, .05, .025, .01}:
        for method in methods:
                new_method = method+'_'+str(q)
                recall[new_method] = []
                fdr[new_method] = []
                #tp_fp_counter[new_method] = np.zeros((number_features,number_features))
    
    s = strudel.Strudel()
    for i in range(number_of_simulation):
        
        #Generate simulated datasets
        number_features = 9 + i
        number_samples = 100 + i*5
        number_blocks = 3 + int(i/3)
        print 'Synthetic Data Generation ...'
        '''X = data.simulateData(number_features,number_samples,number_blocks , .95, .05)
        Y,_ = s.spike( X, strMethod = "line" )
        '''
        X,Y,A = s.double_cholesky_block( number_features, number_samples , number_blocks, fVal = 2.6 , Beta = 3.0 )#, link = "line" )
        #X = np.concatenate((X, np.random.randint(0.0, 1000.0,(number_features*2, number_samples)) ), axis=0)
        #A = np.concatenate((A, np.zeros((number_features*2, number_features))), axis=0)
        #A = np.concatenate((A, np.zeros((number_features*3, number_features*2))), axis=1)
        #Y = np.concatenate((Y, np.random.randint(0.0, 1000.0,(number_features*2, number_samples)) ),axis=0)
        #X,Y,A = s.cholesky_nlblock( number_features, number_samples , number_blocks, fVal = 2.6, Beta = 3.0, link = "half_circle" )
        #X,Y,A = s.cholesky_nlblock( number_features, number_samples , number_blocks, fVal = 2.6, Beta = 3.0, link = "log" )
        #X,Y,A = s.cholesky_nlblock( number_features, number_samples , number_blocks, fVal = 2.6, Beta = 3.0, link = "sine" )
        #X,Y,A = s.cholesky_nlblock( number_features, number_samples , number_blocks, fVal = 2.6, Beta = 3.0, link = "parabola" )
        #X1,Y,A1 = s.double_cholesky_block( number_features/4, number_samples , number_blocks, fVal = .6 , Beta = 3.0 )
        #print A, X, Y
        #return
        halla.data.writeData(X,"X")
        halla.data.writeData(Y,"Y")
        h = halla.HAllA( X,Y)
        #new_methods = set()
        #start_parameter = .05
        #alpha = .3
        #h.set_start_parameter (start_parameter)
        #h.set_alpha (alpha)
        for q in q_cutoff:#, .25, .1, .05, .025, .01}:
            # Setup alpha and q-cutoff and start parameter
            h.set_q(q)
            h.set_alpha(.1)
            #h.iterations = 100
            #h.set_start_parameter(0.5)
            for method in methods:
                print method ,'is running ...with q, cut-off, ',q
                aOut = h.run(method)
                #print "aOut", h.meta_alla
                new_method = method+'_'+str(q)#+'_'+str(alpha)+'_'+str(q)+'_'+str(start_parameter)
                #y_score = 1- h.meta_summary[0].flatten()
                print 'h.meta_summary[0]', h.meta_summary
                #print 'A', A
                print 'h.meta_alla[0]', h.meta_alla
                score  = h.meta_summary[0].flatten()
                y_true =  A.flatten()#[x for sublist in condition for x in sublist]
                #print 'h.meta_summary[0]', zip (score, y_true)
                #print 'A', y_true
                #fpr[new_method], tpr[new_method], _ = roc_curve(y_true, y_score, pos_label= 1)
                condition_positive = sum(1 for i in y_true if i==1.0)
                all_negative_association = len(y_true) - condition_positive
                test_outcome_positive = sum(1 for i in score if i <= q)
                #print 'All positives', condition_positive
                #print 'All negetives', all_negative_association
                number_association_tp  = 0.0
                number_association_fp = 0.0
                for i in range(len(y_true)):
                    #print score[i], '  ', y_true[i]
                    if score[i] <= q and y_true[i] == 1.0 :
                        number_association_tp = number_association_tp + 1.0
                    if score[i] <= q and y_true[i] == 0.0:
                        number_association_fp = number_association_fp + 1.0
                #print 'number_association_tp', number_association_tp
                #print 'number_association_fp:', number_association_fp
                if condition_positive > 0.0:
                    recall[new_method].append((number_association_tp/condition_positive))
                if test_outcome_positive > 0.0:
                    fdr[new_method].append((number_association_fp/test_outcome_positive))
                else:
                    fdr[new_method].append(1.0) 
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
        halla.plot.plot_box(recall_data, figure_name = 'Figure2a', alpha = q, ylabel = 'Recall', labels = labels)
        halla.plot.plot_box(fdr_data,figure_name = 'Figure2b', alpha = q, ylabel = 'FDR', labels = labels)
        halla.plot.scatter_plot(mean_recall,mean_fdr, alpha = q, labels = labels)
        labels = []
        recall_data = []
        fdr_data = []
        #data.append(recall[new_method])
        #data.append(fdr[new_method]) 
    return;
if __name__ == '__main__':
    _main( )
    