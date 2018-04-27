""" This file is the main files for module to invoke other modules in order and log the results and performance """
from matplotlib.pyplot import ylabel
from numpy import dtype
from time import sleep

# Test if matplotlib is installed
try:
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("Please install matplotlib")

# Test if numpy is installed
try:
    from numpy import array
    import numpy as np
except ImportError:
    sys.exit("Please install numpy")

import csv
import itertools
import os
import sys
import shutil 
import time
import math
import datetime
#  Load a halla module to check the installation
try:
    from . import hierarchy
except ImportError:
    sys.exit("CRITICAL ERROR: Unable to find the hierarchy module." + 
        " Please check your halla install.")

# import HAllA's module sthat are needed here
from . import distance
from . import stats
from . import plot
from . import logger
from . import config
import random

def smart_decisoin():
    '''
    This function decides for similarity metric, permutation type
    if not explicitly decided by user.
    '''
    if config.similarity_method =='':
        if all ([val == 'CON' for val in config.data_type[0]]) and all ([val == 'CON' for val in config.data_type[1]]): 
            config.similarity_method = 'spearman'
        else:
            config.similarity_method = 'nmi' 
    if config.permutation_func == '':
        if config.similarity_method in ['spearman', 'pearson']:
            config.permutation_func = 'none'
        else:
            config.permutation_func = 'gpd'
    if config.permutation_func == 'ecdf':
        if  config.p_adjust_method ==  "by" and config.iterations < len(config.data_type[0])* len(config.data_type[1]) * math.log(len(config.data_type[0])* len(config.data_type[1])):
            print ('--- WARNING: HAllA recommends to use 10*number of feature in first data set been used * number of features been used in second data set\n \
                when using ECDF as the permutation method with Benjamini–Yekutieli procedure. In this run HAllA update your requested iterations to %s it will takes more time.')%\
                (int(len(config.data_type[0])* len(config.data_type[1]) * math.log(len(config.data_type[0])* len(config.data_type[1]))))
            config.iterations =int(len(config.data_type[0])* len(config.data_type[1]) * math.log(len(config.data_type[0])* len(config.data_type[1])))
        elif config.p_adjust_method ==  "bh" and config.iterations < len(config.data_type[0])* len(config.data_type[1] * 10):
            print ('--- WARNING: HAllA recommends to use 10*number of feature in first data set been used * number of features been used in second data set\n \
                when using ECDF as the permutation method with Benjamini–Hochberg procedure. In this run HAllA update your requested iterations to %s it will takes more time.')%\
                (int(len(config.data_type[0])* len(config.data_type[1]) * 10))
            config.iterations = int(len(config.data_type[0])* len(config.data_type[1]) * 10)
        sleep(10)
def bypass_discretizing():
    """
    This module decide if the discretizing should by bypassed or not based on 
    similarity metric and decomposition method
    """
    if config.strDiscretizing == "none" or\
        config.similarity_method in ["pearson"] or config.decomposition in ["pca", "ica"] or \
        not distance.c_hash_association_method_discretize[config.similarity_method]:
        return True
    else:
        return False
#==========================================================#
# Static Methods 
#==========================================================# 

def m(dataset, pFunc, strDiscretizing, axis=0,):
    """ 
    Maps pFunc over the array dataset 
    """

    if bool(axis): 
        dataset = dataset.T
        # Set the axis as per numpy convention 
    if isinstance(pFunc , np.ndarray):
        return dataset[pFunc]
    else:  # generic function type
        return array([pFunc(item, strDiscretizing) for item in dataset]) 


#==========================================================#
# Helper Functions 
#==========================================================# 
def name_features():
    #if not config.FeatureNames[0]:
    config.FeatureNames[0] = [str(i) for i in range(len(config.original_dataset[0])) ]
    #if not config.FeatureNames[1]:
    config.FeatureNames[1] = [str(i) for i in range(len(config.original_dataset[1])) ]
def set_parsed_data():
    if bypass_discretizing():
        config.parsed_dataset = config.original_dataset
    else:
        config.parsed_dataset = config.discretized_dataset

def _hclust():
    config.meta_data_tree = []
    tree1, config.Features_order[0] = hierarchy.hclust(config.parsed_dataset[0], labels=config.FeatureNames[0],  dataset_number = 0)
    config.meta_data_tree.append(tree1)
    tree2, config.Features_order[1]= hierarchy.hclust(config.parsed_dataset[1] , labels=config.FeatureNames[1], dataset_number = 1)
    config.meta_data_tree.append(tree2)
    # config.meta_data_tree = config.m( config.parsed_dataset, lambda x: hclust(x , bTree=True) )
    return config.meta_data_tree 

def _couple():
    config.meta_hypothesis_tree = None        
    config.meta_hypothesis_tree = hierarchy.couple_tree(apClusterNode0=[config.meta_data_tree[0]],
            apClusterNode1=[config.meta_data_tree[1]],
            dataset1=config.parsed_dataset[0], dataset2=config.parsed_dataset[1])[0]
    
    # # remember, `couple_tree` returns object wrapped in list 
    #return config.meta_hypothesis_tree 

def _naive_all_against_all(iIter=100):
    config.meta_alla = None
    config.meta_alla = hierarchy.naive_all_against_all()
    return config.meta_alla 
def _test_by_level():
    config.meta_alla = hierarchy.test_by_level(apClusterNode0=[config.meta_data_tree[0]],
            apClusterNode1=[config.meta_data_tree[1]],
            dataset1=config.parsed_dataset[0], dataset2=config.parsed_dataset[1])
def _test_by_family():
    config.meta_alla = hierarchy.test_by_family(apClusterNode0=[config.meta_data_tree[0]],
            apClusterNode1=[config.meta_data_tree[1]],
            dataset1=config.parsed_dataset[0], dataset2=config.parsed_dataset[1])


def _naive_summary_statistics():
    try:
        _, p_values = zip(*config.aOut[0])
    except:
        _, p_values = list(zip(*config.aOut[0]))
    config.meta_summary = []
    config.meta_summary.append(np.reshape([p_values], (int(math.sqrt(len(p_values))), int(math.sqrt(len(p_values))))))


def _summary_statistics(strMethod=None): 
    """
    provides summary statistics on the output given by _hypotheses_testing 
    """

    if not strMethod:
        strMethod = config.summary_method
    # print('meta array:')
    #print(config.original_dataset[0])
    #print(config.original_dataset[1])    
    X = config.original_dataset[0]
    Y = config.original_dataset[1]
    iX, iY = len(X), len(Y)
    S = -1 * np.ones((iX, iY , 2))  # # matrix of all associations; symmetric if using a symmetric measure of association  
    
    Z = config.meta_alla 
    _final, _all = Z#map(array, Z)  # # Z_final is the final bags that passed criteria; Z_all is all the associations delineated throughout computational tree
    Z_final = np.array([[_final[i].m_pData, _final[i].pvalue, _final[i].qvalue] for i in range(len(_final))])
    Z_all = np.array([[_all[i].m_pData, _all[i].pvalue, _all[i].qvalue] for i in range(len(_all))])    
    Z_final_dummy = [-1.0 * (len(line[0][0]) + len(line[0][1])) for line in Z_final]
    args_sorted = np.argsort(Z_final_dummy)
    Z_final = Z_final[args_sorted]
    if config.verbose == 'INFO':
        print (Z_final) 
    
    # assert( Z_all.any() ), "association bags empty." ## Technically, Z_final could be empty 
    def __set_outcome(Z_final):
        config.outcome = np.zeros((len(config.parsed_dataset[0]),len(config.parsed_dataset[1])), dtype = float)
        #config.outcome[:] = np.NAN
        for aLine in Z_final:
            if config.verbose == 'INFO':
                print (aLine) 
            aaBag, _, _ = aLine
            listBag1, listBag2 = aaBag 
            for i, j in itertools.product(listBag1, listBag2):
                config.outcome[i][j] = 1.0
    def __set_pvalues(Z_all):
        config.pvalues = np.empty((len(config.parsed_dataset[0]),len(config.parsed_dataset[1])), dtype = float)
        config.pvalues[:] = np.NAN
        for aLine in Z_all:
            if config.verbose == 'INFO':
                print (aLine) 
            aaBag, pvalue, _ = aLine
            listBag1, listBag2 = aaBag     
            for i, j in itertools.product(listBag1, listBag2):
                config.pvalues[i][j] = pvalue 
                        
    def __add_pval_product_wise(_x, _y, _fP, _fP_adjust):
        S[_x][_y][0] = _fP
        S[_x][_y][1] = _fP_adjust  

    def __get_conditional_pval_from_bags(_Z, _strMethod=None):
        """
        
        _strMethod: str 
            {"default",}

        The default option does the following: go through the bags, treating the p-value for each bag pair as applying to all the variables inside the bag. 
        If new instance arises (e.g. [[3],[5]] following [[3,5,6],[3,5,6]] ), override the p-value to the one with more precision. 
        """

        for aLine in _Z:
            if config.verbose == 'INFO':
                print (aLine) 
            
            aaBag, fAssoc, fP_adjust = aLine
            listBag1, listBag2 = aaBag 
            for i, j in itertools.product(listBag1, listBag2):
                S[i][j][0] = fAssoc 
                S[i][j][1] = fP_adjust

    def __get_pval_from_bags(_Z, _strMethod='final'):
        """
        
        _strMethod: str 
            {"default",}

        The default option does the following: go through the bags, treating the p-value for each bag pair as applying to all the variables inside the bag. 
        If new instance arises (e.g. [[3],[5]] following [[3,5,6],[3,5,6]] ), override the p-value to the one with more precision. 
        """

        for aLine in _Z:
            if config.verbose == 'INFO':
                print (aLine) 
            
            aaBag, fAssoc, P_adjust = aLine
            aBag1, aBag2 = aaBag 
            aBag1, aBag2 = array(aBag1), array(aBag2)
            config.bc(aBag1, aBag2, pFunc=lambda x, y: __add_pval_product_wise(_x=x, _y=y, _fP=fAssoc, _fP_adjust=P_adjust))
    __set_outcome(Z_final)
    __set_pvalues(Z_all)
    if strMethod == "final":
        if config.verbose == 'INFO':
            print ("Using only final p-values")
        __get_conditional_pval_from_bags(Z_final)

        #assert(S.any())
        config.meta_summary = S
        return config.meta_summary

    elif strMethod == "all":
        if config.verbose == 'INFO':
            print ("Using all p-values")
        __get_conditional_pval_from_bags(Z_all)
        #assert(S.any())
        config.meta_summary = S
        return config.meta_summary
global associated_feature_X_indecies
associated_feature_X_indecies = []
global associated_feature_Y_indecies
associated_feature_Y_indecies = []
def _report():
    """
    helper function for reporting the output to the user,
    """
    output_dir = config.output_dir
    aaOut = []

    # config.meta_report = [] 

    aP = config.meta_summary
    iRow1 = len(config.original_dataset[0])
    iRow2 = len(config.original_dataset[1])

    for i, j in itertools.product(list(range(iRow1)), list(range(iRow2))):
        # ## i <= j 
        fQ = aP[i][j][0] 
        fQ_adust = aP[i][j][1] 
        if fQ != -1:
            aaOut.append([[i, j], fQ, fQ_adust ])

    
    
    def _report_all_tests():
        output_file_all  = open(str(config.output_dir)+'/all_association_results_one_by_one.txt', 'w')
        csvw = csv.writer(output_file_all, csv.excel_tab, delimiter='\t')
        #csvw.writerow(["Decomposition method: ", config.decomposition  +"-"+ config.similarity_method , "q value: " + str(config.q), "metric " +config.similarity_method])
        csvw.writerow(["First Dataset", "Second Dataset", "pvalue", "qvalue"])

        for line in aaOut:
            iX, iY = line[0]
            fP = line[1]
            fP_adjust = line[2]
            aLineOut = list(map(str, [config.FeatureNames[0][iX], config.FeatureNames[1][iY], fP, fP_adjust]))
            csvw.writerow(aLineOut)

    def _report_associations():    
        number_of_association = 0
        number_of_association_faeture = 0
        output_file_associations  = open(str(config.output_dir)+'/associations.txt', 'w')
        bcsvw = csv.writer(output_file_associations, csv.excel_tab, delimiter='\t')
        #bcsvw.writerow(["Method: " + config.decomposition +"-"+ config.similarity_method , "q value: " + str(config.q), "metric " + config.similarity_method])
        bcsvw.writerow(["association_rank", "cluster1", "cluster1_similarity_score", \
                        "cluster2", \
                        "cluster2_similarity_score", \
                        "pvalue", "qvalue", "similarity_score_between_clusters"])

        #sorted_associations = sorted(config.meta_alla[0], key=lambda x: math.fabs(x.similarity_score), reverse=True)
        #sorted_associations = sorted(sorted_associations, key=lambda x: x.pvalue)
        sorted_associations = sorted(config.meta_alla[0], key=lambda x: (- math.fabs(x.similarity_score), x.pvalue, x.qvalue ))

        for association in sorted_associations:
            number_of_association += 1
            iX, iY = association.m_pData
            number_of_association_faeture += (len( iX) * len(iY))
            global associated_feature_X_indecies
            associated_feature_X_indecies += iX
            global associated_feature_Y_indecies
            associated_feature_Y_indecies += iY
            fP = association.pvalue
            fP_adjust = association.qvalue
            clusterX_similarity = 1.0 - association.left_distance
            #clusterX_first_rep = association.get_left_first_rep_variance()
            clusterY_similarity = 1.0 - association.right_distance
            #clusterY_first_rep = association.get_right_first_rep_variance()
            association_similarity = association.similarity_score
            
            aLineOut = [number_of_association,
                                 str(';'.join(config.FeatureNames[0][i] for i in iX)),
                                 clusterX_similarity,
                                 #clusterX_first_rep,
                                 str(';'.join(config.FeatureNames[1][i] for i in iY)),
                                 clusterY_similarity,
                                 #clusterY_first_rep,
                                 fP,
                                 fP_adjust,
                                 association_similarity]
            bcsvw.writerow(aLineOut)
        performance_file = open(str(config.output_dir)+'/performance.txt', 'a') 
        csvw = csv.writer(performance_file, csv.excel_tab, delimiter='\t')
        csvw.writerow(["Number of association cluster-by-cluster:", number_of_association])
        csvw.writerow(["Number of association feature-by-feature: ", number_of_association_faeture])
        csvw.writerow([])
        performance_file.close()
        
    #sorted_associations = sorted(config.meta_alla[0], key=lambda x: math.fabs(x.similarity_score), reverse=True)
    #sorted_associations = sorted(sorted_associations, key=lambda x: x.pvalue)
    sorted_associations = sorted(config.meta_alla[0], key=lambda x: (- math.fabs(x.similarity_score), x.pvalue, x.qvalue ))

    if config.descending == "AllA":
        config.Features_order[0]  = [i for i in range(len(config.original_dataset[0]))]   
        config.Features_order[1] = [i for i in range(len(config.original_dataset[1]))]         
    def _plot_associations():
        import pandas as pd    
        association_number = 0
        diagnostics_plot_dir = config.output_dir + '/diagnostics_plot'
        if os.path.isdir(diagnostics_plot_dir):
            try:
                shutil.rmtree(diagnostics_plot_dir)
            except EnvironmentError:
                sys.exit("Unable to remove directory: "+dir)
        try:
            os.mkdir(diagnostics_plot_dir)
        except EnvironmentError:
                sys.exit("Unable to create directory: "+dir)
        for association in sorted_associations:
            association_number += 1
            iX, iY = association.m_pData
            global associated_feature_X_indecies
            associated_feature_X_indecies += iX
            global associated_feature_Y_indecies
            associated_feature_Y_indecies += iY
            print ("--- plotting associations %s %s" %(association_number," ..."))
            cluster1 = [config.original_dataset[0][i] for i in iX]
            discretized_cluster1 = [config.discretized_dataset[0][i] for i in iX]
            X_labels = np.array([config.FeatureNames[0][i] for i in iX])
            
            cluster2 = [config.original_dataset[1][i] for i in iY]
            discretized_cluster2 = [config.discretized_dataset[1][i] for i in iY]
            Y_labels = np.array([config.FeatureNames[1][i] for i in iY])
            
            association_dir = str(diagnostics_plot_dir) + "/association_"+ str(association_number)
            filename = association_dir +"/"
            #discretized_filename = association_dir+"/discretized_data/"
            #dir = os.path.dirname(filename)
            #discretized_dir = os.path.dirname(discretized_filename)
            # remove the directory if it exists
            if os.path.isdir(association_dir):
                try:
                    shutil.rmtree(association_dir)
                    #shutil.rmtree(dir)
                    #shutil.rmtree(discretized_dir)
                except EnvironmentError:
                    sys.exit("Unable to remove directory: "+association_dir)
            
            # create new directory
            try:
                os.mkdir(association_dir)
                #os.mkdir(dir)
                #if not bypass_discretizing():
                    #os.mkdir(discretized_dir)
            except EnvironmentError:
                sys.exit("Unable to create directory: "+association_dir)
            plt.figure()  
            try: 
                if len(discretized_cluster1) < 40:
                    df1 = pd.DataFrame(np.array(cluster1, dtype= float).T ,columns=X_labels )
                    if config.similarity_method in ['spearman', 'pearson']:
                        df1_rank = df1.rank()
                        ax1 = plot.scatter_matrix(df1_rank, filename = filename + 'Dataset_1_cluster_' + str(association_number) + '_scatter_matrix_rank.pdf')

                    ax1 = plot.scatter_matrix(df1, filename = filename + 'Dataset_1_cluster_' + str(association_number) + '_scatter_matrix.pdf')
            except:
                pass
                #print ("Exception in first dataset")
            
            try:
                if len(discretized_cluster2) < 40:
                    df2 = pd.DataFrame(np.array(cluster2, dtype= float).T ,columns=Y_labels )
                    if config.similarity_method in ['spearman', 'pearson']:
                        df2_rank = df2.rank()
                        ax2 = plot.scatter_matrix(df2_rank, filename =filename + 'Dataset_2_cluster_' + str(association_number) + '_scatter_matrix_rank.pdf')

                    ax2 = plot.scatter_matrix(df2, filename =filename + 'Dataset_2_cluster_' + str(association_number) + '_scatter_matrix.pdf')
            except:
                pass
                #print ("Exception in second dataset")
            try:
                if len (iX) + len(iY) <40:
                    two_clusters = cluster1
                    two_clusters.extend(cluster2)
                    two_labels = [config.FeatureNames[0][i] for i in iX]
                    two_labels.extend([config.FeatureNames[1][i] for i in iY])
                    df_all = pd.DataFrame(np.array(two_clusters, dtype= float).T ,columns=np.array(two_labels))
                    if config.similarity_method in ['spearman', 'pearson']:
                        df_all_rank = df_all.rank()
                        axes = plot.scatter_matrix(df_all_rank, x_size = len(iX),filename =filename + 'Scatter_association' + str(association_number) + '_rank.pdf')
                    axes = plot.scatter_matrix(df_all, x_size = len(iX),filename =filename + 'Scatter_association' + str(association_number) + '.pdf')
            except:
                pass
                #print ("Exception in association")
            
            x_label_order = []
            fig = plt.figure(figsize=(5, 4))
            # Create an Axes object.
            ax = fig.add_subplot(1,1,1) # one row, one column, first plot
            plt.rc('xtick', labelsize=6) 
            plt.rc('ytick', labelsize=6) 
            decomposition_method = stats.c_hash_decomposition[config.decomposition]
            discretized_df1 = np.array(discretized_cluster1, dtype=float)
            discretized_df2 = np.array(discretized_cluster2, dtype=float)
            if not bypass_discretizing():
                d_x_d_rep = decomposition_method(discretized_df1)
                d_y_d_rep = decomposition_method(discretized_df2)
                d_x_d_rep, d_y_d_rep = list(zip(*sorted(zip(d_x_d_rep, d_y_d_rep))))
                plot.confusion_matrix(d_x_d_rep, d_y_d_rep, filename = filename + '/association_' + str(association_number) + '_confusion_matrix.pdf' )
            plt.close("all")
            
    def _report_compared_clusters():

        if config.descending == "AllA":
            output_file_compared_clusters  = open(str(config.output_dir)+'/hypotheses_tree.txt', 'w')
            csvwc = csv.writer(output_file_compared_clusters , csv.excel_tab, delimiter='\t')
            csvwc.writerow(['Level', "Dataset 1", "Dataset 2" ])
            aLineOut = list(map(str, ['0', str(';'.join(config.FeatureNames[0][i] for i in range(len(config.FeatureNames[0])))), str(';'.join(config.FeatureNames[1][i] for i in range(len(config.FeatureNames[1]))))]))
            csvwc.writerow(aLineOut)
        elif config.meta_hypothesis_tree:
            output_file_compared_clusters  = open(str(config.output_dir)+'/hypotheses_tree.txt', 'w')
            csvwc = csv.writer(output_file_compared_clusters , csv.excel_tab, delimiter='\t')
            csvwc.writerow(['Level', "Dataset 1", "Dataset 2" ])
            for line in hierarchy.reduce_tree_by_layer([config.meta_hypothesis_tree]):
                (level, clusters) = line
                iX, iY = clusters[0], clusters[1]
                fP = line[1]
                # fP_adjust = line[2]
                aLineOut = list(map(str, [str(level), str(';'.join(config.FeatureNames[0][i] for i in iX)), str(';'.join(config.FeatureNames[1][i] for i in iY))]))
                csvwc.writerow(aLineOut)
        #else:
            #aLineOut = map(str, ['0', str(';'.join(config.FeatureNames[0][i] for i in config.Features_order[0])), str(';'.join(config.FeatureNames[1][i] for i in config.Features_order[1]))])
            #csvwc.writerow(aLineOut)
            #pass
        #output_file_compared_clusters.close()

    def _heatmap_associations():
        print ("--- plotting heatmap of associations  ...")
        global associated_feature_X_indecies
        Xs = list(set(associated_feature_X_indecies))
        X_labels = np.array([config.FeatureNames[0][i] for i in Xs])
        global associated_feature_Y_indecies
        Ys = list(set(associated_feature_Y_indecies))
        Y_labels = np.array([config.FeatureNames[1][i] for i in Ys])
        if len(Xs) > 1 and len(Ys) > 1: 
            cluster1 = [config.parsed_dataset[0][i] for i in Xs]    
            cluster2 = [config.parsed_dataset[1][i] for i in Ys]
            df1 = np.array(cluster1, dtype=float)
            df2 = np.array(cluster2, dtype=float)
            p = np.zeros(shape=(len(Xs), len(Ys)))
            #nmi = np.zeros(shape=(len(Xs), len(Ys)))
            plot.heatmap2(dataset1=cluster1, dataset2=cluster2, xlabels =X_labels, ylabels = Y_labels, filename = str(config.output_dir)+'/all_nmi_heatmap' )
    def _write_hallagram_info():
        global associated_feature_X_indecies
        global associated_feature_Y_indecies
        if len(associated_feature_X_indecies) == 0 or len(associated_feature_Y_indecies) == 0 :
            return
        Xs = list(set(associated_feature_X_indecies)) 
        Ys = list(set(associated_feature_Y_indecies))
        
        config.Features_order[0] = [config.Features_order[0][i] for i in range (len(config.Features_order[0]))  if config.Features_order[0][i] in Xs ] 
        config.Features_order[1]= [config.Features_order[1][i] for i in range (len(config.Features_order[1]))  if config.Features_order[1][i] in Ys ] 
        
        X_labels = np.array([config.FeatureNames[0][i] for i in config.Features_order[0]])
        Y_labels = np.array([config.FeatureNames[1][i] for i in config.Features_order[1]])
        
        import re
        X_labels_circos = np.array([re.sub('[^a-zA-Z0-9  \n\.]', '_', config.FeatureNames[0][i]).replace(' ','_') for i in config.Features_order[0]])
        Y_labels_circos = np.array([re.sub('[^a-zA-Z0-9  \n\.]', '_', config.FeatureNames[1][i]).replace(' ','_') for i in config.Features_order[1]])
        
        similarity_score = np.zeros(shape=(len(config.Features_order[0]), len(config.Features_order[1])))  
        for i in range(len(config.Features_order[0])):
            for j in range(len(config.Features_order[1])):
                similarity_score[i][j] = distance.c_hash_metric[config.similarity_method](config.parsed_dataset[0][config.Features_order[0][i]], config.parsed_dataset[1][config.Features_order[1][j]])
        #sorted_associations = sorted(config.meta_alla[0], key=lambda x: math.fabs(x.similarity_score), reverse=True)
        #sorted_associations = sorted(sorted_associations, key=lambda x: x.pvalue)
        sorted_associations = sorted(config.meta_alla[0], key=lambda x: (- math.fabs(x.similarity_score), x.pvalue, x.qvalue ))
     
        def _is_in_an_assciostions(i,j):
            for n in range(len(sorted_associations)):
                iX, iY = sorted_associations[n].m_pData
                if i in iX and j in iY:
                    return n+1
            return 0

        circos_tabel = np.zeros(shape=(len(config.Features_order[0]), len(config.Features_order[1])))
        for i in range(len(config.Features_order[0])):
            for j in range(len(config.Features_order[1])):
                if _is_in_an_assciostions(config.Features_order[0][i],config.Features_order[1][j])>0: #for association in sorted_associations:
                    try:
                        circos_tabel[i][j] = math.fabs(int(similarity_score[i][j]*100))
                    except:
                        circos_tabel[i][j] = 0
        logger.write_circos_table(circos_tabel, str(config.output_dir)+"/" +"circos_table_"+ config.similarity_method+".txt", rowheader=X_labels_circos, colheader=Y_labels_circos, corner = "Data")         
        logger.write_table(similarity_score,str(config.output_dir)+"/" + "similarity_table.txt", rowheader=X_labels, colheader=Y_labels, corner = "#")
        return
    def _heatmap_associations_R():
        global associated_feature_X_indecies
        global associated_feature_Y_indecies
        if len(associated_feature_X_indecies) == 0 or len(associated_feature_Y_indecies) == 0 :
            return
        
        Xs = list(set(associated_feature_X_indecies)) 
        Ys = list(set(associated_feature_Y_indecies))
        
        config.Features_order[0] = [config.Features_order[0][i] for i in range (len(config.Features_order[0]))  if config.Features_order[0][i] in Xs ] 
        config.Features_order[1]= [config.Features_order[1][i] for i in range (len(config.Features_order[1]))  if config.Features_order[1][i] in Ys ] 
        
        X_labels = np.array([config.FeatureNames[0][i] for i in config.Features_order[0]])
        Y_labels = np.array([config.FeatureNames[1][i] for i in config.Features_order[1]])
        
        import re
        X_labels_circos = np.array([re.sub('[^a-zA-Z0-9  \n\.]', '_', config.FeatureNames[0][i]).replace(' ','_') for i in config.Features_order[0]])
        Y_labels_circos = np.array([re.sub('[^a-zA-Z0-9  \n\.]', '_', config.FeatureNames[1][i]).replace(' ','_') for i in config.Features_order[1]])
        
        similarity_score = np.zeros(shape=(len(config.Features_order[0]), len(config.Features_order[1])))  
        for i in range(len(config.Features_order[0])):
            for j in range(len(config.Features_order[1])):
                similarity_score[i][j] = distance.c_hash_metric[config.similarity_method](config.parsed_dataset[0][config.Features_order[0][i]], config.parsed_dataset[1][config.Features_order[1][j]])
        sorted_associations = sorted(config.meta_alla[0], key=lambda x: (- math.fabs(x.similarity_score), x.pvalue, x.qvalue ))
        
        #sorted_associations = sorted(sorted_associations, key=lambda x: ( x.s)
        for association in sorted_associations:
            iX, iY = association.m_pData
            for i, j in itertools.product(iX, iY):
                #similarity_score[i][j] = similarity_score[i][j]*2
                pass         
       
        def _is_in_an_assciostions(i,j):
            for num, association in enumerate(sorted_associations):
                iX, iY = association.m_pData
                if i in iX and j in iY:
                    return num+1
            return 0

        anottation_cell = np.zeros(shape=(len(config.Features_order[0]), len(config.Features_order[1])))                
        for i in range(len(config.Features_order[0])):
            for j in range(len(config.Features_order[1])):
                association_num = _is_in_an_assciostions(config.Features_order[0][i],config.Features_order[1][j])
                #print association_num
                if association_num > 0: #for association in sorted_associations:
                    anottation_cell[i][j] = association_num
                    
        circos_tabel = np.zeros(shape=(len(config.Features_order[0]), len(config.Features_order[1])))
        for i in range(len(config.Features_order[0])):
            for j in range(len(config.Features_order[1])):
                if _is_in_an_assciostions(config.Features_order[0][i],config.Features_order[1][j])>0: #for association in sorted_associations:
                    try:
                        circos_tabel[i][j] = math.fabs(int(similarity_score[i][j]*100))
                    except:
                        circos_tabel[i][j] = 0
        logger.write_circos_table(circos_tabel, str(config.output_dir)+"/" +"circos_table_"+ config.similarity_method+".txt", rowheader=X_labels_circos, aSampleNames=Y_labels_circos, corner = "Data")         
        logger.write_table(similarity_score,str(config.output_dir)+"/" + "similarity_table.txt", rowheader=X_labels, aSampleNames=Y_labels, corner = "#")
        logger.write_table(anottation_cell,str(config.output_dir)+"/" + "asscoaitaion_table.txt", rowheader=X_labels, aSampleNames=Y_labels, corner = "#")
        #anottation_cell = [config.Features_order[0]]
        #anottation_cell = [ anottation_cell[:][j] for j in config.Features_order[1]]
        #print anottation_cell
        print ("--- plotting heatmap associations using R ...")
        import rpy2.robjects as ro
        #import pandas.rpy.common as com
        import rpy2.robjects.numpy2ri
        rpy2.robjects.numpy2ri.activate()
        ro.globalenv['similarity_score'] = similarity_score
        ro.globalenv['labRow'] = X_labels 
        ro.globalenv['labCol'] = Y_labels
        ro.globalenv['sig_matrix'] = anottation_cell

        ro.r('rownames(similarity_score) = labRow')
        ro.r('colnames(similarity_score) = labCol')
        ro.r('rownames(sig_matrix) = labRow')
        ro.r('colnames(sig_matrix) = labCol')
        ro.globalenv['output_asscoaiation_table'] = str(config.output_dir)+"/" + config.similarity_method+"_asscoaitaion_table.txt"
        ro.globalenv['output_circus_table'] = str(config.output_dir)+"/" + config.similarity_method+"_circos_table.txt"

        if len(Xs) > 1 and len(Ys) > 1: 
            ro.r('library("RColorBrewer")')
            ro.r('library("pheatmap")')
            ro.globalenv['output_heatmap_similarity_score'] = str(config.output_dir)+"/" + "results_heatmap.pdf"
            #ro.globalenv['output_file_Pearson'] = str(config.output_dir)+"/Pearson_heatmap.pdf"
            if distance.c_hash_association_method_discretize[config.similarity_method]:
                ro.r('pheatmap(similarity_score, color = brewer.pal(100,"Reds"),labRow = labRow, labCol = labCol, filename =output_heatmap_similarity_score, cellwidth = 10, cellheight = 10, fontsize = 8, show_rownames = T, show_colnames = T, cluster_rows=FALSE, cluster_cols=FALSE, display_numbers = matrix(ifelse(sig_matrix > 0, sig_matrix, ""), nrow(sig_matrix)))')#,scale="row",  key=TRUE, symkey=FALSE, density.info="none", trace="none", cexRow=0.5
            else:
                ro.r('pheatmap(similarity_score,labRow = labRow, labCol = labCol, filename =output_heatmap_similarity_score, cellwidth = 10, cellheight = 10, fontsize = 8, show_rownames = T, show_colnames = T, cluster_rows=FALSE, cluster_cols=FALSE, display_numbers = matrix(ifelse(sig_matrix > 0, sig_matrix, ""), nrow(sig_matrix)))')#,scale="row",  key=TRUE, symkey=FALSE, density.info="none", trace="none", cexRow=0.5
            ro.r('dev.off()')
            
    def _heatmap_datasets_R():
        if config.hallagram:          
            print ("--- plotting heatmap datasets using R ...")
            X_indecies = len(config.FeatureNames[0])
            X_labels = np.array([config.FeatureNames[0][i] for i in range(X_indecies)])
            Y_indecies = len(config.FeatureNames[1])
            Y_labels = np.array([config.FeatureNames[1][i] for i in range(Y_indecies)])
            if len(X_labels) > 1 and len(Y_labels) > 1: 
                df1 = np.array(config.parsed_dataset[0], dtype=float)
                df2 = np.array(config.parsed_dataset[1], dtype=float)
                drows1 = np.zeros(shape=(X_indecies, X_indecies))
                drows2 = np.zeros(shape=(Y_indecies, Y_indecies))
                if config.Distance[0] == None:
                    for i in range(X_indecies):
                        for j in range(X_indecies):
                            drows1[i][j] = distance.c_hash_metric[config.similarity_method](df1[i], df1[j]) 
                    
                    for i in range(Y_indecies):
                        for j in range(Y_indecies):
                            drows2[i][j] = distance.c_hash_metric[config.similarity_method](df2[i], df2[j])    
                else:
                   drows1 =  config.Distance[0]
                   drows2 =  config.Distance[1]
                   
                    
                import rpy2.robjects as ro
                #import pandas.rpy.common as com
                import rpy2.robjects.numpy2ri
                rpy2.robjects.numpy2ri.activate()
                ro.r('library("pheatmap")')
                ro.globalenv['drows1'] = drows1
                ro.globalenv['labRow'] = X_labels 
                ro.globalenv['D1'] = str(config.output_dir)+"/D1_heatmap.pdf"
                ro.r('rownames(drows1) = labRow')
                ro.r('pheatmap(drows1, filename =D1, cellwidth = 10, cellheight = 10, fontsize = 10, show_rownames = T,  show_colnames = F, cluster_cols=T, dendrogram="row")')#,scale="row",  key=TRUE, symkey=FALSE, density.info="none", trace="none", cexRow=0.5
                ro.r('dev.off()')
                ro.globalenv['drows2'] = drows2
                ro.globalenv['labRow'] = Y_labels
                ro.globalenv['D2'] = str(config.output_dir)+"/D2_heatmap.pdf"
                ro.r('rownames(drows2) = labRow')
                ro.r('pheatmap(drows2, filename =D2, cellwidth = 10, cellheight = 10, fontsize = 10, show_rownames = T,  show_colnames = F, cluster_cols=T, dendrogram="row")')#,scale="row",  key=TRUE, symkey=FALSE, density.info="none", trace="none", cexRow=0.5
                ro.r('dev.off()')
    def _hallagram_strongest(n):
        if config.similarity_method=="nmi":
            sim_color = ' --similarity=\"NMI\" --cmap=YlGnBu'
        else:
            sim_color =''
        #_heatmap_associations()
        #from rpy2.rinterface import RRuntimeError
        output_path = config.output_dir# str(config.output_dir).replace("(","\(").replace(")","\)").replace(" ","\ ")
        print ("--- Writing plotting outputs to  %s " % output_path )
        if os.path.isfile(output_path+"/similarity_table.txt"):
            try:                 
                hallagram_command= "hallagram "+ output_path+"/similarity_table.txt "+\
                          output_path+"/hypotheses_tree.txt "+\
                          output_path+"/associations.txt "+ "--similarity "+ config.similarity_method.title()+\
                          " --outfile="+output_path+"/hallagram_strongest_"+str(n)+".pdf" + " --strongest " + str(n) + sim_color
                os.system(hallagram_command)
                
                hallagram_command_png= "hallagram "+ output_path+"/similarity_table.txt "+\
                          output_path+"/hypotheses_tree.txt "+\
                          output_path+"/associations.txt "+ "--similarity "+ config.similarity_method.title()+\
                          " --outfile="+output_path+"/hallagram_strongest_"+str(n)+".png" + " --strongest " + str(n) + sim_color
                os.system(hallagram_command_png)
                
                hallagram_command_mask = "hallagram "+ output_path+"/similarity_table.txt "+\
                          output_path+"/hypotheses_tree.txt "+\
                          output_path+"/associations.txt " + "--similarity "+ config.similarity_method.title()+\
                          " --outfile="+output_path+"/hallagram_strongest_"+str(n)+"_mask.pdf --mask" + " --strongest " + str(n) + sim_color
                os.system(hallagram_command_mask)
                
                hallagram_command_mask_png = "hallagram "+ output_path+"/similarity_table.txt "+\
                          output_path+"/hypotheses_tree.txt "+\
                          output_path+"/associations.txt " + "--similarity "+ config.similarity_method.title()+\
                          " --outfile="+output_path+"/hallagram_strongest_"+str(n)+"_mask.png --mask" + " --strongest " + str(n) + sim_color
                os.system(hallagram_command_mask_png)
                #_heatmap_associations_R()
                #_heatmap_datasets_R()
                #_plot_associations()
            except IOError:
                print("exception with plotting the final results ")       
    # Execute report functions
    _report_all_tests()
    _report_associations()
    _report_compared_clusters()
    _write_hallagram_info()
    if len(config.meta_alla[0]) > 1:
        n = min(len(config.meta_alla[0]), 100)
        _hallagram_strongest(n)
        
    if config.diagnostics_plot:
        #_heatmap_associations()
        #from rpy2.rinterface import RRuntimeError
        try:
            _plot_associations()
        except IOError:
            print ("exception with plotting in asscociations ")
    return config.meta_report 
def write_config():
    try:    
        performance_file  = open(str(config.output_dir)+'/performance.txt', 'a')
    except IOError:
        sys.exit("IO Exception: "+config.output_dir+"/performance.txt") 
    csvw = csv.writer(performance_file, csv.excel_tab, delimiter='\t')
    csvw.writerow(["HAllA version:", config.version])
    csvw.writerow(["Decomposition method: ", config.decomposition])
    csvw.writerow(["Similarity method: ", config.similarity_method]) 
    csvw.writerow(["Hierarchical linkage method: ", config.linkage_method]) 
    csvw.writerow(["q: FDR cut-off : ", config.q]) 
    csvw.writerow(["FDR adjusting method : ", config.p_adjust_method]) 
    csvw.writerow(["FDR style using : ", config.fdr_style])
    #csvw.writerow(["r: effect size for robustness : ", config.robustness]) 
    csvw.writerow(["Applied stop condition : ", config.apply_stop_condition]) 
    csvw.writerow(["Discretizing method : ", config.strDiscretizing])
    csvw.writerow(["Permutation function: ", config.permutation_func])
    csvw.writerow(["Seed number: ", config.seed]) 
    csvw.writerow(["Number of permutations iterations for estimating pvalues: ", config.iterations]) 
    csvw.writerow(["Minimum entropy for filtering threshold in the first dataset : ", config.entropy_threshold1])
    csvw.writerow(["Minimum entropy for filtering threshold in the second dataset: ", config.entropy_threshold2])
    csvw.writerow([]) 
    performance_file.close() 
def run():
    
    """
    Main run module 

    Returns 
    -----------

        Z : HAllA output object 
    
    * Main steps

        + Parse input and clean data 
        + Feature selection (discretization for MI, beta warping, copula selection)
        + Hierarchical clustering 
        + Hypothesis generation (tree coupling via appropriate step function)
        + Hypothesis testing and agglomeration of test statistics, with multiple hypothesis correction 
        + Parse output 

    * Visually, looks much nicer and is much nicely wrapped if functions are entirely self-contained and we do not have to pass around pointers 

    """ 
    if config.seed != -1:
        random.seed(config.seed)
        np.random.seed(config.seed)
    set_parsed_data()  
    try:    
        performance_file  = open(str(config.output_dir)+'/performance.txt', 'a')
    except IOError:
        sys.exit("IO Exception: "+config.output_dir+"/performance.txt") 
    csvw = csv.writer(performance_file, csv.excel_tab, delimiter='\t')
    write_config()
    #name_features()
    if not is_correct_submethods_combination():
        sys.exit("Please ckeck the combination of your options!!!!")
    execution_time = time.time()
    
    #plot.heatmap2(dataset1=config.parsed_dataset[0], dataset2=config.parsed_dataset[1], xlabels =config.FeatureNames[0], ylabels = config.FeatureNames[1], filename = str(config.output_dir)+'/heatmap2_all' )
    if config.log_input:
        logger.write_table(data=config.parsed_dataset[0], name=config.output_dir+"/X_dataset.txt", rowheader=config.FeatureNames[0] , colheader=config.SampleNames[0], prefix = "label",  corner = '#', delimiter= '\t')
        logger.write_table(data=config.parsed_dataset[1], name=config.output_dir+"/Y_dataset.txt", rowheader=config.FeatureNames[1] , colheader=config.SampleNames[1], prefix = "label",  corner = '#', delimiter= '\t')
    if config.descending == "AllA":
        print("--- association hypotheses testing is started, this task may take longer ...")
        start_time = time.time()
        _naive_all_against_all()
        excution_time_temp = time.time() - start_time
        csvw.writerow(["Hypotheses testing time", str(datetime.timedelta(seconds=excution_time_temp)) ])
        print("--- %s h:m:s hypotheses testing time ---" % str(datetime.timedelta(seconds=excution_time_temp)))
    elif config.descending == "HAllA":
        # hierarchical clustering 
        start_time = time.time()
        _hclust()
        excution_time_temp = time.time() - start_time
        csvw.writerow(["Hierarchical clustering time", str(datetime.timedelta(seconds=excution_time_temp)) ])
        print("--- %s h:m:s hierarchical clustering time ---" % str(datetime.timedelta(seconds=excution_time_temp)))
        
        # coupling clusters hierarchically 
        start_time = time.time()
        #_couple()
        if config.fdr_style == 'level':
            _test_by_level()
        elif config.fdr_style == 'family':
            _test_by_family()
        excution_time_temp = time.time() - start_time
        csvw.writerow(["Level-by-level hypothesis testing", str(datetime.timedelta(seconds=excution_time_temp)) ])
        print("--- %s h:m:s level-by-level hypothesis testing ---" % str(datetime.timedelta(seconds=excution_time_temp)))
        # hypotheses testing
        
        #start_time = time.time()
        #_hypotheses_testing()
        #excution_time_temp = time.time() - start_time
        csvw.writerow(["number of performed permutation tests: ", config.number_of_performed_tests])
        #csvw.writerow(["Hypotheses testing time", str(datetime.timedelta(seconds=excution_time_temp)) ])
        #print("--- %s h:m:s hypotheses testing time ---" % str(datetime.timedelta(seconds=excution_time_temp)))
    
    # Generate a report
    start_time = time.time() 
    _summary_statistics('final') 
    excution_time_temp = time.time() - start_time
    csvw.writerow(["Summary statistics time", excution_time_temp ])
    print("--- %s h:m:s summary statistics time ---" % str(datetime.timedelta(seconds=excution_time_temp)))
    
    start_time = time.time() 
    if config.report_results:
        _report()
    excution_time_temp = time.time() - start_time
    csvw.writerow(["Plotting results time", str(datetime.timedelta(seconds=excution_time_temp)) ])
    print("--- %s h:m:s plotting results time ---" % str(datetime.timedelta(seconds=excution_time_temp)))
    excution_time_temp = time.time() - execution_time
    csvw.writerow(["Total execution time", str(datetime.timedelta(seconds=excution_time_temp))])
    print("--- in %s h:m:s the task is successfully done ---" % str(datetime.timedelta(seconds=excution_time_temp)) )
    performance_file.close()

def view_singleton(pBags):
    aOut = [] 
    for aIndices, fP in pBags:
        if len(aIndices[0]) == 1 and len(aIndices[1]) == 1:
            aOut.append([aIndices, fP])
    return aOut 

def is_correct_submethods_combination():
    if config.descending == "AllA" and config.decomposition in ['medoid']:
        config.decomposition = 'none'        
    if (config.descending == "AllA" and not config.decomposition in ['none', "pls","cca"]) or\
                        (config.descending == "HAllA" and config.decomposition =='none') or\
                        (config.decomposition in ["ica","pca",'pls', 'cca', 'kpca'] and\
                        config.similarity_method not in ["pearson", "spearman","mic","dcor"] ) or\
                        (config.descending == "HAllA" and config.decomposition in  ['pls', 'cca']):
            False
    else:
        return True
