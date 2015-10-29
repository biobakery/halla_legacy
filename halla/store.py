""" Class to store HAllA data """
from matplotlib.pyplot import ylabel

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

from . import distance
from . import stats
from . import plot
from . import logger
from . import config
import random


def _bypass_discretizing():
    if not distance.c_hash_association_method_discretize[config.distance] or  config.strDiscretizing == "none" or\
        config.distance in ["pearson"] or config.decomposition in ["pca", "ica"] :
        return True
    else:
        return False
#==========================================================#
# Static Methods 
#==========================================================# 

def m(pArray, pFunc, strDiscretizing, axis=0,):
    """ 
    Maps pFunc over the array pArray 
    """

    if bool(axis): 
        pArray = pArray.T
        # Set the axis as per numpy convention 
    if isinstance(pFunc , np.ndarray):
        return pArray[pFunc]
    else:  # generic function type
        # print pArray.shape
        return array([pFunc(item, strDiscretizing) for item in pArray]) 

def bp(pArray, pFunc, axis=0):
    """
    Map _by pairs_ ; i.e. apply pFunc over all possible pairs in pArray 
    """

    if bool(axis): 
        pArray = pArray.T

    pIndices = itertools.combinations(range(pArray.shape[0]), 2)

    return array([pFunc(pArray[i], pArray[j]) for i, j in pIndices])

 
def bc(pArray1, pArray2, pFunc, axis=0):
    """
    Map _by cross product_ for ; i.e. apply pFunc over all possible pairs in pArray1 X pArray2 
    """

    if bool(axis): 
        pArray1, pArray2 = pArray1.T, pArray2.T

    pIndices = itertools.product(range(pArray1.shape[0]), range(pArray2.shape[0]))

    return array([pFunc(pArray1[i], pArray2[j]) for i, j in pIndices])

@staticmethod 
def r(pArray, pFunc, axis=0):
    """
    Reduce over array 
    pFunc is X x Y -> R 

    """
    if bool(axis):
        pArray = pArray.T

    return reduce(pFunc, pArray)

#==========================================================#
# Helper Functions 
#==========================================================# 
def name_features():
    #if not config.aOutName1:
    config.aOutName1 = [str(i) for i in range(len(config.meta_array[0])) ]
    #if not config.aOutName2:
    config.aOutName2 = [str(i) for i in range(len(config.meta_array[1])) ]
def _discretize():
    config.meta_feature = m(config.meta_array, stats.discretize, config.strDiscretizing)
    #config.meta_feature = array([stats.discretize(config.meta_array[0], config.strDiscretizing), stats.discretize(config.meta_array[1], config.strDiscretizing)])
    return config.meta_feature

def _featurize(strMethod="_discretize"):
    pMethod = None 
    try:
        pMethod = _discretize#getattr(strMethod)
    except AttributeError:
        raise Exception("Invalid Method.")

    if pMethod:
        return pMethod()

def _hclust():
    # print config.meta_feature
    config.meta_data_tree = [] 
    tree1, config.X_features_cluster_order = hierarchy.hclust(config.meta_feature[0], labels=config.aOutName1)
    config.meta_data_tree.append(tree1)
    #print config.X_features_cluster_order
    tree2, config.Y_features_order= hierarchy.hclust(config.meta_feature[1] , labels=config.aOutName2)
    config.meta_data_tree.append(tree2)
    # config.meta_data_tree = config.m( config.meta_feature, lambda x: hclust(x , bTree=True) )
    #print config.meta_data_tree
    return config.meta_data_tree 

def _couple():
    config.meta_hypothesis_tree = None        
    config.meta_hypothesis_tree = hierarchy.couple_tree(apClusterNode1=[config.meta_data_tree[0]],
            apClusterNode2=[config.meta_data_tree[1]],
            pArray1=config.meta_feature[0], pArray2=config.meta_feature[1])[0]
    
    # # remember, `couple_tree` returns object wrapped in list 
    #return config.meta_hypothesis_tree 

def _naive_all_against_all(iIter=100):
    config.meta_alla = None
    config.meta_alla = hierarchy.naive_all_against_all()
    return config.meta_alla 
def _hypotheses_testing():
    config.meta_alla = None    
    fQ = config.q
        
    if config.verbose:
        print ("HAllA PROMPT: q value", fQ)
        print ("q value is", fQ) 
    
    config.meta_alla = hierarchy.hypotheses_testing() 
    # # Choose to keep to 2 arrays for now -- change later to generalize 
    #return config.meta_alla 

def _naive_summary_statistics():
    _, p_values = zip(*config.aOut[0])
    config.meta_summary = []
    config.meta_summary.append(np.reshape([p_values], (int(math.sqrt(len(p_values))), int(math.sqrt(len(p_values))))))


def _summary_statistics(strMethod=None): 
    """
    provides summary statistics on the output given by _hypotheses_testing 
    """

    if not strMethod:
        strMethod = config.summary_method
    # print('meta array:')
    #print(config.meta_array[0])
    #print(config.meta_array[1])    
    X = config.meta_array[0]
    Y = config.meta_array[1]
    iX, iY = len(X), len(Y)
    S = -1 * np.ones((iX, iY , 2))  # # matrix of all associations; symmetric if using a symmetric measure of association  
    
    Z = config.meta_alla 
    _final, _all = map(array, Z)  # # Z_final is the final bags that passed criteria; Z_all is all the associations delineated throughout computational tree
    Z_final = array([[_final[i].get_data(), _final[i].get_pvalue(), _final[i].get_qvalue()] for i in range(len(_final))])
    Z_all = array([[_all[i].get_data(), _all[i].get_pvalue(), _all[i].get_qvalue()] for i in range(len(_all))])    
        
    # ## Sort the final Z to make sure p-value consolidation happens correctly 
    Z_final_dummy = [-1.0 * (len(line[0][0]) + len(line[0][1])) for line in Z_final]
    args_sorted = np.argsort(Z_final_dummy)
    Z_final = Z_final[args_sorted]
    if config.verbose:
        print (Z_final) 
    # assert( Z_all.any() ), "association bags empty." ## Technically, Z_final could be empty 
    def __set_outcome(Z_final):
        config.outcome = np.zeros((len(config.meta_feature[0]),len(config.meta_feature[1])))
        
        for aLine in Z_final:
            if config.verbose:
                print (aLine) 
            
            aaBag, _, _ = aLine
            listBag1, listBag2 = aaBag     
            for i, j in itertools.product(listBag1, listBag2):
                config.outcome[i][j] = 1.0 
    def __set_pvalues(Z_all):
        config.pvalues = np.zeros((len(config.meta_feature[0]),len(config.meta_feature[1])))
        
        for aLine in Z_all:
            if config.verbose:
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
            if config.verbose:
                print (aLine) 
            
            aaBag, fAssoc, fP_adjust = aLine
            listBag1, listBag2 = aaBag 
            #aBag1, aBag2 = array(listBag1), array(listBag2)
            #print aBag1 , aBag2
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
            if config.verbose:
                print (aLine) 
            
            aaBag, fAssoc, P_adjust = aLine
            aBag1, aBag2 = aaBag 
            aBag1, aBag2 = array(aBag1), array(aBag2)
            config.bc(aBag1, aBag2, pFunc=lambda x, y: __add_pval_product_wise(_x=x, _y=y, _fP=fAssoc, _fP_adjust=P_adjust))
    __set_outcome(Z_final)
    __set_pvalues(Z_all)
    if strMethod == "final":
        if config.verbose:
            print ("Using only final p-values")
        __get_conditional_pval_from_bags(Z_final)
        assert(S.any())
        config.meta_summary = S
        return config.meta_summary

    elif strMethod == "all":
        if config.verbose:
            print ("Using all p-values")
        __get_conditional_pval_from_bags(Z_all)
        assert(S.any())
        config.meta_summary = S
        return config.meta_summary

def _report():
    """
    helper function for reporting the output to the user,
    """
    output_dir = config.output_dir
    aaOut = []

    # config.meta_report = [] 

    aP = config.meta_summary
    iRow1 = len(config.meta_array[0])
    iRow2 = len(config.meta_array[1])

    for i, j in itertools.product(range(iRow1), range(iRow2)):
        # ## i <= j 
        fQ = aP[i][j][0] 
        fQ_adust = aP[i][j][1] 
        if fQ != -1:
            aaOut.append([[i, j], fQ, fQ_adust ])

    config.meta_report = aaOut
    # print "meta summary:", config.meta_report
    global associated_feature_X_indecies
    associated_feature_X_indecies = []
    global associated_feature_Y_indecies
    associated_feature_Y_indecies = []
    
    def _report_all_tests():
        output_file_all  = open(str(config.output_dir)+'/all_association_results_one_by_one.txt', 'w')
        csvw = csv.writer(output_file_all, csv.excel_tab, delimiter='\t')
        #csvw.writerow(["Decomposition method: ", config.decomposition  +"-"+ config.distance , "q value: " + str(config.q), "metric " +config.distance])
        csvw.writerow(["First Dataset", "Second Dataset", "nominal-pvalue", "adjusted-pvalue"])

        for line in aaOut:
            iX, iY = line[0]
            fP = line[1]
            fP_adjust = line[2]
            #print line, config.aOutName1, config.aOutName2
            aLineOut = map(str, [config.aOutName1[iX], config.aOutName2[iY], fP, fP_adjust])
            csvw.writerow(aLineOut)

    def _report_associations():    
        number_of_association = 0
        number_of_association_faeture = 0
        output_file_associations  = open(str(config.output_dir)+'/associations.txt', 'w')
        bcsvw = csv.writer(output_file_associations, csv.excel_tab, delimiter='\t')
        #bcsvw.writerow(["Method: " + config.decomposition +"-"+ config.distance , "q value: " + str(config.q), "metric " + config.distance])
        bcsvw.writerow(["Association Number", "Clusters First Dataset", "Cluster Similarity Score", "Explained Variance by the first representative of the cluster", "Clusters Second Dataset", "Cluster Similarity Score", "Explained Variance by the first representative of the cluster", "nominal-pvalue", "adjusted-pvalue", "Similarity score between Clusters"])

        sorted_associations = sorted(config.meta_alla[0], key=lambda x: x.pvalue)
        for association in sorted_associations:
            number_of_association += 1
            iX, iY = association.get_data()
            number_of_association_faeture += (len( iX) * len(iY))
            global associated_feature_X_indecies
            associated_feature_X_indecies += iX
            global associated_feature_Y_indecies
            associated_feature_Y_indecies += iY
            fP = association.get_pvalue()
            fP_adjust = association.get_qvalue()
            clusterX_similarity = 1.0 - association.get_left_distance()
            clusterX_first_rep = association.get_left_first_rep_variance()
            clusterY_similarity = 1.0 - association.get_right_distance()
            clusterY_first_rep = association.get_right_first_rep_variance()
            association_similarity = association.get_similarity_score()
            
            aLineOut = [number_of_association,
                                 str(';'.join(config.aOutName1[i] for i in iX)),
                                 clusterX_similarity,
                                 clusterX_first_rep,
                                 str(';'.join(config.aOutName2[i] for i in iY)),
                                 clusterY_similarity,
                                 clusterY_first_rep,
                                 fP,
                                 fP_adjust,
                                 association_similarity]
            bcsvw.writerow(aLineOut)
        performance_file = open(str(config.output_dir)+'/performance.txt', 'w') 
        csvw = csv.writer(performance_file, csv.excel_tab, delimiter='\t')
        csvw.writerow(["Number of association cluster-by-cluster:", number_of_association])
        csvw.writerow(["Number of association feature-by-feature: ", number_of_association_faeture])
        csvw.writerow([])
        performance_file.close()
        
    sorted_associations = sorted(config.meta_alla[0], key=lambda x: x.pvalue)
    if config.descending == "AllA":
        config.X_features_cluster_order  = [i for i in range(len(config.meta_array[0]))]   
        config.Y_features_order = [i for i in range(len(config.meta_array[1]))]         
    def _plot_associations():
        import pandas as pd    
        association_number = 0
        #output_file_associations  = open(str(config.output_dir)+'/associations.txt', 'w')
        #bcsvw = csv.writer(output_file_associations, csv.excel_tab)
        #bcsvw.writerow(["Method: " + config.decomposition +"-"+ config.distance , "q value: " + str(config.q), "metric " + config.distance])
        #bcsvw.writerow(["Association Number", "Clusters First Dataset", "Cluster Similarity Score", "Explained Variance by the First PC of the cluster"," ", "Clusters Second Dataset", "Cluster Similarity Score (NMI)", "Explained Variance by the First PC of the cluster"," ", "nominal-pvalue", "adjusted-pvalue", "Similarity score between Clusters"])

        
        for association in sorted_associations:
            association_number += 1
            iX, iY = association.get_data()
            global associated_feature_X_indecies
            associated_feature_X_indecies += iX
            global associated_feature_Y_indecies
            associated_feature_Y_indecies += iY
            #fP = association.get_pvalue()
            #fP_adjust = association.get_qvalue()
            #clusterX_similarity = 1.0 - association.get_left_distance()
            #clusterX_first_rep = association.get_left_first_rep()
            #clusterY_similarity = 1.0 - association.get_right_distance()
           # clusterY_first_rep = association.get_right_first_rep()
          #  association_similarity = association.get_similarity_score()
            
            ''' 
            aLineOut = map(str, [association_number,
                                 str(';'.join(config.aOutName1[i] for i in iX)),
                                 clusterX_similarity,
                                 clusterX_first_rep,
                                 " ", 
                                 str(';'.join(config.aOutName2[i] for i in iY)),
                                 clusterY_similarity,
                                 clusterY_first_rep,
                                 " ",
                                 fP,
                                 fP_adjust,
                                 association_similarity])
            bcsvw.writerow(aLineOut)
           '''
            if config.plotting_results:
                print "--- plotting associations ",association_number," ..."
                cluster1 = [config.meta_array[0][i] for i in iX]
                discretized_cluster1 = [config.meta_feature[0][i] for i in iX]
                X_labels = np.array([config.aOutName1[i] for i in iX])
                association_dir = config.output_dir + "/association_"+ str(association_number) + '/'
                filename = association_dir +"original_data" + '/'
                discretized_filename = association_dir+"discretized_data" + '/'
                dir = os.path.dirname(filename)
                discretized_dir = os.path.dirname(discretized_filename)
                # remove the directory if it exists
                if os.path.isdir(dir):
                    try:
                        shutil.rmtree(association_dir)
                        #shutil.rmtree(dir)
                        #shutil.rmtree(discretized_dir)
                    except EnvironmentError:
                        sys.exit("Unable to remove directory: "+dir)
                
                # create new directory
                try:
                    os.mkdir(association_dir)
                    os.mkdir(dir)
                    os.mkdir(discretized_dir)
                except EnvironmentError:
                    sys.exit("Unable to create directory: "+dir)
                plt.figure()  
                try: 
                    if len(discretized_cluster1) < 40:
                        #print len(discretized_cluster1)
                        df1 = pd.DataFrame(np.array(cluster1, dtype= float).T ,columns=X_labels )
                        axes = plot.scatter_matrix(df1, filename + 'Dataset_1_cluster_' + str(association_number) + '_scatter_matrix.pdf')
                except:
                    pass
                if len(discretized_cluster1) < 40:
                    discretized_df = pd.DataFrame(np.array(discretized_cluster1, dtype= float).T ,columns=X_labels )
                    
                    discretized_axes = plot.scatter_matrix(discretized_df, filename = discretized_filename + 'Dataset_1_cluster_' + str(association_number) + '_scatter_matrix.pdf')
                             
                cluster2 = [config.meta_array[1][i] for i in iY]
                discretized_cluster2 = [config.meta_feature[1][i] for i in iY]
                Y_labels = np.array([config.aOutName2[i] for i in iY])
                try:
                    if len(discretized_cluster2) < 40:
                        df2 = pd.DataFrame(np.array(cluster2, dtype= float).T ,columns=Y_labels )
                        axes = plot.scatter_matrix(df2, filename =filename + 'Dataset_2_cluster_' + str(association_number) + '_scatter_matrix.pdf')
                except:
                    pass
                if len(discretized_cluster2) < 40:
                    discretized_df = pd.DataFrame(np.array(discretized_cluster2, dtype= float).T ,columns=Y_labels )
                    discretized_axes = plot.scatter_matrix(discretized_df, filename = discretized_filename + 'Dataset_2_cluster_' + str(association_number) + '_scatter_matrix.pdf')
                #concatenated_df = pd.concat([df1, df2],axis=1)
                #axes = plot.scatter_matrix(concatenated_df, filename =filename + 'Concatenated_' + str(association_number) + '_scatter_matrix.pdf')
                # heatmap cluster in an association
                x_label_order = []
                if len(discretized_cluster1) >= len(discretized_cluster2):
                    #print len(discretized_cluster1), ">=", len(discretized_cluster2)
                    #print "Before1: ",len(x_label_order)     
                    plot.heatmap(np.array(discretized_cluster1), x_label_order, xlabels =X_labels, filename =discretized_filename + 'Dataset_1_cluster_' + str(association_number) + '_heatmap', )
                    #x_label_order =  x_label_order1
                    #print "after1", x_label_order
                    #print "After1: ",len(x_label_order)
                    plot.heatmap(np.array(discretized_cluster2), x_label_order, xlabels =Y_labels, filename =discretized_filename + 'Dataset_2_cluster_' + str(association_number) + '_heatmap' )
                else:
                    #print len(discretized_cluster2), ">=", len(discretized_cluster1)
                    #print "Before2: ",len(x_label_order) 
                    plot.heatmap(np.array(discretized_cluster2), x_label_order, xlabels =Y_labels, filename =discretized_filename + 'Dataset_2_cluster_' + str(association_number) + '_heatmap'  ) 
                    #x_label_order =  x_label_order
                    #print "after2", x_label_order 
                    #print "After2: ",len(x_label_order)
                    plot.heatmap(np.array(discretized_cluster1), xlabels_order = x_label_order, xlabels =X_labels, filename =discretized_filename + 'Dataset_1_cluster_' + str(association_number) + '_heatmap' )
                x_label_order = []
                #plot.heatmap2(pArray1=discretized_cluster1, pArray2=discretized_cluster2, xlabels =X_labels, ylabels = Y_labels, filename =discretized_filename + 'AllA_association_' + str(association_number) + '_heatmap' )
                #Heatmap on continue datasets
                #plot.heatmap2(pArray1=cluster1, pArray2=cluster2, xlabels =X_labels, ylabels = Y_labels, filename =filename + 'AllA_association_' + str(association_number) + '_heatmap' )  
                #plt.figure()
                fig = plt.figure(figsize=(5, 4))
                # Create an Axes object.
                ax = fig.add_subplot(1,1,1) # one row, one column, first plot
                plt.rc('xtick', labelsize=6) 
                plt.rc('ytick', labelsize=6) 
                #df1 = np.array(cluster1, dtype=float)
                #df2 = np.array(cluster2, dtype=float)
                
                decomposition_method = stats.c_hash_decomposition[config.decomposition]
                #x_rep = decomposition_method(df1)[0]
                #y_rep = decomposition_method(df2)[0]
                #ax.set_xlabel("First Principal Component of the First cluster", fontsize = 6)
                #ax.set_ylabel("First Principal Component of the Second cluster", fontsize = 6)
                #ax.scatter(x_rep, y_rep, alpha=0.5)
                #fig.tight_layout()
                #fig.savefig(filename + '/association_' + str(association_number) + '.pdf')
                
                #plt.figure()
                fig = plt.figure(figsize=(5, 4))
                # Create an Axes object.
                ax = fig.add_subplot(1,1,1) # one row, one column, first plot
                plt.rc('xtick', labelsize=6) 
                plt.rc('ytick', labelsize=6) 
                discretized_df1 = np.array(discretized_cluster1, dtype=float)
                discretized_df2 = np.array(discretized_cluster2, dtype=float)
                ax.set_xlabel("Representative of the First Cluster", fontsize = 10)
                ax.set_ylabel("Representative of the Second Cluster", fontsize = 10)

                d_x_d_rep = association.get_left_rep()#stats.discretize(decomposition_method(discretized_df1))
                d_y_d_rep = association.get_right_rep()#stats.discretize(decomposition_method(discretized_df2))
                
                #d_x_rep = decomposition_method(discretized_df1)
                #d_y_rep = decomposition_method(discretized_df2)
                #plt.scatter( stats.discretize(stats.pca(discretized_df1)),  stats.discretize(stats.pca(discretized_df2)), alpha=0.5)
                #ax.scatter( d_x_d_rep, d_y_d_rep, alpha=0.4)
                ax.scatter( association.get_left_rep(), association.get_right_rep(), alpha=0.4)
                fig.tight_layout()
                fig.savefig(discretized_filename + '/association_' + str(association_number) + '.pdf')
               
                '''x_d_pc1 = stats.discretize(x_rep)
                y_d_pc1 = stats.discretize(y_rep)
                '''
                if _bypass_discretizing():
                    d_x_d_rep = stats.discretize(association.get_left_rep())#stats.discretize(decomposition_method(discretized_df1))
                    d_y_d_rep = stats.discretize(association.get_right_rep())#stats.discretize(decomposition_method(discretized_df2))
                else:
                    d_x_d_rep = association.get_left_rep()#stats.discretize(decomposition_method(discretized_df1))
                    d_y_d_rep = association.get_right_rep()#stats.discretize(decomposition_method(discretized_df2))
                d_x_d_rep, d_y_d_rep = zip(*sorted(zip(d_x_d_rep, d_y_d_rep)))
                plot.confusion_matrix(d_x_d_rep, d_y_d_rep, filename = discretized_filename + '/association_' + str(association_number) + '_confusion_matrix.pdf' )
                #plot.confusion_matrix(x_d_pc1, d_y_d_rep, filename = discretized_filename + '/association_' + str(association_number) + '_confusion_matrix_pc_orginal_data.pdf' )
                plot.heatmap(array([d_x_d_rep]),  xlabels_order = x_label_order, xlabels =X_labels, filename =discretized_filename + 'Rep1_' + str(association_number) + '_heatmap', sortCol = False)
                plot.heatmap(array([d_y_d_rep]), xlabels_order= x_label_order,  xlabels =X_labels, filename =discretized_filename + 'Rep2_' + str(association_number) + '_heatmap', sortCol = False)
                plt.close("all")
            
    def _report_compared_clusters():
        output_file_compared_clusters  = open(str(config.output_dir)+'/all_compared_clusters_hypotheses_tree.txt', 'w')
        csvwc = csv.writer(output_file_compared_clusters , csv.excel_tab, delimiter='\t')
        csvwc.writerow(['Level', "Dataset 1", "Dataset 2" ])
        for line in hierarchy.reduce_tree_by_layer([config.meta_hypothesis_tree]):
            (level, clusters) = line
            iX, iY = clusters[0], clusters[1]
            fP = line[1]
            # fP_adjust = line[2]
            aLineOut = map(str, [str(level), str(';'.join(config.aOutName1[i] for i in iX)), str(';'.join(config.aOutName2[i] for i in iY))])
            csvwc.writerow(aLineOut)
    def _heatmap_associations():
        if config.plotting_results:
            print "--- plotting heatmap of associations  ..."
            global associated_feature_X_indecies
            Xs = list(set(associated_feature_X_indecies))
            X_labels = np.array([config.aOutName1[i] for i in Xs])
            global associated_feature_Y_indecies
            Ys = list(set(associated_feature_Y_indecies))
            Y_labels = np.array([config.aOutName2[i] for i in Ys])
            if len(Xs) > 1 and len(Ys) > 1: 
                cluster1 = [config.meta_feature[0][i] for i in Xs]    
                cluster2 = [config.meta_feature[1][i] for i in Ys]
                df1 = np.array(cluster1, dtype=float)
                df2 = np.array(cluster2, dtype=float)
                p = np.zeros(shape=(len(Xs), len(Ys)))
                #nmi = np.zeros(shape=(len(Xs), len(Ys)))
                plot.heatmap2(pArray1=cluster1, pArray2=cluster2, xlabels =X_labels, ylabels = Y_labels, filename = str(config.output_dir)+'/all_nmi_heatmap' )
    def _heatmap_associations_R():
        if len(associated_feature_X_indecies) == 0 or len(associated_feature_Y_indecies) == 0 :
            return
        from scipy.stats.stats import pearsonr
        global associated_feature_X_indecies
        Xs = list(set(associated_feature_X_indecies)) 
        
        global associated_feature_Y_indecies
        Ys = list(set(associated_feature_Y_indecies))
        
        config.X_features_cluster_order = [config.X_features_cluster_order[i] for i in range (len(config.X_features_cluster_order))  if config.X_features_cluster_order[i] in Xs ] 
        config.Y_features_order= [config.Y_features_order[i] for i in range (len(config.Y_features_order))  if config.Y_features_order[i] in Ys ] 
        
        X_labels = np.array([config.aOutName1[i] for i in config.X_features_cluster_order])
        Y_labels = np.array([config.aOutName2[i] for i in config.Y_features_order])
        
        import re
        X_labels_circos = np.array([re.sub('[^a-zA-Z0-9  \n\.]', '_', config.aOutName1[i]).replace(' ','_') for i in config.X_features_cluster_order])
        Y_labels_circos = np.array([re.sub('[^a-zA-Z0-9  \n\.]', '_', config.aOutName2[i]).replace(' ','_') for i in config.Y_features_order])
        
        similarity_score = np.zeros(shape=(len(config.X_features_cluster_order), len(config.Y_features_order)))  
        for i in range(len(config.X_features_cluster_order)):
            for j in range(len(config.Y_features_order)):
                similarity_score[i][j] = distance.c_hash_metric[config.distance](config.meta_feature[0][config.X_features_cluster_order[i]], config.meta_feature[1][config.Y_features_order[j]])
                
        ''''p = np.zeros(shape=(len(config.X_features_cluster_order), len(config.Y_features_order))) 
        for i in range(len(config.X_features_cluster_order)):
            for j in range(len(config.Y_features_order)):
                p[i][j] = pearsonr(np.array(config.meta_array[0][config.X_features_cluster_order[i]], dtype=float), np.array(config.meta_array[1][config.Y_features_order[j]], dtype=float))[0]
        '''
        def _is_in_an_assciostions(i,j):
            for association in sorted_associations:
                iX, iY = association.get_data()
                if i in iX and j in iY:
                    return True
            return False
         
        '''with open('similarity_score.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(Y_labels)
            [writer.writerow(r) for r in similarity_score] 
        '''
        
        anottation_cell = np.zeros(shape=(len(config.X_features_cluster_order), len(config.Y_features_order)))                
        for i in range(len(config.X_features_cluster_order)):
            for j in range(len(config.Y_features_order)):
                if _is_in_an_assciostions(config.X_features_cluster_order[i],config.Y_features_order[j]): #for association in sorted_associations:
                    anottation_cell[i][j] = 1
                    
        circos_tabel = np.zeros(shape=(len(config.X_features_cluster_order), len(config.Y_features_order)))
        for i in range(len(config.X_features_cluster_order)):
            for j in range(len(config.Y_features_order)):
                if _is_in_an_assciostions(config.X_features_cluster_order[i],config.Y_features_order[j]): #for association in sorted_associations:
                    try:
                        circos_tabel[i][j] = math.fabs(int(similarity_score[i][j]*100))
                    except:
                        circos_tabel[i][j] = 0
        logger.write_circos_table(circos_tabel, str(config.output_dir)+"/" +"circos_table_"+ config.distance+".txt", rowheader=X_labels_circos, colheader=Y_labels_circos, corner = "Data")         
        logger.write_table(similarity_score,str(config.output_dir)+"/" + config.distance+"_similarity_table.txt", rowheader=X_labels, colheader=Y_labels, corner = "#")
        logger.write_table(anottation_cell,str(config.output_dir)+"/" + config.distance+"_asscoaitaion_table.txt", rowheader=X_labels, colheader=Y_labels, corner = "#")
        #anottation_cell = [config.X_features_cluster_order]
        #anottation_cell = [ anottation_cell[:][j] for j in config.Y_features_order]
        #print anottation_cell
        if config.plotting_results:
            print "--- plotting heatmap associations using R ..."
            import rpy2.robjects as ro
            #import pandas.rpy.common as com
            import rpy2.robjects.numpy2ri
            rpy2.robjects.numpy2ri.activate()
            ro.globalenv['similarity_score'] = similarity_score
            ro.globalenv['labRow'] = X_labels 
            ro.globalenv['labCol'] = Y_labels
            ro.globalenv['sig_matrix'] = anottation_cell
            #ro.globalenv['circos_table'] = circos_tabel
            #Ext_Y_labels = ["labels"] + list(Y_labels)
            #ro.globalenv['Ext_labCol'] = Ext_Y_labels
            ro.r('rownames(similarity_score) = labRow')
            ro.r('colnames(similarity_score) = labCol')
            ro.r('rownames(sig_matrix) = labRow')
            ro.r('colnames(sig_matrix) = labCol')
            #ro.r('rownames(circos_table) = labRow')
            #ro.r('colnames(circos_table) = labCol')
           # ro.globalenv['output_table_similarity_score'] = str(config.output_dir)+"/" + config.distance+"_similarity_table.txt"
            ro.globalenv['output_asscoaiation_table'] = str(config.output_dir)+"/" + config.distance+"_asscoaitaion_table.txt"
            ro.globalenv['output_circus_table'] = str(config.output_dir)+"/" + config.distance+"_circos_table.txt"
            #ro.r('write.table(similarity_score , output_table_similarity_score, sep = "\t", eol = "\n", quote = F, col.names = NA, row.names = labRow)')
            #ro.r('write.table(sig_matrix , output_asscoaiation_table, sep = "\t", eol = "\n", quote = F, col.names =NA , row.names = labRow)')
            #ro.r('write.table(circos_table , output_circus_table, sep = "\t", eol = "\n", quote = F, col.names =NA , row.names = labRow)')
            #rev(heat.colors(100))
            if len(Xs) > 1 and len(Ys) > 1: 
                ro.r('library("RColorBrewer")')
                ro.r('library("pheatmap")')
                ro.globalenv['output_heatmap_similarity_score'] = str(config.output_dir)+"/" + config.distance+"_heatmap.pdf"
                #ro.globalenv['output_file_Pearson'] = str(config.output_dir)+"/Pearson_heatmap.pdf"
                if distance.c_hash_association_method_discretize[config.distance]:
                    ro.r('pheatmap(similarity_score, color = brewer.pal(100,"Reds"),labRow = labRow, labCol = labCol, filename =output_heatmap_similarity_score, cellwidth = 10, cellheight = 10, fontsize = 10, show_rownames = T, show_colnames = T, cluster_rows=FALSE, cluster_cols=FALSE, display_numbers = matrix(ifelse(sig_matrix > 0, "*", ""), nrow(sig_matrix)))')#,scale="row",  key=TRUE, symkey=FALSE, density.info="none", trace="none", cexRow=0.5
                else:
                    ro.r('pheatmap(similarity_score,labRow = labRow, labCol = labCol, filename =output_heatmap_similarity_score, cellwidth = 10, cellheight = 10, fontsize = 10, show_rownames = T, show_colnames = T, cluster_rows=FALSE, cluster_cols=FALSE, display_numbers = matrix(ifelse(sig_matrix > 0, "*", ""), nrow(sig_matrix)))')#,scale="row",  key=TRUE, symkey=FALSE, density.info="none", trace="none", cexRow=0.5
                ro.r('dev.off()')
                '''if config.distance != "pearson":
                    ro.globalenv['p'] = p
                    #ro.r('pdf(file = "./output/Pearson_heatmap.pdf")')
                    ro.r('rownames(p) = labRow')
                    ro.r('colnames(p) = labCol')
                    #if distance.c_hash_association_method_discretize[config.distance]:
                    ro.r('pheatmap(p, labRow = labRow, labCol = labCol, filename = output_file_Pearson, cellwidth = 10, cellheight = 10, fontsize = 10, show_rownames = T, show_colnames = T, cluster_rows=F, cluster_cols=F, display_numbers = matrix(ifelse(sig_matrix > 0, "*", ""), nrow(sig_matrix)))')#, scale="column",  key=TRUE, symkey=FALSE, density.info="none", trace="none", cexRow=0.5
                    #else:
                    #ro.r('pheatmap(p, labRow = labRow, labCol = labCol, filename = output_file_Pearson, cellwidth = 10, cellheight = 10, fontsize = 10, show_rownames = T, show_colnames = T, cluster_rows=F, cluster_cols=F, display_numbers = matrix(ifelse(sig_matrix > 0, "*", ""), nrow(sig_matrix)))')#, scale="column",  key=TRUE, symkey=FALSE, density.info="none", trace="none", cexRow=0.5
                    ro.r('dev.off()')
                    '''
    def _heatmap_datasets_R():
        if config.plotting_results:          
            print "--- plotting heatmap datasets using R ..."
            from scipy.stats.stats import pearsonr
            X_indecies = len(config.aOutName1)
            X_labels = np.array([config.aOutName1[i] for i in range(X_indecies)])
            Y_indecies = len(config.aOutName2)
            Y_labels = np.array([config.aOutName2[i] for i in range(Y_indecies)])
            if len(X_labels) > 1 and len(Y_labels) > 1: 
                df1 = np.array(config.meta_feature[0], dtype=float)
                df2 = np.array(config.meta_feature[1], dtype=float)
                drows1 = np.zeros(shape=(X_indecies, X_indecies))
                drows2 = np.zeros(shape=(Y_indecies, Y_indecies))
                
                for i in range(X_indecies):
                    for j in range(X_indecies):
                        drows1[i][j] = distance.c_hash_metric[config.distance](df1[i], df1[j]) 
                
                for i in range(Y_indecies):
                    for j in range(Y_indecies):
                        drows2[i][j] = distance.c_hash_metric[config.distance](df2[i], df2[j])    
                
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
            
    # Execute report functions
    _report_all_tests()
    _report_associations()
    _report_compared_clusters()
    if config.plotting_results:
        #_heatmap_associations()
        _heatmap_associations_R()
        _heatmap_datasets_R()
        _plot_associations()
    
    return config.meta_report 
  
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
    if config.output_dir == "./":
        config.output_dir = "./"+config.descending+'_'+config.distance+'_'+config.decomposition +'_'+ str(len(config.meta_array[0]))+'_'+str(len(config.meta_array[0][1]))
        if not os.path.isdir(config.output_dir):
            try:
                print("Creating output directory: " + config.output_dir)
                os.mkdir(config.output_dir)
            except EnvironmentError:
                sys.exit("CRITICAL ERROR: Unable to create output directory.")
        else:
            try:
                print("Removing the old output directory: " + config.output_dir)
                shutil.rmtree(config.output_dir)
                print("Creating output directory: " + config.output_dir)
                os.mkdir(config.output_dir)
            except EnvironmentError:
                sys.exit("CRITICAL ERROR: Unable to create output directory.")
    
    try:    
        performance_file  = open(str(config.output_dir)+'/performance.txt', 'a')
    except IOError:
        sys.exit("IO Exception: "+config.output_dir+"/performance.txt") 
    csvw = csv.writer(performance_file, csv.excel_tab, delimiter='\t')
    csvw.writerow(["Decomposition method: ", config.decomposition])
    csvw.writerow(["Similarity method: ", config.distance]) 
    csvw.writerow(["q: FDR cut-off : ", config.q]) 
    #csvw.writerow(["r: effect size for robustness : ", config.robustness]) 
    csvw.writerow(["Applied stop condition : ", config.apply_bypass]) 
    csvw.writerow(["Discretizing method : ", config.strDiscretizing])
    csvw.writerow(["Seed number: ", config.seed]) 
    csvw.writerow([])
    #name_features()
    if not is_correct_submethods_combination():
        sys.exit("Please ckeck the combination of your options!!!!")
    execution_time = time.time()
    if _bypass_discretizing():
        config.meta_feature = array([np.asarray(config.meta_array[0], dtype = float), np.asarray(config.meta_array[1], dtype = float)])
        #print config.meta_feature
    else:
        print "Featurize is started using: ", config.strDiscretizing, " style!"
        # featurize 
        start_time = time.time()
        _featurize()
        excution_time_temp = time.time() - start_time
        csvw.writerow(["featurize time", str(datetime.timedelta(seconds=excution_time_temp)) ])
        print("--- %s h:m:s featurize data time ---" % str(datetime.timedelta(seconds=excution_time_temp)))
        #print config.meta_feature
    #plot.heatmap2(pArray1=config.meta_feature[0], pArray2=config.meta_feature[1], xlabels =config.aOutName1, ylabels = config.aOutName2, filename = str(config.output_dir)+'/heatmap2_all' )
    if config.log_input:
        logger.write_table(data=config.meta_feature[0], name=config.output_dir+"/X_dataset.txt", rowheader=config.aOutName1 , colheader=None, prefix = "label",  corner = '#', delimiter= '\t')
        logger.write_table(data=config.meta_feature[1], name=config.output_dir+"/Y_dataset.txt", rowheader=config.aOutName2 , colheader=None, prefix = "label",  corner = '#', delimiter= '\t')
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
        _couple()
        excution_time_temp = time.time() - start_time
        csvw.writerow(["Coupling hypotheses tree time", str(datetime.timedelta(seconds=excution_time_temp)) ])
        print("--- %s h:m:s coupling hypotheses tree time ---" % str(datetime.timedelta(seconds=excution_time_temp)))
        # hypotheses testing
        print("--- association hypotheses testing is started, this task may take longer ...")
        start_time = time.time()
        _hypotheses_testing()
        excution_time_temp = time.time() - start_time
        csvw.writerow(["Hypotheses testing time", str(datetime.timedelta(seconds=excution_time_temp)) ])
        print("--- %s h:m:s hypotheses testing time ---" % str(datetime.timedelta(seconds=excution_time_temp)))
    
    # Generate a report
    start_time = time.time() 
    _summary_statistics('final') 
    excution_time_temp = time.time() - start_time
    csvw.writerow(["Summary statistics time", excution_time_temp ])
    print("--- %s h:m:s summary statistics time ---" % str(datetime.timedelta(seconds=excution_time_temp)))
    
    start_time = time.time() 
    results = _report()
    excution_time_temp = time.time() - start_time
    csvw.writerow(["Plotting results time", str(datetime.timedelta(seconds=excution_time_temp)) ])
    print("--- %s h:m:s plotting results time ---" % str(datetime.timedelta(seconds=excution_time_temp)))
    excution_time_temp = time.time() - execution_time
    csvw.writerow(["Total execution time", str(datetime.timedelta(seconds=excution_time_temp))])
    print("--- in %s h:m:s the task is successfully done ---" % str(datetime.timedelta(seconds=excution_time_temp)) )
    performance_file.close()
    return results

def view_singleton(pBags):
    aOut = [] 
    for aIndices, fP in pBags:
        if len(aIndices[0]) == 1 and len(aIndices[1]) == 1:
            aOut.append([aIndices, fP])
    return aOut 

def is_correct_submethods_combination():
    if config.descending == "AllA" and config.decomposition == 'mca':
        config.decomposition = "none"        
    if (config.descending == "AllA" and not config.decomposition =='none') or\
                        (config.descending == "HAllA" and config.decomposition =='none') or\
                        (config.decomposition in ["ica","pca",'pls', 'cca', 'kpca'] and config.distance not in ["pearson", "spearman","mic","dcor"] ) or\
                        (config.decomposition == "mca" and config.distance in ["pearson", "spearman","dcor"]):
            False
    else:
        return True
