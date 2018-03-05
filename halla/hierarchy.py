#!/usr/bin/env python 
'''
Hiearchy module, used to build trees and other data structures.
Handles clustering and other organization schemes. 
'''
import itertools
import copy
import math 
from numpy import array , rank, median
import numpy 
import scipy.cluster 
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import linkage, to_tree, leaves_list
from scipy.spatial.distance import pdist, squareform
import sys
#import matplotlib.pyplot as plt
import numpy as np
import pandas
import csv
from . import distance
from . import stats
from . import plot
from . import config
from . import logger
from . import HSIC
#try:
#from __builtin__ import 'True'
#except:
#from builtins import True
from matplotlib.sankey import RIGHT
from itertools import product, combinations
from unicodedata import decomposition
from math import fabs
#from profile import Stats
try:
    from functools import reduce
except:
    pass
sys.setrecursionlimit(20000)

# Multi-threading section
def multi_pMethod(args):
    """
    Runs the pMethod function and returns the results plus the id of the node
    """
    
    id, pMethod, dataset1, dataset2 = args
    dP, similarity, left_first_rep_variance, right_first_rep_variance, \
    left_loading, right_loading, left_rep, right_rep = pMethod(dataset1, dataset2)

    return id, dP, similarity, left_first_rep_variance, right_first_rep_variance,\
         left_loading, right_loading, left_rep, right_rep

def multiprocessing_estimate_pvalue(estimate_pvalue, current_level_tests, pMethod, dataset1, dataset2):
    """
    Return the results from applying the data to the estimate_pvalue function
    """
    def _multi_pMethod_args(current_level_tests, pMethod, dataset1, dataset2, ids_to_process):
        for id in ids_to_process:
            aIndicies = current_level_tests[id].m_pData
            aIndiciesMapped = list(map(array, aIndicies))
            yield [id, pMethod, dataset1[aIndiciesMapped[0]], dataset2[aIndiciesMapped[1]]]
    
    if config.NPROC > 1:
        import multiprocessing
        pool = multiprocessing.Pool(config.NPROC)
        
        # check for tests that already have pvalues as these do not need to be recomputed
        ids_to_process=[]
        result = [0.0] * len(current_level_tests)
        for id in range(len(current_level_tests)):
            if current_level_tests[id].pvalue != None:
                result[id]=current_level_tests[id].pvalue
            else:
                # increment the number of permutations tests 
                config.number_of_performed_tests += 1
                ids_to_process.append(id)
        
        
        results_by_id = pool.map(multi_pMethod, _multi_pMethod_args(current_level_tests, 
            pMethod, dataset1, dataset2, ids_to_process))
        pool.close()
        pool.join()
       
        # order the results by id and apply results to nodes
        for id, dP, similarity, left_first_rep_variance, right_first_rep_variance, left_loading,\
         right_loading, left_rep, right_rep in results_by_id:
            result[id]=dP
            current_level_tests[id].similarity_score = similarity
    else:
        result=[]
        for i in range(len(current_level_tests)):
            if current_level_tests[i].pvalue != None:
                result.append(current_level_tests[i].pvalue)
            else: 
                # increment the number of permutations tests 
                config.number_of_performed_tests +=1
                result.append(estimate_pvalue(current_level_tests[i]))

    return result

#==========================================================================#
# DATA STRUCTURES 
#==========================================================================#

# maximum distance in hierarchical  trees.
global max_dist
max_dist = 1.0 

# A number for hierarchy heatmaps
global fig_num 
fig_num = 1

class Hypothesis_Node():
    ''' 
    A hierarchically nested structure containing nodes as
    a basic core unit    
    A general object, tree need not be 2-tree 
    '''    
    __slots__ = ['m_pData', 'm_arrayChildren', 'left_distance', 'right_distance',
                'pvalue', 'qvalue', 'similarity_score','level_number' , 'significance', 'rank', 
                'already_passed', 'already_tested' ]
    def __init__(self, data=None, left_distance=None, right_distance=None, similarity=None):
        self.m_pData = data 
        self.m_arrayChildren = []
        self.left_distance = left_distance
        self.right_distance = right_distance
        self.pvalue = None
        self.qvalue = None
        self.similarity_score = None
        self.already_tested = False
        self.already_passed = False
        self.level_number = 1
        self.significance =  None
        self.rank = None

def left(node):
    return get_child(node, iIndex=0)

def right(node):
    return get_child(node, iIndex=1)

def is_leaf(node):
    return bool(not(node.m_pData and node.m_arrayChildren))

def add_child(node, data):
    if not isinstance(data, Hypothesis_Node):
        pChild = Hypothesis_Node(data)
    else:
        pChild = data 
    node.m_arrayChildren.append(pChild)
    return node 
    
def add_children(node, aData):
    for item in aData:
        node = add_child(node, item)
    return node 

def get_children(node): 
    return node.m_arrayChildren

def get_child(node, iIndex=None):
    return node.m_arrayChildren[iIndex or 0] if node.m_arrayChildren else None 

def add_data(node, pDatum):
    node.m_pData = pDatum 
    return node 
    
def get_data(node):
    return node.m_pData

def stop_decesnding_silhouette_coefficient(Node):
    if len(Node.m_pData[0]) <= 1 and len(Node.m_pData[1]) <= 1:
        return True
    pMe = distance.c_hash_metric[config.similarity_method]
    silhouette_scores = []
    cluster_a = Node.m_pData[0]
    cluster_b = Node.m_pData[1]
    silhouette_coefficient_A = []
    silhouette_coefficient_B = []
    for a_feature in cluster_a:
        if len(cluster_a) ==1:
            a = 0.0
        else:
            temp_a_features = cluster_a[:]
            temp_a_features.remove(a_feature)
            a = np.mean([1.0 - config.Distance[0][i][j] for i,j in product([a_feature], temp_a_features)])

        b = np.mean([1.0 - math.fabs(pMe(config.parsed_dataset[0][i], config.parsed_dataset[1][j])) 
                    for i,j in product([a_feature], cluster_b)])
        s = (b-a)/max([a,b])
        silhouette_coefficient_A.append(s)
    for a_feature in cluster_b:
        if len(cluster_b) ==1:
            a = 0.0
        else:
            temp_a_features = cluster_b[:]
            temp_a_features.remove(a_feature)
            a = np.mean([1.0 - config.Distance[1][i][j] for i,j in product([a_feature], temp_a_features)])
               
        b = np.mean([1.0 - math.fabs(pMe(config.parsed_dataset[1][i], config.parsed_dataset[0][j])) 
                    for i,j in product([a_feature], cluster_a)])
        s = (b-a)/max([a,b])
        silhouette_coefficient_B.append(s)

    silhouette_scores = silhouette_coefficient_A
    silhouette_scores.extend(silhouette_coefficient_B)
    
    if numpy.min(silhouette_scores)  < 0.25:
        return False
    else:
        return True

def Node_clusters_diameter(Node):
    
    number_left_features = len(Node.m_pData[0])
    number_right_features = len(Node.m_pData[1])

    counter = 0
    temp_right_loading = list()
    reps_similarity = Node.similarity_score
    pMe = distance.c_hash_metric[config.similarity_method] 
    if len(Node.m_pData[0]) == 1:
        left_all_sim = [1.0]
    else:
        left_all_sim = [pMe(config.parsed_dataset[0][i], config.parsed_dataset[0][j]) for i,j in combinations(Node.m_pData[0], 2)]
    if len(Node.m_pData[1]) == 1:
        right_all_sim = [1.0]
    else:
        right_all_sim = [pMe(config.parsed_dataset[1][i], config.parsed_dataset[1][j]) for i,j in combinations(Node.m_pData[1],2)]
    diam_A_r = ((1.0 - math.fabs(min(left_all_sim))))# - math.fabs((1.0 - max(left_all_sim))))
    diam_B_r = ((1.0 - math.fabs(min(right_all_sim))))# - math.fabs((1.0 - max(right_all_sim))))
    return diam_A_r, diam_B_r
    
def stop_and_reject(Node):
    
    number_left_features = len(Node.m_pData[0])
    number_right_features = len(Node.m_pData[1])

    if len(Node.m_pData[0]) <= 1 and len(Node.m_pData[1]) <= 1:
        return True
    counter = 0
    temp_right_loading = list()
    reps_similarity = Node.similarity_score
    pMe = distance.c_hash_metric[config.similarity_method] 
    diam_Ar_Br = (1.0 - math.fabs(pMe(Node.left_rep, Node.right_rep)))
    if len(Node.m_pData[0]) == 1:
        left_all_sim = [1.0]
    else:
        left_all_sim = [pMe(config.parsed_dataset[0][i], config.parsed_dataset[0][j]) for i,j in combinations(Node.m_pData[0], 2)]
    if len(Node.m_pData[1]) == 1:
        right_all_sim = [1.0]
    else:
        right_all_sim = [pMe(config.parsed_dataset[1][i], config.parsed_dataset[1][j]) for i,j in combinations(Node.m_pData[1],2)]
    diam_A_r = ((1.0 - math.fabs(min(left_all_sim))))# - math.fabs((1.0 - max(left_all_sim))))
    diam_B_r = ((1.0 - math.fabs(min(right_all_sim))))# - math.fabs((1.0 - max(right_all_sim))))
    if config.verbose == 'DEBUG':
        print ("===================stop and reject check========================")
        #print "Left Exp. Var.: ", Node.left_first_rep_variance
        print ("Left before: ", Node.m_pData[0])
        #print "Right Exp. Var.: ", Node.right_first_rep_variance
        print ("Right before: ", Node.m_pData[1])
        print ("dime_A_r: ", diam_A_r,"  ", "dime_B_r: ", diam_B_r, "diam_Ar_Br: ", diam_Ar_Br)
    if diam_A_r + diam_B_r == 0:
        return True
    if diam_Ar_Br > diam_A_r + diam_B_r:
        return True
    else:
        return False

def is_bypass(Node, method = ''):
    
    if len(Node.m_pData[0]) <= 1 and len(Node.m_pData[1]) <= 1:
        return True
    else: return False
    if config.apply_stop_condition:
        if method == 'HSIC':
            l0 =Node.m_pData[0]
            l1 =Node.m_pData[1]
            if len(l0) == 1 and len(l1)==1:
                return False
            #print l0, l1
            #print config.parsed_dataset[0][Node.m_pData[0]]
            return HSIC.HSIC_pval(config.parsed_dataset[0][l0].T, config.parsed_dataset[1][l1].T)[1] > .05
        else:
            return stop_and_reject(Node)
            #return stop_decesnding_silhouette_coefficient(Node)
    else:
        return False

def report(Node):
    print ("\n--- hypothesis test based on permutation test")        
    print ("---- pvalue                        :", Node.pvalue)
    print ("---- similarity_score score              :", self.similarity_score)
    print ("---- first cluster's features      :", Node.m_pData[0])
    print ("---- first cluster similarity_score      :", 1.0 - Node.left_distance)
    print ("---- second cluster's features     :", Node.m_pData[1])
    print ("---- second cluster similarity_score     :", 1.0 - Node.right_distance)

#==========================================================================#
# METHODS  
#==========================================================================#

def is_tree(pObj):
    """
    duck type implementation for checking if
    object is ClusterNode or Hypothesis_Node, more or less
    """

    try:
        get_data (pObj)
        return True 
    except Exception:
        return False 


def hclust(dataset, labels, dataset_number):
    #linkage_method = config.linkage_method
    Distance_matrix = pdist(dataset, metric=distance.pDistance) 
    config.Distance[dataset_number] =  squareform(Distance_matrix)
    Z= None
    if config.diagnostics_plot:# and len(config.Distance[dataset_number][0]) < 2000:
        print ("--- plotting heatmap for Dataset %s %s" %(str(dataset_number+ 1)," ... "))
        Z = plot.heatmap(data_table = dataset , D = Distance_matrix, xlabels_order = [], xlabels = labels,\
                          filename= config.output_dir+"/"+"hierarchical_heatmap_"+str(config.similarity_method)+"_" + \
                          str(dataset_number+1), linkage_method = config.linkage_method)
    else:
        Z = linkage(Distance_matrix, method = config.linkage_method)
    logger.write_table(data=config.Distance[dataset_number], name=config.output_dir+'/Distance_matrix'+str(dataset_number+1)+'.tsv', rowheader=config.FeatureNames[dataset_number], colheader=config.FeatureNames[dataset_number])
    return to_tree(Z) if len(dataset)>1 else Z, sch.dendrogram(Z)['leaves'] if len(dataset)>1 else sch.dendrogram(Z)['leaves']

def truncate_tree(apClusterNode, level=0, skip=0):
    """
    Chop tree from root, returning smaller tree towards the leaves 

    Parameters
    ---------------
        
        list_clusternode : list of ClusterNode objects 
        level : int 
        skip : int 

    Output 
    ----------

        lC = list of ClusterNode objects 

    """
    iSkip = skip 
    iLevel = level
    if iLevel < iSkip:
        try:
            return truncate_tree(list(filter(lambda x: bool(x), [(p.right if p.right else None) for p in apClusterNode])) \
            + list(filter(lambda x: bool(x), [(q.left if q.left else None) for q in apClusterNode]), level=iLevel + 1, skip=iSkip))
        except:
            return truncate_tree([x for x in [(p.right if p.right else None) for p in apClusterNode] if bool(x)] \
            + [x for x in [(q.left if q.left else None) for q in apClusterNode] if bool(x)], level=iLevel + 1, skip=iSkip) 

    elif iSkip == iLevel:
        if any(apClusterNode):
            try:
                return list(filter(lambda x: bool(x), apClusterNode))
            except:
                return [x for x in apClusterNode if bool(x)]
    
        else:
            return []
            raise Exception("truncated tree is malformed--empty!")


#-------------------------------------#
# Decider Functions                   #
#-------------------------------------#

def _percentage(dist, max_dist):
    if max_dist > 0:
        return float(dist) / float(max_dist)
    else:
        return 0.0

def _is_start(ClusterNode, X, func, distance):
    if _percentage(ClusterNode.dist) <= distance: 
        return True
    else: 
        return False

def _is_stop(ClusterNode):
        if ClusterNode.get_count() == 1 :#or ClusterNode.dist < .4:
            return True
        else:
            return False
        
def _cutree_to_log2 (apNode, X, func, distance, cluster_threshold):
    temp_apChildren = []
    temp_sub_apChildren = []
    print ("Length of ", len(apNode))
    for node in apNode:
        n = node.get_count()
        print ("Number of feature in node: ", n)
        sub_apChildren = truncate_tree([node], level=0, skip=1)
        if sub_apChildren == None:
            sub_apChildren = [node]
        else:
            while len(set(sub_apChildren)) < round(math.log(n)):
                temp_sub_apChildren = truncate_tree(sub_apChildren, level=0, skip=1)
                for i in range(len(sub_apChildren)):
                        if sub_apChildren[i].is_leaf():
                            if temp_sub_apChildren:
                                temp_sub_apChildren.append(sub_apChildren[i])
                            else:
                                temp_sub_apChildren = [sub_apChildren[i]]
                
                if temp_sub_apChildren == None:
                    temp_sub_apChildren = sub_apChildren
                sub_apChildren = temp_sub_apChildren
                temp_sub_apChildren = []
        temp_apChildren += sub_apChildren
    return set(temp_apChildren)

def cutree_to_get_number_of_features (cluster, distance_matrix, number_of_estimated_clusters = None):
    n_features = cluster.get_count()
    if n_features==1:
        return [cluster]
    if number_of_estimated_clusters == None:
        number_of_estimated_clusters = math.log(n_features, 2)
    sub_clusters = []
    sub_clusters = truncate_tree([cluster], level=0, skip=1)
    
    while True:# not all(val <= t for val in distances):
        largest_node = sub_clusters[0]
        index = 0
        for i in range(len(sub_clusters)):
            if largest_node.get_count() < sub_clusters[i].get_count():
                largest_node = sub_clusters[i]
                index = i
        if largest_node.get_count() > (n_features/number_of_estimated_clusters):
            #sub_clusters.remove(largest_node)
            #sub_clusters = sub_clusters[:index] + sub_clusters[index+1 :]
            del sub_clusters[index]
            sub_clusters += truncate_tree([largest_node], level=0, skip=1)
        else:
            break
    return sub_clusters

def cutree_to_get_below_threshold_distance_of_clusters (cluster, t = None):
    n_features = cluster.get_count()
    if t == None:
        t = config.cut_distance_thrd
    if n_features==1:# or cluster.dist <= t:
        return [cluster]
    sub_clusters = []
    #sub_clusters = cutree_to_get_number_of_clusters ([cluster])
    sub_clusters = truncate_tree([cluster], level=0, skip=1)
    #distances = [sub_clusters[i].dist for i in range(len(sub_clusters))]
    while True:# not all(val <= t for val in distances):
        max_dist_node = sub_clusters[0]
        index = 0
        for i in range(len(sub_clusters)):
            #if sub_clusters[i].dist > 0.0:
                #aDist += [sub_clusters[i].dist]
            if max_dist_node.dist < sub_clusters[i].dist:
                max_dist_node = sub_clusters[i]
                index = i
        if max_dist_node.dist > t:
            del sub_clusters[index]
            sub_clusters += truncate_tree([max_dist_node], level=0, skip=1)
        else:
            break
    return sub_clusters
def cutree_to_get_number_of_clusters (cluster, distance_matrix, number_of_estimated_clusters = None):
    n_features = cluster.get_count()
    if n_features==1:
        return [cluster]
    if number_of_estimated_clusters ==None:
        number_of_sub_cluters_threshold, _ = predict_best_number_of_clusters(cluster, distance_matrix)
        #round(math.log(n_features, 2))        
    else:
        number_of_sub_cluters_threshold = number_of_estimated_clusters
    sub_clusters = []
    sub_clusters = truncate_tree([cluster], level=0, skip=1)
    while len(sub_clusters) < number_of_sub_cluters_threshold:
        max_dist_node = sub_clusters[0]
        max_dist_node_index = 0
        for i in range(len(sub_clusters)):
            if max_dist_node.dist < sub_clusters[i].dist:
                max_dist_node = sub_clusters[i]
                max_dist_node_index = i
        if not max_dist_node.is_leaf():
            sub_clusters_to_add = truncate_tree([max_dist_node], level=0, skip=1)
            del sub_clusters[max_dist_node_index]
            sub_clusters.insert(max_dist_node_index,sub_clusters_to_add[0])
            if len(sub_clusters_to_add) ==2:
                sub_clusters.insert(max_dist_node_index+1,sub_clusters_to_add[1])
        else:
            break
    return sub_clusters
def descending_silhouette_coefficient(cluster, dataset_number):
    #====check within class homogeniety
    #Ref: http://scikit-learn.org/stable/modules/clustering.html#homogeneity-completeness-and-v-measure
    pMe = distance.c_hash_metric[config.similarity_method]
    sub_cluster = truncate_tree([cluster], level=0, skip=1)
    all_a_clusters = sub_cluster[0].pre_order(lambda x: x.id)
    all_b_clusters = sub_cluster[1].pre_order(lambda x: x.id)
    s_all_a = []
    s_all_b = []
    temp_all_a_clusters = []
    from copy import deepcopy
    for a_cluster in all_a_clusters:
        if len(all_a_clusters) ==1:
            # math.fabs(pMe(config.parsed_dataset[dataset_number][i], config.parsed_dataset[dataset_number][j])
            a = np.mean([1.0 - config.Distance[dataset_number][i][j] for i,j in product([a_cluster], all_a_clusters)])
        else:
            temp_all_a_clusters = all_a_clusters[:]#deepcopy(all_a_clusters)
            temp_all_a_clusters.remove(a_cluster)
            a = np.mean([1.0 - config.Distance[dataset_number][i][j] for i,j in product([a_cluster], temp_all_a_clusters)])            
        b = np.mean([1.0 - config.Distance[dataset_number][i][j] for i,j in product([a_cluster], all_b_clusters)])
        s = (b-a)/max([a,b])
        s_all_a.append(s)
    if any(val <= 0.0 for val in s_all_a) and not len(s_all_a) == 1:
        return True
    for b_cluster in all_b_clusters:
        if len(all_b_clusters) ==1:
            
            a = np.mean([1.0 - math.fabs(pMe(config.parsed_dataset[dataset_number][i], config.parsed_dataset[dataset_number][j])) for i,j in product([b_cluster], all_b_clusters)])
        else:
            temp_all_b_clusters = all_b_clusters[:]#deepcopy(all_a_clusters)
            temp_all_b_clusters.remove(b_cluster)
            a = np.mean([1.0 - math.fabs(pMe(config.parsed_dataset[dataset_number][i], config.parsed_dataset[dataset_number][j])) for i,j in product([b_cluster], temp_all_b_clusters)])            
        b = np.mean([1.0 -  math.fabs(pMe(config.parsed_dataset[dataset_number][i], config.parsed_dataset[dataset_number][j])) for i,j in product([b_cluster], all_a_clusters)])
        s = (b-a)/max([a,b])
        s_all_b.append(s)
    if any(val <= 0.0 for val in s_all_b) and not len(s_all_b) == 1:
        return True
    return False
def silhouette_coefficient(clusters, distance_matrix):
    #====check within class homogeniety
    #Ref: http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    #pMe = distance.c_hash_metric[config.Distance]
    distance_matrix = pandas.DataFrame(distance_matrix)
    silhouette_scores = []
    if len(clusters) <= 1:
        sys.exit("silhouette method needs at least two clusters!")
        
    for i in range(len(clusters)):
        cluster_a = clusters[i].pre_order(lambda x: x.id)
        
        # find the next and previous clusters for each cluster as a
        # potential closest clusters to the cluster[i]
        if i==0:
            next_cluster = prev_cluster = clusters[i+1].pre_order(lambda x: x.id)
        elif i == len((clusters))-1:
            next_cluster = prev_cluster = clusters[i-1].pre_order(lambda x: x.id) 
        else:
            next_cluster = clusters[i+1].pre_order(lambda x: x.id)
            prev_cluster = clusters[i-1].pre_order(lambda x: x.id)

        s_all_a = []
        for a_feature in cluster_a:
            if len(cluster_a) ==1:
                a = 0.0
            else:
                temp_a_features = cluster_a[:]#deepcopy(all_a_clusters)
                temp_a_features.remove(a_feature)
                a = np.mean([distance_matrix.iloc[i, j] for i,j in product([a_feature], temp_a_features)])            
            b1 = np.mean([ distance_matrix.iloc[i, j] for i,j in product([a_feature], next_cluster)])
            b2 = np.mean([ distance_matrix.iloc[i, j] for i,j in product([a_feature], prev_cluster)])
            b = min(b1,b2)
            #print a, b
            s = (b-a)/max(a,b)
            s_all_a.append(s)
        silhouette_scores.append(np.mean(s_all_a))
    return silhouette_scores

def get_medoid(features, distance_matrix):
    med = features[0]#max(distance_matrix)
    medoid_index = med
    for i in features:
        temp_mean = numpy.mean(distance_matrix.iloc[i])
        if temp_mean <= med:
            med = temp_mean
            medoid_index = i
    return medoid_index
def wss_heirarchy(clusters, distance_matrix):
    wss = numpy.zeros(len(clusters))
    temp_wss = 0.0
    for i in range(len(clusters)):
        if clusters[i].get_count() == 1:
            wss[i] = 0.0
        else:
            cluster_a = clusters[i].pre_order(lambda x: x.id)
            
            temp_a_features = cluster_a[:]
            medoid_feature = get_medoid(temp_a_features, distance_matrix)#temp_a_features[len(temp_a_features)-1]
            # remove medoid
            temp_a_features.remove(medoid_feature)
            
            temp_wss = sum([distance_matrix.iloc[i_t,j_t]* distance_matrix.iloc[i_t,j_t] 
                            for i_t,j_t in product([medoid_feature], temp_a_features)])
            wss[i] = temp_wss# * clusters[i].get_count()
    avgWithinSS = np.sum(wss) #[sum(d)/X.shape[0] for d in dist]
    return avgWithinSS

def predict_best_number_of_clusters_wss(hierarchy_tree, distance_matrix):
    distance_matrix = pandas.DataFrame(distance_matrix)
    min_num_cluster = 2  
    max_num_cluster = int(math.floor((math.log(len(distance_matrix),2))))
    wss = numpy.zeros(max_num_cluster+1)
    best_clust_size = 1
    best_wss = 0.0
    wss[1] = math.sqrt((len(distance_matrix)-1)*sum(distance_matrix.var(axis=1)))#apply(distance_matrix,2,var)))
    best_wss = wss[1]
    best_drop = .8
    #TSS = wss_heirarchy([hierarchy_tree], distance_matrix)
    #wss[1] = TSS
    #R=0.0
    #best_drop = R
    for i in range(min_num_cluster,max_num_cluster):
        clusters = cutree_to_get_number_of_clusters(hierarchy_tree, distance_matrix, number_of_estimated_clusters= i)
        wss[i] = wss_heirarchy(clusters, distance_matrix)
        wss[i] = math.sqrt(wss[i])
        if wss[i]/wss[i-1] < best_drop :
            print (wss[i]/wss[i-1])
            best_clust_size = i
            best_wss = wss[i]
            best_drop = wss[i]/wss[i-1]
    print ("The best guess for the number of clusters is: ", best_clust_size)
    return  best_clust_size 
def predict_best_number_of_clusters(hierarchy_tree, distance_matrix):
    #distance_matrix = pandas.DataFrame(distance_matrix)
    features = get_leaves(hierarchy_tree)
    clusters= [] #[hierarchy_tree]
    min_num_cluster = 2  
    max_num_cluster = int(len(features)/math.ceil(math.log(len(features), 2)))
    best_sil_score = 0.0
    best_clust_size = 1
    for i in range(min_num_cluster,max_num_cluster):
        clusters = cutree_to_get_number_of_clusters(hierarchy_tree, distance_matrix, number_of_estimated_clusters= i)
        removed_singlton_clusters = [cluster for cluster in clusters if cluster.get_count()>1]
        if len(removed_singlton_clusters) < 2:
            removed_singlton_clusters = clusters

        sil_scores = [sil for sil in silhouette_coefficient(removed_singlton_clusters, distance_matrix) if sil < 1.0 ]
        sil_score = numpy.mean(sil_scores)
        if best_sil_score < sil_score:
            best_sil_score = sil_score
            best_clust_size = len(clusters)
            result_sub_clusters = clusters
                
    print ("The best guess for the number of clusters is: ", best_clust_size)
    return best_clust_size, clusters       
def get_leaves(cluster):
    return cluster.pre_order(lambda x: x.id)  
    
def get_homogenous_clusters_silhouette(cluster, distance_matrix, number_of_estimated_clusters=None, resolution= 'high'):
    n = cluster.get_count()
    if n==1:
        return [cluster]
    if resolution == 'low' :
        sub_clusters = cutree_to_get_number_of_clusters(cluster, distance_matrix, number_of_estimated_clusters= number_of_estimated_clusters)    
    else:
        sub_clusters = cutree_to_get_number_of_features(cluster, distance_matrix, number_of_estimated_clusters= number_of_estimated_clusters)#truncate_tree([cluster], level=0, skip=1)#
    sub_silhouette_coefficient = silhouette_coefficient(sub_clusters, distance_matrix) 
    while True:
        min_silhouette_node = sub_clusters[0]
        min_silhouette_node_index = 0
        
        # find cluster with minimum homogeneity 
        for i in range(len(sub_clusters)):
            if sub_silhouette_coefficient[min_silhouette_node_index] > sub_silhouette_coefficient[i]:
                min_silhouette_node = sub_clusters[i]
                min_silhouette_node_index = i
        # if the cluster with the minimum homogeneity has silhouette_coefficient
        # it means all cluster has passed the minimum homogeneity threshold  
        if sub_silhouette_coefficient[min_silhouette_node_index] == 1.0:
            break
        sub_clusters_to_check = cutree_to_get_number_of_features(min_silhouette_node, distance_matrix, number_of_estimated_clusters= number_of_estimated_clusters) #truncate_tree([min_silhouette_node], level=0, skip=1) #
        clusters_to_add = truncate_tree([min_silhouette_node], level=0, skip=1)
        if len(clusters_to_add) < 2:
            break
        temp_silhouette_coefficient = silhouette_coefficient(clusters_to_add, distance_matrix)
        if len(sub_clusters_to_check) < 2:
            break
        sub_silhouette_coefficient_to_check = silhouette_coefficient(sub_clusters_to_check, distance_matrix)
        temp_sub_silhouette_coefficient_to_check = sub_silhouette_coefficient_to_check[:]
        temp_sub_silhouette_coefficient_to_check = [value for value in temp_sub_silhouette_coefficient_to_check if value != 1.0]

        if len(temp_sub_silhouette_coefficient_to_check) == 0 or sub_silhouette_coefficient[min_silhouette_node_index] >= np.max(temp_sub_silhouette_coefficient_to_check):
            sub_silhouette_coefficient[min_silhouette_node_index] =  1.0
        else:
            del sub_clusters[min_silhouette_node_index]#min_silhouette_node)
            del sub_silhouette_coefficient[min_silhouette_node_index]
            sub_silhouette_coefficient.extend(temp_silhouette_coefficient)
            sub_clusters.extend(clusters_to_add)
   
    return sub_clusters
    
def couple_tree(apClusterNode0, apClusterNode1, dataset1, dataset2, strMethod="uniform", strLinkage="min", robustness = None):
    
    func = config.similarity_method
    """
    Couples two data trees to produce a hypothesis tree 

    Parameters
    ------------
    pClusterNode1, pClusterNode2 : ClusterNode objects
    method : str 
        {"uniform", "2-uniform", "log-uniform"}
    linkage : str 
        {"max", "min"}

    Returns
    -----------
    tH : Hypothesis_Node object 

    Examples
    ----------------
    """
    
    X, Y = dataset1, dataset2
    global max_dist_cluster1 
    max_dist_cluster1 = max (node.dist for node in apClusterNode0)
    
    global max_dist_cluster2 
    max_dist_cluster2 = max (node.dist for node in apClusterNode1)

    # Create the root of the coupling tree
    for a, b in itertools.product(apClusterNode0, apClusterNode1):
        try:
            data1 = a.pre_order(lambda x: x.id)
            data2 = b.pre_order(lambda x: x.id)
        except:
            data1 = reduce_tree(a)
            data2 = reduce_tree(b)
    Hypothesis_Tree_Root = Hypothesis_Node([data1, data2])
    Hypothesis_Tree_Root.level_number = 0
    
    # Get the first level homogeneous clusters
    apChildren0 = get_homogenous_clusters_silhouette (apClusterNode0[0], config.Distance[0])
    apChildren1 = get_homogenous_clusters_silhouette (apClusterNode1[0], config.Distance[1])
    
    childList = []
    L = []    
    for a, b in itertools.product(apChildren0, apChildren1):
        try:
            data1 = a.pre_order(lambda x: x.id)
            data2 = b.pre_order(lambda x: x.id)
        except:
            data1 = reduce_tree(a)
            data2 = reduce_tree(b)
        tempTree = Hypothesis_Node(data=[data1, data2], left_distance=a.dist, right_distance=b.dist)
        tempTree.level_number = 1
        childList.append(tempTree)
        L.append((tempTree, (a, b)))
    Hypothesis_Tree_Root = add_children(Hypothesis_Tree_Root, childList)
    next_L = []
    level_number = 2
    while L:
        (pStump, (a, b)) = L.pop(0)
        try:
            data1 = a.pre_order(lambda x: x.id)
            data2 = b.pre_order(lambda x: x.id)
        except:
            data1 = reduce_tree(a)
            data2 = reduce_tree(b)        
        bTauX = _is_stop(a)  # ( _min_tau(X[array(data1)], func) >= x_threshold ) ### parametrize by mean, min, or max
        bTauY = _is_stop(b)  # ( _min_tau(Y[array(data2)], func) >= y_threshold ) ### parametrize by mean, min, or max
        if bTauX and bTauY :
            if L:
                continue
            else:
                if next_L:
                    L = next_L
                    next_L = []
                    level_number += 1
                continue
        if not bTauX:
            apChildren0 = get_homogenous_clusters_silhouette(a,config.Distance[0])
        else:
            apChildren0 = [a]
        if not bTauY:
            apChildren1 = get_homogenous_clusters_silhouette(b,config.Distance[1])#cutree_to_get_number_of_clusters([b])
            #cutree_to_get_number_of_features(b)
            ##
            #get_homogenous_clusters_silhouette(b,1)#
        else:
            apChildren1 = [b]

        LChild = [(c1, c2) for c1, c2 in itertools.product(apChildren0, apChildren1)] 
        childList = []
        while LChild:
            (a1, b1) = LChild.pop(0)
            try:
                data1 = a1.pre_order(lambda x: x.id)
                data2 = b1.pre_order(lambda x: x.id)
            except:
                data1 = reduce_tree(a1)
                data2 = reduce_tree(b1)
            
            tempTree = Hypothesis_Node(data=[data1, data2], left_distance=a1.dist, right_distance=b1.dist)
            tempTree.level_number = level_number
            childList.append(tempTree)
            next_L.append((tempTree, (a1, b1)))
        pStump = add_children(pStump, childList)
        if not L:
            if next_L:
                L = next_L
                next_L = []
                level_number += 1

    return [Hypothesis_Tree_Root]
def number_of_pause(n1,n2):
    #number of pause
    n = 0
    if n1 == n2:
        return 0
    else:
        a = max(n1, n2)
        b = min(n1, n2)
        if a/math.log(a,2) >= b:
            return 1 + number_of_pause(a/math.log(a,2), b)
        else:
            return 0
def number_of_level(n):
    #number of possible levels
    #print(n)
    if n <= 1:
        return 0
    elif n ==2:
        return 1
    else:
        return 1 + number_of_level(int(n/math.log(n,2)))
        
def test_by_level(apClusterNode0, apClusterNode1, dataset1, dataset2, strMethod="uniform", strLinkage="min", robustness = None):
    
    func = config.similarity_method
    X, Y = dataset1, dataset2
    significant_hypotheses = []
    tested_hypotheses = []
    config.number_of_performed_tests = 0
    # Define the speed of cutting hierarchies
    # e.g. 1 means we cut in each iteration
    # e.g. 2 means we cut in each 2 iteration 
    n1 = apClusterNode0[0].get_count()
    n2 = apClusterNode1[0].get_count()
    n_pause = number_of_pause(n1, n2)
    #print ("Number of pause:", n_pause)
    n1 = apClusterNode0[0].get_count()
    n2 = apClusterNode1[0].get_count()
    if n1 > n2 :
        cut_speed_1 = 1
    else:
        #cut_speed_1 = max(math.log(n2,2)/math.log(n1,2) ,1) 
        cut_speed_1 = max(number_of_level(n2)*1.0/number_of_level(n1) ,1)
    if n2 > n1:
        cut_speed_2 = 1
    else:
        #cut_speed_2 = max(math.log(n1,2) / math.log(n2, 2),1)
        cut_speed_2 = max(number_of_level(n1)*1.0/number_of_level(n2) ,1)
    # Write the hypothesis that has been tested
    output_file_compared_clusters  = open(str(config.output_dir)+'/hypotheses_tree.txt', 'w')
    csvwc = csv.writer(output_file_compared_clusters , csv.excel_tab, delimiter='\t')
    csvwc.writerow(['Level', "Dataset 1", "Dataset 2" ])
    data1 = apClusterNode0[0].pre_order(lambda x: x.id)
    data2 = apClusterNode1[0].pre_order(lambda x: x.id)
    aLineOut = list(map(str, [str(0), str(';'.join([config.FeatureNames[0][i] for i in data1])), str(';'.join([config.FeatureNames[1][i] for i in data2]))]))
    csvwc.writerow(aLineOut)
    
    # Get the first level homogeneous clusters
    apChildren0 = get_homogenous_clusters_silhouette (apClusterNode0[0], config.Distance[0])
    apChildren1 = get_homogenous_clusters_silhouette (apClusterNode1[0], config.Distance[1])
    current_level_nodes = []
    current_level_tests = []    
    level_number = 1
      
    for a, b in itertools.product(apChildren0, apChildren1):
        try:
            data1 = a.pre_order(lambda x: x.id)
            data2 = b.pre_order(lambda x: x.id)
        except:
            data1 = reduce_tree(a)
            data2 = reduce_tree(b)
        tempTree = Hypothesis_Node(data=[data1, data2], left_distance=a.dist, right_distance=b.dist)
        tempTree.level_number = 1
        #current_level_tests.append(tempTree)
        current_level_nodes.append((tempTree, (a, b)))
        if config.write_hypothesis_tree:
            aLineOut = list(map(str, [str(level_number), str(';'.join([config.FeatureNames[0][i] for i in data1])), str(';'.join([config.FeatureNames[1][i] for i in data2]))]))
            csvwc.writerow(aLineOut)
    do_next_level = True
    while do_next_level  :
        current_level_tests = [ hypothesis for (hypothesis, _ ) in current_level_nodes]
        print ("--- Testing hypothesis level %s with %s hypotheses ... " % (level_number, len(current_level_tests)))
        temp_significant_hypotheses, temp_tested_hypotheses = significance_testing(current_level_tests, level = level_number)
        #print len(temp_tested_hypotheses)
        significant_hypotheses.extend(temp_significant_hypotheses)
        tested_hypotheses.extend(temp_tested_hypotheses)
        from_prev_hypothesis =  []
        from_prev_hypothesis_node = []
        do_next_level = False
        change_level_flag = True
        next_level = []
        leaf_nodes = []
        level_number += 1
        level_number_2 = 1
        for i in range(len(current_level_nodes)):
            hypothesis_node= current_level_nodes[i]
            (hypothesis, (a, b)) = hypothesis_node

            # Add leaves from current level to next level
            # Add significant or non significant hypothesis from previous level
            #if len(hypothesis.m_pData[0]) == 1 and  len(hypothesis.m_pData[1]) == 1 :
            bTauX = _is_stop(a) # currently if a is a tip 
            bTauY = _is_stop(b) # currently if b is a tip 
            if bTauX and  bTauY :
                # pass hypothesis tests with individual features to next levels 
                # to participate in FDR correction and increase the power 
                #Pairwise test is a test between clusters with only one feature
                if hypothesis.significance != True:
                    hypothesis.significance = None
                    leaf_nodes.append(hypothesis_node)
            elif hypothesis.significance != None:
                from_prev_hypothesis_node.append(hypothesis_node)
            else:
                #bTauX = _is_stop(a) # currently if a is a tip 
                #bTauY = _is_stop(b) # currently if b is a tip 
                do_next_level = True
                # Pair clusters between relevant levels for two hierarchies 
                #diam_A_r, diam_B_r = Node_clusters_diameter(hypothesis)
                #print(diam_A_r, diam_B_r)
                if cut_speed_1 != 1:# or diam_A_r > 2* diam_B_r:
                    if level_number  / cut_speed_1 > level_number_2:# or diam_A_r > 1.0 * diam_B_r :
                        if change_level_flag:
                            level_number_2 += 1
                            change_level_flag = False
                        if not bTauX:
                            #print (level_number  / cut_speed_1 , level_number, level_number_2)
                            apChildren0 = get_homogenous_clusters_silhouette(a,config.Distance[0])
                        else:
                           apChildren0 = [a] 
                    else:
                        apChildren0 = [a]
                else:
                    if not bTauX:
                        apChildren0 = get_homogenous_clusters_silhouette(a,config.Distance[0])
                    elif not bTauY:
                        apChildren0 = [a]
                    
                if cut_speed_2 != 1:# or diam_B_r > 2* diam_A_r:
                    if level_number  / cut_speed_2 > level_number_2:# or diam_B_r > 1.0* diam_A_r: 
                        if change_level_flag:
                            level_number_2 += 1
                            change_level_flag = False
                        if not bTauY:
                            #print (level_number  / cut_speed_2 , level_number, level_number_2)
                            apChildren1 = get_homogenous_clusters_silhouette(b,config.Distance[1])
                        else:
                            apChildren1 = [b]
                    else:
                        apChildren1 = [b]
                else:
                    if not bTauY:
                        apChildren1 = get_homogenous_clusters_silhouette(b,config.Distance[1])
                    elif not bTauX:
                        apChildren1 = [b]
                LChild = [(c1, c2) for c1, c2 in itertools.product(apChildren0, apChildren1)] 
                while LChild:
                    (a1, b1) = LChild.pop(0)
                    try:
                        data1 = a1.pre_order(lambda x: x.id)
                        data2 = b1.pre_order(lambda x: x.id)
                    except:
                        data1 = reduce_tree(a1)
                        data2 = reduce_tree(b1)
                    tempTree = Hypothesis_Node(data=[data1, data2], left_distance=a1.dist, right_distance=b1.dist)
                    tempTree.level_number = level_number
                    next_level.append((tempTree, (a1, b1)))
                    if config.write_hypothesis_tree:
                        aLineOut = list(map(str, [str(level_number), str(';'.join([config.FeatureNames[0][i] for i in data1])), str(';'.join([config.FeatureNames[1][i] for i in data2]))]))
                        csvwc.writerow(aLineOut)
        current_level_nodes = next_level
        #print ("--- number of new tests: %s leafs: %s  prev: %s" % (len(next_level), len(leaf_nodes), len(from_prev_hypothesis_node)) )
        if len(current_level_nodes) > 0:
            current_level_nodes.extend(leaf_nodes)
            current_level_nodes.extend(from_prev_hypothesis_node)
            
    print ("--- number of performed tests: %s") % (config.number_of_performed_tests)
    print ("--- number of passed tests after FDR controlling: %s" % len(significant_hypotheses))
    return significant_hypotheses, tested_hypotheses

def test_by_family(apClusterNode0, apClusterNode1, dataset1, dataset2, strMethod="uniform", strLinkage="min", robustness = None):
    
    func = config.similarity_method
    X, Y = dataset1, dataset2
    significant_hypotheses = []
    tested_hypotheses = []
    config.number_of_performed_tests = 0
    
    # Define the speed of cutting hierarchies
    # e.g. 1 means we cut in each iteration
    # e.g. 2 means we cut in each 2 iteration 
    n1 = apClusterNode0[0].get_count()
    n2 = apClusterNode1[0].get_count()
    n_pause = number_of_pause(n1, n2)
    #print ("Number of pause:", n_pause)
    n1 = apClusterNode0[0].get_count()
    n2 = apClusterNode1[0].get_count()
    if n1 > n2 :
        cut_speed_1 = 1
    else:
        #cut_speed_1 = max(math.log(n2,2)/math.log(n1,2) ,1) 
        cut_speed_1 = max(number_of_level(n2)*1.0/number_of_level(n1) ,1)
    if n2 > n1:
        cut_speed_2 = 1
    else:
        #cut_speed_2 = max(math.log(n1,2) / math.log(n2, 2),1)
        cut_speed_2 = max(number_of_level(n1)*1.0/number_of_level(n2) ,1)
    # Write the hypothesis that has been tested
    output_file_compared_clusters  = open(str(config.output_dir)+'/hypotheses_tree.txt', 'w')
    csvwc = csv.writer(output_file_compared_clusters , csv.excel_tab, delimiter='\t')
    csvwc.writerow(['Level', "Dataset 1", "Dataset 2" ])
    data1 = apClusterNode0[0].pre_order(lambda x: x.id)
    data2 = apClusterNode1[0].pre_order(lambda x: x.id)
    aLineOut = list(map(str, [str(0), str(';'.join([config.FeatureNames[0][i] for i in data1])), str(';'.join([config.FeatureNames[1][i] for i in data2]))]))
    csvwc.writerow(aLineOut)
    
    # Get the first level homogeneous clusters
    apChildren0 = get_homogenous_clusters_silhouette (apClusterNode0[0], config.Distance[0])
    apChildren1 = get_homogenous_clusters_silhouette (apClusterNode1[0], config.Distance[1])
    current_level_nodes = []
    current_level_tests = []    
    level_number = 1
      
    for a, b in itertools.product(apChildren0, apChildren1):
        try:
            data1 = a.pre_order(lambda x: x.id)
            data2 = b.pre_order(lambda x: x.id)
        except:
            data1 = reduce_tree(a)
            data2 = reduce_tree(b)
        tempTree = Hypothesis_Node(data=[data1, data2], left_distance=a.dist, right_distance=b.dist)
        tempTree.level_number = 1
        #current_level_tests.append(tempTree)
        current_level_nodes.append((tempTree, (a, b)))
        if config.write_hypothesis_tree:
            aLineOut = list(map(str, [str(level_number), str(';'.join([config.FeatureNames[0][i] for i in data1])), str(';'.join([config.FeatureNames[1][i] for i in data2]))]))
            csvwc.writerow(aLineOut)
    do_next_level = True
    
    while do_next_level  :
        next_level_nodes =[]
        if level_number ==1 :
            print ("--- Testing hypothesis level %s with %s hypotheses ... " % (level_number, len(current_level_nodes)))
            current_level_tests = [ hypothesis for (hypothesis, _ ) in current_level_nodes]
            temp_significant_hypotheses, temp_tested_hypotheses = significance_testing(current_level_tests)
            significant_hypotheses.extend(temp_significant_hypotheses)
            tested_hypotheses.extend(temp_tested_hypotheses)
            for i in range(len(current_level_tests)):
                if current_level_tests[i] not in significant_hypotheses:
                    next_level_nodes.append(current_level_nodes[i])
        else:
            next_level_nodes =   current_level_nodes 
            
        current_level_nodes = []     
        do_next_level = False
        change_level_flag = True
        current_family_tests = []
        next_set = []
        level_number += 1
        level_number_2 = 1
        for i in range(len(next_level_nodes)):
            current_family_tests = []
            next_set = []
            hypothesis_node= next_level_nodes[i]
            (hypothesis, (a, b)) = hypothesis_node
            if hypothesis in significant_hypotheses:
                continue
            # Add leaves from current level to next level
            # Add significant or non significant hypothesis from previous level
            #if len(hypothesis.m_pData[0]) == 1 and  len(hypothesis.m_pData[1]) == 1 :
            bTauX = _is_stop(a) # currently if a is a tip 
            bTauY = _is_stop(b) # currently if b is a tip 
            if bTauX and  bTauY :
                continue
            else:
                do_next_level = True
                
                if cut_speed_1 != 1:# or diam_A_r > 2* diam_B_r:
                    if level_number  / cut_speed_1 > level_number_2:# or diam_A_r > 1.0 * diam_B_r :
                        if change_level_flag:
                            level_number_2 += 1
                            change_level_flag = False
                        if not bTauX:
                            #print (level_number  / cut_speed_1 , level_number, level_number_2)
                            apChildren0 = get_homogenous_clusters_silhouette(a,config.Distance[0])
                        else:
                           apChildren0 = [a] 
                    else:
                        apChildren0 = [a]
                else:
                    if not bTauX:
                        apChildren0 = get_homogenous_clusters_silhouette(a,config.Distance[0])
                    elif not bTauY:
                        apChildren0 = [a]
                    
                if cut_speed_2 != 1:# or diam_B_r > 2* diam_A_r:
                    if level_number  / cut_speed_2 > level_number_2:# or diam_B_r > 1.0* diam_A_r: 
                        if change_level_flag:
                            level_number_2 += 1
                            change_level_flag = False
                        if not bTauY:
                            #print (level_number  / cut_speed_2 , level_number, level_number_2)
                            apChildren1 = get_homogenous_clusters_silhouette(b,config.Distance[1])
                        else:
                            apChildren1 = [b]
                    else:
                        apChildren1 = [b]
                else:
                    if not bTauY:
                        apChildren1 = get_homogenous_clusters_silhouette(b,config.Distance[1])
                    elif not bTauX:
                        apChildren1 = [b]
                LChild = [(c1, c2) for c1, c2 in itertools.product(apChildren0, apChildren1)] 
                while LChild:
                    (a1, b1) = LChild.pop(0)
                    try:
                        data1 = a1.pre_order(lambda x: x.id)
                        data2 = b1.pre_order(lambda x: x.id)
                    except:
                        data1 = reduce_tree(a1)
                        data2 = reduce_tree(b1)
                    tempTree = Hypothesis_Node(data=[data1, data2], left_distance=a1.dist, right_distance=b1.dist)
                    tempTree.level_number = level_number
                    current_family_tests.append(tempTree)
                    next_set.append((tempTree, (a1, b1)))
                    if config.write_hypothesis_tree:
                        aLineOut = list(map(str, [str(level_number), str(';'.join([config.FeatureNames[0][i] for i in data1])), str(';'.join([config.FeatureNames[1][i] for i in data2]))]))
                        csvwc.writerow(aLineOut)
                temp_significant_hypotheses, temp_tested_hypotheses = significance_testing(current_family_tests)
                significant_hypotheses.extend(temp_significant_hypotheses)
                for h in range(len(current_family_tests)):
                    if current_family_tests[h] not in significant_hypotheses:
                        current_level_nodes.append(next_set[h])
                current_family_tests = []
                next_set = []
        #current_level_nodes = next_set

    print ("--- number of performed tests: %s" % config.number_of_performed_tests)
    print ("--- number of passed tests after FDR controlling: %s" % len(significant_hypotheses))
    return significant_hypotheses, tested_hypotheses



pHashMethods = {"permutation" : stats.permutation_test,
                        "permutation_test_by_medoid": stats.permutation_test_by_medoid,
                        
                        # parametric tests
                        "parametric_test_by_pls_pearson": stats.parametric_test_by_pls_pearson,
                        "parametric_test_by_representative": stats.parametric_test_by_representative,
                        "parametric_test" : stats.parametric_test,
                        
                        # G-Test
                        "g-test":stats.g_test
                        }

strMethod = config.randomization_method
pMethod = pHashMethods[strMethod]
def estimate_pvalue(pNode):
    dataset1 = config.parsed_dataset[0]
    dataset2 = config.parsed_dataset[1]
    """
    Performs a certain action at the node
    
        * E.g. compares two bags, reports distance and p-values 
    """
    aIndicies = pNode.m_pData 
    aIndiciesMapped = list(map(array, aIndicies))  # So we can vectorize over numpy arrays
    X = dataset1[aIndiciesMapped[0]]
    Y = dataset2[aIndiciesMapped[1]]
    est_pvalue, similarity, left_first_rep_variance, right_first_rep_variance, left_loading, right_loading, left_rep, right_rep = pMethod(X, Y)
    pNode.similarity_score = similarity
    return est_pvalue        
def naive_all_against_all():
    dataset1 = config.parsed_dataset[0]
    dataset2 = config.parsed_dataset[1]
    p_adjusting_method = config.p_adjust_method
    decomposition = config.decomposition
    method = config.randomization_method
    metric = config.similarity_method
    iIter= config.iterations
    discretize_style = config.strDiscretizing
    
    iRow = len(dataset1)
    iCol = len(dataset2)
    
    tested_hypotheses = [] 
    significant_hypotheses = []
    tests = []
    for i, j in itertools.product(list(range(iRow)), list(range(iCol))):
        test =  Hypothesis_Node(left_distance=0.0, right_distance=0.0)
        data = [[i], [j]]
        test = add_data(test, data)
        tests.append(test)
    
    p_values = multiprocessing_estimate_pvalue(estimate_pvalue, tests, pMethod, dataset1, dataset2)
    aP_adjusted, pRank = stats.p_adjust(p_values, config.q)
    q_values = stats.pvalues2qvalues (p_values, adjusted=True)
    for i in range(len(tests)):
        tests[i].pvalue = p_values[i]
        tests[i].qvalue = q_values[i]
        tests[i].rank = pRank[i]
    def _get_passed_fdr_tests():
        if p_adjusting_method in ["bh", "by"]:
            max_r_t = 0
            for i in range(len(tests)):
                if tests[i].pvalue <= aP_adjusted[i] and max_r_t <= tests[i].rank:
                    max_r_t = tests[i].rank
            for i in range(len(tests)):
                if tests[i].rank <= max_r_t:
                    tested_hypotheses.append(tests[i])
                    significant_hypotheses.append(tests[i])
                    tests[i].significance = True
                    print ("-- association after %s fdr correction" % p_adjusting_method)
                else:
                    tests[i].significance = False
                    tested_hypotheses.append(tests[i])
        elif p_adjusting_method == "bonferroni":
            for i in range(len(tests)):
                if tests[i].pvalue <= aP_adjusted[i]:
                    tested_hypotheses.append(tests[i])
                    significant_hypotheses.append(tests[i])
                    tests[i].significance = True
                    print ("-- association after %s fdr correction" % p_adjusting_method)
                else:
                    tests[i].significance = False
                    tested_hypotheses.append(tests[i])
        elif p_adjusting_method == "no_adjusting":
            for i in range(len(tests)):
                if tests[i].pvalue <= config.q:
                    tested_hypotheses.append(tests[i])
                    significant_hypotheses.append(tests[i])
                    tests[i].significance = True
                    print ("-- association after %s fdr correction" % p_adjusting_method)
                else:
                    tests[i].significance = False
                    tested_hypotheses.append(tests[i])
    _get_passed_fdr_tests()
    config.number_of_performed_tests =len(tested_hypotheses)
    print("--- number of performed tests: %s" % config.number_of_performed_tests)
    print("--- number of passed tests after FDR controlling: %s "%len(significant_hypotheses)) 
    return significant_hypotheses, tested_hypotheses


def significance_testing(current_level_tests, level = None):
    tested_hypotheses = []  # # Full log 
    significant_hypotheses = []  # # Only the final reported values 
    dataset1 = config.parsed_dataset[0]
    dataset2 = config.parsed_dataset[1]

    # Calculate p-value for all hypotheses
    p_values = multiprocessing_estimate_pvalue(estimate_pvalue, current_level_tests, pMethod, dataset1, dataset2)
    # Calculate q-value for all hypotheses
    q_values = stats.pvalues2qvalues (p_values, adjusted=True)
    aP_adjusted, pRank = stats.p_adjust(p_values, config.q)#config.q)
    
    for i in range(len(current_level_tests)):
       current_level_tests[i].rank = pRank[i]
       current_level_tests[i].pvalue = p_values[i]
       current_level_tests[i].already_tested = True
       if current_level_tests[i].significance == None: 
           current_level_tests[i].qvalue = q_values[i]
   
    max_r_t = 0
    for i in range(len(current_level_tests)):
        if current_level_tests[i].pvalue <= aP_adjusted[i] and max_r_t <= current_level_tests[i].rank:
             max_r_t = current_level_tests[i].rank
            #print "max_r_t", max_r_t
    hsci_within_pvalues = []
    hsci_between_pvalues = []
    hsci_between_significant =[]
    passed_tests = []
    if config.p_adjust_method in ["bh", "by"]:
        for i in range(len(current_level_tests)):
            if current_level_tests[i].rank <= max_r_t:
                if current_level_tests[i].significance == None:
                    current_level_tests[i].significance = True
                    tested_hypotheses.append(current_level_tests[i])
                    significant_hypotheses.append(current_level_tests[i])
                    print ("-- association after %s fdr correction" % config.p_adjust_method)
                    #hsci_between_significant.append('Significant')
            else:
                if current_level_tests[i].significance == None and is_bypass(current_level_tests[i]):
                    current_level_tests[i].significance = False
                    tested_hypotheses.append(current_level_tests[i])
                #hsci_between_significant.append('Not significant')
           
        '''print current_level_tests[i].m_pData
            if len(current_level_tests[i].m_pData[0]) > 1:
                hsci_within_pvalues.append(HSIC.HSIC_pval(config.parsed_dataset[0][current_level_tests[i].m_pData[0]].T,\
                                                          config.parsed_dataset[1][current_level_tests[i].m_pData[0]].T)[1])
            if len(current_level_tests[i].m_pData[1]) > 1:
                hsci_within_pvalues.append(HSIC.HSIC_pval(config.parsed_dataset[0][current_level_tests[i].m_pData[1]].T,\
                                                          config.parsed_dataset[1][current_level_tests[i].m_pData[1]].T)[1])
            if len(current_level_tests[i].m_pData[0]) > 1 and len(current_level_tests[i].m_pData[1]) > 1:
                hsci_between_pvalues.append(HSIC.HSIC_pval(config.parsed_dataset[0][current_level_tests[i].m_pData[0]].T,\
                                                          config.parsed_dataset[1][current_level_tests[i].m_pData[1]].T)[1])
            
        with open("clusters_pvalues.txt", "a") as text_file:
            text_file.write("Level" + "\t" + "pvalue" + "\t" + 'Category' + "\t" + "Significance" + "\n")
            for i in range(len(hsci_within_pvalues)):
                text_file.write(str(level) + "\t" + str(hsci_within_pvalues[i]) + "\t" + 'Within cluster' + "\t" + "Homogeneous" + "\n")
        with open("clusters_pvalues.txt", "a") as text_file:
            for i in range(len(hsci_between_pvalues)):
                text_file.write(str(level) + "\t" + str(hsci_between_pvalues[i]) + "\t" + 'Between cluster' + "\t" + str(hsci_between_significant[i]) + "\n")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1)
        ax.hist(hsci_within_pvalues, normed=True, histtype='stepfilled', alpha=0.2)
        ax.legend(loc='best', frameon=False)
        plt.savefig("hsci_within_pvalues_" + str(level)+".pdf")
        plt.show()
        #exit()'''
               
    elif config.p_adjust_method == "bonferroni":
        for i in range(len(current_level_tests)):
            if current_level_tests[i].pvalue <= aP_adjusted[i]:
                if current_level_tests[i].significance == None:
                    tested_hypotheses.append(current_level_tests[i])
                    significant_hypotheses.append(current_level_tests[i])
                    current_level_tests[i].significance = True
                    print ("-- association after %s fdr correction" % config.p_adjust_method)
                    
            else:
                if current_level_tests[i].significance == None and is_bypass(current_level_tests[i]):
                    current_level_tests[i].significance = False
                    tested_hypotheses.append(current_level_tests[i])
    elif config.p_adjust_method == "no_adjusting":
        for i in range(len(current_level_tests)):
            if current_level_tests[i].pvalue <= config.q:
                if current_level_tests[i].significance == None:
                    tested_hypotheses.append(current_level_tests[i])
                    significant_hypotheses.append(current_level_tests[i])
                    current_level_tests[i].significance = True
                    print ("-- association after %s fdr correction" % config.p_adjust_method)
                    
            else:
                if current_level_tests[i].significance == None and is_bypass(current_level_tests[i]):
                    current_level_tests[i].significance = False
                    tested_hypotheses.append(current_level_tests[i])
                    temp_sub_hypotheses = get_children(current_level_tests[i])
                    if len(temp_sub_hypotheses) > 0:
                        for j in range(len(temp_sub_hypotheses)):
                            temp_sub_hypotheses[j].pvalue = current_level_tests[i].pvalue
    return significant_hypotheses, tested_hypotheses

