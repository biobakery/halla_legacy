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
from scipy.cluster.hierarchy import linkage, to_tree, leaves_list
from scipy.spatial.distance import pdist, squareform
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas

from . import distance
from . import stats
from . import plot
from . import config
from . import logger
from __builtin__ import True
from matplotlib.sankey import RIGHT
from itertools import product, combinations
from unicodedata import decomposition
from math import fabs
from profile import Stats
sys.setrecursionlimit(20000)

# Multi-threading section
def multi_pMethod(args):
    """
    Runs the pMethod function and returns the results plus the id of the node
    """
    
    id, pMethod, dataset1, dataset2 = args
    dP, similarity, left_first_rep_variance, right_first_rep_variance, 
    left_loading, right_loading, left_rep, right_rep = pMethod(dataset1, dataset2)

    return id, dP, similarity, left_first_rep_variance, right_first_rep_variance,\
         left_loading, right_loading, left_rep, right_rep

def multiprocessing_actor(_actor, current_level_tests, pMethod, dataset1, dataset2):
    """
    Return the results from applying the data to the actor function
    """
    def _multi_pMethod_args(current_level_tests, pMethod, dataset1, dataset2, ids_to_process):
        for id in ids_to_process:
            aIndicies = current_level_tests[id].m_pData
            aIndiciesMapped = map(array, aIndicies)
            yield [id, pMethod, dataset1[aIndiciesMapped[0]], dataset2[aIndiciesMapped[1]]]
    
    if config.NPROC > 1:
        import multiprocessing
        pool = multiprocessing.Pool(config.NPROC)
        
        # check for tests that already have pvalues as these do not need to be recomputed
        ids_to_process=[]
        result = [0] * len(current_level_tests)
        for id in xrange(len(current_level_tests)):
            if current_level_tests[id].significance != None:
                result[id]=current_level_tests[id].pvalue
            else:
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
        for i in xrange(len(current_level_tests)):
            if current_level_tests[i].significance != None:
                result.append(current_level_tests[i].pvalue)
            else: 
                result.append(_actor(current_level_tests[i]))

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

def pop(node):
    # pop one of the children, else return none, since this amounts to killing the singleton 
    if node.m_arrayChildren:
        return node.m_arrayChildren.pop()

def l(node):
    return node.left()

def r(node):
    return node.right()

def left(node):
    return node.get_child(iIndex=0)

def right(node):
    return node.get_child(iIndex=1)

def is_leaf(node):
    return bool(not(node.m_pData and node.m_arrayChildren))

def is_degenerate(node):
    return (not(node.m_pData) and not(node.m_arrayChildren))            

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
        print "===================stop and reject check========================"
        #print "Left Exp. Var.: ", Node.left_first_rep_variance
        print "Left before: ", Node.m_pData[0]
        #print "Right Exp. Var.: ", Node.right_first_rep_variance
        print "Right before: ", Node.m_pData[1]
        print "dime_A_r: ", diam_A_r,"  ", "dime_B_r: ", diam_B_r, "diam_Ar_Br: ", diam_Ar_Br
    if diam_A_r + diam_B_r == 0:
        return True
    if diam_Ar_Br > diam_A_r + diam_B_r:
        return True
    else:
        return False

def is_bypass(Node ):
    if config.apply_stop_condition:
        return stop_decesnding_silhouette_coefficient(Node)
        if stop_and_reject(Node):
            if config.verbose == 'DEBUG':
                print "q: ", Node.qvalue, " p: ", Node.pvalue
            return True
        else:
            return False
    return False

def report(Node):
    print "\n--- hypothesis test based on permutation test"        
    print "---- pvalue                        :", Node.pvalue
    #if Node.qvalue <> 0.0:
    #    print "--- adjusted pvalue     :", Node.qvalue
    print "---- similarity_score score              :", self.similarity_score
    print "---- first cluster's features      :", Node.m_pData[0]
    print "---- first cluster similarity_score      :", 1.0 - Node.left_distance
    print "---- second cluster's features     :", Node.m_pData[1]
    print "---- second cluster similarity_score     :", 1.0 - Node.right_distance

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
    bTree=True
    linkage_method = 'single'
    D = pdist(dataset, metric=distance.pDistance) 
    config.Distance[dataset_number] =  copy.deepcopy(squareform(D))
    if config.diagnostics_plot:
        print "--- plotting heatmap for Dataset", str(dataset_number)," ... "
        Z = plot.heatmap(data_table = dataset , D = D, xlabels_order = [], xlabels = labels, filename= config.output_dir+"/"+"hierarchical_heatmap_"+str(config.similarity_method)+"_" + str(dataset_number), method =linkage_method, dataset_number= None)
    else:
        Z = linkage(D, method= linkage_method)
    import scipy.cluster.hierarchy as sch
    logger.write_table(data=config.Distance[dataset_number], name=config.output_dir+'/Distance_matrix'+str(dataset_number)+'.tsv', rowheader=config.FeatureNames[dataset_number], colheader=config.FeatureNames[dataset_number])
    return to_tree(Z) if (bTree and len(dataset)>1) else Z, sch.dendrogram(Z, orientation='right')['leaves'] if len(dataset)>1 else sch.dendrogram(Z)['leaves']

def dendrogram(Z):
    return scipy.cluster.hierarchy.dendrogram(Z)

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
        return truncate_tree(filter(lambda x: bool(x), [(p.right if p.right else None) for p in apClusterNode]) \
            + filter(lambda x: bool(x), [(q.left if p.left else None) for q in apClusterNode]), level=iLevel + 1, skip=iSkip) 

    elif iSkip == iLevel:
        if any(apClusterNode):
            return filter(lambda x: bool(x), apClusterNode)
    
        else:
            return []
            # print "truncated tree is malformed--empty!"
            raise Exception("truncated tree is malformed--empty!")

def reduce_tree(pClusterNode, pFunction=lambda x: x.id, aOut=[]):
    """
    Recursive

    Input: pClusterNode, pFunction = lambda x: x.id, aOut = []

    Output: a list of pFunction calls (node ids by default)

    Should be designed to handle both ClusterNode and Hypothesis_Node types 
    """ 

    bTree = is_tree(pClusterNode)

    func = pFunction if not bTree else lambda x: x.m_pData 

    if pClusterNode:

        if not bTree:
            if pClusterNode.is_leaf():
                return (aOut + [func(pClusterNode)])
            else:
                return reduce_tree(pClusterNode.left, func, aOut) + reduce_tree(pClusterNode.right, func, aOut) 
        elif bTree:
            if pClusterNode.is_leaf():
                return (aOut + [func(pClusterNode)])
            else:
                pChildren = pClusterNode.get_children()
                iChildren = len(pChildren)
                return reduce(lambda x, y: x + y, [reduce_tree(pClusterNode.get_child(i), func, aOut) for i in range(iChildren)], [])
    else:
        return [] 

def reduce_tree_by_layer(apParents, iLevel=0, iStop=None):
    """

    Traverse one tree. 

    Input: apParents, iLevel = 0, iStop = None

    Output: a list of (iLevel, list_of_nodes_at_iLevel)

        Ex. 

        [(0, [0, 2, 6, 7, 4, 8, 9, 5, 1, 3]),
         (1, [0, 2, 6, 7]),
         (1, [4, 8, 9, 5, 1, 3]),
         (2, [0]),
         (2, [2, 6, 7]),
         (2, [4]),
         (2, [8, 9, 5, 1, 3]),
         (3, [2]),
         (3, [6, 7]),
         (3, [8, 9]),
         (3, [5, 1, 3]),
         (4, [6]),
         (4, [7]),
         (4, [8]),
         (4, [9]),
         (4, [5]),
         (4, [1, 3]),
         (5, [1]),
         (5, [3])]

    """

    apParents = list(apParents)
    apParents = filter(bool, apParents)

    bTree = False 
    
    if not isinstance(apParents, list):
        bTree = is_tree(apParents)
        apParents = [apParents]
    else:
        try:
            bTree = is_tree(apParents[0])
        except IndexError:
            pass 

    if (iStop and (iLevel > iStop)) or not(apParents):
        return [] 
    else:
        filtered_apParents = filter(lambda x: not(is_leaf(x)) , apParents)
        new_apParents = [] 
        for q in filtered_apParents:
            if not bTree:
                new_apParents.append(q.left); new_apParents.append(q.right)
            else:
                for item in get_children(q):
                    new_apParents.append(item)
        if not bTree:
            return [(iLevel, reduce_tree(p)) for p in apParents ] + reduce_tree_by_layer(new_apParents, iLevel=iLevel + 1)
        else:
            return [(iLevel, p.m_pData) for p in apParents ] + reduce_tree_by_layer(new_apParents, iLevel=iLevel + 1)
#-------------------------------------#
# Threshold Helpers                   #
#-------------------------------------# 

def _min_tau(X, func):
    X = numpy.array(X) 
    D = stats.discretize(X)
    A = numpy.array([func(D[i], D[j]) for i, j in itertools.combinations(range(len(X)), 2)])

    # assert(numpy.any(A))

    if X.shape[0] < 2:
        return numpy.min([func(D[0], D[0])])

    else:
        return numpy.min(A)

def _max_tau(X, func):
    X = numpy.array(X) 
    D = stats.discretize(X)
    A = numpy.array([func(D[i], D[j]) for i, j in itertools.combinations(range(len(X)), 2)])

    # assert(numpy.any(A))

    if X.shape[0] < 2:
        return numpy.max([func(D[0], D[0])])

    else:
        return numpy.max(A)

def _mean_tau(X, func):
    X = numpy.array(X) 
    D = stats.discretize(X)
    A = numpy.array([func(D[i], D[j]) for i, j in itertools.combinations(range(len(X)), 2)])

    if X.shape[0] < 2:
        return numpy.mean([func(D[0], D[0])])

    else:
        return numpy.mean(A)

#-------------------------------------#
# Decider Functions                   #
#-------------------------------------#

def _filter_true(x):
    return filter(lambda y: bool(y), x)

def _decider_min(node1, node2):
    return (not(_filter_true([node1.left, node1.right])) or not(_filter_true([node2.left, node2.right])))

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

def _is_stop(ClusterNode, dataSet, max_dist_cluster):
        if ClusterNode.is_leaf() or ClusterNode.get_count() == 1:
            return True
        else:
            return False
        
def _cutree_to_log2 (apNode, X, func, distance, cluster_threshold):
    temp_apChildren = []
    temp_sub_apChildren = []
    print "Length of ", len(apNode)
    for node in apNode:
        n = node.get_count()
        print "Number of feature in node: ", n
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
    # print "Number of sub-clusters: ", len(set(temp_apChildren))
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
                #print largest_node.get_count()
                largest_node = sub_clusters[i]
                index = i
        if largest_node.get_count() > (n_features/number_of_estimated_clusters):
            #sub_clusters.remove(largest_node)
            #sub_clusters = sub_clusters[:index] + sub_clusters[index+1 :]
            del sub_clusters[index]
            sub_clusters += truncate_tree([largest_node], level=0, skip=1)
            #print "first cut", [a.pre_order(lambda x: x.id) for a in sub_clusters ]
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
    #print distances
    while True:# not all(val <= t for val in distances):
        max_dist_node = sub_clusters[0]
        index = 0
        for i in range(len(sub_clusters)):
            #if sub_clusters[i].dist > 0.0:
                #aDist += [sub_clusters[i].dist]
            if max_dist_node.dist < sub_clusters[i].dist:
                max_dist_node = sub_clusters[i]
                index = i
        # print "Max Distance in this level", _percentage(max_dist_node.dist)
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
        # print "Max Distance in this level", _percentage(max_dist_node.dist)
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
            #print 'before', all_a_clusters
            temp_all_a_clusters.remove(a_cluster)
            #print 'after', all_a_clusters
            a = np.mean([1.0 - config.Distance[dataset_number][i][j] for i,j in product([a_cluster], temp_all_a_clusters)])            
        b = np.mean([1.0 - config.Distance[dataset_number][i][j] for i,j in product([a_cluster], all_b_clusters)])
        s = (b-a)/max([a,b])
        #print 's a', s, a, b
        s_all_a.append(s)
    if any(val <= 0.0 for val in s_all_a) and not len(s_all_a) == 1:
        return True
    #print "silhouette_coefficient a", np.mean(s_all_a)
    #print "child _a", all_a_clusters, " b_child", all_b_clusters 
    for b_cluster in all_b_clusters:
        if len(all_b_clusters) ==1:
            
            a = np.mean([1.0 - math.fabs(pMe(config.parsed_dataset[dataset_number][i], config.parsed_dataset[dataset_number][j])) for i,j in product([b_cluster], all_b_clusters)])
        else:
            temp_all_b_clusters = all_b_clusters[:]#deepcopy(all_a_clusters)
            #print 'before', all_a_clusters
            temp_all_b_clusters.remove(b_cluster)
            #print 'after', all_a_clusters
            a = np.mean([1.0 - math.fabs(pMe(config.parsed_dataset[dataset_number][i], config.parsed_dataset[dataset_number][j])) for i,j in product([b_cluster], temp_all_b_clusters)])            
        b = np.mean([1.0 -  math.fabs(pMe(config.parsed_dataset[dataset_number][i], config.parsed_dataset[dataset_number][j])) for i,j in product([b_cluster], all_a_clusters)])
        s = (b-a)/max([a,b])
        #print 's b', s
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
        #cluster_b = [val for val in range(len(config.Distance[dataset_number])) if val not in cluster_a]
        #print cluster_a, cluster_b
        if i%2 == 0 and i<len(clusters)-1:
            next_cluster = clusters[i+1].pre_order(lambda x: x.id)
        else:
            next_cluster = clusters[i-1].pre_order(lambda x: x.id)
        #all_features = [a for a in range(len(distance_matrix))] 
        #cluster_b = [item for item in all_features if item not in cluster_a]  
         
        if i%2 != 0 and i> 0:
            prev_cluster = clusters[i-1].pre_order(lambda x: x.id)
        elif i < len((clusters))-1:
            prev_cluster = clusters[i+1].pre_order(lambda x: x.id)
        else: 
            prev_cluster = clusters[i-1].pre_order(lambda x: x.id)
        #next_cluster = [clusters[num].pre_order(lambda x: x.id) for num in range(i+1, len(clusters)) if clusters[num].pre_order(lambda x: x.id)>1 ]
        #prev_cluster = [clusters[num].pre_order(lambda x: x.id) for num in range(0, i-1) if clusters[num].pre_order(lambda x: x.id)>1 and i>0  ]
        #print next_cluster, cluster_a
        #silhouette_score.append(silhouette_coefficient(cluster))
        s_all_a = []
        for a_feature in cluster_a:
            if len(cluster_a) ==1:
                a = 0.0
            else:
                temp_a_features = cluster_a[:]#deepcopy(all_a_clusters)
                #print 'before', temp_a_features
                temp_a_features.remove(a_feature)
                a = np.mean([distance_matrix.iloc[i, j] for i,j in product([a_feature], temp_a_features)])            
            b1 = np.mean([ distance_matrix.iloc[i, j] for i,j in product([a_feature], next_cluster)])
            b2 = np.mean([ distance_matrix.iloc[i, j] for i,j in product([a_feature], prev_cluster)])
            b = min(b1,b2)
            s = (b-a)/max([a,b])
            #print 's a', s, a, b
            s_all_a.append(s)
        silhouette_scores.append(np.mean(s_all_a))
    return silhouette_scores

def get_medoid(features, distance_matrix):
    med = features[0]#max(distance_matrix)
    #print features#, distance_matrix.iloc[features[0]]
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
    #print wss
    avgWithinSS = np.sum(wss) #[sum(d)/X.shape[0] for d in dist]
    #print avgWithinSS 
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
    #print wss[1], TSS
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
    print "The best guess for the number of clusters is: ", best_clust_size
    return  best_clust_size 
def predict_best_number_of_clusters(hierarchy_tree, distance_matrix):
    #distance_matrix = pandas.DataFrame(distance_matrix)
    features = get_leaves(hierarchy_tree)
    clusters= [] #[hierarchy_tree]
    min_num_cluster = 2  
    max_num_cluster = int(math.ceil(math.log(len(features), 2)))
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
        #print best_sil_score, best_clust_size
                
    print "The best guess for the number of clusters is: ", best_clust_size
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
        sub_clusters = cutree_to_get_number_of_features(cluster, distance_matrix, number_of_estimated_clusters= number_of_estimated_clusters)
    sub_silhouette_coefficient = silhouette_coefficient(sub_clusters, distance_matrix) 
    while True:
        min_silhouette_node = sub_clusters[0]
        min_silhouette_node_index = 0

        for i in range(len(sub_clusters)):
            if sub_silhouette_coefficient[min_silhouette_node_index] > sub_silhouette_coefficient[i]:
                min_silhouette_node = sub_clusters[i]
                min_silhouette_node_index = i
        if sub_silhouette_coefficient[min_silhouette_node_index] == 1.0:
            break
        sub_clusters_to_add = truncate_tree([min_silhouette_node], level=0, skip=1)#cutree_to_get_number_of_clusters([min_silhouette_node])##
        if len(sub_clusters_to_add) < 2:
            break
        sub_silhouette_coefficient_to_add = silhouette_coefficient(sub_clusters_to_add, distance_matrix)
        temp_sub_silhouette_coefficient_to_add = sub_silhouette_coefficient_to_add[:]
        temp_sub_silhouette_coefficient_to_add = [value for value in temp_sub_silhouette_coefficient_to_add if value != 1.0]

        if len(temp_sub_silhouette_coefficient_to_add) == 0 or sub_silhouette_coefficient[min_silhouette_node_index] >= np.max(temp_sub_silhouette_coefficient_to_add) :
            sub_silhouette_coefficient[min_silhouette_node_index] =  1.0
        else:
            del sub_clusters[min_silhouette_node_index]#min_silhouette_node)
            del sub_silhouette_coefficient[min_silhouette_node_index]
            sub_silhouette_coefficient.extend(sub_silhouette_coefficient_to_add)
            sub_clusters.extend(sub_clusters_to_add)
   
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
    #cutree_to_get_number_of_features(apClusterNode0[0])
    #get_homogenous_clusters_silhouette (apClusterNode0[0], dataset_number = 0)
    apChildren1 = get_homogenous_clusters_silhouette (apClusterNode1[0], config.Distance[1])
    #cutree_to_get_number_of_features(apClusterNode[0])
    #
    #print "first cut", [a.pre_order(lambda x: x.id) for a in apChildren0]
    #print "first cut", [a.pre_order(lambda x: x.id) for a in apChildren1]
    
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
    #print "child list:", childList
    next_L = []
    level_number = 2
    while L:
        (pStump, (a, b)) = L.pop(0)
        #print "child list:", tempNode
        #continue
        try:
            data1 = a.pre_order(lambda x: x.id)
            data2 = b.pre_order(lambda x: x.id)
        except:
            data1 = reduce_tree(a)
            data2 = reduce_tree(b)        
        bTauX = _is_stop(a, X, max_dist_cluster1)  # ( _min_tau(X[array(data1)], func) >= x_threshold ) ### parametrize by mean, min, or max
        bTauY = _is_stop(b, Y, max_dist_cluster2)  # ( _min_tau(Y[array(data2)], func) >= y_threshold ) ### parametrize by mean, min, or max
        if bTauX and bTauY :
            #print"leaf both"
            if L:
                continue
            else:
                #print "*****Finished coupling for level: ", level_number
                if next_L:
                    L = next_L
                    next_L = []
                    level_number += 1
                    #print "******************len: ",len(L)
                continue
        if not bTauX:
            apChildren0 = get_homogenous_clusters_silhouette(a,config.Distance[0])
            #cutree_to_get_number_of_clusters([a])
            #cutree_to_get_number_of_features(a)
            #get_homogenous_clusters_silhouette(a,0)
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
            #print "*****Finished coupling for level: ", level_number
            if next_L:
                L = next_L
                next_L = []
                level_number += 1
                #print "******************len: ",len(L)
    #print "Coupled Hypothesis_Node", reduce_tree_by_la
    #print "Number of levels after coupling", level_number-1
    return [Hypothesis_Tree_Root]
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
def _actor(pNode):
    dataset1 = config.parsed_dataset[0]
    dataset2 = config.parsed_dataset[1]
    """
    Performs a certain action at the node
    
        * E.g. compares two bags, reports distance and p-values 
    """
    aIndicies = pNode.m_pData 
    aIndiciesMapped = map(array, aIndicies)  # # So we can vectorize over numpy arrays
    X = dataset1[aIndiciesMapped[0]]
    Y = dataset2[aIndiciesMapped[1]]
    dP, similarity, left_first_rep_variance, right_first_rep_variance, left_loading, right_loading, left_rep, right_rep = pMethod(X, Y)
    pNode.similarity_score = similarity
    return dP        
def naive_all_against_all():
    dataset1 = config.parsed_dataset[0]
    dataset2 = config.parsed_dataset[1]
    p_adjusting_method = config.p_adjust_method
    decomposition = config.decomposition
    method = config.randomization_method
    metric = config.similarity_method
    fQ = config.q
    iIter= config.iterations
    discretize_style = config.strDiscretizing
    
    iRow = len(dataset1)
    iCol = len(dataset2)
    
    aOut = [] 
    aFinal = []
    aP = []
    tests = []
    passed_tests = []
    #print iRow, iCol
    for i, j in itertools.product(range(iRow), range(iCol)):
        test =  Hypothesis_Node(left_distance=0.0, right_distance=0.0)
        data = [[i], [j]]
        test = add_data(test, data)
        tests.append(test)
    
    p_values = multiprocessing_actor(_actor, tests, pMethod, dataset1, dataset2)
    cluster_size = [1 for i in range(len(p_values))]
    aP_adjusted, pRank = stats.p_adjust(p_values, config.q, cluster_size )
    #print aP_adjusted, pRank
    q_values = stats.pvalues2qvalues (p_values, adjusted=True)
    for i in range(len(tests)):
        tests[i].pvalue = p_values[i]
        tests[i].qvalue = q_values[i]
        tests[i].rank = pRank[i]
    def _get_passed_fdr_tests():
        if p_adjusting_method in ["bh", "bhy"]:
            max_r_t = 0
            for i in range(len(tests)):
                if tests[i].pvalue <= aP_adjusted[i] and max_r_t <= tests[i].rank:
                    max_r_t = tests[i].rank
                    #print tests[i].rank
            for i in range(len(tests)):
                if tests[i].rank <= max_r_t:
                    passed_tests.append(tests[i])
                    aOut.append(tests[i])
                    aFinal.append(tests[i])
                    tests[i].significance = True
                    print ("-- association after %s fdr correction" % p_adjusting_method)
                else:
                    tests[i].significance = False
                    aOut.append(tests[i])
        elif p_adjusting_method == "bonferroni":
            for i in range(len(tests)):
                if tests[i].pvalue <= aP_adjusted[i]:
                    passed_tests.append(tests[i])
                    aOut.append(tests[i])
                    aFinal.append(tests[i])
                    tests[i].significance = True
                    print ("-- association after %s fdr correction" % p_adjusting_method)
                else:
                    tests[i].significance = False
                    aOut.append(tests[i])
        elif p_adjusting_method == "no_adjusting":
            for i in range(len(tests)):
                if tests[i].pvalue <= fQ:
                    passed_tests.append(tests[i])
                    aOut.append(tests[i])
                    aFinal.append(tests[i])
                    tests[i].significance = True
                    print ("-- association after %s fdr correction" % p_adjusting_method)
                else:
                    tests[i].significance = False
                    aOut.append(tests[i])
    _get_passed_fdr_tests()
    config.number_of_performed_tests =len(aOut)
    print "--- number of performed tests:", config.number_of_performed_tests
    print "--- number of passed tests after FDR controlling:", len(aFinal) 
    return aFinal, aOut

def hypotheses_testing():
    pTree = config.meta_hypothesis_tree
    dataset1 = config.parsed_dataset[0]
    dataset2 = config.parsed_dataset[1]
    """
    pTree, dataset1, dataset2, seed, method="permutation", metric="nmi", fdr= "BHY", p_adjust="BH", fQ=0.1,
    iIter=1000, pursuer_method="nonparameteric", decomposition = "mca", bVerbose=False, robustness = None, fAlpha=0.05, apply_bypass = True, discretize_style= 'equal-area'
    
    Perform all-against-all on a hypothesis tree.

    Notes:
        Right now, return aFinal, aOut 


    Parameters
    ---------------

        pTree 
        dataset1
        dataset2
        method 
        metric
        p_adjust
        pursuer_method 
        verbose 

    Returns 
    ----------
        Z_final, Z_all: numpy.ndarray
            Bags of associations of _final_ associations, and _all_ associations respectively. 
    ----------
        
    """
    X, Y = dataset1, dataset2 
    aOut = []  # # Full log 
    aFinal = []  # # Only the final reported values 
    def _level_by_level_testing():
        apChildren = [pTree]
        level = 1
        number_performed_tests = 0 
        number_passed_tests = 0
        next_level_apChildren = []
        current_level_tests = []
        #temp_current_level_tests = []
        leaves_hypotheses = []
        from_prev_hypotheses = []
        #previous_unqualified_hypotheses = []
        while apChildren:
            temp_sub_hypotheses = []
            
            temp_hypothesis = apChildren.pop(0)
            #use the signifantly rejected or accepte dhypotheses from previouse level 
            if config.p_adjust_method != "bhy2":
                if temp_hypothesis.significance != None:
                    from_prev_hypotheses.append(temp_hypothesis)
                else:
                    temp_sub_hypotheses = get_children(temp_hypothesis)
                    if len(temp_sub_hypotheses) == 0:
                        leaves_hypotheses.append(temp_hypothesis)
            else:
                temp_sub_hypotheses = temp_hypothesis.get_children()
                # Repeat leaves in the next levels  
                if len(temp_sub_hypotheses) == 0:
                    leaves_hypotheses.append(temp_hypothesis)
                else:
                    #for i in range(len(temp_sub_hypotheses)):
                    #    print(temp_sub_hypotheses[i].m_pData)
                    if temp_hypothesis.significance != None:
                        for i in range(len(temp_sub_hypotheses)):
                            temp_sub_hypotheses[i].significance = temp_hypothesis.significance
                            temp_sub_hypotheses[i].pvalue = temp_hypothesis.pvalue 
                            temp_sub_hypotheses[i].qvalue = temp_hypothesis.qvalue
                  
            current_level_tests.extend(temp_sub_hypotheses)
            
            if len (apChildren) > 0:
                continue
            if len(current_level_tests) == 0 :
                break
            if len(from_prev_hypotheses) > 0 :
                current_level_tests.extend(from_prev_hypotheses)
                from_prev_hypotheses = []
            current_level_tests.extend(leaves_hypotheses)
            print "number of hypotheses in level %s: %s" % (level, len(current_level_tests))
            p_values = multiprocessing_actor(_actor, current_level_tests, pMethod, dataset1, dataset2)
            for i in range(len(current_level_tests)):
                current_level_tests[i].pvalue = p_values[i]
            cluster_size = [ len(current_level_tests[i].m_pData[0])*len(current_level_tests[i].m_pData[1]) for i in range(len(current_level_tests)) ]
            total_cluster_size = numpy.sum(cluster_size)
            q = config.q 
            aP_adjusted, pRank = stats.p_adjust(p_values, q, cluster_size)#config.q)
            for i in range(len(current_level_tests)):
                current_level_tests[i].rank = pRank[i]
            
            max_r_t = 0
            for i in range(len(current_level_tests)):
                if current_level_tests[i].pvalue <= aP_adjusted[i] and max_r_t <= current_level_tests[i].rank:
                    max_r_t = current_level_tests[i].rank
                    #print "max_r_t", max_r_t
                       
            passed_tests = []
            def _get_passed_fdr_tests():
                if config.p_adjust_method in ["bh", "bhy"]:
                  
                    for i in range(len(current_level_tests)):
                        if current_level_tests[i].rank <= max_r_t:#(level ==1000 and sum([1 if current_level_tests[i].pvalue<= val else 0 for val  in list_p_trshld])>= (1.0-config.q)*number_of_bootstrap)\
                            passed_tests.append(current_level_tests[i])
                            if current_level_tests[i].significance == None:
                                current_level_tests[i].significance = True
                                aOut.append(current_level_tests[i])
                                aFinal.append(current_level_tests[i])
                                print ("-- association after %s fdr correction" % config.p_adjust_method)
                                #print (current_level_tests[i].m_pData)
                        else:
                            if current_level_tests[i].significance == None and is_bypass(current_level_tests[i]):
                                current_level_tests[i].significance = False
                                aOut.append(current_level_tests[i])
                            elif is_leaf(current_level_tests[i]):
                                if current_level_tests[i].significance == None:
                                    current_level_tests[i].significance = False
                                    aOut.append(current_level_tests[i])
                elif config.p_adjust_method == "bonferroni":
                    print len(current_level_tests)
                    for i in range(len(current_level_tests)):
                        if current_level_tests[i].pvalue <= aP_adjusted[i]:
                            passed_tests.append(current_level_tests[i])
                            if current_level_tests[i].significance == None:
                                aOut.append(current_level_tests[i])
                                aFinal.append(current_level_tests[i])
                                current_level_tests[i].significance = True
                                print ("-- association after %s fdr correction" % config.p_adjust_method)
                        else:
                            if current_level_tests[i].significance == None and is_bypass(current_level_tests[i]):
                                current_level_tests[i].significance = False
                                aOut.append(current_level_tests[i])
                            elif is_leaf(current_level_tests[i]):
                                if current_level_tests[i].significance == None:
                                    current_level_tests[i].significance = False
                                    aOut.append(current_level_tests[i])
                elif config.p_adjust_method == "no_adjusting":
                    for i in range(len(current_level_tests)):
                        if current_level_tests[i].pvalue <= q:
                            passed_tests.append(current_level_tests[i])
                            if current_level_tests[i].significance == None:
                                aOut.append(current_level_tests[i])
                                aFinal.append(current_level_tests[i])
                                current_level_tests[i].significance = True
                                print ("-- association after %s fdr correction" % config.p_adjust_method)
                        else:
                            if current_level_tests[i].significance == None and is_bypass(current_level_tests[i]):
                                current_level_tests[i].significance = False
                                aOut.append(current_level_tests[i])
                            elif is_leaf(current_level_tests[i]):
                                if current_level_tests[i].significance == None:
                                    current_level_tests[i].significance = False
                                    aOut.append(current_level_tests[i])
                
                q_values = stats.pvalues2qvalues ([current_level_tests[i].pvalue for i in range(len(current_level_tests))], adjusted=True)
                for i in range(len(current_level_tests)):
                    if current_level_tests[i].qvalue == None and current_level_tests[i] in passed_tests: 
                        current_level_tests[i].qvalue = q_values[i]
            _get_passed_fdr_tests()
            
            #return aFinal, aOut                
            apChildren = current_level_tests #next_level_apChildren #
            hist_pvalues = [t.pvalue for t in current_level_tests]
            plt.clf()
            plt.hist(hist_pvalues, bins=20)  # plt.hist passes it's arguments to np.histogram
            plt.title("Histogram of nominal p-values in level "+ str(level))
            #plt.show()
            plt.savefig(str(level)+'_hist.pdf', pad_inches = .05, dpi=300) 
            plt.close()
            print "Hypotheses testing level", level, "with ",len(current_level_tests), "hypotheses is finished."
            level += 1
            next_level_apChildren = []
            current_level_tests = []
            temp_current_level_tests = []
            from_prev_hypotheses = []
            leaves_hypotheses = []
            aP = []
        config.number_of_performed_tests = len(aOut)
        print "--- number of performed tests:", config.number_of_performed_tests #len(aOut)#number_performed_tests
        print "--- number of passed tests after FDR controlling:", len(aFinal)#number_passed_tests                                  
        return aFinal, aOut

    fdr_function = {"default": _level_by_level_testing,
                            "level":_level_by_level_testing}
    #======================================#
    # Execute 
    #======================================#
    strFDR = config.fdr_function
    pFDR = fdr_function[strFDR]
    aFinal, aOut = pFDR()
    return aFinal, aOut 