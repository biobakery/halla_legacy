#!/usr/bin/env python 
'''
Hiearchy module, used to build trees and other data structures.
Handles clustering and other organization schemes. 
'''
# # structural packages 
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
    dP, similarity, left_first_rep_variance, right_first_rep_variance, left_loading, right_loading, left_rep, right_rep = pMethod(dataset1, dataset2)
    #dP, similarity, left_first_rep_variance, right_first_rep_variance = pMethod(dataset1, dataset2,  metric = metric, decomposition = decomposition, iIter=iIter)

    return id, dP, similarity, left_first_rep_variance, right_first_rep_variance, left_loading, right_loading, left_rep, right_rep

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
        for id, dP, similarity, left_first_rep_variance, right_first_rep_variance, left_loading, right_loading, left_rep, right_rep in results_by_id:
            result[id]=dP
            current_level_tests[id].similarity_score = similarity
            #current_level_tests[id].set_left_first_rep_variance(left_first_rep_variance)
            #current_level_tests[id].set_right_first_rep_variance(right_first_rep_variance)
            #current_level_tests[id].set_right_loading(right_loading)
            #current_level_tests[id].set_left_loading(left_loading)
            #current_level_tests[id].set_left_rep(left_rep)
            #current_level_tests[id].set_right_rep(right_rep)
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

global fig_num 
fig_num = 1

# class ClusterNode #scipy.cluster.hierarchy.ClusterNode
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
        #self.left_first_rep_variance = None
        #self.right_first_rep_variance = None
        self.already_tested = False
        self.already_passed = False
        self.level_number = 1
        self.significance =  None
        self.rank = None
        #self.left_loading= None
        #self.right_loading = None
        #self.right_rep = None
        #self.left_rep = None
    ''' 
    def set_right_loading(self, right_loading):
         self.right_loading = right_loading
    '''
    def pop(self):
        # pop one of the children, else return none, since this amounts to killing the singleton 
        if self.m_arrayChildren:
            return self.m_arrayChildren.pop()

    def l(self):
        return self.left()

    def r(self):
        return self.right()

    def left(self):
        # assert( len(self.m_arrayChildren) == 2 )
        return self.get_child(iIndex=0)
    
    def right(self):
        # assert( len(self.m_arrayChildren) == 2 )
        return self.get_child(iIndex=1)

    def is_leaf(self):
        return bool(not(self.m_pData and self.m_arrayChildren))

    def is_degenerate(self):
        return (not(self.m_pData) and not(self.m_arrayChildren))            

    def add_child(self, data):
        if not isinstance(data, Hypothesis_Node):
            pChild = Hypothesis_Node(data)
        else:
            pChild = data 
        self.m_arrayChildren.append(pChild)
        return self 
        
    def add_children(self, aData):
        for item in aData:
            self.add_child(item)
        return self 

    def get_children(self): 
        return self.m_arrayChildren
    
    def get_child(self, iIndex=None):
        return self.m_arrayChildren[iIndex or 0] if self.m_arrayChildren else None 
    
    def add_data(self, pDatum):
        self.m_pData = pDatum 
        return self 
    
    def get_data(self):
        return self.m_pData

    def get_right_distance(self):
        return self.right_distance
    
    #def get_left_distance(self):
     #   return self.left_distance
    
    #def get_similarity_score(self):
    #    return self.similarity_score
    
    #def set_similarity_score(self, similarity_score=None):
    #    self.similarity_score = similarity_score
        
    #def set_left_first_rep_variance(self, pc):
    #    self.left_first_rep_variance = pc
    
    #def set_right_first_rep_variance(self, pc):
    #    self.right_first_rep_variance = pc
        
    #def get_left_first_rep_variance(self):
    #    return self.left_first_rep_variance
    
    #def get_right_first_rep_variance(self):
    #    return self.right_first_rep_variance
    
    #def set_left_loading(self, left_loading):
     #   self.left_loading = left_loading
     
    #def set_right_loading(self, right_loading):
    #     self.right_loading = right_loading
    #def get_left_loading(self):
    #    return self.left_loading 
     
    #def get_right_loading(self):
    #    return self.right_loading 
    
    '''
    def set_pvalue(self, pvalue):
        self.pvalue = pvalue
    
    def get_pvalue(self):
        return self.pvalue
    
    def set_qvalue(self, qvalue):
        self.qvalue = qvalue
    
    def get_qvalue(self):
        return self.qvalue
   
    def set_left_rep(self, left_rep):
        self.left_rep = left_rep
    
    def set_right_rep(self, right_rep):
        self.right_rep = right_rep
        
    def get_left_rep(self):
        return self.left_rep
    
    def get_right_rep(self):
        return self.right_rep
    '''
def is_representative(Node, pvalue_threshold, decomp):
    return True
    number_left_features = len(Node.m_pData[0])
    number_right_features = len(Node.m_pData[1])
    if len(Node.m_pData[0]) <= 1 and len(Node.m_pData[1]) <= 1:
        return True
    counter = 0
    temp_right_loading = list()
    reps_similarity = Node.similarity_score
    pMe = distance.c_hash_metric[config.similarity_method] 
    left_threshold = [pMe(config.parsed_dataset[0][Node.m_pData[0][i]], Node.left_rep) for i in range(len(Node.m_pData[0]))]
    right_threshold = [pMe(config.parsed_dataset[1][Node.m_pData[1][i]], Node.right_rep) for i in range(len( Node.m_pData[1]))]
    left_rep_similarity_to_right_cluster = np.median([pMe(Node.left_rep, config.parsed_dataset[1][Node.m_pData[1][i]]) for i in range(len(Node.m_pData[1]))])
    right_rep_similarity_to_left_cluster = np.median([pMe(Node.right_rep, config.parsed_dataset[0][Node.m_pData[0][i]]) for i in range(len(Node.m_pData[0]))])
    for i in range(len(Node.m_pData[1])):
        if right_threshold[i]< (right_rep_similarity_to_left_cluster):# - np.std(right_threshold)):#scipy.stats.spearmanr(config.parsed_dataset[1][Node.m_pData[1][i]], Node.right_rep)[1] >.05:# 
            counter += 1
            temp_right_loading.append(i)
            #print "right:", Node.get_right_loading()
            if (counter >= number_right_features) or counter > (number_right_features/(math.log(number_right_features,2))):#math.log(number_right_features,2)):
                if config.verbose == 'DEBUG':
                    print "#Outlier right cluster:",counter
                return False
    counter = 0
    temp_left_loading = list()
    for i in range(len(Node.m_pData[0])):
        if left_threshold[i]< (left_rep_similarity_to_right_cluster):# - np.std(left_threshold)): 
        #scipy.stats.spearmanr(config.parsed_dataset[0][Node.m_pData[0][i]], Node.right_rep)[1] >.05:
            temp_left_loading.append(i)
            #print "after:", temp_left_loading
            counter += 1
            if (counter >= number_left_features) or counter > (number_left_features/(math.log(number_left_features,2))): # (number_left_features/2):#math.log(number_left_features,2)):
                if config.verbose == 'DEBUG':
                    print "#Outlier left cluster:",counter
                return False

    return True
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
            #print config.Features_order[0]
            #print temp_a_features#, [config.Features_order[0][i] for i in temp_a_features]
            temp_a_features.remove(a_feature)
            a = np.mean([1.0 - config.Distance[0][i][j] for i,j in product([a_feature], temp_a_features)])
            #a = np.mean([1.0 - math.fabs(pMe(config.parsed_dataset[0][i], config.parsed_dataset[0][j]))
            #             for i,j in product([a_feature], temp_a_features)])
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
            #a = np.mean([1.0 - math.fabs(pMe(config.parsed_dataset[1][i], config.parsed_dataset[1][j]))
            #             for i,j in product([a_feature], temp_a_features)])
               
        b = np.mean([1.0 - math.fabs(pMe(config.parsed_dataset[1][i], config.parsed_dataset[0][j])) 
                    for i,j in product([a_feature], cluster_a)])
        s = (b-a)/max([a,b])
        #print 's a', s, a, b
        silhouette_coefficient_B.append(s)
    #print "Silhouette coefficient A", silhouette_coefficient_A
    #print "Silhouette coefficient B", silhouette_coefficient_B
    silhouette_scores = silhouette_coefficient_A
    silhouette_scores.extend(silhouette_coefficient_B)
    
    if numpy.min(silhouette_scores)  < 0.25:
        #print "Silhouette coefficient all", silhouette_scores
        #print cluster_a
        #print cluster_b
        return False
    else:
        #print "Silhouette coefficient all", silhouette_scores 
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

    #def set_rank(Node, rank= None):
    #    Node.rank = rank
        
    #def get_rank(Node):
    #    return Node.rank
 
    #def set_level_number(Node, level_number):
    #    Node.level_number = level_number
        
    #def get_level_number(Node):
    #    return Node.level_number
    
    #def significance = Node, significance):
    #    Node.significance = significance
    
    #def get_significance(Node):
    #    return Node.significance
    
class Gardener():
    """
    A gardener object is a handler for the different types of hierarchical data structures ("trees")
    
    Always return a copied version of the tree being modified. 

    """

     

    def __init__(self, apTree=None):
        import copy
        self.delta = 1.0  # #step parameter 
        self.sigma = 0.5  # #start parameter 

        self.apTree = [copy.deepcopy(ap) for ap in apTree]  # # the list of tree objects that is going to be modified 
        # # this is a list instead of a single tree object because it needs to handle any cross section of a given tree 

        assert(0.0 <= self.delta <= 1.0)
        assert(0.0 <= self.sigma <= 1.0)

    def next(self):
        '''
        return the data of the tree, layer by layer
        input: None 
        output: a list of data pointers  
        '''
        
        if self.is_leaf():
            return Exception("Empty Hypothesis_Node")

        elif self.m_pData:
            pTmp = self.m_pData 
            self.m_queue.extend(self.m_arrayChildren)
            self.m_arrayChildren = None 
            self = self.m_queue 
            assert(self.is_degenerate())
            return pTmp     
        
        else:
            assert(self.is_degenerate())
            aOut = [] 
    
            for pTree in self.m_queue:
                aOut.append(pTree.m_pData)

        if self.m_queue:
            self = self.m_queue.pop()
        elif self.m_arrayChildren:
            pSelf = self.m_arrayChildren.pop() 
            self.m_queue = self.m_arrayChildren
            self = pSelf 
        return pTmp 

 
#==========================================================================#
# FUNCTORS   
#==========================================================================#

def lf2tree(lf):
    """
    Functor converting from a layerform object to py Hypothesis_Node object 

    Parameters
    ------------
        lf : layerform 

    Returns 
    -------------
        t : py Hypothesis_Node object 
    """ 

    pass 

def tree2clust(pTree, exact=True):
    """
    Functor converting from a py Hypothesis_Node object to a scipy ClusterNode object.
    When exact is True, gives error when the map Hypothesis_Node() -> ClusterNode is not injective (more than 2 children per node)
    """

    pass 


#==========================================================================#
# METHODS  
#==========================================================================#

def is_tree(pObj):
    """
    duck type implementation for checking if
    object is ClusterNode or Hypothesis_Node, more or less
    """

    try:
        pObj.get_data ()
        return True 
    except Exception:
        return False 


def hclust(dataset, labels, dataset_number):
    bTree=True
    """
    Notes 
    -----------

        This hclust function is not quite right for the MI case. Need a generic MI function that can take in clusters of RV's, not just single ones 
        Use the "grouping property" as discussed by Kraskov paper. 
    """
    
    
    '''
    D = np.zeros(shape=(len(dataset), len(dataset)))  
    for i in range(len(dataset)):
        for j in range(i,len(dataset)):
            if i == j:
                D[i][j] = 0
                continue
            D[i][j] = pDistance(dataset[i], dataset[j])
            D[j][i] = D[i][j]
    #print dataset.shape  
    #D = squareform(D)
    #print D
    '''
    linkage_method = 'single'
    D = pdist(dataset, metric=distance.pDistance) 
    #D = squareform(D)
    #print D
    
    config.Distance[dataset_number] =  copy.deepcopy(squareform(D))
    
    #print D.shape,  D
    if config.diagnostics_plot:
        print "--- plotting heatmap for Dataset", str(dataset_number)," ... "
        Z = plot.heatmap(data_table = dataset , D = D, xlabels_order = [], xlabels = labels, filename= config.output_dir+"/"+"hierarchical_heatmap_"+str(config.similarity_method)+"_" + str(dataset_number), method =linkage_method, dataset_number= None)
        
    else:
        Z = linkage(D, method= linkage_method)
    import scipy.cluster.hierarchy as sch
    logger.write_table(data=config.Distance[dataset_number], name=config.output_dir+'/Distance_matrix'+str(dataset_number)+'.tsv', rowheader=config.FeatureNames[dataset_number], colheader=config.FeatureNames[dataset_number])
    
    #resoltion_hclust(distance_matrix= D)
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
        filtered_apParents = filter(lambda x: not(x.is_leaf()) , apParents)
        new_apParents = [] 
        for q in filtered_apParents:
            if not bTree:
                new_apParents.append(q.left); new_apParents.append(q.right)
            else:
                for item in q.get_children():
                    new_apParents.append(item)
        if not bTree:
            return [(iLevel, reduce_tree(p)) for p in apParents ] + reduce_tree_by_layer(new_apParents, iLevel=iLevel + 1)
        else:
            return [(iLevel, p.m_pData) for p in apParents ] + reduce_tree_by_layer(new_apParents, iLevel=iLevel + 1)

def tree2lf(apParents, iLevel=0, iStop=None):
    """
    An alias of reduce_tree_by_layer, for consistency with functor definitions 
    """
    return reduce_tree_by_layer(apParents) 

def fix_layerform(lf, iExtend=0):
    """ 
    
    iExtend is when you want to extend layerform beyond normal global depth 

    There is undesired behavior when descending down singletons
    Fix this behavior 

        Example 

        [(0, [0, 7, 4, 6, 8, 2, 9, 1, 3, 5]),
         (1, [0]),
         (1, [7, 4, 6, 8, 2, 9, 1, 3, 5]),
         (2, [7, 4, 6, 8, 2, 9]),
         (2, [1, 3, 5]),
         (3, [7]),
         (3, [4, 6, 8, 2, 9]),
         (3, [1]),
         (3, [3, 5]),
         (4, [4]),
         (4, [6, 8, 2, 9]),
         (4, [3]),
         (4, [5]),
         (5, [6]),
         (5, [8, 2, 9]),
         (6, [8]),
         (6, [2, 9]),
         (7, [2]),
         (7, [9])]

    """

    aOut = [] 

    iDepth = depth_tree(lf, bLayerform=True)  # # how many layers? 
    iLevel = iDepth - 1  # #layer level 

    for tD in lf:  # #tuple data  
        iCurrent, aiIndices = tD[:2] 
        if len(aiIndices) == 1:  # #if singleton 
            aOut += [(i, aiIndices) for i in range(iCurrent + 1, iLevel + 1)]

    lf += aOut 

    # # Need to sort to ensure correct layerform  
    # # source: http://docs.scipy.org/doc/numpy/reference/generated/numpy.sort.html
    
    dtype = [('layer', int), ('indices', list)]
    return filter(bool, list(numpy.sort(array(lf, dtype=dtype), order='layer')))

def fix_clusternode(pClusterNode, iExtend=0):
    """
    Same as fix_layerform, but for ClusterNode objects 

    Note: should NOT alter original ClusterNode object; make a deep copy of it instead 
    """

     

    def _fix_clusternode(pChild):
        # pChildUpdate = copy.deepcopy( pChild )
        iChildDepth = get_depth(pChild)
        iDiff = iGlobalDepth - iChildDepth 
        if iChildDepth == 1:
            # print "singleton"
            # print "original", reduce_tree_by_layer( [pChild] ) 
            # print "difference", iDiff 
            assert(pChild.id == reduce_tree(pChild)[0])
            pChild = spawn_clusternode(pData=pChild.id, iCopy=iDiff) 
            # print "fixed", reduce_tree_by_layer( [pChild])
            # pChild = pChildUpdate 
            return pChild
        else:
            # print "non-singleton"
            # print reduce_tree_by_layer( [pChild] )
            pChild = fix_clusternode(pChild, iExtend=iExtend)
            return pChild 
            
    pClusterNode = copy.deepcopy(pClusterNode)  # #make a fresh instance 
    iGlobalDepth = get_depth(pClusterNode) + iExtend 
    if iGlobalDepth == 1:
        return pClusterNode
    else:
        pClusterNode.left = _fix_clusternode(pClusterNode.left)
        pClusterNode.right = _fix_clusternode(pClusterNode.right)
            
        return pClusterNode 

def get_depth(pClusterNode, bLayerform=False):
    """
    Get the depth of a tree 

    Parameters
    --------------

        pClusterNode: clusternode or layerform object 
        bLayerform: bool 


    Returns
    -------------

        iDepth: int 
    """

    aOut = reduce_tree_by_layer([pClusterNode]) if not bLayerform else pClusterNode 
    aZip = zip(*aOut)[0]
    return max(aZip) - min(aZip) + 1

def depth_tree(pClusterNode, bLayerform=False):
    """
    alias for get_depth
    """

    return get_depth(pClusterNode, bLayerform=bLayerform)

def depth_min(pClusterNode, bLayerform=False):
    """
    Get the index for the minimnum layer 
    """
    aOut = reduce_tree_by_layer([pClusterNode]) if not bLayerform else pClusterNode 
    aZip = zip(*aOut)[0]
    return min(aZip)

def depth_max(pClusterNode, bLayerform=False):
    """
    Get the index for the maximum layer
    """
    aOut = reduce_tree_by_layer([pClusterNode]) if not bLayerform else pClusterNode 
    aZip = zip(*aOut)[0]
    return max(aZip)

def get_layer(atData, iLayer=None, bTuple=False, bIndex=False):
    """
    Get output from `reduce_tree_by_layer` and parse 

    Input: atData = a list of (iLevel, list_of_nodes_at_iLevel), iLayer = zero-indexed layer number 

    BUGBUG: Need get_layer to work with ClusterNode and Hypothesis_Node objects as well! 
    """

    if not atData:
        return None 

    dummyOut = [] 

    if not isinstance(atData, list): 
        atData = reduce_tree_by_layer([atData])

    if not iLayer:
        iLayer = atData[0][0]

    for couple in atData:
        if couple[0] < iLayer:
            continue 
        elif couple[0] == iLayer:
            dummyOut.append(couple[1])
            atData = atData[1:]
        else:
            break

    if bIndex:
        dummyOut = (iLayer, dummyOut)
    return (dummyOut, atData) if bTuple else dummyOut 

def cross_section_tree(pClusterNode, method="uniform", cuts="complete"):
    """
    Returns cross sections of the tree depths in layer_form
    """
    aOut = [] 

    layer_form = reduce_tree_by_layer(pClusterNode)
    iDepth = depth_tree(layer_form, bLayerform=True)
    pCuts = stats.uniform_cut(range(iDepth), iDepth if cuts == "complete" else cuts)
    aCuts = [x[0] for x in pCuts]

    for item in layer_form:
        iDepth, pBag = item
        if iDepth in aCuts:
            aOut.append((iDepth, pBag))

    return aOut 

def spawn_clusternode(pData, iCopy=1, iDecider=-1):
    """
    Parameters
    -----------------

        pData: any data in node

        iCopy: int
            When iCopy > 0, makes a copy of pData and spawns children `iCopy` number of times 

        iDecider: int
            When adding copies of nodes, need to know if 
            -1-> only add to left 
            0 -> add both to left and right 
            1 -> only add to right 

    Returns 
    ------------------

        C: scipy.cluster.hierarchy.ClusterNode instance 

    """

    assert(iCopy >= 1), "iCopy must be a positive integer!"

    def _spawn_clusternode(pData, iDecider=-1):
        """
        spawns a "clusterstump" 
        """
        return scipy.cluster.hierarchy.ClusterNode(pData)  # #should return a new instance **each time**

    if iCopy == 1:
        return _spawn_clusternode(pData)

    else: 
        pOut = _spawn_clusternode(pData)
        pLeft = spawn_clusternode(pData, iCopy=iCopy - 1, iDecider=iDecider)
        pOut.left = pLeft 
        return pOut 

def spawn_tree(pData, iCopy=0, iDecider=-1):
    """
    Extends `spawn_clusternode` to the py.hierarchy.Hypothesis_Node object 
    """
    return None 
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

def _decider_max(node1, node2):
    pass

def _next():
    """
    gives the next node on the chain of linkages 
    """
    pass
def _percentage(dist, max_dist):
    if max_dist > 0:
        return float(dist) / float(max_dist)
    else:
        return 0.0

def _is_start(ClusterNode, X, func, distance):
    # node_indeces = reduce_tree(ClusterNode)
    # print "Node: ",node_indeces
    # if halla.pca_explained_variance_ratio_(X[array(node_indeces)])[0] > .65 or len(node_indeces) ==1:# and _min_tau(X[array(node_indeces)], func) <= x_threshold:
    if _percentage(ClusterNode.dist) <= distance:  # and py.stats.pca_explained_variance_ratio_(X[array(node_indeces)])[0] > .60  :#and _min_tau(X[array(node_indeces)], func) <= cluster_threshold :# and halla.pca_explained_variance_ratio_(X[array(node_indeces)])[0] > .60 :#and ClusterNode.get_count() >2 :
        return True
    else: 
        return False

def _is_stop(ClusterNode, dataSet, max_dist_cluster):
        #node_indeces = reduce_tree(ClusterNode)
        #first_PC = stats.pca_explained_variance_ratio_(dataSet[array(node_indeces)])[0]
        if ClusterNode.is_leaf():# or _percentage(ClusterNode.dist, max_dist_cluster) < .1:# or first_PC > .9:
            #print "Node: ",node_indeces
            #print "dist:", ClusterNode.dist, " first_PC:", first_PC,"\n"
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
def _cutree_overall (clusterNodelist, X, func, distance):
    clusterNode = clusterNodelist
    n = clusterNode[0].get_count()
    sub_clusters = []
    while clusterNode :
        temp_apChildren = []
        sub_clusterNode = truncate_tree(clusterNode, level=0, skip=1)
        for node in clusterNode:
            if node.is_leaf():
                sub_clusterNode += [node]
        if sub_clusterNode:
            for node in sub_clusterNode:
                if _is_start(node , X, func, distance):
                    sub_clusters.append(node)
                else:
                    if not node.is_leaf():
                        truncated_result = truncate_tree([node], level=0, skip=1)    
                    if truncated_result:
                        temp_apChildren += truncated_result
    
        clusterNode = temp_apChildren
    if distance > .1:
        next_dist = distance - 0.1 
    elif distance > .01:
        next_dist = distance - 0.01
    else:
        next_dist = 0.0
    aDist = []
    for i in range(len(sub_clusters)):
        if sub_clusters[i].dist > 0.0:
            aDist += [sub_clusters[i].dist]
    while len(sub_clusters) < round(math.sqrt(n)):
        aDist = []
        max_dist_node = sub_clusters[0]
        index = 0 
        for i in range(len(sub_clusters)):
            if sub_clusters[i].dist > 0.0:
                aDist += [sub_clusters[i].dist]
            if max_dist_node.dist < sub_clusters[i].dist:
                max_dist_node = sub_clusters[i]
                index = i
        # print "Max Distance in this level", _percentage(max_dist_node.dist)
        if not max_dist_node.is_leaf():
            del sub_clusters[index]
            sub_clusters += truncate_tree([max_dist_node], level=0, skip=1)
            
        else:
            break
    if aDist:
        next_dist = _percentage(numpy.min(aDist))
    # print len(sub_clusters), n            
    return sub_clusters , next_dist
def cutree_to_get_below_threshold_number_of_features (cluster, number_of_estimated_clusters = None):
    n_features = cluster.get_count()

    if number_of_estimated_clusters == None:
        number_of_estimated_clusters = math.log(n_features, 2)
    if n_features==1:# or cluster.dist <= t:
        return [cluster]
    sub_clusters = []
    #print n_features, t
    #sub_clusters = cutree_to_get_number_of_clusters ([cluster])
    sub_clusters = truncate_tree([cluster], level=0, skip=1)
    
    while True:# not all(val <= t for val in distances):
        largest_node = sub_clusters[0]
        index = 0
        for i in range(len(sub_clusters)):
            if largest_node.get_count() < sub_clusters[i].get_count():
                #print largest_node.get_count()
                largest_node = sub_clusters[i]
                index = i
        if largest_node.get_count() > n_features/number_of_estimated_clusters:
            #sub_clusters.remove(largest_node)
            #sub_clusters = sub_clusters[:index] + sub_clusters[index+1 :]
            del sub_clusters[index]
            sub_clusters += truncate_tree([largest_node], level=0, skip=1)
            #print "first cut", [a.pre_order(lambda x: x.id) for a in sub_clusters ]
        else:
            break
        '''elif len(sub_clusters) < t:
            max_dist_node = sub_clusters[0]
            for i in range(len(sub_clusters)):
                if max_dist_node.dist < sub_clusters[i].dist:
                    max_dist_node = sub_clusters[i]
            sub_clusters += truncate_tree([max_dist_node], level=0, skip=1)
            sub_clusters.remove(max_dist_node)
            # print "Max Distance in this level", _percentage(max_dist_node.dist)'''
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
def cutree_to_get_number_of_clusters (cluster, n = None):
    n_features = cluster[0].get_count()
    if n_features==1:
        return cluster
    if n ==None:
        number_of_sub_cluters_threshold = round(math.log(n_features, 2))
    else:
        number_of_sub_cluters_threshold = n
    sub_clusters = []
    sub_clusters = truncate_tree(cluster, level=0, skip=1)
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
    silhouette_scores = []
    if len(clusters) <= 1:
        sys.exit("silhouette method needs at least two clusters!")
        
    for i in range(len(clusters)):
        cluster_a = clusters[i].pre_order(lambda x: x.id)
        #cluster_b = [val for val in range(len(config.Distance[dataset_number])) if val not in cluster_a]
        #print cluster_a, cluster_b
        if i%2 == 0 and i<len(clusters)-1:
            cluster_b = clusters[i+1].pre_order(lambda x: x.id)
        else:
            cluster_b = clusters[i-1].pre_order(lambda x: x.id)
            
        if i%2 != 0 and i> 0:
            cluster_b_2 = clusters[i-1].pre_order(lambda x: x.id)
        elif i < len((clusters))-1:
            cluster_b_2 = clusters[i+1].pre_order(lambda x: x.id)
        else: 
            cluster_b_2 = clusters[i-1].pre_order(lambda x: x.id)
        #silhouette_score.append(silhouette_coefficient(cluster))
        s_all_a = []
        for a_feature in cluster_a:
            if len(cluster_a) ==1:
                a = 0.0
            else:
                temp_a_features = cluster_a[:]#deepcopy(all_a_clusters)
                #print 'before', all_a_clusters
                temp_a_features.remove(a_feature)
                #print 'a feature ', a_feature, temp_a_features
                #a = np.mean([1.0 - math.fabs(pMe(config.parsed_dataset[dataset_number][i], config.parsed_dataset[dataset_number][j]))
                #             for i,j in product([a_feature], temp_a_features)])
                a = np.mean([distance_matrix[i][j] for i,j in product([a_feature], temp_a_features)])            
            #b = np.mean([1.0 - math.fabs(pMe(config.parsed_dataset[dataset_number][i], config.parsed_dataset[dataset_number][j])) 
             #            for i,j in product([a_feature], cluster_b)])
            b1 = np.mean([ distance_matrix[i][j] for i,j in product([a_feature], cluster_b)])
            b2 = np.mean([ distance_matrix[i][j] for i,j in product([a_feature], cluster_b_2)])
            b = min(b1,b2)
            s = (b-a)/max([a,b])
            #print 's a', s, a, b
            s_all_a.append(s)
        silhouette_scores.append(np.mean(s_all_a))
    return silhouette_scores
def wss_heirarchy(clusters, dataset_number):
    wss = []
    for i in range(len(clusters)):
        cluster_a = clusters[i].pre_order(lambda x: x.id)
        
        temp_a_features = cluster_a[:]
        medoid_feature = temp_a_features[len(temp_a_features)-1]
        # remove medoid
        temp_a_features.remove(medoid_feature)
        
        wss.append(sum([config.Distance[dataset_number][i][j]* config.Distance[dataset_number][i][j] 
                        for i,j in product([medoid_feature], temp_a_features)]))
    
    avgWithinSS = np.sum(wss)#[sum(d)/X.shape[0] for d in dist]
    return avgWithinSS
def get_homogenous_clusters_silhouette_log(cluster, distance_matrix, number_of_estimated_clusters=None):
    n = cluster.get_count()
    if n==1:
        return cluster
    #number_of_sub_cluters_threshold = round(math.log(n, 2)) 
    #t = 1.0 - np.percentile(config.Distance[dataset_number].flatten(), config.q*100 - 1.0/len(config.Distance[dataset_number])*100) #config.cut_distance_thrd
    #print "t",t
    #print "first cut", [cluster.pre_order(lambda x: x.id)]
    sub_clusters = cutree_to_get_below_threshold_number_of_features(cluster, number_of_estimated_clusters)
    #cutree_to_get_below_threshold_number_of_features(cluster, t)
    #cutree_to_get_below_threshold_distance_of_clusters(cluster, t)
    
    
    #print "first cut", [a.pre_order(lambda x: x.id) for a in sub_clusters]
   # sub_clusters = cutree_to_get_number_of_clusters([cluster])#truncate_tree([cluster], level=0, skip=1)truncate_tree([cluster], level=0, skip=1)#
    #print "before sil sub:", len(sub_clusters)
    sub_silhouette_coefficient = silhouette_coefficient(sub_clusters, distance_matrix) 
    #print sub_silhouette_coefficient
    while True:#len(sub_clusters) < number_of_sub_cluters_threshold and
        min_silhouette_node = sub_clusters[0]
        min_silhouette_node_index = 0
        #print sum(sub_silhouette_coefficient), len(sub_silhouette_coefficient)
        #if sum(sub_silhouette_coefficient) == float(len(sub_silhouette_coefficient)):
        #    print sum(sub_silhouette_coefficient), len(sub_silhouette_coefficient)
        #    break
        for i in range(len(sub_clusters)):
            #if math.isnan(sub_silhouette_coefficient[min_silhouette_node_index]):
            #    print min_silhouette_node_index, min_silhouette_node, sub_silhouette_coefficient[min_silhouette_node_index]
            #    sys.exit()
            if sub_silhouette_coefficient[min_silhouette_node_index] > sub_silhouette_coefficient[i]:
                min_silhouette_node = sub_clusters[i]
                min_silhouette_node_index = i
        if sub_silhouette_coefficient[min_silhouette_node_index] == 1.0:
            break
        sub_clusters_to_add = truncate_tree([min_silhouette_node], level=0, skip=1)#cutree_to_get_number_of_clusters([min_silhouette_node])##
        #sub_clusters_to_check = cutree_to_get_below_threshold_number_of_features(min_silhouette_node)
        if len(sub_clusters_to_add) < 2:
            break
        sub_silhouette_coefficient_to_add = silhouette_coefficient(sub_clusters_to_add, distance_matrix)
        #sub_silhouette_coefficient_to_check = silhouette_coefficient(sub_clusters_to_check, dataset_number)
        temp_sub_silhouette_coefficient_to_add = sub_silhouette_coefficient_to_add[:]
        #temp_sub_silhouette_coefficient_to_check = sub_silhouette_coefficient_to_check[:]
        temp_sub_silhouette_coefficient_to_add = [value for value in temp_sub_silhouette_coefficient_to_add if value != 1.0]
            
        #if len(sub_clusters_to_add) ==0:
        #    sub_silhouette_coefficient[min_silhouette_node_index] =  1.0
            
        if len(temp_sub_silhouette_coefficient_to_add) == 0 or sub_silhouette_coefficient[min_silhouette_node_index] >= np.max(temp_sub_silhouette_coefficient_to_add) :
            sub_silhouette_coefficient[min_silhouette_node_index] =  1.0
        else:
            #print temp_sub_silhouette_coefficient_to_add
            #print np.max(temp_sub_silhouette_coefficient_to_add)
            #print "first cut", [min_silhouette_node.pre_order(lambda x: x.id)]
            #print "befor", [a.pre_order(lambda x: x.id) for a in sub_clusters ]
            del sub_clusters[min_silhouette_node_index]#min_silhouette_node)
            #print "after", [a.pre_order(lambda x: x.id) for a in sub_clusters ]
            del sub_silhouette_coefficient[min_silhouette_node_index]
            #sub_silhouette_coefficient=[sub_silhouette_coefficient != "nan"]
            #sub_clusters = [sub_clusters != "nan"]
            sub_silhouette_coefficient.extend(sub_silhouette_coefficient_to_add)
            sub_clusters.extend(sub_clusters_to_add)
            '''if len(sub_clusters_to_add) == 2:
                sub_clusters.insert(min_silhouette_node_index,sub_clusters_to_add[0])
                sub_silhouette_coefficient.insert(min_silhouette_node_index,sub_silhouette_coefficient_to_add[0])
            elif len(sub_clusters_to_add) == 1:
                sub_clusters.insert(min_silhouette_node_index+1,sub_clusters_to_add[1])
                sub_silhouette_coefficient.insert(min_silhouette_node_index+1,sub_silhouette_coefficient_to_add[1])
            '''  
    #print "After sil sub:", len(sub_clusters)
    return sub_clusters
def get_homogenous_clusters(cluster, dataset_number, prev_silhouette_coefficient):
    
    #pMe = distance.c_hash_metric[config.similarity_method]
    
    cluster_features = cluster.pre_order(lambda x: x.id)
    if len(cluster_features) == 1:
        return [cluster]
    sub_homogenous_clusters = []
    sub_clusters = truncate_tree([cluster], level=0, skip=1)#cutree_to_get_number_of_clusters([cluster])#cutree_to_get_number_of_clusters([cluster])#
    for sub_cluster in sub_clusters:
        if sub_cluster.get_count() == 1:
            sub_homogenous_clusters.append(sub_cluster)
            sub_clusters.remove(sub_cluster)
    sub_silhouette_coefficient = silhouette_coefficient(sub_clusters, distance_matrix) 
    #print sub_silhouette_coefficient
    temp_sub_silhouette_coefficient= sub_silhouette_coefficient[:]
    '''
    try:
        temp_sub_silhouette_coefficient.remove(1.0)
        #temp_sub_silhouette_coefficient= [.5 if x == 1.0 else x for x in temp_sub_silhouette_coefficient]
    except:
        pass
        '''
        
    if prev_silhouette_coefficient >= np.max(temp_sub_silhouette_coefficient):
        return [cluster]
    else:
        for i in range(len(sub_clusters)):
            sub_homogenous_clusters.extend(get_homogenous_clusters(sub_clusters[i], dataset_number, sub_silhouette_coefficient[i]))
    #print [cluster.pre_order(lambda x: x.id) for cluster in sub_homogenous_clusters]
    return sub_homogenous_clusters
    
    #cluster_medoid = config.parsed_dataset[dataset_number][cluster_features[len(cluster_features)-1]]
    all_sim = [math.fabs(pMe(config.parsed_dataset[dataset_number][i], config.parsed_dataset[dataset_number][j])) for i,j in combinations(cluster_features, 2)]
    #print "all_sim ", all_sim
    #all_sim = [1.0 - config.D[dataset_number][i][j] for i,j in combinations(cluster_features, 2)]
    #print "all_sim ",all_sim
    #S = squareform([math.exp((all_sim[i]*all_sim[i])/(2*np.std(all_sim))) for i in range(len(all_sim))])
    #print "S ",S
    '''A = numpy.zeros((len(S), len(S)))
    for i in range(len(S)):
        for j in range(len(S)):
            A[i][j] = S[i][j] if i != j else S[i][j] - sum(S[:][j])
    try:        
        eigen_values, _ = numpy.linalg.eig(A)
        print eigen_values
    except:
        pass
        '''
    #all_dist = [math.fabs(pMe(config.parsed_dataset[dataset_number][i], cluster_medoid)) for i in cluster_features[0: len(cluster_features)-1]]
    #clutser_95_percentile = np.percentile(dist_to_medoid,50)
    #print math.sqrt(len(A))
    #print "A ",A
    
    k = config.K
    Q1 = np.percentile(all_sim, 25)
    Q3 = np.percentile(all_sim, 75)
    IQR = Q3 - Q1
    upper_fence = Q3 + k * IQR
    lower_fence = Q1 - k * IQR
    #====check within class homogeniety
    coefficient_of_variation = np.std(all_sim)/np.mean(all_sim)
    if descending_silhouette_coefficient(cluster, dataset_number):
    #if all (val<= upper_fence and val >= lower_fence for val in all_sim) and coefficient_of_variation < .5:
        print "coefficient_of_variation:", coefficient_of_variation , cluster_features
        #descending_silhouette_coefficient(cluster, dataset_number):
        #not (max(all_dist)-median(all_dist)> k * math.fabs(median(all_dist)-min(all_dist))):
        #descending_silhouette_coefficient(cluster, dataset_number):
        #stats.kstest(all_dist, 'norm', mode='asymp')[1] < .05:
    
        # Homogeneous
        sub_homogenous_clusters.extend([cluster])
        if config.verbose == 'DEBUG':
            print "Homogenous cluster!!!"
            print "cluster:", cluster_features, all_sim
            print "Q1: ",Q1
            print "Q3: ",Q3
            print "========================"
            #import matplotlib.pyplot as plt
            #fig, ax = plt.subplots(1, 1)
            #ax.hist(all_dist, normed=True, histtype='stepfilled', alpha=0.2)
            #ax.legend(loc='best', frameon=False)
            #plt.savefig("/Users/rah/Documents/Hutlab/halla/hist"+str(stats.kstest(all_dist, 'norm')[1])+".pdf")
        
    else:
        # Too heterogeneous 
        print "heterogeneous coefficient_of_variation:", coefficient_of_variation, cluster_features
        for sub_cluster in truncate_tree([cluster], level=0, skip=1):
           sub_homogenous_clusters.extend(get_homogenous_clusters(sub_cluster, dataset_number)) 
    return sub_homogenous_clusters
def cutree_to_get_homogenous_clusters (clusterNodelist, dataset_number):
    #clusterNode = clusterNodelist
    #sub_clusters = truncate_tree(clusterNodelist, level=0, skip=1)
    #homogenous_clusters = []
    #for sub_cluster in sub_clusters:
    homogenous_clusters = get_homogenous_clusters_silhouette_log(clusterNodelist[0], config.Distance[dataset_number])
    #homogenous_clusters.extend(get_homogenous_clusters(clusterNodelist[0], dataset_number, prev_silhouette_coefficient = -1))

    for cluster in homogenous_clusters:
        print cluster.pre_order(lambda x: x.id) 
    print '====================================='
    return homogenous_clusters

    
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
    apChildren0 = get_homogenous_clusters_silhouette_log (apClusterNode0[0], config.Distance[0])
    #cutree_to_get_below_threshold_number_of_features(apClusterNode0[0])
    #get_homogenous_clusters_silhouette_log (apClusterNode0[0], dataset_number = 0)
    apChildren1 = get_homogenous_clusters_silhouette_log (apClusterNode1[0], config.Distance[1])
    #cutree_to_get_below_threshold_number_of_features(apClusterNode[0])
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
    Hypothesis_Tree_Root.add_children(childList)
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
            apChildren0 = get_homogenous_clusters_silhouette_log(a,config.Distance[0])
            #cutree_to_get_number_of_clusters([a])
            #cutree_to_get_below_threshold_number_of_features(a)
            #get_homogenous_clusters_silhouette_log(a,0)
        else:
            apChildren0 = [a]
        if not bTauY:
            apChildren1 = get_homogenous_clusters_silhouette_log(b,config.Distance[1])#cutree_to_get_number_of_clusters([b])
            #cutree_to_get_below_threshold_number_of_features(b)
            ##
            #get_homogenous_clusters_silhouette_log(b,1)#
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
        pStump.add_children(childList)
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
    '''if decomposition not in ["pca", "ica"]:
        X = dataset1[aIndiciesMapped[0]]
        Y = dataset2[aIndiciesMapped[1]]
    else:
        orginal_data_0 = array(orginal_data[0])
        orginal_data_1 = array(orginal_data[1])
        X = orginal_data_0[aIndiciesMapped[0]]
        Y = orginal_data_1[aIndiciesMapped[1]]
    '''
    X = dataset1[aIndiciesMapped[0]]
    Y = dataset2[aIndiciesMapped[1]]
    dP, similarity, left_first_rep_variance, right_first_rep_variance, left_loading, right_loading, left_rep, right_rep = pMethod(X, Y)
    pNode.similarity_score = similarity
    #pNode.set_left_first_rep_variance(left_first_rep_variance)
    #pNode.set_right_first_rep_variance(right_first_rep_variance)
    #pNode.set_left_loading(left_loading)
    #pNode.set_right_loading(right_loading)
    #pNode.set_left_rep(left_rep)
    #pNode.set_right_rep(right_rep)
    
        
    # aOut.append( [aIndicies, dP] ) #### dP needs to appended AFTER multiple hypothesis correction
    
    return dP        
def naive_all_against_all():
    dataset1 = config.parsed_dataset[0]
    dataset2 = config.parsed_dataset[1]
    seed = config.seed
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
        test.add_data(data)
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

def traverse_by_layer(pClusterNode1, pClusterNode2, dataset1, dataset2, pFunction=None):
    """

    Useful function for doing all-against-all comparison between nodes in each layer 

    traverse two trees at once, applying function `pFunction` to each layer pair 

    latex: $pFunction: index1 \times index2 \times data1 \times data2 \rightarrow \mathbb{R}^k, $ for $k$ the size of the cross-product set per layer 

    Parameters
    ----------------
        pClusterNode1, pClusterNode2, dataset1, dataset2, pFunction
    
    Returns 
    ---------
        All-against-all per layer 

        Ex. 

        [[([0, 2, 6, 7, 4, 8, 9, 5, 1, 3], [0, 2, 6, 7, 4, 8, 9, 5, 1, 3])],
         [([0, 2, 6, 7], [0, 2, 6, 7]),
          ([0, 2, 6, 7], [4, 8, 9, 5, 1, 3]),
          ([4, 8, 9, 5, 1, 3], [0, 2, 6, 7]),
          ([4, 8, 9, 5, 1, 3], [4, 8, 9, 5, 1, 3])],
         [([0], [0]),
          ([0], [2, 6, 7]),
          ([0], [4]),
          ([0], [8, 9, 5, 1, 3]),
          ([2, 6, 7], [0]),
          ([2, 6, 7], [2, 6, 7]),
          ([2, 6, 7], [4]),
          ([2, 6, 7], [8, 9, 5, 1, 3]),
          ([4], [0]),
          ([4], [2, 6, 7]),
          ([4], [4]),
          ([4], [8, 9, 5, 1, 3]),
          ([8, 9, 5, 1, 3], [0]),
          ([8, 9, 5, 1, 3], [2, 6, 7]),
          ([8, 9, 5, 1, 3], [4]),
          ([8, 9, 5, 1, 3], [8, 9, 5, 1, 3])],
         [([2], [2]),
          ([2], [6, 7]),
          ([2], [8, 9]),
          ([2], [5, 1, 3]),
          ([6, 7], [2]),
          ([6, 7], [6, 7]),
          ([6, 7], [8, 9]),
          ([6, 7], [5, 1, 3]),
          ([8, 9], [2]),
          ([8, 9], [6, 7]),
          ([8, 9], [8, 9]),
          ([8, 9], [5, 1, 3]),
          ([5, 1, 3], [2]),
          ([5, 1, 3], [6, 7]),
          ([5, 1, 3], [8, 9]),
          ([5, 1, 3], [5, 1, 3])],
         [([6], [6]),
          ([6], [7]),
          ([6], [8]),
          ([6], [9]),
          ([6], [5]),
          ([6], [1, 3]),
          ([7], [6]),
          ([7], [7]),
          ([7], [8]),
          ([7], [9]),
          ([7], [5]),
          ([7], [1, 3]),
          ([8], [6]),
          ([8], [7]),
          ([8], [8]),
          ([8], [9]),
          ([8], [5]),
          ([8], [1, 3]),
          ([9], [6]),
          ([9], [7]),
          ([9], [8]),
          ([9], [9]),
          ([9], [5]),
          ([9], [1, 3]),
          ([5], [6]),
          ([5], [7]),
          ([5], [8]),
          ([5], [9]),
          ([5], [5]),
          ([5], [1, 3]),
          ([1, 3], [6]),
          ([1, 3], [7]),
          ([1, 3], [8]),
          ([1, 3], [9]),
          ([1, 3], [5]),
          ([1, 3], [1, 3])],
         [([1], [1]), ([1], [3]), ([3], [1]), ([3], [3])]]

    """

    aOut = [] 

    def _link(i1, i2, a1, a2):
        return (i1, i2)

    if not pFunction:
        pFunction = _link 

    tData1, tData2 = [ fix_layerform(tree2lf([pT])) for pT in [pClusterNode1, pClusterNode2] ]  # # adjusted layerforms 

    iMin = np.min([depth_tree(tData1, bLayerform=True), depth_tree(tData2, bLayerform=True)]) 

    for iLevel in range(iMin + 1):  # # min formulation 
        
        aLayerOut = [] 

        aLayer1, aLayer2 = get_layer(tData1, iLevel), get_layer(tData2, iLevel)
        iLayer1, iLayer2 = len(aLayer1), len(aLayer2)

        for i, j in itertools.product(range(iLayer1), range(iLayer2)):
            aLayerOut.append(pFunction(aLayer1[i], aLayer2[j], dataset1, dataset2))

        aOut.append(aLayerOut)

    return aOut 

#### Perform all-against-all per layer, without adherence to hierarchical structure at first
def layerwise_all_against_all(pClusterNode1, pClusterNode2, dataset1, dataset2, adjust_method="BH"):
    """
    Perform layer-wise all against all 

    Notes
    ---------

        New behavior for coupling trees 

        CALCULATE iMaxLayer
        IF SingletonNode:
            CALCULATE iLayer 
            EXTEND (iMaxLayer - iLayer) times 
        ELSE:
            Compare bags 
    """
    aOut = [] 

    pPTBR = lambda ai, aj, X, Y : stats.permutation_test_by_representative(X[array(ai)], Y[array(aj)]) 

    traverse_out = traverse_by_layer(pClusterNode1, pClusterNode2, dataset1, dataset2)  # #just gives me the coupled indices 

    for layer in traverse_out:
        aLayerOut = [] 
        aPval = [] 
        for item in layer:
            fPval = stats.permutation_test_by_representative(dataset1[array(item[0])], dataset2[array(item[1])])
            aPval.append(fPval)
        
        adjusted_pval = stats.p_adjust(aPval)
        if not isinstance(adjusted_pval, list):
            # # keep type consistency 
            adjusted_pval = [adjusted_pval]
    
        for i, item in enumerate(layer):
            aLayerOut.append(([item[0], item[1]], adjusted_pval[i]))
        
        aOut.append(aLayerOut)

    return aOut

#### BUGBUG: When q = 1.0, results should be _exactly the same_ as naive hypotheses_testing, but something is going on that messes this up
#### Need to figure out what -- it's probably in the p-value consolidation stage 
#### Need to reverse sort by the sum of the two sizes of the bags; the problem should be fixed afterwards 

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


    Notes 
    ----------

        
    """
    X, Y = dataset1, dataset2 

    if config.verbose == 'INFO':
        print reduce_tree_by_layer([pTree])

    def _start_parameter_to_iskip(start_parameter):
        """
        takes start_parameter, determines how many to skip
        """

        assert(type(start_parameter) == float)

        iDepth = get_depth(pTree)
        iSkip = int(start_parameter * (iDepth - 1))

        return iSkip 

    def _step_parameter_to_aislice(step_parameter):
        """
        takes in step_parameter, returns a list of indices for which all-against-all will take place 
        """

        pass 

    # print "layers to skip:", iSkip 

    aOut = []  # # Full log 
    aFinal = []  # # Only the final reported values 

    iGlobalDepth = depth_tree(pTree)
    # iSkip = _start_parameter_to_iskip( start_parameter )
    
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
                    temp_sub_hypotheses = temp_hypothesis.get_children()
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
                            
                    #else:
                        #number_performed_tests += len(temp_sub_hypotheses)
                    #temp_current_level_tests.extend(temp_sub_hypotheses)
            current_level_tests.extend(temp_sub_hypotheses)
            
            if len (apChildren) > 0:
                continue
            if len(current_level_tests) == 0 :
                break
            #if config.p_adjust_method != "bhy":
            if len(from_prev_hypotheses) > 0 :
                current_level_tests.extend(from_prev_hypotheses)
                from_prev_hypotheses = []
            #number_performed_tests += len(current_level_tests)
            #if n1 < 2 and n2 < 2:
            current_level_tests.extend(leaves_hypotheses)
            print "number of hypotheses in level %s: %s" % (level, len(current_level_tests))
            #if not len(current_level_tests):
            #    continue
         
            p_values = multiprocessing_actor(_actor, current_level_tests, pMethod, dataset1, dataset2)
            
            for i in range(len(current_level_tests)):
                current_level_tests[i].pvalue = p_values[i]
                #print "Pvalue", i, " :", p_values[i]
                
            cluster_size = [ len(current_level_tests[i].m_pData[0])*len(current_level_tests[i].m_pData[1]) for i in range(len(current_level_tests)) ]
            #if current_level_tests[i].significance == None else 1 
            total_cluster_size = numpy.sum(cluster_size)
            # claculate adjusted p-value
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
                            elif current_level_tests[i].is_leaf():
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
                            elif current_level_tests[i].is_leaf():
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
                            elif current_level_tests[i].is_leaf():
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
            print "Hypotheses testing level", level, "with ",len(current_level_tests), "hypotheses is finished."
            level += 1
            #q = fQ - fQ*max_r_t/100.0
            #if len(current_level_tests)>0:
                #q = fQ - fQ*max_r_t/len(current_level_tests)
                #print "Next level q:", q
            #    last_current_level_tests = current_level_tests
            #else:
            #    last_current_level_tests = leaves_hypotheses
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

    #print "____Number of performed test:", number_performed_test
    return aFinal, aOut 
