#!/usr/bin/env python 
'''
Hiearchy module, used to build trees and other data structures.
Handles clustering and other organization schemes. 
'''
# # structural packages 
import itertools
import math 
from numpy import array , rank
import numpy 
import scipy.cluster 
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import pdist
import sys
import matplotlib.pyplot as plt
import numpy as np

from . import distance
from . import stats
from . import plot
from __builtin__ import True

# number of available processors
NPROC = 1

def multi_pMethod(args):
    """
    Runs the pMethod function and returns the results plus the id of the node
    """
    
    id, pMethod, pArray1, pArray2, metric, decomposition, iIter = args
    
    dP, similarity, left_first_rep, right_first_rep = pMethod(pArray1, pArray2,  metric = metric, decomposition = decomposition, iIter=iIter)

    return id, dP, similarity, left_first_rep, right_first_rep

def multiprocessing_actor(_actor, current_level_tests, pMethod, pArray1, pArray2, metric, decomposition, iIter):
    """
    Return the results from applying the data to the actor function
    """
    
    def _multi_pMethod_args(current_level_tests, pMethod, pArray1, pArray2, metric, decomposition, iIter):
        for i in xrange(len(current_level_tests)):
            aIndicies = current_level_tests[i].get_data()
            aIndiciesMapped = map(array, aIndicies)
            yield [i, pMethod, pArray1[aIndiciesMapped[0]], pArray2[aIndiciesMapped[1]], metric, decomposition, iIter]
    
    if NPROC > 1:
        import multiprocessing
        
        pool = multiprocessing.Pool(NPROC)
        
        results_by_id = pool.map(multi_pMethod, _multi_pMethod_args(current_level_tests, 
            pMethod, pArray1, pArray2, metric, decomposition, iIter))
        pool.close()
        pool.join()
        
        # order the results by id and apply results to nodes
        result = [0] * len(results_by_id)
        for id, dP, similarity, left_first_rep, right_first_rep in results_by_id:
            result[id]=dP
            current_level_tests[id].set_similarity_score(similarity)
            current_level_tests[id].set_left_first_rep(left_first_rep)
            current_level_tests[id].set_right_first_rep(right_first_rep)
    
    else:
        result=[]
        for i in xrange(len(current_level_tests)):
            if current_level_tests[i].get_significance_status() != None:
                result.append(current_level_tests[i].get_pvalue())
            else: 
                result.append(_actor(current_level_tests[i]))

    return result

from unicodedata import decomposition

# # statistics packages 
sys.setrecursionlimit(20000)
#==========================================================================#
# DATA STRUCTURES 
#==========================================================================#

# maximum distance in hierarchical  trees.
global max_dist
max_dist = 1.0 

global fig_num 
fig_num = 1

global hypotheses_tree_heigth 
hypotheses_tree_heigth = 0

# class ClusterNode #scipy.cluster.hierarchy.ClusterNode
class Tree():
    ''' 
    A hierarchically nested structure containing nodes as
    a basic core unit    

    A general object, tree need not be 2-tree 
    '''    

    def __init__(self, data=None, left_distance=None, right_distance=None, similarity=None):
        self.m_pData = data 
        self.m_arrayChildren = []
        self.left_distance = left_distance
        self.right_distance = right_distance
        self.pvalue = None
        self.qvalue = None
        self.similarity_score = None
        self.left_first_rep = None
        self.right_first_rep = None
        self.already_tested = False
        self.already_passed = False
        self.level_number = 1
        self.significance =  None
        self.rank = None

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
        if not isinstance(data, Tree):
            pChild = Tree(data)
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
    
    def get_left_distance(self):
        return self.left_distance
    
    def get_similarity_score(self):
        return self.similarity_score
    
    def set_similarity_score(self, similarity_score=None):
        self.similarity_score = similarity_score
        
    def set_left_first_rep(self, pc):
        self.left_first_rep = pc
    
    def set_right_first_rep(self, pc):
        self.right_first_rep = pc
        
    def get_left_first_rep(self):
        return self.left_first_rep
    
    def get_right_first_rep(self):
        return self.right_first_rep
    
    def set_pvalue(self, pvalue):
        self.pvalue = pvalue
    
    def get_pvalue(self):
        return self.pvalue
    
    def set_qvalue(self, qvalue):
        self.qvalue = qvalue
    
    def get_qvalue(self):
        return self.qvalue
    
    def is_qualified_association(self, pvalue_threshold, pc_threshold, sim_threshold):
        #return True
    
        if (1.0 - self.get_left_distance() >= .25 or self.get_left_first_rep() >= .35) and \
            (1.0 - self.get_right_distance() >= .25 or self.get_right_first_rep() >= .35) :
            return True
        else:
            return False
        
    def is_bypass(self ):#
        
        #return False
        #sub_hepotheses = math.log(len(self.get_data()[0]) * len(self.get_data()[1]), 2)
        #/ len(self.get_data()[0])* len(self.get_data()[1]) or\
        #  test_level/ (hypotheses_tree_heigth - self.get_level_number()+1) or\
        #if self.get_qvalue()  >  self.get_pvalue() / test_level * self.get_level_number()/ hypotheses_tree_heigth or\
        #/ sub_hepotheses * (hypotheses_tree_heigth -self.get_level_number() +1):#/ self.get_level_number():#
        if self.get_qvalue()  > 1.0 - self.get_pvalue():# and\
            #(self.get_left_first_rep() > .6 and \
            #self.get_right_first_rep()> .6):
            print "bypass "#, sub_hepotheses, "log ", math.log(sub_hepotheses, 2), " l ",self.get_level_number()," q", self.get_qvalue()," p", self.get_pvalue()
            return True
        else:
            return False
    def report(self):
        print "\n--- hypothesis test based on permutation test"        
        print "---- pvalue                        :", self.get_pvalue()
        #if self.get_qvalue() <> 0.0:
        #    print "--- adjusted pvalue     :", self.get_qvalue()
        print "---- similarity_score score              :", self.get_similarity_score()
        print "---- first cluster's features      :", self.get_data()[0]
        print "---- first cluster similarity_score      :", 1.0 - self.get_left_distance()
        print "---- first pc of the first cluster :", self.get_left_first_rep()
        print "---- second cluster's features     :", self.get_data()[1]
        print "---- second cluster similarity_score     :", 1.0 - self.get_right_distance()
        print "---- first pc of the second cluster:", self.get_right_first_rep(), "\n"
    
    def set_rank(self, rank= None):
        self.rank = rank
        
    def get_rank(self):
        return self.rank
    
    #def is_qualified_association(self):
    #    return self.significance
    
    #def set_significance_status(self):
    #    self.significance = True
        
    def set_level_number(self, level_number):
        self.level_number = level_number
        
    def get_level_number(self):
        return self.level_number
    
    def set_significance_status(self, significance):
        self.significance = significance
    
    def get_significance_status(self):
        return self.significance
    
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
            return Exception("Empty Tree")

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
                aOut.append(pTree.get_data())

        if self.m_queue:
            self = self.m_queue.pop()
        elif self.m_arrayChildren:
            pSelf = self.m_arrayChildren.pop() 
            self.m_queue = self.m_arrayChildren
            self = pSelf 
        return pTmp 

    def prune(self,):
        """
        Return a pruned version of the tree, with new starting node(s) 

        """
        pass

    def slice(self,):
        """
        Slice tree, giving arise to thinner cross sections of the tree, still put together by hierarchy 
        """
        pass 


    # ##NOTE: I think doing the modification at run-time makes a lot more sense, and is a lot less complicated 

#==========================================================================#
# FUNCTORS   
#==========================================================================#

def lf2tree(lf):
    """
    Functor converting from a layerform object to py Tree object 

    Parameters
    ------------
        lf : layerform 

    Returns 
    -------------
        t : py Tree object 
    """ 

    pass 

def tree2clust(pTree, exact=True):
    """
    Functor converting from a py Tree object to a scipy ClusterNode object.
    When exact is True, gives error when the map Tree() -> ClusterNode is not injective (more than 2 children per node)
    """

    pass 

def clust2tree(pTree):
    """
    Functor converting from a scipy ClusterNode to a py Tree object; 
    can always be done 
    """

    pass 

#==========================================================================#
# METHODS  
#==========================================================================#

def is_tree(pObj):
    """
    duck type implementation for checking if
    object is ClusterNode or Tree, more or less
    """

    try:
        pObj.get_data 
        return True 
    except Exception:
        return False 


def hclust(pArray, labels=None, strMetric="nmi", cluster_method="single", bTree=False, plotting_result = False , output_dir = "./"):
    """
    Performs hierarchical clustering on an numpy array 

    Parameters
    ------------
        pArray : numpy array 
        pdist_metric : str 
        cluster_method : str
        bTree : boolean 

    Returns
    -------------

        Z : numpy array or ClusterNode object 

    Examples
    ---------------

    * Pearson correlation 1::

        from numpy import array 
        from scipy.cluster.hierarchy import dendrogram, linkage
        import scipy.cluster.hierarchy 
        import py

        x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])

        lxpearson = py.hierarchy.hclust( x, pdist_metric = py.distance.cord )

        dendrogram(lxpearson)    

    .. plot::

        from numpy import array 
        from scipy.cluster.hierarchy import dendrogram, linkage
        import scipy.cluster.hierarchy 
        import py

        x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])

        lxpearson = py.hierarchy.hclust( x, pdist_metric = py.distance.cord )

        dendrogram(lxpearson)    

    * Pearson correlation 2::

        from numpy import array 
        from scipy.cluster.hierarchy import dendrogram, linkage
        import scipy.cluster.hierarchy 
        import py

        y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0], [0.015625,0.125,0.421875,1.0]])

        lypearson = py.hierarchy.hclust( y, pdist_metric = py.distance.cord )

        dendrogram(lypearson)
    
    .. plot::

        from numpy import array 
        from scipy.cluster.hierarchy import dendrogram, linkage
        import scipy.cluster.hierarchy 
        import py

        y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0], [0.015625,0.125,0.421875,1.0]])

        lypearson = py.hierarchy.hclust( y, pdist_metric = py.distance.cord )

        dendrogram(lypearson)

    * Spearman correlation 1::

        from numpy import array 
        from scipy.cluster.hierarchy import dendrogram, linkage
        import scipy.cluster.hierarchy 
        import py

        x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])

        lxspearman = py.hierarchy.hclust( x, pdist_metric = lambda u,v: py.distance.cord(u,v,method="spearman") )

        dendrogram(lxspearman)

    .. plot::

        from numpy import array 
        from scipy.cluster.hierarchy import dendrogram, linkage
        import scipy.cluster.hierarchy 
        import py

        x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])

        lxspearman = py.hierarchy.hclust( x, pdist_metric = lambda u,v: py.distance.cord(u,v,method="spearman") )

        dendrogram(lxspearman)

    * Spearman correlation 2::

        from numpy import array 
        from scipy.cluster.hierarchy import dendrogram, linkage
        import scipy.cluster.hierarchy 
        import py

        y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0], [0.015625,0.125,0.421875,1.0]])

        lyspearman = py.hierarchy.hclust( y, pdist_metric = lambda u,v: py.distance.cord(u,v,method="spearman") )

        dendrogram(lyspearman)

    .. plot::
        
        from numpy import array 
        from scipy.cluster.hierarchy import dendrogram, linkage
        import scipy.cluster.hierarchy 
        import py

        y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0], [0.015625,0.125,0.421875,1.0]])

        lyspearman = py.hierarchy.hclust( y, pdist_metric = lambda u,v: py.distance.cord(u,v,method="spearman") )

        dendrogram(lyspearman)
    
    * Mutual Information 1::

        from numpy import array 
        from scipy.cluster.hierarchy import dendrogram, linkage
        import scipy.cluster.hierarchy 
        import py

        x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])
        dx = halla.discretize( x, iN = None, method = None, aiSkip = [1,3] )

        lxmi = py.hierarchy.hclust( dx, pdist_metric = py.distance.nmid )

        dendrogram(lxmi)

    .. plot::

        from numpy import array 
        from scipy.cluster.hierarchy import dendrogram, linkage
        import scipy.cluster.hierarchy 
        import py

        x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])
        dx = py.discretize( x, iN = None, method = None, aiSkip = [1,3] )

        lxmi = py.hierarchy.hclust( dx, pdist_metric = py.distance.nmid )

        dendrogram(lxmi)

    * Mutual Information 2::

        from numpy import array 
        from scipy.cluster.hierarchy import dendrogram 
        import py

        y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0], [0.015625,0.125,0.421875,1.0]])
        dy = halla.discretize( y, iN = None, method = None, aiSkip = [1] )        

        lymi = py.hierarchy.hclust( dy, pdist_metric = py.distance.nmid )

        dendrogram( lymi )    

    .. plot::

        from numpy import array 
        from scipy.cluster.hierarchy import dendrogram 
        import py

        y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0], [0.015625,0.125,0.421875,1.0]])
        dy = py.discretize( y, iN = None, method = None, aiSkip = [1] )        

        lymi = py.hierarchy.hclust( dy, pdist_metric = py.distance.nmid )

        dendrogram( lymi )    

    Notes 
    -----------

        This hclust function is not quite right for the MI case. Need a generic MI function that can take in clusters of RV's, not just single ones 
        Use the "grouping property" as discussed by Kraskov paper. 
    """
    pMetric = distance.c_hash_metric[strMetric] 
    # # Remember, pMetric is a notion of _strength_, not _distance_ 
    # print str(pMetric)
    def pDistance(x, y):
        return  1.0 - pMetric(x, y)

    
    # print "Distance",D
    #plt.figure(figsize=(len(labels)/10.0 + 5.0, 5.0))
    D = pdist(pArray, metric=pDistance)
    if plotting_result:
        global fig_num
        print "--- plotting heatmap for Dataset", str(fig_num)," ... "
        #plt.figure("Hierarcichal clusters of Dataset "+ str(fig_num),  dpi=300)
        #if labels:
        #    scipy.cluster.hierarchy.dendrogram(Z, labels=labels, leaf_rotation=90)
        #else:
        #    scipy.cluster.hierarchy.dendrogram(Z)
        #plt.gcf()
        #global fig_num
        Z = plot.heatmap(pArray, D, xlabels_order = [], xlabels = labels, filename= output_dir+"/hierarchical_heatmap_" + str(fig_num))
        #plt.savefig(output_dir+"/Dendrogram1_" + str(fig_num) + ".pdf")
        fig_num += 1
    else:
        
        Z = linkage(D, metric=pDistance, method= "single")
        #plt.close("all")
    # print "Linkage Matrix:", Z
    # print fcluster(Z, .75 )
    # print fcluster(Z, .9 )
    # print fcluster(Z, .3 )
    
    # cutted_Z = np.where(Z[:,2]<.7)
    # print  cutted_Z 
    #print  Z
    # scipy.all( (Z[:,3] >= .4, Z[:,3] <= .6), axis=0 ).nonzero()
    # print pos.distance()
    return to_tree(Z) if bTree else Z 


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

    # if isinstance( list_clusternode, scipy.cluster.hierarchy.ClusterNode ):
    #     apClusterNode = [list_clusternode]
    # else:
    #     apClusterNode = list_clusternode

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

    Should be designed to handle both ClusterNode and Tree types 
    """ 

    bTree = is_tree(pClusterNode)

    func = pFunction if not bTree else lambda x: x.get_data() 

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
            return [(iLevel, p.get_data()) for p in apParents ] + reduce_tree_by_layer(new_apParents, iLevel=iLevel + 1)

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

    import copy 

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

    BUGBUG: Need get_layer to work with ClusterNode and Tree objects as well! 
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
    Extends `spawn_clusternode` to the py.hierarchy.Tree object 
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

    # print "X:"
    # print X

    # print "D:"
    # print D

    # print "Mean Tau:"
    # print A 
    
    # assert(numpy.any(A))



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

def _is_stop(ClusterNode, dataSet, max_dist_cluster, threshold = None):
        node_indeces = reduce_tree(ClusterNode)
        first_PC = stats.pca_explained_variance_ratio_(dataSet[array(node_indeces)])[0]
        if ClusterNode.is_leaf():# or _percentage(ClusterNode.dist, max_dist_cluster) < .1 or first_PC > .9:
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
        for i in range(len(sub_clusters)):
            if sub_clusters[i].dist > 0.0:
                aDist += [sub_clusters[i].dist]
            if max_dist_node.dist < sub_clusters[i].dist:
                max_dist_node = sub_clusters[i]
        # print "Max Distance in this level", _percentage(max_dist_node.dist)
        if not max_dist_node.is_leaf():
            sub_clusters += truncate_tree([max_dist_node], level=0, skip=1)
            sub_clusters.remove(max_dist_node)
        else:
            break
    if     aDist:
        next_dist = _percentage(numpy.min(aDist))
    # print len(sub_clusters), n            
    return sub_clusters , next_dist
def _cutree (clusterNodelist, first = False):
    clusterNode = clusterNodelist
    n = clusterNode[0].get_count()
    number_of_sub_cluters_threshold = round(math.log(n, 2))#* 2 #if first else round(math.log(n, 2)) 
    #print "n: ", n
    sub_clusters = []
    while clusterNode :
        temp_apChildren = []
        sub_clusterNode = truncate_tree(clusterNode, level=0, skip=1)
        for node in clusterNode:
            if node.is_leaf():
                sub_clusterNode += [node]
        sub_clusters = sub_clusterNode
        clusterNode = temp_apChildren
    while len(sub_clusters) < number_of_sub_cluters_threshold:
        max_dist_node = sub_clusters[0]
        for i in range(len(sub_clusters)):
            if max_dist_node.dist < sub_clusters[i].dist:
                max_dist_node = sub_clusters[i]
        # print "Max Distance in this level", _percentage(max_dist_node.dist)
        if not max_dist_node.is_leaf():
            sub_clusters += truncate_tree([max_dist_node], level=0, skip=1)
            sub_clusters.remove(max_dist_node)
        else:
            break
    #print "len of subcluster: ", len(sub_clusters)
    return sub_clusters

    
def couple_tree(apClusterNode1, apClusterNode2, pArray1, pArray2, strMethod="uniform", strLinkage="min", func="nmi", threshold = None):
    
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
    tH : Tree object 

    Examples
    ----------------
    """
    
    X, Y = pArray1, pArray2
    global max_dist_cluster1 
    max_dist_cluster1 = max (node.dist for node in apClusterNode1)
    
    global max_dist_cluster2 
    max_dist_cluster2 = max (node.dist for node in apClusterNode2)
    # print "Max distance:", max_dist
    # if not afThreshold:    
    #     afThreshold = [stats.alpha_threshold(a, fAlpha, func ) for a in [pArray1,pArray2]]
    
    # x_threshold, y_threshold = afThreshold[0], afThreshold[1]
    # print "x_threshold, y_threshold:", x_threshold, y_threshold
    # aTau = [] ### Did the child meet the intra-dataset confidence cut-off? If not, its children will continue to be itself. 
        #### tau_hat <= tau 
        #### e.g.[(True,False),(True,True),]

    # # hash containing string mappings for deciders 

    hashMethod = {"min": _decider_min,
                "max": _decider_max,
                }

    pMethod = hashMethod[strLinkage]  # #returns 'aNode x aNode -> bool' object 

    #-------------------------------------#
    # Parsing Steps                       #
    #-------------------------------------#
    
    # # Unalias data structure so this does not alter the original data type
    # # Fix for depth 
    aiGlobalDepth1 = [get_depth(ap) for ap in apClusterNode1]
    aiGlobalDepth2 = [get_depth(ap) for ap in apClusterNode2]
    
    iMaxDepth = max(max(aiGlobalDepth1), max(aiGlobalDepth2))
    iMinDepth = min(min(aiGlobalDepth1), min(aiGlobalDepth2))
    
    # apClusterNode1 = [fix_clusternode(apClusterNode1[i], iExtend = iMaxDepth - aiGlobalDepth1[i]) for i in range(len(apClusterNode1))]
    # apClusterNode2 = [fix_clusternode(apClusterNode2[i], iExtend = iMaxDepth - aiGlobalDepth2[i]) for i in range(len(apClusterNode2))]

    # print "Hierarchical TREE 1 ", reduce_tree_by_layer(apClusterNode1)
    # print "Hierarchical TREE 2 ", reduce_tree_by_layer(apClusterNode2)

    aOut = []

    # Create the root of the coupling tree
    for a, b in itertools.product(apClusterNode1, apClusterNode2):
        data1 = reduce_tree(a)
        data2 = reduce_tree(b)
    pStump = Tree([data1, data2])
    pStump.set_level_number(0)
    aOut.append(pStump)
    
    apChildren1 = _cutree (apClusterNode1, first = True)
    apChildren2 = _cutree (apClusterNode2, first =  True)
    #print apChildren1
    #print apChildren2
    childList = []
    L = []    
    for a, b in itertools.product(apChildren1, apChildren2):
        data1 = reduce_tree(a)
        data2 = reduce_tree(b)
        tempTree = Tree(data=[data1, data2], left_distance=a.dist, right_distance=b.dist)
        tempTree.set_level_number(1)
        childList.append(tempTree)
        L.append((tempTree, (a, b)))
    pStump.add_children(childList)
    #print "child list:", childList
    next_L = []
    level_number = 2
    while L:
        (pStump, (a, b)) = L.pop(0)
        #print "child list:", tempNode
        #continue
        data1 = reduce_tree(a)
        data2 = reduce_tree(b)
                
        bTauX = _is_stop(a, X, max_dist_cluster1, threshold)  # ( _min_tau(X[array(data1)], func) >= x_threshold ) ### parametrize by mean, min, or max
        bTauY = _is_stop(b, Y, max_dist_cluster2, threshold)  # ( _min_tau(Y[array(data2)], func) >= y_threshold ) ### parametrize by mean, min, or max
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
            apChildren1 = _cutree([a])  
        else:
            apChildren1 = [a]
        if not bTauY:
            apChildren2 = _cutree([b])
        else:
            apChildren2 = [b]

        LChild = [(c1, c2) for c1, c2 in itertools.product(apChildren1, apChildren2)] 
        childList = []
        while LChild:
            (a1, b1) = LChild.pop(0)
            data1 = reduce_tree(a1)
            data2 = reduce_tree(b1)
            tempTree = Tree(data=[data1, data2], left_distance=a1.dist, right_distance=b1.dist)
            tempTree.set_level_number(level_number)
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
    #print "Coupled Tree", reduce_tree_by_layer(aOut)
    global hypotheses_tree_heigth
    hypotheses_tree_heigth = level_number
    #print "Number of levels after coupling", level_number-1
    return aOut
        
def naive_all_against_all(pArray1, pArray2, fdr= "BH", decomposition = "pca", method="permutation", metric="nmi", fQ=0.1,
    bVerbose=False, iIter=1000):
    
    pHashMethods = {"permutation" : stats.permutation_test,
                        "permutation_test_by_medoid": stats.permutation_test_by_medoid,
                        
                        # parametric tests
                        "parametric_test_by_pls_pearson": stats.parametric_test_by_pls_pearson,
                        "parametric_test_by_representative": stats.parametric_test_by_representative,
                        "parametric_test" : stats.parametric_test,
                        
                        # G-Test
                        "g-test":stats.g_test
                        }

    strMethod = method
    pMethod = pHashMethods[strMethod]
    iRow = len(pArray1)
    iCol = len(pArray2)
    
    aOut = [] 
    aFinal = []
    aP = []
    tests = []
    t = 0
    #print iRow, iCol
    for i, j in itertools.product(range(iRow), range(iCol)):
        test =  Tree(left_distance=0.0, right_distance=0.0)
        data = [[i], [j]]
        test.add_data(data)
        #print i, j
        fP, similarity, left_rep, right_rep = pMethod(array([pArray1[i]]), array([pArray2[j]]), metric = metric, decomposition = decomposition, iIter=iIter)
        test.set_pvalue(fP)
        test.set_similarity_score(similarity)
        test.set_left_first_rep(1.0)
        test.set_right_first_rep(1.0)
        test.set_qvalue(fP)
        aP.append(fP)
        tests.append(test)
        
        if fdr == "simple":
            if fP <= fQ:
                print "-- association after BH fdr controlling"
                if bVerbose:
                    tests[t].report()
                aOut.append(tests[t])
                #aFinal.append([Current_Family_Children[i].get_data(), float(aP[i]), aP_adjusted[i]])
                aFinal.append(tests[t])
            else :
                #aOut.append([Current_Family_Children[i].get_data(), float(aP[i]), aP_adjusted[i]])
                aOut.append(tests[t])
        t += 1    
    #print "number of tetsts", t    
    if fdr  in ["BH", "BHF", "BHL", "BHY"]:    
        aP_adjusted, pRank = stats.p_adjust(aP, fQ)
        for i in range(len(tests)):
            tests[i].set_qvalue(aP_adjusted[i])
            tests[i].set_rank(pRank[i])
        max_r_t = 0
                #print "aP", aP
                #print "aP_adjusted: ", aP_adjusted  
        for i in range(len(tests)):
            if aP[i] <= aP_adjusted[i] and max_r_t <= pRank[i]:
                max_r_t = pRank[i]
                #print "max_r_t", max_r_t
        for i in range(len(aP)):
            if pRank[i] <= max_r_t:
                print "-- associations after BH fdr controlling"
                if bVerbose:
                    tests[i].report()
                aOut.append(tests[i])
                #aFinal.append([Current_Family_Children[i].get_data(), float(aP[i]), aP_adjusted[i]])
                aFinal.append(tests[i])
            else :
                #aOut.append([Current_Family_Children[i].get_data(), float(aP[i]), aP_adjusted[i]])
                aOut.append(tests[i])
    #aOut_header = zip(*aOut)[0]
    #aOut_adjusted = stats.p_adjust(zip(*aOut)[1])

    #return zip(aOut_header, aOut_adjusted)
    # return numpy.reshape( aOut, (iRow,iCol) )
    return aFinal, aOut

def traverse_by_layer(pClusterNode1, pClusterNode2, pArray1, pArray2, pFunction=None):
    """

    Useful function for doing all-against-all comparison between nodes in each layer 

    traverse two trees at once, applying function `pFunction` to each layer pair 

    latex: $pFunction: index1 \times index2 \times data1 \times data2 \rightarrow \mathbb{R}^k, $ for $k$ the size of the cross-product set per layer 

    Parameters
    ----------------
        pClusterNode1, pClusterNode2, pArray1, pArray2, pFunction
    
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
            aLayerOut.append(pFunction(aLayer1[i], aLayer2[j], pArray1, pArray2))

        aOut.append(aLayerOut)

    return aOut 

#### Perform all-against-all per layer, without adherence to hierarchical structure at first
def layerwise_all_against_all(pClusterNode1, pClusterNode2, pArray1, pArray2, adjust_method="BH"):
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

    traverse_out = traverse_by_layer(pClusterNode1, pClusterNode2, pArray1, pArray2)  # #just gives me the coupled indices 

    for layer in traverse_out:
        aLayerOut = [] 
        aPval = [] 
        for item in layer:
            fPval = stats.permutation_test_by_representative(pArray1[array(item[0])], pArray2[array(item[1])])
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

def hypotheses_testing(pTree, pArray1, pArray2, method="permutation", metric="nmi", fdr= "BHY", p_adjust="BH", fQ=0.1,
    iIter=1000, pursuer_method="nonparameteric", decomposition = "pca", bVerbose=False, afThreshold=.2, fAlpha=0.05):
    """
    Perform all-against-all on a hypothesis tree.

    Notes:
        Right now, return aFinal, aOut 


    Parameters
    ---------------

        pTree 
        pArray1
        pArray2
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
    X, Y = pArray1, pArray2 

    if bVerbose:
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
    
        
    pHashMethods = {"permutation" : stats.permutation_test,
                        "permutation_test_by_multiple_representative" : stats.permutation_test_by_multiple_representative,
                        "permutation_test_by_medoid": stats.permutation_test_by_medoid,
                        
                        # parametric tests
                        "parametric_test_by_pls_pearson": stats.parametric_test_by_pls_pearson,
                        "parametric_test_by_representative": stats.parametric_test_by_representative,
                        "parametric_test" : stats.parametric_test,
                        
                        # G-Test
                        "g-test":stats.g_test
                        }

    strMethod = method
    pMethod = pHashMethods[strMethod]
    
    def _simple_hypothesis_testing():
        apChildren = [pTree]
        number_performed_tests = 0
        number_passed_tests = 0
        next_level_apChildren = []
        level = 1
        while apChildren:
            Current_Family_Children = apChildren.pop(0).get_children()
            number_performed_tests += len(Current_Family_Children)
            
            # claculate nominal p-value
            for i in range(len(Current_Family_Children)):
                Current_Family_Children[i].set_pvalue(_actor(Current_Family_Children[i]))
            
                aOut.append([Current_Family_Children[i].get_data(), Current_Family_Children[i].get_pvalue(), Current_Family_Children[i].get_pvalue()])
                if Current_Family_Children[i].is_qualified_association(pvalue_threshold = fQ, pc_threshold = afThreshold , sim_threshold = afThreshold):
                    Current_Family_Children[i].report()
                    number_passed_tests += 1
                    aFinal.append([Current_Family_Children[i].get_data(), Current_Family_Children[i].get_pvalue(), Current_Family_Children[i].get_pvalue()])
                elif Current_Family_Children[i].get_pvalue() > fQ and Current_Family_Children[i].get_pvalue() <= 1.0 - fQ:
                    next_level_apChildren.append(Current_Family_Children[i])
                    if bVerbose: 
                        print "Conitinue, gray area with p-value:", Current_Family_Children[i].get_pvalue()
                elif Current_Family_Children[i].is_bypass():
                    if bVerbose:
                        print "Stop: no chance of association by descending", Current_Family_Children[i].get_pvalue()
            if not apChildren:
                if bVerbose:
                    print "Hypotheses testing level ", level, " is finished."
                # number_performed_test += len(next_level_apChildren)
                apChildren = next_level_apChildren
                level += 1
                next_level_apChildren = []
            
        print "--- number of performed tests:", number_performed_tests
        print "--- number of passed tests without FDR controlling:", number_passed_tests        
        return aFinal, aOut
    
    def _bh_family_testing():
        apChildren = [pTree]
        level = 1
        number_performed_tests = 0 
        number_passed_tests = 0
        next_level_apChildren = []
        while apChildren:
            Current_Family_Children = apChildren.pop(0).get_children()
            number_performed_tests += len(Current_Family_Children)
            #aP = []
            # claculate nominal p-value
            for i in range(len(Current_Family_Children)):
                #print "aP", i, _actor(Current_Family_Children[i])
                #print Current_Family_Children[i]
                Current_Family_Children[i].set_pvalue(_actor(Current_Family_Children[i]))
            aP = [ Current_Family_Children[i].get_pvalue() for i in range(len(Current_Family_Children)) ]
            # claculate adjusted p-value
            aP_adjusted, pRank = stats.p_adjust(aP, fQ)
            for i in range(len(Current_Family_Children)):
                Current_Family_Children[i].set_qvalue(aP_adjusted[i])
                Current_Family_Children[i].set_rank(pRank[i])
                
                
            max_r_t = 0
            #print "aP", aP
            #print "aP_adjusted: ", aP_adjusted  
            for i in range(len(Current_Family_Children)):
                if aP[i] <= aP_adjusted[i] and max_r_t <= pRank[i]:
                    max_r_t = pRank[i]
                    # print "max_r_t", max_r_t
            for i in range(len(aP)):
                if pRank[i] <= max_r_t and Current_Family_Children[i].is_qualified_association(pc_threshold = afThreshold, sim_threshold = afThreshold, pvalue_threshold = fQ):
                    number_passed_tests += 1
                    print "-- associations after fdr correction"
                    if bVerbose:
                        Current_Family_Children[i].report()
                    #aOut.append([Current_Family_Children[i].get_data(), float(aP[i]), aP_adjusted[i]])
                    aOut.append(Current_Family_Children[i])
                    #aFinal.append([Current_Family_Children[i].get_data(), float(aP[i]), aP_adjusted[i]])
                    aFinal.append(Current_Family_Children[i])
                else :
                    #aOut.append([Current_Family_Children[i].get_data(), float(aP[i]), aP_adjusted[i]])
                    aOut.append(Current_Family_Children[i])
                    #if not Current_Family_Children[i].is_leaf():  # and aP[i] <= 1.0-fQ:#aP[i]/math.sqrt((len(Current_Family_Children[i].get_data()[0]) * len(Current_Family_Children[i].get_data()[1]))) <= 1.0-fQ:#
                    if Current_Family_Children[i].is_bypass() :
                        if bVerbose:
                            print "Bypass, no hope to find an association in the branch with p-value: ", \
                    aP[i], " and ", len(Current_Family_Children[i].get_children()), \
                     " sub-hypotheses.", Current_Family_Children[i].get_data()[0], \
                      "   ", Current_Family_Children[i].get_data()[1]
                        
                    elif Current_Family_Children[i].is_leaf():
                        if bVerbose:
                            print "End of branch, leaf!"
                        # aOut.append( [Current_Family_Children[i].get_data(), float(aP[i]), float(aP[i])] )
                    else:
                        if bVerbose:
                            print "Gray area with p-value:", aP[i]
                        next_level_apChildren.append(Current_Family_Children[i])
                    
            if not apChildren:
                if bVerbose:
                    print "Family Hypotheses testing at level ", level, " is finished."
                print "Family hypotheses testing at level ", level, " is finished."
                # number_performed_test += len(next_level_apChildren)
                apChildren = next_level_apChildren
                level += 1
                next_level_apChildren = []
        print "--- number of performed tests:", number_performed_tests
        print "--- number of passed tests after FDR controlling:", number_passed_tests                                        
        return aFinal, aOut
    def _bh_all_testing():
        apChildren = [pTree]
        level = 1
        number_performed_tests = 0 
        number_passed_tests = 0
        next_level_apChildren = []
        current_level_tests = []
        all_performed_tests = []
        all_aP = []
        while apChildren:
            current_level_tests.extend(apChildren.pop(0).get_children())
            
            if len (apChildren) != 0 :
                continue
            else:
                number_performed_tests += len(current_level_tests)
                for i in range(len(current_level_tests)):
                    current_level_tests[i].set_pvalue(_actor(current_level_tests[i]))
                aP = [ current_level_tests[i].get_pvalue() for i in range(len(current_level_tests)) ]
                all_aP.extend(aP)
                all_performed_tests.extend(current_level_tests)
                # claculate adjusted p-value
                aP_adjusted, pRank = stats.p_adjust(all_aP, fQ)
                for i in range(len(all_performed_tests)):
                    all_performed_tests[i].set_qvalue(aP_adjusted[i])
                    all_performed_tests[i].set_rank(pRank[i])

                max_r_t = 0
                for i in range(len(all_performed_tests)):
                    if all_aP[i] <= aP_adjusted[i] and max_r_t <= pRank[i]:
                        max_r_t = pRank[i]
                        # print "max_r_t", max_r_t
                for i in range(len(all_aP)):
                    if pRank[i] <= max_r_t and not all_performed_tests[i].already_passed:
                        number_passed_tests += 1
                        all_performed_tests[i].already_passed = True
                        all_performed_tests[i].already_tested = True
                        print "-- associations after fdr correction"
                        if bVerbose:
                            all_performed_tests[i].report()
                        #aOut.append([Current_Family_Children[i].get_data(), float(aP[i]), aP_adjusted[i]])
                        aOut.append(all_performed_tests[i])
                        #aFinal.append([Current_Family_Children[i].get_data(), float(aP[i]), aP_adjusted[i]])
                        aFinal.append(all_performed_tests[i])
                    elif not all_performed_tests[i].already_tested:
                        all_performed_tests[i].already_tested = True
                        #aOut.append([Current_Family_Children[i].get_data(), float(aP[i]), aP_adjusted[i]])
                        #print i, range(len(current_level_tests)), current_level_tests[i]
                        aOut.append(all_performed_tests[i])
                        #if not Current_Family_Children[i].is_leaf():  # and aP[i] <= 1.0-fQ:#aP[i]/math.sqrt((len(Current_Family_Children[i].get_data()[0]) * len(Current_Family_Children[i].get_data()[1]))) <= 1.0-fQ:#
                        if all_performed_tests[i].is_bypass() :
                            if bVerbose:
                                print "Bypass, no hope to find an association in the branch with p-value: ", \
                        aP[i], " and ", len(all_performed_tests[i].get_children()), \
                         " sub-hypotheses.", all_performed_tests[i].get_data()[0], \
                          "   ", all_performed_tests[i].get_data()[1]
                            
                        elif all_performed_tests[i].is_leaf():
                            if bVerbose:
                                print "End of branch, leaf!"
                            # aOut.append( [Current_Family_Children[i].get_data(), float(aP[i]), float(aP[i])] )
                        else:
                            if bVerbose:
                                print "Gray area with p-value:", aP[i]
                            next_level_apChildren.append(all_performed_tests[i])
                        
                        if bVerbose:
                            print "Hypotheses testing level ", level, " is finished."
                        # number_performed_test += len(next_level_apChildren)
                apChildren = next_level_apChildren
                #print "level: ", level, "#test: ", number_performed_tests
                level += 1
                next_level_apChildren = []
                current_level_tests = []
                aP = []

        print "--- number of performed tests:", number_performed_tests
        print "--- number of passed tests after FDR controlling:", number_passed_tests                                        
        return aFinal, aOut

    def _bh_level_testing():
        q = fQ
        apChildren = [pTree]
        n1 = len(pTree.get_data()[0])
        n2 = len(pTree.get_data()[1])
        n1 = n1/math.log(n1,2)
        n2 = n2/math.log(n2,2)
        level = 1
        number_performed_tests = 0 
        number_passed_tests = 0
        next_level_apChildren = []
        current_level_tests = []
        #temp_current_level_tests = []
        leaves_hypotheses = []
        #previous_unqualified_hypotheses = []
        while apChildren:
            temp_hypothesis = apChildren.pop(0)
            #if temp_hypothesis.get_children():
            temp_sub_hypotheses = temp_hypothesis.get_children()
            if len(temp_sub_hypotheses) == 0:
                leaves_hypotheses.append(temp_hypothesis)
            if temp_hypothesis.get_significance_status() != None:
                for i in range(len(temp_sub_hypotheses)):
                    temp_sub_hypotheses[i].set_significance_status(temp_hypothesis.get_significance_status())
                    temp_sub_hypotheses[i].set_pvalue(temp_hypothesis.get_pvalue()) 
                    temp_sub_hypotheses[i].set_qvalue(temp_hypothesis.get_qvalue())
            else:
                number_performed_tests += len(temp_sub_hypotheses)
            #temp_current_level_tests.extend(temp_sub_hypotheses)
            current_level_tests.extend(temp_sub_hypotheses)
            if len (apChildren) > 0:
                continue
            '''             
            current_level_tests.extend(previous_unqualified_hypotheses)
            previous_unqualified_hypotheses = []
            for i in range(len(temp_current_level_tests)):
                if len(temp_current_level_tests[i].get_data()[0]) >= n1/2 and len(temp_current_level_tests[i].get_data()[1]) >= n2/2:
                    current_level_tests.append(temp_current_level_tests[i]) 
                else:
                    previous_unqualified_hypotheses.append(temp_current_level_tests[i])
             '''   
            if len(current_level_tests) > 0 :
                #number_performed_tests += len(current_level_tests)
                #if n1 < 2 and n2 < 2:
                current_level_tests.extend(leaves_hypotheses)
                print "number of hypotheses in level:", len(current_level_tests)
                #if not len(current_level_tests):
                #    continue
                p_values = multiprocessing_actor(_actor, current_level_tests, pMethod, pArray1, pArray2, metric, decomposition, iIter)
                
                for i in range(len(current_level_tests)):
                    current_level_tests[i].set_pvalue(p_values[i])
                    #print "Pvalue", i, " :", p_values[i]
                    
                aP = [ current_level_tests[i].get_pvalue() for i in range(len(current_level_tests)) ]
                # claculate adjusted p-value
                aP_adjusted, pRank = stats.p_adjust(aP, q)
                for i in range(len(current_level_tests)):
                    if current_level_tests[i].get_qvalue() == None:
                        current_level_tests[i].set_qvalue(aP_adjusted[i])
                    current_level_tests[i].set_rank(pRank[i])

                max_r_t = 0
                for i in range(len(current_level_tests)):
                    if current_level_tests[i].get_pvalue() <= current_level_tests[i].get_qvalue() and max_r_t <= current_level_tests[i].get_rank():
                        max_r_t = current_level_tests[i].get_rank()
                        #print "max_r_t", max_r_t
                for i in range(len(current_level_tests)):
                    if current_level_tests[i].get_significance_status() == None and\
                    current_level_tests[i].get_rank() <= max_r_t and\
                    current_level_tests[i].is_qualified_association(pc_threshold = afThreshold, sim_threshold = afThreshold, pvalue_threshold = fQ):
                        number_passed_tests += 1
                        if bVerbose:
                            current_level_tests[i].report()
                        print "-- associations after fdr correction"
                        current_level_tests[i].set_significance_status(True)
                        aOut.append(current_level_tests[i])
                        aFinal.append(current_level_tests[i])
                        #next_level_apChildren.append(current_level_tests[i])
                    else:
                        if current_level_tests[i].get_significance_status() == None and current_level_tests[i].is_bypass():# and current_level_tests[i].get_significance_status() == None:
                            current_level_tests[i].set_significance_status(False)
                            aOut.append(current_level_tests[i])
                            if bVerbose:
                                print "Bypass, no hope to find an association in the branch with p-value: ", \
                        aP[i], " and ", len(current_level_tests[i].get_children()), \
                         " sub-hypotheses.", current_level_tests[i].get_data()[0], \
                          "   ", current_level_tests[i].get_data()[1]
                            #next_level_apChildren.append(current_level_tests[i])
                            
                        elif current_level_tests[i].is_leaf():
                            if bVerbose:
                                print "End of branch, leaf!"
                            #next_level_apChildren.append(current_level_tests[i])
                            if current_level_tests[i].get_significance_status() == None:
                                current_level_tests[i].set_significance_status(False)
                                aOut.append(current_level_tests[i])
                            
                        else:
                            if bVerbose:
                                print "Gray area with p-value:", aP[i]
                            #next_level_apChildren.append(current_level_tests[i])
                        
                        if bVerbose:
                            print "Hypotheses testing level ", level, " is finished."                        
                            
            apChildren = current_level_tests #next_level_apChildren #
            print "Hypotheses testing level ", level, "with ",len(current_level_tests), "hypotheses is finished."
            level += 1
            #q = fQ - fQ*max_r_t/100.0
            if len(current_level_tests)>0:
                #q = fQ - fQ*max_r_t/len(current_level_tests)
                #print "Next level q:", q
                last_current_level_tests = current_level_tests
            next_level_apChildren = []
            current_level_tests = []
            #temp_current_level_tests = []
            last_current_level_tests = leaves_hypotheses
            leaves_hypotheses = []
            aP = []
            #return aFinal, aOut
            #n1 = n1 /math.log(n1,2) if n1 > 2 else 1
            #n2 = n2 /math.log(n2, 2) if n2 > 2 else 1
            #q = fQ - fQ*max_r_t/100.0 

        print "--- number of performed tests:", number_performed_tests
        print "--- number of passed tests after FDR controlling:", number_passed_tests 
        '''aFinal1= []
        aOut1=[]
        aP = [ last_current_level_tests[i].get_pvalue() for i in range(len(last_current_level_tests)) ]
                # claculate adjusted p-value
        aP_adjusted, pRank = stats.p_adjust(aP, q)
        max_r_t = 0
        for i in range(len(last_current_level_tests)):
            if last_current_level_tests[i].get_pvalue() <= aP_adjusted[i] and\
             max_r_t <= pRank[i]:
                max_r_t = pRank[i]
                #print "max_r_t", max_r_t
        for i in range(len(last_current_level_tests)):
            if pRank[i] <= max_r_t:
                aOut1.append(last_current_level_tests[i])
                aFinal1.append(last_current_level_tests[i])
            else:
                aOut1.append(last_current_level_tests[i])
        '''                                  
        return aFinal, aOut
    
    def _rh_hypothesis_testing():
        apChildren = [pTree]
        level = 1
        end_level_tests = []
        performed_tests = []
        round1_passed_tests = []
        global number_performed_test
        next_level_apChildren = []
        number_performed_test = 0
        while apChildren:
            Current_Family_Children = apChildren.pop(0).get_children()
            
            # print "Number of children:", len(Current_Family_Children)
            number_performed_test += len(Current_Family_Children)
            
            # claculate nominal p-value
            for i in range(len(Current_Family_Children)):
                Current_Family_Children[i].set_pvalue(_actor(Current_Family_Children[i]))
            aP = [ Current_Family_Children[i].get_pvalue() for i in range(len(Current_Family_Children)) ]
            
            for i in range(len(aP)):
                # print "NMI", Current_Family_Children[i].get_similarity_score()
                performed_tests.append([Current_Family_Children[i], float(aP[i])])    
                if Current_Family_Children[i].is_qualified_association(pc_threshold = afThreshold, sim_threshold = afThreshold, pvalue_threshold = fQ):
                    Current_Family_Children[i].report()
                    end_level_tests.append([Current_Family_Children[i], float(aP[i])])
                    round1_passed_tests.append([Current_Family_Children[i], float(aP[i])])
                elif Current_Family_Children[i].is_leaf():
                    end_level_tests.append([Current_Family_Children[i], float(aP[i])])
                    if bVerbose:
                        print "End of branch, leaf!"
                elif Current_Family_Children[i].is_bypass() :
                    if bVerbose:
                        print "Bypass, no hope to find an association in the branch with p-value: ", \
                    aP[i], " and ", len(Current_Family_Children[i].get_children()), \
                     " sub-hypotheses.", Current_Family_Children[i].get_data()[0], \
                      "   ", Current_Family_Children[i].get_data()[1]
                else:
                    if bVerbose:
                        print "Gray area with p-value:", aP[i]
                    next_level_apChildren.append(Current_Family_Children[i])
                
            if not apChildren:
                if bVerbose:
                    print "Hypotheses testing level ", level, " is finished.", "number of hypotheses in the next level: ", len(next_level_apChildren)
                apChildren = next_level_apChildren
                level += 1
                next_level_apChildren = []
        max_r_t = 0
        print "--- number of performed tests:", len(performed_tests)
        print "--- number of passed from nominal tests:", len(round1_passed_tests)
        print "--- number of tests in the end of branches:", len(end_level_tests)
        end_level_tests = array(end_level_tests)
        performed_tests = array(performed_tests)
        print "Nominal p-values", end_level_tests[:, 1]
        aP_adjusted, pRank = stats.p_adjust(end_level_tests[:, 1], fQ)
        print "ajusted pvalue: ", aP_adjusted
        for i in range(len(end_level_tests)):
            if end_level_tests[i][1] <= aP_adjusted[i] and max_r_t <= pRank[i]:
                max_r_t = pRank[i]
                #print "max_r_t", max_r_t
        for i in range(len(end_level_tests[:, 1])):
            if pRank[i] <= max_r_t:
                print "--- Pass with p-value:", end_level_tests[i][1], " adjusted_pvalue: ", aP_adjusted[i], end_level_tests[i][0].get_data()[0], end_level_tests[i][0].get_data()[1]
                aOut.append([end_level_tests[i][0].get_data(), float(end_level_tests[i][1]) , aP_adjusted[i]])
                aFinal.append([end_level_tests[i][0].get_data(), float(end_level_tests[i][1]) , aP_adjusted[i]])
            else :
                aOut.append([end_level_tests[i][0].get_data(), float(end_level_tests[i][1]) , aP_adjusted[i]])
        print "--- number of passed tests after FDR controllin:", len(aFinal)                                        
        return aFinal, aOut
    
    def _actor(pNode):
        """
        Performs a certain action at the node
        
            * E.g. compares two bags, reports distance and p-values 
        """
    
        aIndicies = pNode.get_data() 
        aIndiciesMapped = map(array, aIndicies)  # # So we can vectorize over numpy arrays
        dP, similarity, left_first_rep, right_first_rep = pMethod(pArray1[aIndiciesMapped[0]], pArray2[aIndiciesMapped[1]],  metric = metric, decomposition = decomposition, iIter=iIter)
        pNode.set_similarity_score(similarity)
        pNode.set_left_first_rep(left_first_rep)
        pNode.set_right_first_rep(right_first_rep)
        
            
        # aOut.append( [aIndicies, dP] ) #### dP needs to appended AFTER multiple hypothesis correction

        return dP 
    

    fdr_function = {"default": _bh_family_testing,
                            "BHF": _bh_family_testing,
                            "BHL":_bh_level_testing,
                            "BHA":_bh_all_testing,
                            "RH": _rh_hypothesis_testing,
                            "simple":_simple_hypothesis_testing}
    #======================================#
    # Execute 
    #======================================#
    strFDR = fdr
    pFDR = fdr_function[strFDR]
    aFinal, aOut = pFDR()

    #print "____Number of performed test:", number_performed_test
    return aFinal, aOut 
