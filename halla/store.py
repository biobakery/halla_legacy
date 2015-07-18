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

global D1_features_order , D2_features_order 
D1_features_order  = []
D2_features_order  = []

class HAllA():
    
    def __init__(self, X, Y): 
        self.distance = "nmi"
        self.decomposition = "pca" 
        self.fdr_function = "BHL"
        self.step_parameter = 1.0  # # a value between 0.0 and 1.0; a fractional value of the layers to be tested 
        self.q = .2  
        self.iterations = 1000
        self.p_adjust_method = "BH"
        self.randomization_method = "permutation"  # method to generate error bars 
        self.strStep = "uniform"
        self.verbose = False
        self.descending = "HAllA" 
            
        self.summary_method = "final"
        self.step_parameter = 1.0  # # a value between 0.0 and 1.0; a fractional value of the layers to be tested 
        self.output_dir = "./"
        self.plotting_results = False
        self.heatmap_all = False
        self.bypass_discretizing = False
        self.apply_stop_condition = False
        #==================================================================#
        # Mutable Meta Objects  
        #==================================================================#
        self.meta_array = array([None, None])
        self.meta_array[0] = X  # .append(X)
        self.meta_array[1] = Y  # .append(Y)
        #print self.meta_array[0]
        #print self.meta_array[1]
        self.meta_feature = None
        self.meta_threshold = None 
        self.meta_data_tree = [] 
        self.meta_hypothesis_tree = None 
        self.meta_alla = None  # results of all-against-all
        self.meta_out = None  # final output array; some methods (e.g. hypotheses_testing) have multiple outputs piped to both self.meta_alla and self.meta_out 
        self.meta_summary = None  # summary statistics 
        self.meta_report = None  # summary report 
        self.aOut = None  # summary output for naive approaches_
        self.aOutName1 = None 
        self.aOutName2 = None 
        self.robustness = .5
        self.strFile1 = None
        self.strFile2 = None
        self.outcome = None
        self.pvalues = None
        
        #==================================================================#
        # Static Objects  
        #==================================================================#

        self.__description__      = """
          _    _          _ _          
         | |  | |   /\   | | |   /\    
         | |__| |  /  \  | | |  /  \   
         |  __  | / /\ \ | | | / /\ \  
         | |  | |/ ____ \| | |/ ____ \ 
         |_|  |_/_/    \_\_|_/_/    \_\
                                       

        HAllA Object for hierarchical all-against-all association testing 
        """

        self.__doc__             = __doc__ 
        self.__version__          = "0.1.0"
        self.__author__             = ["Gholamali.Rahnavard", "YS Joseph Moon", "Curtis Huttenhower"]
        self.__contact__         = "gholamali.rahnavard@gmail.com"

        self.hash_decomposition = stats.c_hash_decomposition

        self.hash_metric          = distance.c_hash_metric 

        self.keys_attribute = ["__description__", "__version__", "__author__", "__contact__", "q", "distance", "iterations", "decomposition", "p_adjust_method", "randomization_method"]
        # # END INIT 

    #==================================================================#
    # Type Checking
    #==================================================================#

    def _check(self, pObject, pType, pFun=isinstance, pClause="or"):
        """
        Wrapper for type checking 
        """

        if (isinstance(pType, list) or isinstance(pType, tuple) or isinstance(pType, np.ndarray)):
            aType = pType 
        else:
            aType = [pType]

        return reduce(lambda x, y: x or y, [isinstance(pObject, t) for t in aType], False)

    def _cross_check(self, pX, pY, pFun=len):
        """
        Checks that pX and pY are consistent with each other, in terms of specified function pFun. 
        """

    def _is_meta(self, pObject):
        """    
        Is pObject an iterable of iterable? 
        """

        try: 
            pObject[0]
            return self._is_iter(pObject[0])    
        except IndexError:
            return False 

    def _is_empty(self, pObject):
        """
        Wrapper for both numpy arrays and regular lists 
        """
        
        aObject = array(pObject)

        return not aObject.any()

    # ## These functions are absolutely unncessary; get rid of them! 
    def _is_list(self, pObject):
        return self._check(pObject, list)

    def _is_tuple(self, pObject):
        return self._check(pObject, tuple)

    def _is_str(self, pObject):
        return self._check(pObject, str)

    def _is_int(self, pObject):
        return self._check(pObject, int)    

    def _is_array(self, pObject):
        return self._check(pObject, np.ndarray)

    def _is_1d(self, pObject):
    
        strErrorMessage = "Object empty; cannot determine type"
        bEmpty = self._is_empty(pObject)

        # # Enforce non-empty invariance 
        if bEmpty:
            raise Exception(strErrorMessage)

        # # Assume that pObject is non-empty 
        try:
            iRow, iCol = pObject.shape 
            return(iRow == 1) 
        except ValueError:  # # actual arrays but are 1-dimensional
            return True
        except AttributeError:  # # not actual arrays but python lists 
            return not self._is_iter(pObject[0])

    def _is_iter(self, pObject):
        """
        Is the object a list or tuple? 
        Disqualify string as a true "iterable" in this sense 
        """

        return self._check(pObject, [list, tuple, np.ndarray])
   
    def _bypass_discretizing(self):
        if not distance.c_hash_association_method_discretize[self.distance] or  self.bypass_discretizing or\
            self.distance in ["pearson"] or self.decomposition in ["pca", "ica"] :
            return True
        else:
            return False
    #==========================================================#
    # Static Methods 
    #==========================================================# 

    @staticmethod 
    def m(pArray, pFunc, axis=0):
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
            return array([pFunc(item) for item in pArray]) 

    @staticmethod 
    def bp(pArray, pFunc, axis=0):
        """
        Map _by pairs_ ; i.e. apply pFunc over all possible pairs in pArray 
        """

        if bool(axis): 
            pArray = pArray.T

        pIndices = itertools.combinations(range(pArray.shape[0]), 2)

        return array([pFunc(pArray[i], pArray[j]) for i, j in pIndices])

    @staticmethod 
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

    @staticmethod 
    def rd():
        """
        General reduce-dimension method 
        """
        pass 

    #==========================================================#
    # Helper Functions 
    #==========================================================# 
    def _name_features(self):
        if not self.aOutName1:
            self.aOutName1 = [str(i) for i in range(len(self.meta_array[0])) ]
        if not self.aOutName2:
            self.aOutName2 = [str(i) for i in range(len(self.meta_array[1])) ]
    def _discretize(self):
        self.meta_feature = self.m(self.meta_array, stats.discretize)
        return self.meta_feature

    def _featurize(self, strMethod="_discretize"):
        pMethod = None 
        try:
            pMethod = getattr(self, strMethod)
        except AttributeError:
            raise Exception("Invalid Method.")

        if pMethod:
            return pMethod()
    
    def _hclust(self):
        # print self.meta_feature
        global D1_features_order
        tree1, D1_features_order = hierarchy.hclust(self.meta_feature[0] , strMetric= self.distance, labels=self.aOutName1, bTree=True, plotting_result = self.plotting_results , output_dir = self.output_dir)
        self.meta_data_tree.append(tree1)
        #print D1_features_order
        global D2_features_order
        tree2, D2_features_order = hierarchy.hclust(self.meta_feature[1] , strMetric= self.distance, labels=self.aOutName2, bTree=True, plotting_result = self.plotting_results , output_dir = self.output_dir)
        self.meta_data_tree.append(tree2)
        # self.meta_data_tree = self.m( self.meta_feature, lambda x: hclust(x , bTree=True) )
        #print self.meta_data_tree
        return self.meta_data_tree 

    def _couple(self):
                
        self.meta_hypothesis_tree = hierarchy.couple_tree(apClusterNode1=[self.meta_data_tree[0]],
                apClusterNode2=[self.meta_data_tree[1]],
                pArray1=self.meta_feature[0], pArray2=self.meta_feature[1], func=self.distance, robustness = self.robustness)[0]
        
        # # remember, `couple_tree` returns object wrapped in list 
        #return self.meta_hypothesis_tree 

    def _naive_all_against_all(self, iIter=100):
        self.meta_alla = hierarchy.naive_all_against_all(self.meta_feature[0], self.meta_feature[1], decomposition = self.decomposition, method=self.randomization_method, metric=self.distance, fQ=self.q, fdr = self.fdr_function,
    bVerbose=False, iIter = self.iterations)
        return self.meta_alla 
    def _hypotheses_testing(self):
            
        fQ = self.q
            
        if self.verbose:
            print ("HAllA PROMPT: q value", fQ)
            print ("q value is", fQ) 
        
        self.meta_alla = hierarchy.hypotheses_testing(self.meta_hypothesis_tree, self.meta_feature[0], self.meta_feature[1], method=self.randomization_method, fdr=self.fdr_function, decomposition=self.decomposition, metric= self.distance, fQ=self.q, iIter = self.iterations, robustness=self.robustness, bVerbose=self.verbose, apply_stop_condition = self.apply_stop_condition) 
        # # Choose to keep to 2 arrays for now -- change later to generalize 
        #return self.meta_alla 

    def _layerwise_all_against_all(self):

        X, Y = self.meta_array[0], self.meta_array[1]
        dX, dY = self.meta_feature[0], self.meta_feature[1]
        tX, tY = self.meta_data_tree[0], self.meta_data_tree[1]
        iX, iY = X.shape[0], Y.shape[0]

        aOut = filter(bool, list(hierarchy.layerwise_all_against_all(tX, tY, X, Y)))

        aMetaOut = [] 

        def _layer(Z):

            S = -1 * np.ones((iX, iY))  # # matrix of all associations; symmetric if using a symmetric measure of association  

            def __add_pval_product_wise(_x, _y, _fP):
                S[_x][_y] = _fP ; S[_y][_x] = _fP 

            def __get_pval_from_bags(_Z, _strMethod='final'):
                """
                
                _strMethod: str 
                    {"default",}

                The default option does the following: go through the bags, treating the p-value for each bag pair as applying to all the variables inside the bag. 
                If new instance arises (e.g. [[3],[5]] following [[3,5,6],[3,5,6]] ), override the p-value to the one with more precision. 
                """

                for aLine in _Z:
                    if self.verbose:
                        print (aLine) 
                    # break
                    aaBag, fAssoc = aLine
                    aBag1, aBag2 = aaBag 
                    aBag1, aBag2 = array(aBag1), array(aBag2)
                    self.bc(aBag1, aBag2, pFunc=lambda x, y: __add_pval_product_wise(_x=x, _y=y, _fP=fAssoc))

            __get_pval_from_bags(Z)
            return S         

        for Z in aOut:
            aMetaOut.append(_layer(Z))

        return aMetaOut
    def _naive_summary_statistics(self):
        _, p_values = zip(*self.aOut[0])
        self.meta_summary = []
        self.meta_summary.append(np.reshape([p_values], (int(math.sqrt(len(p_values))), int(math.sqrt(len(p_values))))))


    def _summary_statistics(self, strMethod=None): 
        """
        provides summary statistics on the output given by _hypotheses_testing 
        """

        if not strMethod:
            strMethod = self.summary_method
        # print('meta array:')
        #print(self.meta_array[0])
        #print(self.meta_array[1])    
        X = self.meta_array[0]
        Y = self.meta_array[1]
        iX, iY = len(X), len(Y)
        S = -1 * np.ones((iX, iY , 2))  # # matrix of all associations; symmetric if using a symmetric measure of association  
        
        Z = self.meta_alla 
        _final, _all = map(array, Z)  # # Z_final is the final bags that passed criteria; Z_all is all the associations delineated throughout computational tree
        Z_final = array([[_final[i].get_data(), _final[i].get_pvalue(), _final[i].get_qvalue()] for i in range(len(_final))])
        Z_all = array([[_all[i].get_data(), _all[i].get_pvalue(), _all[i].get_qvalue()] for i in range(len(_all))])    
            
        # ## Sort the final Z to make sure p-value consolidation happens correctly 
        Z_final_dummy = [-1.0 * (len(line[0][0]) + len(line[0][1])) for line in Z_final]
        args_sorted = np.argsort(Z_final_dummy)
        Z_final = Z_final[args_sorted]
        if self.verbose:
            print (Z_final) 
        # assert( Z_all.any() ), "association bags empty." ## Technically, Z_final could be empty 
        def __set_outcome(Z_final):
            self.outcome = np.zeros((len(self.meta_feature[0]),len(self.meta_feature[1])))
            
            for aLine in Z_final:
                if self.verbose:
                    print (aLine) 
                
                aaBag, _, _ = aLine
                listBag1, listBag2 = aaBag     
                for i, j in itertools.product(listBag1, listBag2):
                    self.outcome[i][j] = 1.0 
        def __set_pvalues(Z_all):
            self.pvalues = np.zeros((len(self.meta_feature[0]),len(self.meta_feature[1])))
            
            for aLine in Z_all:
                if self.verbose:
                    print (aLine) 
                
                aaBag, pvalue, _ = aLine
                listBag1, listBag2 = aaBag     
                for i, j in itertools.product(listBag1, listBag2):
                    self.pvalues[i][j] = pvalue 
            
            
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
                if self.verbose:
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
                if self.verbose:
                    print (aLine) 
                
                aaBag, fAssoc, P_adjust = aLine
                aBag1, aBag2 = aaBag 
                aBag1, aBag2 = array(aBag1), array(aBag2)
                self.bc(aBag1, aBag2, pFunc=lambda x, y: __add_pval_product_wise(_x=x, _y=y, _fP=fAssoc, _fP_adjust=P_adjust))
        __set_outcome(Z_final)
        __set_pvalues(Z_all)
        if strMethod == "final":
            if self.verbose:
                print ("Using only final p-values")
            __get_conditional_pval_from_bags(Z_final)
            assert(S.any())
            self.meta_summary = S
            return self.meta_summary

        elif strMethod == "all":
            if self.verbose:
                print ("Using all p-values")
            __get_conditional_pval_from_bags(Z_all)
            assert(S.any())
            self.meta_summary = S
            return self.meta_summary

    def _plot(self):
        """
        Wrapper for plotting facilities
        """

    def _report(self):
        """
        helper function for reporting the output to the user,
        """
        output_dir = self.output_dir
        aaOut = []

        # self.meta_report = [] 

        aP = self.meta_summary
        iRow1 = len(self.meta_array[0])
        iRow2 = len(self.meta_array[1])

        for i, j in itertools.product(range(iRow1), range(iRow2)):
            # ## i <= j 
            fQ = aP[i][j][0] 
            fQ_adust = aP[i][j][1] 
            if fQ != -1:
                aaOut.append([[i, j], fQ, fQ_adust ])

        self.meta_report = aaOut
        # print "meta summary:", self.meta_report
        global associated_feature_X_indecies
        associated_feature_X_indecies = []
        global associated_feature_Y_indecies
        associated_feature_Y_indecies = []
        
        def _report_all_tests():
            output_file_all  = open(str(self.output_dir)+'/all_association_results_one_by_one.txt', 'w')
            csvw = csv.writer(output_file_all, csv.excel_tab)
            #csvw.writerow(["Decomposition method: ", self.decomposition  +"-"+ self.distance , "q value: " + str(self.q), "metric " +self.distance])
            csvw.writerow(["First Dataset", "Second Dataset", "nominal-pvalue", "adjusted-pvalue"])
    
            for line in aaOut:
                iX, iY = line[0]
                fP = line[1]
                fP_adjust = line[2]
                #print line
                aLineOut = map(str, [self.aOutName1[iX], self.aOutName2[iY], fP, fP_adjust])
                csvw.writerow(aLineOut)

        def _report_associations():    
            association_number = 0
            output_file_associations  = open(str(self.output_dir)+'/associations.txt', 'w')
            bcsvw = csv.writer(output_file_associations, csv.excel_tab)
            #bcsvw.writerow(["Method: " + self.decomposition +"-"+ self.distance , "q value: " + str(self.q), "metric " + self.distance])
            bcsvw.writerow(["Association Number", "Clusters First Dataset", "Cluster Similarity Score", "Explained Variance by the first representative of the cluster"," ", "Clusters Second Dataset", "Cluster Similarity Score", "Explained Variance by the first representative of the cluster"," ", "nominal-pvalue", "adjusted-pvalue", "Similarity score between Clusters"])
    
            sorted_associations = sorted(self.meta_alla[0], key=lambda x: x.pvalue)
            for association in sorted_associations:
                association_number += 1
                iX, iY = association.get_data()
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
                
                aLineOut = map(str, [association_number,
                                     str(';'.join(self.aOutName1[i] for i in iX)),
                                     clusterX_similarity,
                                     clusterX_first_rep,
                                     " ", 
                                     str(';'.join(self.aOutName2[i] for i in iY)),
                                     clusterY_similarity,
                                     clusterY_first_rep,
                                     " ",
                                     fP,
                                     fP_adjust,
                                     association_similarity])
                bcsvw.writerow(aLineOut)
        sorted_associations = sorted(self.meta_alla[0], key=lambda x: x.pvalue)
        if self.descending == "AllA":
            global D1_features_order, D2_features_order
            D1_features_order  = [i for i in range(len(self.meta_array[0]))]   
            D2_features_order  = [i for i in range(len(self.meta_array[1]))]         
        def _plot_associations ():
            import pandas as pd    
            association_number = 0
            #output_file_associations  = open(str(self.output_dir)+'/associations.txt', 'w')
            #bcsvw = csv.writer(output_file_associations, csv.excel_tab)
            #bcsvw.writerow(["Method: " + self.decomposition +"-"+ self.distance , "q value: " + str(self.q), "metric " + self.distance])
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
                                     str(';'.join(self.aOutName1[i] for i in iX)),
                                     clusterX_similarity,
                                     clusterX_first_rep,
                                     " ", 
                                     str(';'.join(self.aOutName2[i] for i in iY)),
                                     clusterY_similarity,
                                     clusterY_first_rep,
                                     " ",
                                     fP,
                                     fP_adjust,
                                     association_similarity])
                bcsvw.writerow(aLineOut)
               '''
                if self.plotting_results:
                    print "--- plotting associations ",association_number," ..."
                    cluster1 = [self.meta_array[0][i] for i in iX]
                    discretized_cluster1 = [self.meta_feature[0][i] for i in iX]
                    X_labels = np.array([self.aOutName1[i] for i in iX])
                    association_dir = self.output_dir + "/association_"+ str(association_number) + '/'
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
                                 
                    cluster2 = [self.meta_array[1][i] for i in iY]
                    discretized_cluster2 = [self.meta_feature[1][i] for i in iY]
                    Y_labels = np.array([self.aOutName2[i] for i in iY])
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
                    
                    decomposition_method = self.hash_decomposition[self.decomposition]
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
                    
                    
                    '''for i in range(len(x)):
                        x += np.random.normal(x[i], 3, 100)
                    for i in range(len(y)):
                        y += np.random.normal(y[i], 3, 100)
                    '''
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
                    if self._bypass_discretizing():
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
            output_file_compared_clusters  = open(str(self.output_dir)+'/all_compared_clusters_hypotheses_tree.txt', 'w')
            csvwc = csv.writer(output_file_compared_clusters , csv.excel_tab)
            csvwc.writerow(['Level', "Dataset 1", "Dataset 2" ])
            for line in hierarchy.reduce_tree_by_layer([self.meta_hypothesis_tree]):
                (level, clusters) = line
                iX, iY = clusters[0], clusters[1]
                fP = line[1]
                # fP_adjust = line[2]
                aLineOut = map(str, [str(level), str(';'.join(self.aOutName1[i] for i in iX)), str(';'.join(self.aOutName2[i] for i in iY))])
                csvwc.writerow(aLineOut)
        def _heatmap_associations():
            if self.plotting_results:
                print "--- plotting heatmap of associations  ..."
                global associated_feature_X_indecies
                Xs = list(set(associated_feature_X_indecies))
                X_labels = np.array([self.aOutName1[i] for i in Xs])
                global associated_feature_Y_indecies
                Ys = list(set(associated_feature_Y_indecies))
                Y_labels = np.array([self.aOutName2[i] for i in Ys])
                if len(Xs) > 1 and len(Ys) > 1: 
                    cluster1 = [self.meta_feature[0][i] for i in Xs]    
                    cluster2 = [self.meta_feature[1][i] for i in Ys]
                    df1 = np.array(cluster1, dtype=float)
                    df2 = np.array(cluster2, dtype=float)
                    p = np.zeros(shape=(len(Xs), len(Ys)))
                    #nmi = np.zeros(shape=(len(Xs), len(Ys)))
                    plot.heatmap2(pArray1=cluster1, pArray2=cluster2, xlabels =X_labels, ylabels = Y_labels, filename = str(self.output_dir)+'/all_nmi_heatmap' )
        def _heatmap_associations_R():
            if self.plotting_results:
                print "--- plotting heatmap associations using R ..."
                from scipy.stats.stats import pearsonr
                global associated_feature_X_indecies
                Xs = list(set(associated_feature_X_indecies)) 
                
                global associated_feature_Y_indecies
                Ys = list(set(associated_feature_Y_indecies))
                
                if len(Xs) > 1 and len(Ys) > 1: 
                    global D1_features_order, D2_features_order
                    D1_features_order = [D1_features_order[i] for i in range (len(D1_features_order))  if D1_features_order[i] in Xs ] 
                    D2_features_order = [D2_features_order[i] for i in range (len(D2_features_order))  if D2_features_order[i] in Ys ] 
                    
                    X_labels = np.array([self.aOutName1[i] for i in D1_features_order])
                    Y_labels = np.array([self.aOutName2[i] for i in D2_features_order])
                    
                    p = np.zeros(shape=(len(D1_features_order), len(D2_features_order))) 
                    
                    for i in range(len(D1_features_order)):
                        for j in range(len(D2_features_order)):
                            p[i][j] = pearsonr(np.array(self.meta_array[0][D1_features_order[i]], dtype=float), np.array(self.meta_array[1][D2_features_order[j]], dtype=float))[0]
                    nmi = np.zeros(shape=(len(D1_features_order), len(D2_features_order)))
                    def _is_in_an_assciostions(i,j):
                        for association in sorted_associations:
                            iX, iY = association.get_data()
                            if i in iX and j in iY:
                                return True
                        return False
                      
                    for i in range(len(D1_features_order)):
                        for j in range(len(D2_features_order)):
                            nmi[i][j] = self.hash_metric[self.distance](self.meta_feature[0][D1_features_order[i]], self.meta_feature[1][D2_features_order[j]])
                     
                   
                    anottation_cell = np.zeros(shape=(len(D1_features_order), len(D2_features_order)))                
                    for i in range(len(D1_features_order)):
                        for j in range(len(D2_features_order)):
                            if _is_in_an_assciostions(D1_features_order[i],D2_features_order[j]): #for association in sorted_associations:
                                anottation_cell[i][j] = 1        
                    #anottation_cell = [D1_features_order]
                    #anottation_cell = [ anottation_cell[:][j] for j in D2_features_order]
                    #print anottation_cell
                    import rpy2.robjects as ro
                    #import pandas.rpy.common as com
                    import rpy2.robjects.numpy2ri
                    rpy2.robjects.numpy2ri.activate()
                    ro.r('library("pheatmap")')
                    ro.globalenv['nmi'] = nmi
                    ro.globalenv['sig_matrix1'] = anottation_cell
                    ro.globalenv['sig_matrix2'] = anottation_cell
                    ro.globalenv['labRow'] = X_labels 
                    ro.globalenv['labCol'] = Y_labels
                    #ro.r('pdf(file = "./output/NMI_heatmap.pdf")')
                    ro.globalenv['output_file_NMI'] = str(self.output_dir)+"/" + self.distance+"_heatmap.pdf"
                    ro.globalenv['output_file_Pearson'] = str(self.output_dir)+"/Pearson_heatmap.pdf"
                    ro.r('rownames(nmi) = labRow')
                    ro.r('colnames(nmi) = labCol')
                    if distance.c_hash_association_method_discretize[self.distance]:
                        ro.r('pheatmap(nmi, color = rev(heat.colors(100)),labRow = labRow, labCol = labCol, filename =output_file_NMI, cellwidth = 10, cellheight = 10, fontsize = 10, show_rownames = T, show_colnames = T, cluster_rows=FALSE, cluster_cols=FALSE, display_numbers = matrix(ifelse(sig_matrix1 > 0, "*", ""), nrow(sig_matrix1)))')#,scale="row",  key=TRUE, symkey=FALSE, density.info="none", trace="none", cexRow=0.5
                    else:
                        ro.r('pheatmap(nmi,labRow = labRow, labCol = labCol, filename =output_file_NMI, cellwidth = 10, cellheight = 10, fontsize = 10, show_rownames = T, show_colnames = T, cluster_rows=FALSE, cluster_cols=FALSE, display_numbers = matrix(ifelse(sig_matrix1 > 0, "*", ""), nrow(sig_matrix1)))')#,scale="row",  key=TRUE, symkey=FALSE, density.info="none", trace="none", cexRow=0.5

                    ro.r('dev.off()')
                    if self.distance != "pearson":
                        ro.globalenv['p'] = p
                        #ro.r('pdf(file = "./output/Pearson_heatmap.pdf")')
                        ro.r('rownames(p) = labRow')
                        ro.r('colnames(p) = labCol')
                        if distance.c_hash_association_method_discretize[self.distance]:
                            ro.r('pheatmap(p, color = rev(heat.colors(100)), labRow = labRow, labCol = labCol, filename = output_file_Pearson, cellwidth = 10, cellheight = 10, fontsize = 10, show_rownames = T, show_colnames = T, cluster_rows=F, cluster_cols=F, display_numbers = matrix(ifelse(sig_matrix2 > 0, "*", ""), nrow(sig_matrix2)))')#, scale="column",  key=TRUE, symkey=FALSE, density.info="none", trace="none", cexRow=0.5
                        else:
                            ro.r('pheatmap(p, labRow = labRow, labCol = labCol, filename = output_file_Pearson, cellwidth = 10, cellheight = 10, fontsize = 10, show_rownames = T, show_colnames = T, cluster_rows=F, cluster_cols=F, display_numbers = matrix(ifelse(sig_matrix2 > 0, "*", ""), nrow(sig_matrix2)))')#, scale="column",  key=TRUE, symkey=FALSE, density.info="none", trace="none", cexRow=0.5

                            
                        ro.r('dev.off()')
        def _heatmap_datasets_R():
            if self.plotting_results:
                
                
                print "--- plotting heatmap datasets using R ..."
                from scipy.stats.stats import pearsonr
                X_indecies = len(self.aOutName1)
                X_labels = np.array([self.aOutName1[i] for i in range(X_indecies)])
                Y_indecies = len(self.aOutName2)
                Y_labels = np.array([self.aOutName2[i] for i in range(Y_indecies)])
                if len(X_labels) > 1 and len(Y_labels) > 1: 
                    df1 = np.array(self.meta_feature[0], dtype=float)
                    df2 = np.array(self.meta_feature[1], dtype=float)
                    drows1 = np.zeros(shape=(X_indecies, X_indecies))
                    drows2 = np.zeros(shape=(Y_indecies, Y_indecies))
                    
                    for i in range(X_indecies):
                        for j in range(X_indecies):
                            drows1[i][j] = self.hash_metric[self.distance](df1[i], df1[j]) 
                    
                    for i in range(Y_indecies):
                        for j in range(Y_indecies):
                            drows2[i][j] = self.hash_metric[self.distance](df2[i], df2[j])    
                    
                    import rpy2.robjects as ro
                    #import pandas.rpy.common as com
                    import rpy2.robjects.numpy2ri
                    rpy2.robjects.numpy2ri.activate()
                    ro.r('library("pheatmap")')
                    ro.globalenv['drows1'] = drows1
                    ro.globalenv['labRow'] = X_labels 
                    ro.globalenv['D1'] = str(self.output_dir)+"/D1_heatmap.pdf"
                    ro.r('rownames(drows1) = labRow')
                    ro.r('pheatmap(drows1, filename =D1, cellwidth = 10, cellheight = 10, fontsize = 10, show_rownames = T,  show_colnames = F, cluster_cols=T, dendrogram="row")')#,scale="row",  key=TRUE, symkey=FALSE, density.info="none", trace="none", cexRow=0.5
                    ro.r('dev.off()')
                    ro.globalenv['drows2'] = drows2
                    ro.globalenv['labRow'] = Y_labels
                    ro.globalenv['D2'] = str(self.output_dir)+"/D2_heatmap.pdf"
                    ro.r('rownames(drows2) = labRow')
                    ro.r('pheatmap(drows2, filename =D2, cellwidth = 10, cellheight = 10, fontsize = 10, show_rownames = T,  show_colnames = F, cluster_cols=T, dendrogram="row")')#,scale="row",  key=TRUE, symkey=FALSE, density.info="none", trace="none", cexRow=0.5
                    ro.r('dev.off()')
                
        # Execute report functions
        _report_all_tests()
        _report_associations()
        _report_compared_clusters()
        _plot_associations()
        if self.heatmap_all:
            _heatmap_associations()
            _heatmap_associations_R()
            _heatmap_datasets_R()
        
        return self.meta_report 

    def _run(self):
        """
        helper function: runs vanilla run of HAllA _as is_. 
        """

        pass 

    #==========================================================#
    # Load and set data 
    #==========================================================# 

    def set_data(self, *ta):
        if ta:
            self.meta_array = ta 
            return self.meta_array 
        else:
            raise Exception("Data empty")


    #==========================================================#
    # Set parameters 
    #==========================================================# 

    def set_q(self, fQ):
        self.q = fQ

    def set_summary_method(self, strMethod):
        self.summary_method = strMethod 
        return self.summary_method 

    def set_p_adjust_method(self, strMethod):
        """
        Set multiple hypothesis test correction method 

            {"BH", "FDR", "Bonferroni", "BHY"}
        """

        self.p_adjust_method = strMethod 
        return self.p_adjust_method 

    def set_metric(self, pMetric):
        if isinstance(pMetric, str):
            self.distance = self.hash_metric[pMetric]
        else:
            self.distance = pMetric 
        return self.distance 

    def set_decomposition(self, strMethod):
        if isinstance(strMethod, str):
            self.decomposition = self.hash_decomposition[strMethod]
        else:
            self.decomposition = strMethod 
        return self.decomposition

    def set_iterations(self, iIterations):
        self.iterations = iIterations
        return self.iterations 

    def set_randomization_method(self, strMethod):

        self.randomization_method = strMethod 
        return self.randomization_method 

    def set_fdr_function(self, strFunction):
        self.fdr_function = strFunction

    def set_verbose(self, bBool=True):
        self.verbose = bBool 

    def set_preset(self, strPreset):
        try:
            pPreset = self.hash_preset[strPreset] 
            pPreset()  # # run method 
        except KeyError:
            raise Exception("Preset not found. For the default preset, try set_preset('default')")

    #==========================================================#
    # Presets  
    #==========================================================# 
    """
    These are hard-coded presets deemed useful for the user 
    """

    def load_data(self):
        pass 

    def get_data(self):
        return self.meta_array 

    def get_feature(self):
        return self.meta_feature

    def get_tree(self):
        return self.meta_data_tree

    def get_hypothesis(self):
        return self.meta_hypothesis_tree

    def get_association(self):
        return self.meta_alla 

    def get_attribute(self):
        """
        returns current attributes and statistics about HAllA object implementation 

            * Print parameters in a text-table style 
        """
        
        for item in self.keys_attribute:
            sys.stderr.write("\t".join([item, str(getattr(self, item))]) + "\n") 
    
    def run(self):
        
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
        if self.output_dir == "./":
            self.output_dir = "./"+self.descending+'_'+self.distance+'_'+self.decomposition +'_'+ str(len(self.meta_array[0]))+'_'+str(len(self.meta_array[0][1]))
            if not os.path.isdir(self.output_dir):
                try:
                    print("Creating output directory: " + self.output_dir)
                    os.mkdir(self.output_dir)
                except EnvironmentError:
                    sys.exit("CRITICAL ERROR: Unable to create output directory.")
            else:
                try:
                    print("Removing the old output directory: " + self.output_dir)
                    shutil.rmtree(self.output_dir)
                    print("Creating output directory: " + self.output_dir)
                    os.mkdir(self.output_dir)
                except EnvironmentError:
                    sys.exit("CRITICAL ERROR: Unable to create output directory.")
        
        try:    
            performance_file  = open(str(self.output_dir)+'/performance.txt', 'w')
        except IOError:
            sys.exit("IO Exception: "+self.output_dir+"/performance.txt") 
        csvw = csv.writer(performance_file, csv.excel_tab)
        csvw.writerow(["Decomposition method: ", self.decomposition])
        csvw.writerow(["Similarity method: ", self.distance]) 
        csvw.writerow(["q: FDR cut-off : ", self.q]) 
        csvw.writerow(["r: effect size for robustness : ", self.robustness]) 
        csvw.writerow(["Applied stop condition : ", self.apply_stop_condition]) 
        
        self._name_features()
        if not self.is_correct_submethods_combination():
            sys.exit("Please ckeck the combination of your options!!!!")
        execution_time = time.time()
        if self._bypass_discretizing():
            self.meta_feature = array([np.asarray(self.meta_array[0], dtype = float), np.asarray(self.meta_array[1], dtype = float)])
            #print self.meta_feature
        else:
            #print "featurize is started!"
            # featurize 
            start_time = time.time()
            self._featurize()
            excution_time_temp = time.time() - start_time
            csvw.writerow(["featurize time", str(datetime.timedelta(seconds=excution_time_temp)) ])
            print("--- %s h:m:s featurize data time ---" % str(datetime.timedelta(seconds=excution_time_temp)))
            #print self.meta_feature
        #plot.heatmap2(pArray1=self.meta_feature[0], pArray2=self.meta_feature[1], xlabels =self.aOutName1, ylabels = self.aOutName2, filename = str(self.output_dir)+'/heatmap2_all' )
        if self.descending == "AllA":
            print("--- association hypotheses testing is started, this task may take longer ...")
            start_time = time.time()
            self._naive_all_against_all()
            excution_time_temp = time.time() - start_time
            csvw.writerow(["Hypotheses testing time", str(datetime.timedelta(seconds=excution_time_temp)) ])
            print("--- %s h:m:s hypotheses testing time ---" % str(datetime.timedelta(seconds=excution_time_temp)))
        elif self.descending == "HAllA":
            # hierarchical clustering 
            start_time = time.time()
            self._hclust()
            excution_time_temp = time.time() - start_time
            csvw.writerow(["Hierarchical clustering time", str(datetime.timedelta(seconds=excution_time_temp)) ])
            print("--- %s h:m:s hierarchical clustering time ---" % str(datetime.timedelta(seconds=excution_time_temp)))
            
            # coupling clusters hierarchically 
            start_time = time.time()
            self._couple()
            excution_time_temp = time.time() - start_time
            csvw.writerow(["Coupling hypotheses tree time", str(datetime.timedelta(seconds=excution_time_temp)) ])
            print("--- %s h:m:s coupling hypotheses tree time ---" % str(datetime.timedelta(seconds=excution_time_temp)))
            # hypotheses testing
            print("--- association hypotheses testing is started, this task may take longer ...")
            start_time = time.time()
            self._hypotheses_testing()
            excution_time_temp = time.time() - start_time
            csvw.writerow(["Hypotheses testing time", str(datetime.timedelta(seconds=excution_time_temp)) ])
            print("--- %s h:m:s hypotheses testing time ---" % str(datetime.timedelta(seconds=excution_time_temp)))
        
        # Generate a report
        start_time = time.time() 
        self._summary_statistics('final') 
        excution_time_temp = time.time() - start_time
        csvw.writerow(["Summary statistics time", excution_time_temp ])
        print("--- %s h:m:s summary statistics time ---" % str(datetime.timedelta(seconds=excution_time_temp)))
        
        start_time = time.time() 
        results = self._report()
        excution_time_temp = time.time() - start_time
        csvw.writerow(["Plotting results time", str(datetime.timedelta(seconds=excution_time_temp)) ])
        print("--- %s h:m:s plotting results time ---" % str(datetime.timedelta(seconds=excution_time_temp)))
        excution_time_temp = time.time() - execution_time
        csvw.writerow(["Total execution time", str(datetime.timedelta(seconds=excution_time_temp))])
        print("--- in %s h:m:s the task is successfully done ---" % str(datetime.timedelta(seconds=excution_time_temp)) )
        return results
    
    def view_singleton(self, pBags):
        aOut = [] 
        for aIndices, fP in pBags:
            if len(aIndices[0]) == 1 and len(aIndices[1]) == 1:
                aOut.append([aIndices, fP])
        return aOut 
    
    def is_correct_submethods_combination(self ): 
        if (self.descending == "AllA" and not self.decomposition =='none') or\
                            (self.descending == "HAllA" and self.decomposition =='none') or\
                            (self.decomposition in ["ica","pca",'pls', 'cca', 'kpca'] and self.distance != "pearson") or\
                            (self.decomposition == "mca" and self.distance == "pearson"):
                False
        else:
            return True
