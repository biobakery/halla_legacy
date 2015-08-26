


NBIN = None # number of bins specified by user
NPROC = 1 # number of threads to use 
'''
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
#self.heatmap_all = False
self.strDiscretizing = 'equal-area'
self.apply_stop_condition = False
self.seed = random.randint(1,10000)        #==================================================================#
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
'''