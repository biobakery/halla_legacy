import random
from numpy import array
__description__      = """
  _    _          _ _          
 | |  | |   /\   | | |   /\    
 | |__| |  /  \  | | |  /  \   
 |  __  | / /\ \ | | | / /\ \  
 | |  | |/ ____ \| | |/ ____ \ 
 |_|  |_/_/    \_\_|_/_/    \_\
                               

HAllA Object for hierarchical all-against-all association testing 
"""

__doc__             = __doc__ 
__version__          = "0.1.0"
__author__             = ["Gholamali.Rahnavard", "YS Joseph Moon", "Curtis Huttenhower"]
__contact__         = "gholamali.rahnavard@gmail.com"


keys_attribute = ["__description__", "__version__", "__author__", "__contact__", "q", "distance", "iterations", "decomposition", "p_adjust_method", "randomization_method"]

NBIN = None # number of bins specified by user
NPROC = 1 # number of threads to use 

distance = "nmi"
decomposition = "medoid" 
fdr_function = "level"
q = .2  
iterations = 10000
p_adjust_method = "bh"
randomization_method = "permutation"  # method to generate error bars 
sstrStep = "uniform"
verbose = 'CRITICAL' #"DEBUG","INFO","WARNING","ERROR","CRITICAL"
descending = "HAllA" 
Distance = [None, None] # Distance Matrices 
summary_method = "final"
output_dir = "./"
log_input = True
plotting_results = False
diagnostics_plot = False
strDiscretizing = 'equal-area'
apply_stop_condition = False
missing_char =" "
missing_method = None
seed = 0 #random.randint(1,10000)
K = 1.5 # constant for homogeneity 
cut_distance_thrd = .5
use_one_null_dist = False
null_dist = []
number_of_performed_tests = 0
#==================================================================#
# Mutable Meta Objects  
#==================================================================#
meta_array = array([None, None])
#meta_array[0] = X  # .append(X)
#meta_array[1] = Y  # .append(Y)
meta_feature = None
meta_threshold = None 
Features_order = array([None, None])
#X_features_cluster_order  = []
#Y_features_cluster_order  = []
meta_data_tree = [] 
meta_hypothesis_tree = None 
meta_alla = None  # results of all-against-all
meta_out = None  # final output array; some methods (e.g. hypotheses_testing) have multiple outputs piped to both meta_alla and meta_out 
meta_summary = None  # summary statistics 
meta_report = None  # summary report 
aOut = None  # summary output for naive approaches_
aOutName1 = None 
aOutName2 = None 
robustness = .5
strFile1 = None
strFile2 = None
outcome = None
pvalues = None
