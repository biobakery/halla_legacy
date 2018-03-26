import random
from numpy import array
version = '0.8.1'
__description__      = """
  _    _          _ _          
 | |  | |   /\   | | |   /\    
 | |__| |  /  \  | | |  /  \   
 |  __  | / /\ \ | | | / /\ \  
 | |  | |/ ____ \| | |/ ____ \ 
 |_|  |_/_/    \_\_|_/_/    \_\
                               

HAllA for hierarchical all-against-all association testing 
configuration file
"""

__doc__             = __doc__ 
__version__          = version
__author__             = ["Gholamali Rahnavard", "Curtis Huttenhower"]
__contact__         = "gholamali.rahnavard@gmail.com"


keys_attribute = ["__description__", "__version__", "__author__", "__contact__", "q", "distance", "iterations", "decomposition", "p_adjust_method", "randomization_method"]

NBIN = None # number of bins specified by user
NPROC = 1 # number of threads to use 

similarity_method = "nmi"
decomposition = "medoid" 
fdr_style = "level"
permutation_func = 'gpd'
linkage_method = 'average'
q = .1  # FDR/multiple testing threshold
iterations = 1000 # number of iteration to be used for permutations p-value estimation
p_adjust_method = "bh" # multiple testing correction method
randomization_method = "permutation"  # method to generate error bars 
sstrStep = "uniform" # not been used!
verbose = 'CRITICAL' #"DEBUG","INFO","WARNING","ERROR","CRITICAL"
descending = "HAllA" # the other otion in all-against-all (AllA) 
Distance = array([None, None]) # Distance Matrices 
summary_method = "final"
output_dir = "./"
log_input = True
diagnostics_plot = False
strDiscretizing = 'equal-freq'
apply_stop_condition = False
missing_char =""
missing_method = None
seed = 0 #random.randint(1,10000)
use_one_null_dist = False # use one null sampling for permutation test in synthetic evaluation but not in real applications
gp = None 
Nexc = None
nullsamples = []
number_of_performed_tests = 0
min_var = 0.0 # it was designed in a case we use minimum variation to filter features
entropy_threshold = 0.0
entropy_threshold1 = 0.0
entropy_threshold2 = 0.0
missing_char_category = False
format_feature_names = False
write_hypothesis_tree = False
report_results =  True
transform_method = ''
#==================================================================#
# Mutable Meta Objects  
#==================================================================#
original_dataset = array([None, None])
discretized_dataset = array([None, None])
parsed_dataset = array([None, None])
meta_threshold = None 
Features_order = array([None, None])
data_type = array([None, None])
meta_data_tree = [] 
meta_hypothesis_tree = None 
meta_alla = None  # results of all-against-all
meta_out = None  # final output array; some methods (e.g. hypotheses_testing) have multiple outputs piped to both meta_alla and meta_out 
meta_summary = None  # summary statistics 
meta_report = None  # summary report 
aOut = None  # summary output for naive approaches_
FeatureNames = array([None, None])
SampleNames = array([None, None])
robustness = .5
strFile1 = None
strFile2 = None
outcome = None
pvalues = None
def reset_default():
    print ("reset to default parameters!")
    use_one_null_dist = False
    gp = None
    Nexc = None
    nullsamples = []
    Distance = [None, None] 
    number_of_performed_tests = 0
    outcome = None
    pvalues = None
    meta_summary = None  # summary statistics 
    meta_report = None  # summary report 
    original_dataset = array([None, None])
    meta_data_tree = [] 
    meta_hypothesis_tree = None 
    meta_alla = None  # results of all-against-all
    meta_out = None  # final output array; some methods (e.g. hypotheses_testing) have multiple outputs piped to both meta_alla and meta_out 
