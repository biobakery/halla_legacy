import argparse
import csv
import itertools
import logging
from numpy import array
import os
import shutil 
import sys

from src import hallaclass, hierarchy, stats, distance, parser, plot
from src.parser import Input, Output

# set path
try:
    config_file_location = os.path.dirname(os.path.realpath(__file__))
    # The halla base directory is parent directory of the config file location
    halla_base_directory = os.path.abspath(os.path.join(config_file_location, os.pardir))
    sys.path.append(halla_base_directory + '/src')
    sys.path.append('/Users/rah/Documents/Hutlab/strudel')
except :
    sys.exit("CRITICAL ERROR: Unable to find the HAllA src directory." + 
        " Please check your install.") 


   
# Try to load one of the halla src modules to check the installation
try:
    from src import config
except ImportError:
    sys.exit("CRITICAL ERROR: Unable to find the config under src directory." + 
        " Please check your install.") 

# Check the python version
import sys
try:
    if (sys.version_info[0] != config.required_python_version_major or
        sys.version_info[1] < config.required_python_version_minor):
        sys.exit("CRITICAL ERROR: The python version found (version "+
            str(sys.version_info[0])+"."+str(sys.version_info[1])+") "+
            "does not match the version required (version "+
            str(config.required_python_version_major)+"."+
            str(config.required_python_version_minor)+"+)")
except (AttributeError,IndexError):
    sys.exit("CRITICAL ERROR: The python version found (version 1) " +
        "does not match the version required (version "+
        str(config.required_python_version_major)+"."+
        str(config.required_python_version_minor)+"+)")  
    
def get_halla_base_directory():
    """ 
    Return the location of the halla base directory
    """
    
    config_file_location = os.path.dirname(os.path.realpath(__file__))
    
    # The halla base directory is parent directory of the config file location
    halla_base_directory = os.path.abspath(os.path.join(config_file_location, os.pardir))
    
    return halla_base_directory

def check_requirements(args):
    """
    Check requirements (file format, dependencies, permissions)
    """
    # check the third party softwares for plotting the results
    if args.plotting_results:
        try: 
            import pandas as pd
        except ImportError: 
            sys.exit("--- Please check your installation for pandas library")
        try:
            from rpy2.robjects.packages import importr
        except ImportError: 
            sys.exit("--- Please check your installation for rpy2 library")
        
        
    
    # Check that the output directory is writeable
    output_dir = os.path.abspath(args.output_dir)
    args.output_dir = output_dir
    if not os.path.isdir(output_dir):
        try:
            print("Creating output directory: " + output_dir)
            os.mkdir(output_dir)
        except EnvironmentError:
            sys.exit("CRITICAL ERROR: Unable to create output directory.")
    else:
        try:
            print("Removing the old output directory: " + output_dir)
            shutil.rmtree(output_dir)
            print("Creating output directory: " + output_dir)
            os.mkdir(output_dir)
        except EnvironmentError:
            sys.exit("CRITICAL ERROR: Unable to create output directory.")
        
    
    if not os.access(output_dir, os.W_OK):
        sys.exit("CRITICAL ERROR: The output directory is not " + 
            "writeable. This software needs to write files to this directory.\n" +
            "Please select another directory.")
        
    print("Output files will be written to: " + output_dir) 
    

def parse_arguments (args):
    """ 
    Parse the arguments from the user
    """
    argp = argparse.ArgumentParser(prog="halla.py",
        description="Hierarchical All-against-All significance association testing.")
            
    argp.add_argument("-X", metavar="first input dataset",
            type=argparse.FileType("r"), default=sys.stdin,
            help="First file: Tab-delimited text input file, one row per feature, one column per measurement.")        
            
    argp.add_argument("-Y", metavar="second input dataset",
            type=argparse.FileType("r"), default=None,
            help="Second file: Tab-delimited text input file, one row per feature, one column per measurement - If not selected, we will use the first file (-X).")
    
    argp.add_argument(
        "-o", "--output",
        dest = "output_dir", 
        help="directory to write output files\n[REQUIRED]", 
        metavar="<output>", 
        required=True)
    
    argp.add_argument(
        "--plotting_results", 
        help="bypass the plotting results step\n", 
        action="store_true",
        default=False)

    argp.add_argument("-q", dest="dQ", metavar="q_value",
            type=float, default=0.1,
            help="Q-value for overall significance tests (cut-off for false discovery rate).")
    
    argp.add_argument("-s",         dest="dThreshold_similiarity",     metavar="similarity threshold",
            type=str,         default=.5,
            help="A threshold for similarity to count a cluster as one unit and no consider sub-clusters as sub-unit.")    
    
    argp.add_argument("-f",     dest="strFDR",     metavar="fdr",
        type=str,         default = "RH",
        help="function for maximize statistical power and control false discovery rate, simple, BHY, BH, RH.")

    argp.add_argument("-i", dest="iIter", metavar="iterations",
            type=int, default=1000,
            help="Number of iterations for nonparametric significance testing (permutation test)")

    argp.add_argument("-m",         dest="strMetric",     metavar="metric",
            type=str,         default="norm_mi",
            help="Metric to be used for similarity measurement, NMI, MIC, Pearson.")
    
    argp.add_argument("-d",         dest="strReduce",     metavar="decomposition",
            type=str,         default="pca",
            help="The approach for reducing dimensions (or decomposition)[default = pca, options are pca, cca, kpca, ica]")    
    
    argp.add_argument("-j",         dest="strAdjust",     metavar="adjusting",
            type=str,         default="BH",
            help="The approach for calculating adjusted p-value [default = BH]")
    
    argp.add_argument("-t",         dest="strRandomization",     metavar="test",
            type=str,         default="permutation",
            help="The approach for association test, [default is permutation, options are permutation and G-test]")  
     
    argp.add_argument("-v", dest="iDebug", metavar="verbosity",
            type=int, default= 0,#10 - (logging.WARNING / 10)
            help="Debug logging level; increase for greater verbosity")

    return argp.parse_args()
    '''
argp.add_argument("-o", dest="bostm", metavar="output",  # default = "sys.stdout
            type=argparse.FileType("w"), default="./output/associations.txt",
            help="output file for significance tests (associations).")

    argp.add_argument("-a", dest="ostm", metavar="one_by_one feature output",
            type=argparse.FileType("w"), default="./output/all_association_results_one_by_one.txt",
            help="Optional output file to report features one bye one for association significance tests.")
   
    argp.add_argument("-c", dest="costm", metavar="all compared clusters ",
            type=argparse.FileType("w"), default="./output/all_compared_clusters_hypotheses_tree.txt",
            help="Optional output file for hypothesis tree which includes all compared clusters.")
    '''

def _main():
    import strudel
    #Generate simulated datasets
    number_features = 4 
    number_samples = 100
    number_blocks = 2 
    s = strudel.Strudel()
    print 'Synthetic Data Generation ...'
    
    X,Y,A = s.double_cholesky_block( number_features, number_samples , number_blocks, fVal = 2.6 , Beta = 3.0 )#, link = "line" )
#       
    strudel.writeData(X,"./input/X" )
    strudel.writeData(Y,"./input/Y")
    # Parse arguments from command line
    args=parse_arguments(sys.argv)
    check_requirements(args)
    istm = list()  # X and Y are used to store datasets
 	
	# If Y was not set - we use X
    if args.Y == None:
		istm = [args.X, args.X]  # Use X  
    else:
		istm = [args.X, args.Y]  # Use X and Y

	
    if len(istm) > 1:
		strFile1, strFile2 = istm[:2]
    else:
		strFile1, strFile2 = istm[0], istm[0]
		
    aOut1, aOut2 = Input (strFile1.name, strFile2.name).get()

    (aOutData1, aOutName1, aOutType1, aOutHead1) = aOut1 
    (aOutData2, aOutName2, aOutType2, aOutHead2) = aOut2 

    H = hallaclass.HAllA(args, aOut1, aOut2)
    aaOut = H.run()	
    
if __name__ == '__main__':
	_main()

