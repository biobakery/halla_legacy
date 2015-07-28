#!/usr/bin/env python

import sys

# Check the python version
REQUIRED_PYTHON_VERSION_MAJOR = 2
REQUIRED_PYTHON_VERSION_MINOR = 7
try:
    if (sys.version_info[0] != REQUIRED_PYTHON_VERSION_MAJOR or
        sys.version_info[1] < REQUIRED_PYTHON_VERSION_MINOR):
        sys.exit("CRITICAL ERROR: The python version found (version "+
            str(sys.version_info[0])+"."+str(sys.version_info[1])+") "+
            "does not match the version required (version "+
            str(REQUIRED_PYTHON_VERSION_MAJOR)+"."+
            str(REQUIRED_PYTHON_VERSION_MINOR)+"+)")
except (AttributeError,IndexError):
    sys.exit("CRITICAL ERROR: The python version found (version 1) " +
        "does not match the version required (version "+
        str(REQUIRED_PYTHON_VERSION_MAJOR)+"."+
        str(REQUIRED_PYTHON_VERSION_MINOR)+"+)")  

import argparse
import csv
import itertools
import logging
import os
import shutil 
import time
import math

# Test if numpy is installed
try:
    from numpy import array
    import numpy as np
except ImportError:
    sys.exit("Please install numpy")
    
# Test if matplotlib is installed
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("Please install matplotlib")
    
#  Load a halla module to check the installation
try:
    from . import store
except ImportError:
    sys.exit("CRITICAL ERROR: Unable to find the store module." + 
        " Please check your halla install.") 

from . import parser
from . import hierarchy

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
        '''try:
            from rpy2.robjects.packages import importr
        except ImportError: 
            sys.exit("--- Please check your installation for rpy2 library")
        '''
        
    
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
    argp = argparse.ArgumentParser(
        description="Hierarchical All-against-All significance association testing",
        formatter_class=argparse.RawTextHelpFormatter)
            
    argp.add_argument(
        "-X", metavar="<input_dataset_1.txt>",
        type=argparse.FileType("r"), default=sys.stdin,
        help="first file: Tab-delimited text input file, one row per feature, one column per measurement\n[REQUIRED]",
        required=True)        
            
    argp.add_argument(
        "-Y", metavar="<input_dataset_2.txt>",
        type=argparse.FileType("r"),
        default=None,
        help="second file: Tab-delimited text input file, one row per feature, one column per measurement\n[default = the first file (-X)]")
    
    argp.add_argument(
        "-o", "--output",
        dest = "output_dir", 
        help="directory to write output files\n[REQUIRED]", 
        metavar="<output>", 
        required=True)

    argp.add_argument(
        "-q", "--q-value",
        metavar="<0.2>",
        dest="dQ",
        type=float,
        default=0.2,
        help="q-value for overall significance tests (cut-off for false discovery rate)\n[default = 0.1]")
    
    argp.add_argument(
        "-r", "--robustness",
         #metavar="<0.001>",
         dest="robustness", 
         type=float,
         default=0.20,
         help="threshold for robustness\n[default = 0.20]")    
    
    argp.add_argument(
        "-a","--descending",
        dest="strDescending",
        default = "HAllA",
        choices=["HAllA","AllA"],
        help="descending approach\n[default = HAllA for hierarchical all-against-all]")
    
    argp.add_argument(
        "-f","--fdr",
        dest="strFDR",
        default = "BHL",
        choices=["BHL","BHF","BHA"],
        help="function to maximize statistical power and control false discovery rate\n[default = BHL]")

    argp.add_argument(
        "-i","--iterations", metavar="<1000>",
        dest="iIter",
        type=int,
        default=1000,
        help="iterations for nonparametric significance testing (permutation test)\n[default = 1000]")

    argp.add_argument(
        "-m","--metric",
        dest="strMetric",
        default="nmi",
        choices=["nmi","ami","mic","pearson", "spearman"],
        help="metric to be used for similarity measurement\n[default = nmi]")
    
    argp.add_argument(
        "-d","--decomposition",
        dest="strDecomposition",
        default="mca",
        choices=["none", "mca", "pca", "ica", "cca","kpca","pls","medoid", "mean"],
        help="approach for reducing dimensions (or decomposition)\n[default = mca]")    
    
    '''argp.add_argument(
        "-a","--adjusting",
        dest="strAdjust",    
        default="BH",
        choices=["BH", "FDR", "Bonferroni", "BHY"],
        help="approach for calculating adjusted p-value\n[default = BH]")
    '''
    argp.add_argument(
        "-t","--test",
        dest="strRandomization",
        default="permutation",
        choices=["permutation","g-test"],
        help="approach for association test\n[default = permutation]")  
     
    argp.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="additional output is printed")
    
    argp.add_argument(
        "--plotting-results", 
        help="plot the results", 
        action="store_true")
    argp.add_argument(
        "--heatmap-all", 
        help="plot a heatmap for all participated features in at least one association", 
        action="store_true")
    
    argp.add_argument(
        "--bypass-discretizing", 
        help="bypass the discretizing step", 
        action="store_true")
    argp.add_argument(
        "--apply-stop-condition",
        dest ="apply_stop_condition", 
        help="stops when p_value > 1 - adusted_pavlue", 
        action="store_true")
    
    argp.add_argument(
        "--header",
        action="store_true",
        help="the input files contain a header line") 
    
    argp.add_argument(
        "--nproc", metavar="<1>",
        type=int,
        default=1,
        help="the number of processing units available\n[default = 1]")

    return argp.parse_args()

def set_HAllA_object (H, args):
    H.distance = args.strMetric 
    H.decomposition = args.strDecomposition 
    H.fdr_function = args.strFDR
    H.q = args.dQ  
    H.iterations = args.iIter
    #H.p_adjust_method = args.strAdjust
    H.randomization_method = args.strRandomization  # method to generate error bars 
    H.strStep = "uniform"
    H.verbose = args.verbose
    H.robustness = float(args.robustness)
    H.output_dir = args.output_dir
    H.plotting_results = args.plotting_results
    H.heatmap_all = args.heatmap_all
    H.descending = args.strDescending
    istm = list()  # X and Y are used to store datasets
    H.apply_stop_condition = args.apply_stop_condition
    # If Y was not set - we use X
    if args.Y == None:
        istm = [args.X, args.X]  # Use X  
    else:
        istm = [args.X, args.Y]  # Use X and Y

    
    if len(istm) > 1:
        H.strFile1, H.strFile2 = istm[:2]
    else:
        H.strFile1, H.strFile2 = istm[0], istm[0]
        
    aOut1, aOut2 = parser.Input (H.strFile1.name, H.strFile2.name, headers=args.header).get()
    H.plotting_results = args.plotting_results
    (H.meta_array[0], H.aOutName1, H.aOutType1, H.aOutHead1) = aOut1 
    (H.meta_array[1], H.aOutName2, H.aOutType2, H.aOutHead2) = aOut2 
    
def _main():
    
    # Parse arguments from command line
    args=parse_arguments(sys.argv)
    check_requirements(args)
    
    # Set the number of processing units available
    hierarchy.NPROC = args.nproc
    
    H = store.HAllA(X = None, Y = None)
    set_HAllA_object(H, args)         
    aaOut = H.run()	
    
if __name__ == '__main__':
	_main()

