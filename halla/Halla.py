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
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("Please install matplotlib")
    
#  Load a halla module to check the installation
try:
    import halla.hierarchy
except ImportError:
    sys.exit("CRITICAL ERROR: Unable to find the hierarchy module." + 
        " Please check your install.") 

import halla.stats, halla.distance, halla.store
from halla.parser import Input, Output


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
        description="Hierarchical All-against-All significance association testing.")
            
    argp.add_argument(
        "-X", metavar="<input_dataset_1.txt>",
        type=argparse.FileType("r"), default=sys.stdin,
        help="First file: Tab-delimited text input file, one row per feature, one column per measurement.",
        required=True)        
            
    argp.add_argument(
        "-Y", metavar="input_dataset_2.txt",
        type=argparse.FileType("r"),
        default=None,
        help="Second file: Tab-delimited text input file, one row per feature, one column per measurement - If not selected, we will use the first file (-X).")
    
    argp.add_argument(
        "-o", "--output",
        dest = "output_dir", 
        help="directory to write output files\n[REQUIRED]", 
        metavar="<output>", 
        required=True)
    
    argp.add_argument(
        "--plotting_results", 
        help="plotting results\n", 
        action="store_true",
        default=False)

    argp.add_argument(
        "-q", metavar="q_value",
        dest="dQ",
        type=float,
        default=0.1,
        help="Q-value for overall significance tests (cut-off for false discovery rate).")
    
    argp.add_argument(
        "-s", metavar="similarity threshold",
         dest="dThreshold_similiarity", 
         type=str,
         default=.01,
         help="A threshold for similarity to count a cluster as one unit and no consider sub-clusters as sub-unit.")    
    
    argp.add_argument(
        "-f", metavar="fdr",
        dest="strFDR",
        type=str,
        default = "BHY",
        help="function for maximize statistical power and control false discovery rate, simple, BHY, BH, RH.")

    argp.add_argument(
        "-i", metavar="iterations",
        dest="iIter",
        type=int,
        default=1000,
        help="Number of iterations for nonparametric significance testing (permutation test)")

    argp.add_argument(
        "-m", metavar="metric",
        dest="strMetric",
        type=str,
        default="nmi",
        help="Metric to be used for similarity measurement, NMI, MIC, Pearson.")
    
    argp.add_argument(
        "-d", metavar="decomposition",
        dest="strDecomposition",
        type=str,
        default="pca",
        help="The approach for reducing dimensions (or decomposition)[default = pca, options are pca, cca, kpca, ica]")    
    
    argp.add_argument(
        "-j",  metavar="adjusting",
        dest="strAdjust",    
        type=str,
        default="BH",
        help="The approach for calculating adjusted p-value [default = BH]")
    
    argp.add_argument(
        "-t", metavar="test",
        dest="strRandomization",
        type=str,
        default="permutation",
        help="The approach for association test, [default is permutation, options are permutation and G-test]")  
     
    argp.add_argument(
        "-v", "--verbose", metavar="verbosity",
        dest="iDebug",
        type=int,
        default= False,#10 - (logging.WARNING / 10)
        help="Debug logging level; increase for greater verbosity")
    
    return argp.parse_args()

def set_HAllA_object (H, args):
    H.distance = args.strMetric 
    H.decomposition = args.strDecomposition 
    H.fdr_function = args.strFDR
    H.q = args.dQ  
    H.iterations = args.iIter
    H.p_adjust_method = args.strAdjust
    H.randomization_method = args.strRandomization  # method to generate error bars 
    H.strStep = "uniform"
    H.verbose = args.iDebug
    H.threshold = args.dThreshold_similiarity
    H.output_dir = args.output_dir
    H.plotting_results = args.plotting_results
    istm = list()  # X and Y are used to store datasets
     
    # If Y was not set - we use X
    if args.Y == None:
        istm = [args.X, args.X]  # Use X  
    else:
        istm = [args.X, args.Y]  # Use X and Y

    
    if len(istm) > 1:
        H.strFile1, H.strFile2 = istm[:2]
    else:
        H.strFile1, H.strFile2 = istm[0], istm[0]
        
    aOut1, aOut2 = Input (H.strFile1.name, H.strFile2.name).get()
    H.plotting_results = args.plotting_results
    (H.meta_array[0], H.aOutName1, H.aOutType1, H.aOutHead1) = aOut1 
    (H.meta_array[1], H.aOutName2, H.aOutType2, H.aOutHead2) = aOut2 
    
def _main():
    
    # Parse arguments from command line
    args=parse_arguments(sys.argv)
    check_requirements(args)
    H = halla.store.HAllA(X = None, Y = None)
    set_HAllA_object(H, args)         
    aaOut = H.run()	
    
if __name__ == '__main__':
	_main()

