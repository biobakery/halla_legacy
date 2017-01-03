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
import random

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
from . import config

def get_halla_base_directory():
    """ 
    Return the location of the halla base directory
    """
    
    config_file_location = os.path.dirname(os.path.realpath(__file__))
    
    # The halla base directory is parent directory of the config file location
    halla_base_directory = os.path.abspath(os.path.join(config_file_location, os.pardir))
    
    return halla_base_directory

def check_requirements():
    """
    Check requirements (file format, dependencies, permissions)
    """
    # check the third party softwares for plotting the results
    try: 
        import pandas as pd
    except ImportError: 
        sys.exit("--- Please check your installation for pandas library")
    # Check that the output directory is writeable
    output_dir = os.path.abspath(config.output_dir)
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
    if config.similarity_method =='mic':
        try: 
            import minepy
        except ImportError: 
            sys.exit("--- Please check minepy installation for MIC library")
    # if mca is chosen as decomposition method check if it's R package and dependencies are installed
    if config.decomposition == 'mca':
        try: 
            from rpy2 import robjects as ro
            from rpy2.robjects import r
            from rpy2.robjects.packages import importr
            import rpy2.robjects.numpy2ri
            rpy2.robjects.numpy2ri.activate()
            ro.r('library(FactoMineR)')
        except : 
            sys.exit("--- Please check R, rpy2,  and  FactoMineR installation for MCA library")
  
def parse_arguments (args):
    """ 
    Parse the arguments from the user
    """
    argp = argparse.ArgumentParser(
        description="HAllA: Hierarchical All-against-All significance association testing",
        formatter_class=argparse.RawTextHelpFormatter,
        prog="halla")
    argp.add_argument(
        "--version",
        action="version",
        version="%(prog)s "+config.version)        
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
        metavar="<.1>",
        dest="dQ",
        type=float,
        default=0.1,
        help="q-value for overall significance tests (cut-off for false discovery rate)\n[default = 0.1]")
    argp.add_argument(
        "-p", "--permutation",
        #metavar="<0.001>",
        dest="permutation_func", 
        choices=['ecdf', 'gpd', 'none'], 
        default='',
        help="permutation function \n[default = none for Spearman and Pearson and gpd for other]")    
    argp.add_argument(
        "-a","--descending",
        dest="strDescending",
        default = "HAllA",
        choices=["HAllA","AllA"],
        help="descending approach\n[default = HAllA for hierarchical all-against-all]")
    
    '''argp.add_argument(
        "-f","--fdr",
        dest="strFDR",
        default = "level",
        choices=["level","simple","all"],
        help="hypothesis function to  maximize statistical power and control false discovery rate\n[default = level]")
    '''
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
        choices=["nmi","ami","mic","dmic","dcor","pearson", "spearman"],
        help="metric to be used for similarity measurement\n[default = nmi]")
    
    argp.add_argument(
        "-d","--decomposition",
        dest="strDecomposition",
        default=config.decomposition,
        choices=["none", "mca", "pca", "ica", "cca","kpca","pls","medoid"], #mean
        help="approach for reducing dimensions (or decomposition)\n[default = medoid]")    
    
    argp.add_argument(
        "--fdr",
        dest="strAdjust",    
        default="bh",
        choices=["bh", "by", "bonferroni", "no_adjusting"],
        help="approach for FDR correction\n[default = bh]")
    argp.add_argument(
        "-v", "--verbose",
        dest="verbose",
        default=config.verbose,
        help="additional output is printed")
    
    '''argp.add_argument(
        "--hallagram", 
        help="plot the results", 
        action="store_true")'''
    argp.add_argument(
        "--diagnostics-plot", 
        dest="diagnostics_plot",
        help="Diagnostics plot for associations ", 
        action="store_true")
    argp.add_argument(
        "--discretizing", 
        dest="strDiscretizing",
        default="equal-area",
        choices=["equal-area", "hclust", "none"], #"jenks", "hclust", "kmeans", 
        help="approach for discretizing continuous data\n[default = equal-area]")
    argp.add_argument(
        "--linkage",
        dest ="linkage_method",
        default='complete',
        choices=["single", "average", "complete", "weighted" ],
        help="The method to be used in linkage hierarchical clustering.")
    
    argp.add_argument(
        "--apply-stop-condition",
        dest ="apply_stop_condition", 
        help="stops when two clusters are two far from each other", 
        action="store_true")
    argp.add_argument(
        "--generate-one-null-samples", "--fast",
        dest ="use_one_null_distribution", 
        help="Use one null distribution for permutation test", 
        action="store_true")
    argp.add_argument(
        "--header",
        action="store_true",
        help="the input files contain a header line") 
    argp.add_argument(
        "--format-feature-names",
        dest ="format_feature_names", 
        help="Replaces special characters and for OTUs separated  by | uses the known end of a clade", 
        action="store_true")
    argp.add_argument(
        "--nproc", metavar="<1>",
        type=int,
        default=1,
        help="the number of processing units available\n[default = 1]")
    argp.add_argument(
        "--nbin", metavar="<None>",
        type=int,
        default=None,
        help="the number of bins for discretizing \n[default = None]")
    argp.add_argument(
        "-s", "--seed",
        type=int,
        default= 0,#random.randint(1,10000),
        help="a seed number to make the random permutation reproducible\n[default = 0,and -1 for random number]")
    argp.add_argument(
        "-e", "--entropy",
        dest="entropy_threshold",
        type=float,
        default=0.5,
        help="Minimum entropy threshold to filter features with low information\n[default = 0.5]")
    argp.add_argument(
        "-e1", "--entropy1",
        dest="entropy_threshold1",
        type=float,
        default=None,
        help="Minimum entropy threshold for the first dataset \n[default = None]")
    argp.add_argument(
        "-e2", "--entropy2",
        dest="entropy_threshold2",
        type=float,
        default=None,
        help="Minimum entropy threshold for the second dataset \n[default = None]")
    argp.add_argument(
        "--missing-char",
        dest ="missing_char",
        default=None,
        help="defines missing characters\n[default = " "]")
    argp.add_argument(
        "--missing-method",
        dest ="missing_method",
        default=None,
        choices=["mean", "median", "most_frequent"],
        help="defines missing strategy to fill missing data.\nFor categorical data puts all missing data in one new category.")
    argp.add_argument(
        "--missing-data-category",
        dest = "missing_char_category",
        #default = False,
        action="store_true",
        help="To count the missing data as a category")
    argp.add_argument(
        "--write-hypothesis-tree",
        dest = "write_hypothesis_tree",
        #default = False,
        action="store_true",
        help="To write levels of hypothesis tree in the file")
    return argp.parse_args()

def set_parameters(args):
    '''
    Set the user command line options to config file 
    to be used in the program
    '''
    config.similarity_method = args.strMetric.lower()
    config.decomposition = args.strDecomposition.lower()
    #config.fdr_function = args.strFDR
    config.q = args.dQ  
    config.entropy_threshold = args.entropy_threshold
    if args.entropy_threshold1 == None:
        config.entropy_threshold1 = args.entropy_threshold
    else:
        config.entropy_threshold1 = args.entropy_threshold1
    if args.entropy_threshold2 == None:
        config.entropy_threshold2 = args.entropy_threshold
    else:
        config.entropy_threshold2 = args.entropy_threshold2
    config.permutation_func = args.permutation_func
    if config.permutation_func == '':
        if args.strMetric in ['spearman', 'pearson']:
            config.permutation_func = 'none'
        else:
            config.permutation_func = 'gpd'
    config.write_hypothesis_tree = args.write_hypothesis_tree
    config.strStep = "uniform"
    config.verbose = args.verbose
    config.format_feature_names = args.format_feature_names
    config.output_dir = args.output_dir
    config.diagnostics_plot = args.diagnostics_plot
    config.descending = args.strDescending
    # X and Y are used to store datasets
    istm = list() 
    config.apply_stop_condition = args.apply_stop_condition
    config.use_one_null_dist = args.use_one_null_distribution
    config.strDiscretizing = args.strDiscretizing
    if args.seed == -1:
        config.seed = random.randint(1,10000)
    else:
        config.seed = args.seed
    config.NPROC = args.nproc
    config.NBIN = args.nbin
    config.missing_char = args.missing_char
    config.missing_method = args.missing_method
    config.missing_char_category = args.missing_char_category
    config.p_adjust_method = args.strAdjust.lower()
    config.linkage_method = args.linkage_method
    if args.Y == None:
        istm = [args.X, args.X]  # Use X  
    else:
        istm = [args.X, args.Y]  # Use X and Y
    if len(istm) > 1:
        config.strFile1, config.strFile2 = istm[:2]
    else:
        config.strFile1, config.strFile2 = istm[0], istm[0]
    aOut1, aOut2 = parser.Input (config.strFile1.name, config.strFile2.name, headers=args.header).get()
    (config.discretized_dataset[0], config.original_dataset[0], config.FeatureNames[0], config.aOutType1, config.SampleNames[0]) = aOut1 
    (config.discretized_dataset[1], config.original_dataset[1], config.FeatureNames[1], config.aOutType2, config.SampleNames[1]) = aOut2 
    config.iterations = args.iIter
def main():
    # Parse arguments from command line
    args=parse_arguments(sys.argv)
    set_parameters(args) 
    check_requirements()
    aaOut = store.run()	
if __name__ == '__main__':
	main()

