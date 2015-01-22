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


# set output path
def make_directory(dir = halla_base_directory+'/halla/input'):
    
    '''
    make a directory
    '''
    try:
        shutil.rmtree(dir)
        os.mkdir(dir)
        return dir
    except:
        os.mkdir(dir)
        return dir
    
try:
    dir=make_directory(halla_base_directory+'/halla/output')
except:
    sys.exit("CRITICAL ERROR: Unable to make an output directory." + 
        " Please check your permission.") 
            
# Try to load one of the halla src modules to check the installation
try:
    from src import config
except ImportError:
    sys.exit("CRITICAL ERROR: Unable to find the config under src directory." + 
        " Please check your install.") 


    
# Check the python version
try:
    if (sys.version_info[0] != config.required_python_version_major or
        sys.version_info[1] < config.required_python_version_minor):
        sys.exit("CRITICAL ERROR: The python version found (version " + 
            str(sys.version_info[0]) + "." + str(sys.version_info[1]) + ") " + 
            "does not match the version required (version " + 
            str(config.required_python_version_major) + "." + 
            str(config.required_python_version_minor) + "+)")
except (AttributeError, IndexError):
    sys.exit("CRITICAL ERROR: The python version found (version 1) " + 
        "does not match the version required (version " + 
        str(config.required_python_version_major) + "." + 
        str(config.required_python_version_minor) + "+)") 

def get_halla_base_directory():
    """ 
    Return the location of the halla base directory
    """
    
    config_file_location = os.path.dirname(os.path.realpath(__file__))
    
    # The halla base directory is parent directory of the config file location
    halla_base_directory = os.path.abspath(os.path.join(config_file_location, os.pardir))
    
    return halla_base_directory

def parse_arguments (args):
    """ 
    Parse the arguments from the user
    """
    argp = argparse.ArgumentParser(prog="halla.py",
        description="Hierarchical All-against-All significance association testing.")
            
    argp.add_argument("-X", metavar="Xinput.txt",
            type=argparse.FileType("r"), default=sys.stdin,
            help="First file: Tab-delimited text input file, one row per feature, one column per measurement")        
            
    argp.add_argument("-Y", metavar="Yinput.txt",
            type=argparse.FileType("r"), default=None,
            help="Second file: Tab-delimited text input file, one row per feature, one column per measurement - If not selected, we will use the first file (-X)")
    
    argp.add_argument("-o", dest="bostm", metavar="output.txt",  # default = "sys.stdout
            type=argparse.FileType("w"), default="./output/associations.txt",
            help="Optional output file for blocked association significance tests")

    argp.add_argument("-a", dest="ostm", metavar="one_by_one.txt",
            type=argparse.FileType("w"), default="./output/all_association_results_one_by_one.txt",
            help="Optional output file to report features one bye one for association significance tests")
   
    argp.add_argument("-c", dest="costm", metavar="clusters_output.txt",
            type=argparse.FileType("w"), default="./output/all_compared_clusters_hypotheses_tree.txt",
            help="Optional output file for hypothesis tree which includes all compared clusters")
    
    argp.add_argument("-q", dest="dQ", metavar="q_value",
            type=float, default=0.1,
            help="Q-value for overall significance tests")

    argp.add_argument("-i", dest="iIter", metavar="iterations",
            type=int, default=1000,
            help="Number of iterations for nonparametric significance testing")

    argp.add_argument("-v", dest="iDebug", metavar="verbosity",
            type=int, default= 0,#10 - (logging.WARNING / 10)
            help="Debug logging level; increase for greater verbosity")

    argp.add_argument("-e", "--exploration",     dest="strExploration",     metavar="exploration",
        type=str,         default = "BHY",
        help="Exploration function for maximize power and control false discovery rate, BHY, BH, RAH")

    argp.add_argument("-m",         dest="strMetric",     metavar="metric",
            type=str,         default="norm_mi",
            help="Metric to be used for similarity measurement clustering")
    
    argp.add_argument("-d",         dest="strReduce",     metavar="decomposition",
            type=str,         default="pca",
            help="The approach for reducing dimensions (or decomposition)[default = pca, options are pca, cca, kpca, ica]")    
    
    argp.add_argument("-j",         dest="strAdjust",     metavar="adjusting",
            type=str,         default="BH",
            help="The approach for controlling FDR [default = BH, options are BH, BHY")
    
    argp.add_argument("-r",         dest="strRandomization",     metavar="randomization",
            type=str,         default="permutation",
            help="The approach for randomization, [default is permutation, options are permutation and G-test]")    
          
    return argp.parse_args()
    

def _main():
    
    
    # Parse arguments from command line
    args=parse_arguments(sys.argv)
    
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

