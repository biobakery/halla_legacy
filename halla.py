import sys
# Try to load one of the humann2 src modules to check the installation
try:
    from src import config
except ImportError:
    sys.exit("CRITICAL ERROR: Unable to find the HAllA src directory." +
        " Please check your install.") 

# Check the python version
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

# set path
halla_base_directory=config.get_halla_base_directory()
 
# structural packages 
import argparse
import csv
import itertools
import logging
import os
sys.path.append(halla_base_directory)
sys.path.append('/Users/rah/Documents/Hutlab/strudel')
import strudel
from numpy import array
import distance
import stats
import parser
import distance
import hallaclass
import data
import hierarchy
from parser import Input, Output
import plot 
import shutil 



## internal dependencies 
#=============================================#
# Wrapper  
#=============================================#
def _main(  ):
    s = strudel.Strudel()
    
    number_features = 4 
    number_samples = 40 
    number_blocks = 2 
    print 'Synthetic Data Generation ...'
        
    X,Y,A = s.double_cholesky_block( number_features, number_samples , number_blocks, fVal = 2.6 , Beta = 3.0 )#, link = "line" )
    filename = "./output/"
    dir = os.path.dirname(filename) 
    try:
        shutil.rmtree(dir)
        os.mkdir(dir)
    except:
        os.mkdir(dir)    
    data.writeData(X,"./input/X" )
    data.writeData(Y,"./input/Y")     

    argp = argparse.ArgumentParser( prog = "halla_cli.py",
			description = "Hierarchical All-against-All significance association testing." )
			
    argp.add_argument( "-X",              metavar = "Xinput.txt",   
			type  =   argparse.FileType( "r" ),        default = sys.stdin,      
			help = "First file: Tab-delimited text input file, one row per feature, one column per measurement" )		
			
    argp.add_argument( "-Y",              metavar = "Yinput.txt",   
			type  =   argparse.FileType( "r" ),        default = None,    
			help = "Second file: Tab-delimited text input file, one row per feature, one column per measurement - If not selected, we will use the first file (-X)" )
	
    argp.add_argument( "-o",                dest = "bostm",                  metavar = "output.txt", #default = "sys.stdout
			type = argparse.FileType( "w" ),        default = "./output/associations.txt",
			help = "Optional output file for blocked association significance tests" )

    argp.add_argument( "-a",                dest = "ostm",                  metavar = "one_by_one.txt",
			type = argparse.FileType( "w" ),        default = "./output/all_association_results_one_by_one.txt",
			help = "Optional output file to report features one bye one for association significance tests" )
    argp.add_argument( "-c",                dest = "costm",                  metavar = "clusters_output.txt",
			type = argparse.FileType( "w" ),        default = "./output/all_compared_clusters_hypotheses_tree.txt",
			help = "Optional output file for hypothesis tree which includes all compared clusters" )
	
    argp.add_argument( "-q",                dest = "dQ",                    metavar = "q_value",
			type = float,   default = 0.1,
			help = "Q-value for overall significance tests" )

    argp.add_argument( "-i",                dest = "iIter",    metavar = "iterations",
			type = int,             default = 1000,
			help = "Number of iterations for nonparametric significance testing" )

    argp.add_argument( "-v",                dest = "iDebug",                metavar = "verbosity",
			type = int,             default = 10 - ( logging.WARNING / 10 ),
			help = "Debug logging level; increase for greater verbosity" )

    argp.add_argument( "-e", "--exploration", 	dest = "strExploration", 	metavar = "exploration",
		type  = str, 		default = "default",
        help = "Exploration function" )

    argp.add_argument( "-x", 		dest = "strPreset", 	metavar = "preset",
			type  = str, 		default = "HAllA-PCA-NMI",
			help = "Instead of specifying parameters separately, use a preset" )

    argp.add_argument( "-m", 		dest = "strMetric", 	metavar = "metric",
			type  = str, 		default = "norm_mi",
			help = "Metric to be used for similarity measurement clustering" )
		
			
    args = argp.parse_args( )
    istm = list()						#We are using now X and Y 
 	
	#***************************************************************
	#We are using now X and Y - If Y was not set - we use X        *
	#***************************************************************
    if args.Y == None:
		istm = [args.X,  args.X]						#Use X  
    else:
		istm = [args.X,  args.Y]						#Use X and Y

	
    if len(istm) > 1:
		strFile1, strFile2 = istm[:2]
    else:
		strFile1, strFile2 = istm[0], istm[0]
		
	
	#aOut1, aOut2 = Input( strFile1.name, strFile2.name ).get()

	#aOutData1, aOutName1, aOutType1, aOutHead1 = aOut1 
	#aOutData2, aOutName2, aOutType2, aOutHead2 = aOut2 

	#H = HAllA( aOutData1, aOutData2 )

	#H.set_q( args.dQ )
	#H.set_iterations( args.iIter )
	#H.set_metric( args.strMetric )

	#aOut = H.run()

	#csvw = csv.writer( args.ostm, csv.excel_tab )

	#for line in aOut:
	#	csvw.writerow( line )


	##
	
    aOut1, aOut2 = Input (strFile1.name, strFile2.name ).get()

    (aOutData1, aOutName1, aOutType1, aOutHead1) = aOut1 
    (aOutData2, aOutName2, aOutType2, aOutHead2) = aOut2 

    H = hallaclass.HAllA(args, aOut1, aOut2)
	
    H.set_q( args.dQ )
    H.set_iterations( args.iIter )
    H.set_metric( args.strMetric )
	#print "First Row", halla.stats.discretize(aOutData1)
	#return
    if args.strPreset: 
		aaOut = H.run( strMethod = args.strPreset )
    else:
		aaOut = H.run( )	
if __name__ == '__main__':
	_main(  )

