"""
Run-time behavior of halla -- "batch mode" behavior

Run `halla.py --help` for more details 

"""

## structural packages 
import itertools 
import logging 
import argparse 
from numpy import array 
import csv 
import sys 
import os 

## internal dependencies 
import halla
from . import HAllA
from . import parser  
from parser import Input, Output 


#=============================================#
# Wrapper  
#=============================================#

def _main( istm, ostm, dQ, iIter, strMetric ):
	""" 
	
	Design principle: be as flexible and modularized as possible. 

	Have native Input/Ouput objects in the halla.parser module.

	All the different runs and tests I am doing in batch mode should be available in the "run" directory with separate files, 

	then loaded into memory like so: `execute( strFile ) ; function_call(  )`
	
	"""

	if len(istm) > 1:
		strFile1, strFile2 = istm[:2]
	else:
		strFile1, strFile2 = istm[0], istm[0]

	aOut1, aOut2 = Input( strFile1.name, strFile2.name ).get()

	aOutData1, aOutName1, aOutType1, aOutHead1 = aOut1 
	aOutData2, aOutName2, aOutType2, aOutHead2 = aOut2 

	H = HAllA( aOutData1, aOutData2 )

	H.set_q( dQ )
	H.set_iterations( iIter )
	H.set_metric( strMetric )

	aOut = H.run()

	csvw = csv.writer( ostm, csv.excel_tab )

	for line in aOut:
		csvw.writerow( line )


#=============================================#
# Execute 
#=============================================#

argp = argparse.ArgumentParser( prog = "halla.py",
        description = "Hierarchical All-against-All significance association testing." )
argp.add_argument( "istm",              metavar = "input.txt",
        type = argparse.FileType( "r" ),        default = sys.stdin,    nargs = "+",
        help = "Tab-delimited text input file, one row per feature, one column per measurement" )

argp.add_argument( "-o",                dest = "ostm",                  metavar = "output.txt",
        type = argparse.FileType( "w" ),        default = sys.stdout,
        help = "Optional output file for association significance tests" )

argp.add_argument( "-q",                dest = "dQ",                    metavar = "q_value",
        type = float,   default = 0.05,
        help = "Q-value for overall significance tests" )

argp.add_argument( "-i",                dest = "iIter",    metavar = "iterations",
        type = int,             default = 100,
        help = "Number of iterations for nonparametric significance testing" )

argp.add_argument( "-v",                dest = "iDebug",                metavar = "verbosity",
        type = int,             default = 10 - ( logging.WARNING / 10 ),
        help = "Debug logging level; increase for greater verbosity" )

argp.add_argument( "-f",                dest = "fFlag",         action = "store_true",
        help = "A flag set to true if provided" )
argp.add_argument( "-x", 		dest = "strPreset", 	metavar = "preset",
		type  = str, 		default = None,
        help = "Instead of specifying parameters separately, use a preset" )
argp.add_argument( "-m", 		dest = "strMetric", 	metavar = "metric",
		type  = str, 		default = "norm_mid",
        help = "Metric to be used for hierarchical clustering" )


args = argp.parse_args( ) 
_main( args.istm, args.ostm, args.dQ, args.iIter, args.strMetric )


