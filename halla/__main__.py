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
from . import parser  
from parser import Input, Output 


#=============================================#
# Wrapper  
#=============================================#

def _main():
	""" 
	
	Design principle: be as flexible and modularized as possible. 

	Have native Input/Ouput objects in the halla.parser module.

	All the different runs and tests I am doing in batch mode should be available in the "run" directory with separate files, 

	then loaded into memory like so: `execute( strFile ) ; function_call(  )`
	
	"""

	if len(sys.argv[1:]) > 1:
		strFile1, strFile2 = sys.argv[1:3]
	elif len(sys.argv[1:]) == 1:
		strFile1, strFile2 = sys.argv[1], sys.argv[1]

	
	aOut1, aOut2 = Input( strFile1, strFile2 ).get()

	aOutData1, aOutName1, aOutType1, aOutHead1 = aOut1 
	aOutData2, aOutName2, aOutType2, aOutHead2 = aOut2 


	H = halla.HAllA( aOutData1, aOutData2 )


#=============================================#
# Execute 
#=============================================#

argp = argparse.ArgumentParser( prog = "halla.py",
        description = "Hierarchical All-against-All significance association testing." )
argp.add_argument( "istm",              metavar = "input.txt",
        type = argparse.FileType( "r" ),        default = sys.stdin,    nargs = "?",
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
argp.add_argument( "strString", 		dest = "strPreset", 	metavar = "preset",
		type  = string, 		default = None,
        help = "Instead of specifying parameters separately, use a preset" )


args = argp.parse_args( ) 
_main( args.foo, args.foo, args.foo, args.foo )


"""

"""

