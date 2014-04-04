"""
Author: Yo Sup Moon,  George Weingart
Description: Halla command python wrapper.
"""

#####################################################################################
#Copyright (C) <2012>
#
#Permission is hereby granted, free of charge, to any person obtaining a copy of
#this software and associated documentation files (the "Software"), to deal in the
#Software without restriction, including without limitation the rights to use, copy,
#modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
#and to permit persons to whom the Software is furnished to do so, subject to
#the following conditions:
#
#The above copyright notice and this permission notice shall be included in all copies
#or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
#PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
#HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
#OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#####################################################################################

__author__ = "Yo Sup Moon, George Weingart"
__copyright__ = "Copyright 2014"
__credits__ = ["Yo Sup Moon","George Weingart"]
__license__ = "MIT"
__maintainer__ = "George Weingart"
__email__ = "george.weingart@gmail.com"
__status__ = "Development"

#************************************************************************************
#*   halla.py                                                                       *
#*   Feb. 12, 2014:   George Weingart                                               *
#*   The objective of this program is to serve as a simple wrapper for the halla    *
#*   command.                                                                       *
#*   It accepts as input the parameters that the halla command would accept and     *
#*    passes them verbatum to the halla command                                     *
#*    NOTE:  the   halla directory has have to be in $PYTHONPATH, e.g.              *
#*   March 18, 2014:   George Weingart                                              *
#*   Split istm into -X and -Y                                                      *
#************************************************************************************

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
from halla import HAllA
from halla import stats
from halla import distance
import halla.parser  
from halla.parser import Input, Output 

from halla.test import *
from halla.stats import *
from halla.distance import *
from halla.hierarchy import *
from halla.plot import *


#=============================================#
# Wrapper  
#=============================================#

def _main(  ):
	""" 
	
	Design principle: be as flexible and modularized as possible. 

	Have native Input/Ouput objects in the halla.parser module.

	All the different runs and tests I am doing in batch mode should be available in the "run" directory with separate files, 

	then loaded into memory like so: `execute( strFile ) ; function_call(  )`
	
	"""
 
 #**************************************************************************************
 #*  Change by George Weingart 2014/03/21                                              *
 #*  Moved the parsing into the _main subroutine                                       *
 #**************************************************************************************
 	argp = argparse.ArgumentParser( prog = "halla.py",
			description = "Hierarchical All-against-All significance association testing." )


	argp.add_argument( "-o",                dest = "ostm",                  metavar = "output.txt",
			type = argparse.FileType( "w" ),        default = sys.stdout,
			help = "Optional output file for association significance tests" )

	argp.add_argument( "-q",                dest = "dQ",                    metavar = "q_value",
			type = float,   default = 0.05,
			help = "Q-value for overall significance tests" )

	argp.add_argument( "-s",                dest = "fS",                    metavar = "start_parameter",
			type = float,   default = 0.25,
			help = "Start parameter; [0.0,1.0]" )

	argp.add_argument( "-i",                dest = "iIter",    metavar = "iterations",
			type = int,             default = 100,
			help = "Number of iterations for nonparametric significance testing" )

	argp.add_argument( "-v",                dest = "iDebug",                metavar = "verbosity",
			type = int,             default = 10 - ( logging.WARNING / 10 ),
			help = "Debug logging level; increase for greater verbosity" )

	argp.add_argument( "-e", "--exploration", 	dest = "strExploration", 	metavar = "exploration",
		type  = str, 		default = "default",
        help = "Exploration function" )

	#argp.add_argument( "-f",                dest = "fFlag",         action = "store_true",
	#		help = "A flag set to true if provided" )

	argp.add_argument( "-x", 		dest = "strPreset", 	metavar = "preset",
			type  = str, 		default = "default",
			help = "Instead of specifying parameters separately, use a preset" )

	argp.add_argument( "-m", 		dest = "strMetric", 	metavar = "metric",
			type  = str, 		default = "norm_mi",
			help = "Metric to be used for hierarchical clustering" )
			
	argp.add_argument( "-X",              metavar = "Xinput.txt",   
			type  =   argparse.FileType( "r" ),        default = sys.stdin,      
			help = "First file: Tab-delimited text input file, one row per feature, one column per measurement" )		
			
	argp.add_argument( "-Y",              metavar = "Yinput.txt",   
			type  =   argparse.FileType( "r" ),        default = None,    
			help = "Second file: Tab-delimited text input file, one row per feature, one column per measurement - If not selected, we will use the first file (-X)" )		
			
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
	aOut1, aOut2 = Input( strFile1.name, strFile2.name ).get()

	aOutData1, aOutName1, aOutType1, aOutHead1 = aOut1 
	aOutData2, aOutName2, aOutType2, aOutHead2 = aOut2 

	H = HAllA( aOutData1, aOutData2 )

	H.set_q( args.dQ )
	H.set_iterations( args.iIter )
	H.set_metric( args.strMetric )
	H.set_start_parameter( args.fS )

	if args.strPreset: 
		H.set_preset( args.strPreset )
		aaOut = H.run( strMethod = args.strPreset )

	else:
		aaOut = H.run( )

	csvw = csv.writer( args.ostm, csv.excel_tab )

	csvw.writerow( ["## HAllA preset: " + args.strPreset, "q value: " + str(args.dQ), "start parameter " + str(args.fS), "metric " + args.strMetric] )

	#if H._is_meta( aaOut ):
	#	if H._is_meta( aaOut[0] ):
	#		for i,aOut in enumerate(aaOut):
	#			csvw.writerow( ["p-value matrix: " + str(i+1)] )
	#			for line in aOut:
	#				csvw.writerow( line )
	#	else:
	#		aOut = aaOut
	#		for line in aOut:
	#			csvw.writerow( line )	
	#else:
	#	aOut = aaOut
	#	for line in aOut:
	#			csvw.writerow( line )

	csvw.writerow( [args.X.name, args.Y.name, "p-value"] )

	for line in aaOut[0]:
		iX, iY = line[0]
		fP = line[1]
		aLineOut = map(str,[aOutName1[iX], aOutName2[iY], fP])
		csvw.writerow( aLineOut )


if __name__ == '__main__':

	_main(  )
