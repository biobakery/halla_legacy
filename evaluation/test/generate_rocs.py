#!/usr/bin/env python 
#*******************************************************************************************************
#*   Generate roc curves                                                                               *
#*  This program reads the json file with the store information for the rocs and generates them        *
#*******************************************************************************************************
import matplotlib 
matplotlib.use("Agg")
import strudel, halla, numpy 
import sys 
import multiprocessing 
import argparse
import subprocess

from multiprocessing import Process, Queue
from time import gmtime, strftime
from time import *
import random
import json

##global parameters 

def generate_title( strMethod, strBase, strSpike, iRow, iCol, bParam, iPval, fSparsity, fNoise, iIter, fPermutationNoise ):
	
	strMeta = None 
	if iPval == -1:
		strMeta = "association"
	elif iPval == 0:
		strMeta = "association+pval"
	else:
		strMeta = "pval"

	strDelim = "_"
	strTitle = strDelim.join( [strMethod, strBase, strSpike, "s"+str(fSparsity), "n"+str(fNoise), "pn" + str(fPermutationNoise), "i"+str(iIter), str(iRow)+"x"+str(iCol), ("parametric" if bParam == True else "nonparametric"), strMeta])

	return strTitle
	
#***********************************************************************************************
#*  Convert  arrays to lists and post them to the directory so they can be put in json file    *
#***********************************************************************************************
def  Convert_array_to_list_and_push_to_dict(AInput,dEmp, dEmpEntryName):
	lOutput = list()
	for a in AInput:
		lEntry = a.tolist()
		lOutput.append(lEntry)
	dEmp[dEmpEntryName] = lOutput
	return dEmp 
	

def _main( strosj ):
	a=7

	def __main( ):
 		s = strudel.Strudel()

	 
		##########################################################################################################
		#*    Read the json                                                                                      #
		##########################################################################################################
		strInfileName = strosj  
		InputJson  = json.loads(open(strInfileName ).read())
		A_flatten = InputJson['A_flatten']   
		A_emp_flatten = InputJson['A_emp_flatten'] 
		iIter = InputJson['iIter']
		strFile = InputJson['strFile']
		strTitle = InputJson['strTitle']		
		strFileAlpha  = InputJson['strFileAlpha']
		strTitleAlpha = InputJson['strTitleAlpha']
		
	
	 


		##Generate roc curves 
		aROC = s.roc(A_flatten, A_emp_flatten, astrLabel = ["run " + str(i) for i in range(iIter)],
			strTitle = strTitle, strFile = strFile )
		
		aAlpha = s.alpha_fpr( A_flatten, A_emp_flatten, astrLabel = ["run " + str(i) for i in range(iIter)],
			strTitle = strTitleAlpha, strFile = strFileAlpha )

		sys.stderr.write("ROC AUC values: " + "\n")
		sys.stderr.write("\t".join(map(str,aROC)) + "\n")
		sys.stderr.write("Alpha AUC values: " + "\n")
		sys.stderr.write("\t".join(map(str,aAlpha)) + "\n")

		return aROC, aAlpha  
		
	bException = False
	if bException:
		return __main()
	else:
		try:
			__main()
		except Exception:
			sys.stderr.write("ERROR!\n")
			return subprocess.call( ["touch",strFile] )
			### Exception handling added for sfle runs 

	sys.stdout.write("Program Ending "   + strftime("%a, %d %b %Y %H:%M:%S ", gmtime()) + "\n")

  

if __name__ == "__main__":


	argp = argparse.ArgumentParser( prog = "generate_rocs.py",
	        description = "Test different types of associations in strudel + halla." )

	### BUGBUG: can we run on real files?  
	#argp.add_argument( "istm",              metavar = "input.txt",
	#        type = argparse.FileType( "r" ),        default = sys.stdin,    nargs = "+",
	#        help = "Tab-delimited text input file, one row per feature, one column per measurement" )

	
	        
	        
	        
	argp.add_argument( "-osj",                dest = "strosj",             metavar = "Output_Synthetic_Json",
		type = str,   default = "data",
		help = "Name of the output file where the synthetic data will be stored (In JSON format) - default: data" )        
			
			

	args = argp.parse_args( ) 
	sys.stdout.write("Program Starting "   + strftime("%a, %d %b %Y %H:%M:%S ", gmtime()) + "\n")
	 

	_main( args.strosj )
