#!/usr/bin/env python 

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


##global parameters 
c_num_cores = 8 

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
	

def _main( strFile, strFileAlpha, iRow, iCol, strMethod, iIter, fSparsity, fNoise, strSpike, strBase, bParam, iPval, fPermutationNoise, bException, strMultiProcessing ):

	strTitle = generate_title( strMethod, strBase, strSpike, iRow, iCol, bParam, iPval, fSparsity, fNoise, iIter, fPermutationNoise )
	strTitleAlpha = strTitle + "_alpha"

	strFile = strFile or (strTitle + ".pdf") 
	strFileAlpha = strFileAlpha or (strTitleAlpha + ".pdf")

	def __main( ):
 		s = strudel.Strudel()
		s.set_base(strBase)
		s.set_noise(fNoise)

		##Generate synthetic data 
		X = s.randmat( shape = (iRow, iCol) )
		Y,A = s.spike_synthetic_data( X, sparsity = fSparsity, spike_method = strSpike )

		##Run iIterations of pipeline 
		A_emp = [] 

		if strMultiProcessing == "N":
			for i in range(iIter):
				sys.stderr.write("Running iteration " + str(i) + "\n")
				aOut = s.association( X,Y, strMethod = strMethod, bParam = bParam, bPval = iPval )
				A_emp.append(aOut)
		else:
			A_emp = Calc_Associations_Using_Multiprocessing(s, X,Y, strMethod,bParam, iPval, iIter, A_emp) #* Perform calculations of associations using multiprocessing

		##Set meta objects 
		A_emp_flatten = None 
		if iPval == -1:
			A_emp_flatten = [numpy.reshape( numpy.abs(a), iRow**2 ) for a in A_emp] 
		elif iPval == 1:
			## Remember, associations are _always_ notion of strength, not closeness; this is an invariant I will strictly enforce 
			A_emp_flatten = [1.0 - numpy.reshape( numpy.abs(a), iRow**2 ) for a in A_emp] 
		
		A_flatten = [numpy.reshape( A, iRow**2 ) for _ in range(iIter)]

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

	if bException:
		return __main()
	else:
		try:
			__main()
		except Exception:
			sys.stderr.write("ERROR!\n")
			return subprocess.call( ["touch",strFile] )
			### Exception handling added for sfle runs 

	sys.stdout.write(args.strMethod +  " " + "Multiprocessing=" + args.strMultiProcessing + "  : Program Ending "   + strftime("%a, %d %b %Y %H:%M:%S ", gmtime()) + "\n")



#********************************************************************************
#*  Calculate the associations using Multiprocessing                            *
#********************************************************************************
def Calc_Associations_Using_Multiprocessing(s, X,Y, strMethod,bParam, iPval, iIter, A_emp):
	procs = []
	out_q = Queue()

	for i in range(iIter):
		sys.stderr.write(strMethod + " : Starting Process  " + str(i+1) + "  - " + strftime("%a, %d %b %Y %H:%M:%S ", gmtime())+ "\n")

		#X1 = s.randmat( shape = (2, 2) )
		#Y1,A1 = s.spike_synthetic_data( X1, sparsity = 0.1, spike_method = 'sine' )
		#CalcResult1 = s.association( X1,Y1, strMethod = strMethod, bParam = bParam, bPval = iPval )
		
		
		p = multiprocessing.Process(
			target=calc_assoc_worker,
			args=(s,X,Y, strMethod,bParam, iPval, out_q))
		procs.append(p)
		p.start()

	# Collect all results  

	for i in range(iIter):
		sys.stderr.write(strMethod + " : Collecting results from  Process #" + str(i+1) + "  - " + strftime("%a, %d %b %Y %H:%M:%S ", gmtime()) +"\n")
		aOut = out_q.get(['block', None])
		A_emp.append(aOut)
 



	# Wait for all worker processes to finish
	i = 0
	for p in procs:
		i+=1
		sys.stderr.write(strMethod + " : Joining  Process " + str(i)+ "  - " + strftime("%a, %d %b %Y %H:%M:%S ", gmtime()) + "\n")
		p.join()

	return  A_emp


#********************************************************************************
#*  Calculate the associations worker                                           *
#********************************************************************************
def  calc_assoc_worker(s,X,Y, strMethod,bParam, iPval,   out_q):
	numpy.random.seed()      # To Generate different results
	CalcResult = s.association( X,Y, strMethod = strMethod, bParam = bParam, bPval = iPval )
	out_q.put(CalcResult)
	return 0

if __name__ == "__main__":

	argp = argparse.ArgumentParser( prog = "test_associations.py",
	        description = "Test different types of associations in strudel + halla." )

	### BUGBUG: can we run on real files?  
	#argp.add_argument( "istm",              metavar = "input.txt",
	#        type = argparse.FileType( "r" ),        default = sys.stdin,    nargs = "+",
	#        help = "Tab-delimited text input file, one row per feature, one column per measurement" )

	argp.add_argument( "-o",                dest = "strFile",                  metavar = "output_plot",
	       type = str,        default = None,
	        help = "Optional output file name for script-generated plot" )

	argp.add_argument( "--output_alpha", "-oa",                dest = "strFileAlpha",                  metavar = "output_plot_alpha",
	       type = str,        default = None,
	        help = "Optional output file name for FPR vs. alpha plot" )

	argp.add_argument( "-m",                dest = "strMethod",             metavar = "method_name",
	        type = str,   default = "pearson",
	        help = "Association method. [pearson, spearman, mi, norm_mi, kw, x2, halla]" )

	argp.add_argument( "--row",                dest = "iRow",             metavar = "num_rows",
	        type = int,   default = "20",
	        help = "Number of rows" )

	argp.add_argument( "--col",                dest = "iCol",             metavar = "num_cols",
	        type = int,   default = "20",
	        help = "Number of columns" )

	argp.add_argument( "--parametric",                dest = "bParam",
	        action = "store_true",
	        help = "Parametric pvalue generation? else permutation based error bar generation. The only ones with parametric error bars are pearson, spearman, x2" )

	argp.add_argument( "--exception",                dest = "bException",
	        action = "store_true",
	        help = "Be less lenient about exception handling. Used in non-sfle mode" )

	argp.add_argument( "-i",                dest = "iIter",             metavar = "num_iteration",
	        type = int,   default = "3",
	        help = "Number of iterations for each association method" )

	argp.add_argument( "-s",                dest = "fSparsity",             metavar = "sparsity",
	        type = float,   default = "0.5",
	        help = "Sparsity parameter, value in [0.0,1.0]" )

	argp.add_argument( "-n",                dest = "fNoise",             metavar = "noise",
	        type = float,   default = "0.1",
	        help = "Noise parameter, value in [0.0,1.0]" )

	argp.add_argument( "--permutation_noise",                dest = "fPermutationNoise",             metavar = "permutation_noise",
	        type = float,   default = "0.1",
	        help = "Permutation noise parameter, value in [0.0,1.0]" )

	argp.add_argument( "--spike_method",                dest = "strSpike",             metavar = "spike_method",
	        type = str,   default = "parabola",
	        help = "Spike method: [linear, vee, sine, parabola, cubic, log, half_circle]" )

	argp.add_argument( "-b",                dest = "strBase",             metavar = "base_distribution",
	        type = str,   default = "normal",
	        help = "Base distribution: [normal, uniform]" )

	argp.add_argument( "-p",                dest = "iPval",             metavar = "use_pval",
	        type = int,   default = 1,
	        help = "Parameter to request for association values or p-values. -1 -> only association value, 0 -> both association and p-value (do not use for now), 1 -> only p-value" )

			
	argp.add_argument( "--multiprocessing", "-mp",                dest = "strMultiProcessing",             metavar = "MultiprocessingRequest",
	        type = str,   default = "Y",
	        help = "Request to process iterations using multiprocessing - default: Y" )			
			

	args = argp.parse_args( ) 
	sys.stdout.write(args.strMethod +  " " + "Multiprocessing=" + args.strMultiProcessing + "  : Program Starting "   + strftime("%a, %d %b %Y %H:%M:%S ", gmtime()) + "\n")


	_main( args.strFile, args.strFileAlpha, args.iRow, args.iCol, args.strMethod, args.iIter, args.fSparsity, args.fNoise, args.strSpike, args.strBase, args.bParam, args.iPval, args.fPermutationNoise, args.bException,  args.strMultiProcessing )
