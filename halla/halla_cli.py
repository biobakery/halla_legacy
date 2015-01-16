"""
Author: Gholamali Rahnavard, Yo Sup Moon,  George Weingart
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

__author__ = "Gholamali Rahnavard, Yo Sup Moon, George Weingart"
__copyright__ = "Copyright 2014"
__credits__ = ["Yo Sup Moon","George Weingart"]
__license__ = "MIT"
__hallatainer__ = "George Weingart"
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
import argparse
import csv
import itertools
import logging
import os
import sys
sys.path.append('//Users/rah/Documents/Hutlab/halla')
sys.path.append('/Users/rah/Documents/Hutlab/strudel')
from numpy import array
from halla import HAllA, distance, stats
import halla
from halla.distance import *
from halla.hierarchy import *
from halla.parser import Input, Output
import halla.parser
from halla.plot import *
from halla.stats import *
#from halla.test import *
import shutil 



## internal dependencies 
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
 #*  Moved the parsing into the _halla subroutine                                       *
 #**************************************************************************************
 	filename = "./output/"
	dir = os.path.dirname(filename)
	try:
		shutil.rmtree(dir)
		os.mkdir(dir)
	except:
		os.mkdir(dir)
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

	H = HAllA(args, aOut1, aOut2)
	
	H.set_q( args.dQ )
	H.set_iterations( args.iIter )
	H.set_metric( args.strMetric )
	#print "First Row", halla.stats.discretize(aOutData1)
	#return
	if args.strPreset: 
		aaOut = H.run( strMethod = args.strPreset )
	else:
		aaOut = H.run( )
	
	def _report(args, aOutData2, aOutName2, aOutType2, aOutHead2):
		csvw = csv.writer( args.ostm, csv.excel_tab )
		bcsvw = csv.writer( args.bostm, csv.excel_tab )
		csvw.writerow( ["Method: " + args.strPreset, "q value: " + str(args.dQ), "metric " + args.strMetric] )
		bcsvw.writerow( ["Method: " + args.strPreset, "q value: " + str(args.dQ), "metric " + args.strMetric] )
		
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
		
		# Columns title
		# if we have just one input file 
		
		if args.Y ==None:
			csvw.writerow( [istm[0].name, istm[0].name, "nominal-pvalue", "adjusted-pvalue"] )
			bcsvw.writerow( [istm[0].name, istm[0].name, "nominal-pvalue", "adjusted-pvalue"] )
		else:
			csvw.writerow( [args.X.name, args.Y.name, "nominal-pvalue", "adjusted-pvalue"] )
			bcsvw.writerow( ["Association Number", args.X.name, args.Y.name, "nominal-pvalue", "adjusted-pvalue"] )
		
		
		#print 'aaOut:', aaOut
		#print 'aaOut[0]', aaOut[0]
		for line in aaOut:
			iX, iY = line[0]
			fP = line[1]
			fP_adjust = line[2]
			aLineOut = map(str,[aOutName1[iX], aOutName2[iY], fP, fP_adjust])
			csvw.writerow( aLineOut )
		#print 'H:', H.meta_alla
		#print 'H[0]', H.meta_alla[0]
		associated_feature_X_indecies =  []
		associated_feature_Y_indecies = []
		association_number = 0 
		
		for line in H.meta_alla[0]:
			association_number += 1
			filename = "./output/"+"association"+str(association_number)+'/'
			dir = os.path.dirname(filename)
			try:
				shutil.rmtree(dir)
				os.mkdir(dir)
			except:
				os.mkdir(dir)
				
			iX, iY = line[0]
			associated_feature_X_indecies += iX
			associated_feature_Y_indecies += iY
			fP = line[1]
			fP_adjust = line[2]
			aLineOut = map(str,[association_number, str(';'.join(aOutName1[i] for i in iX)),str(';'.join(aOutName2[i] for i in iY)), fP, fP_adjust])
			bcsvw.writerow( aLineOut )
			import numpy as np
			import pandas as pd
			import matplotlib.pyplot as plt 
			plt.figure()
			cluster1 = [aOutData1[i] for i in iX]
			X_labels = np.array([aOutName1[i] for i in iX])
			#cluster = np.array([aOutData1[i] for i in iX]
			df = pd.DataFrame(np.array(cluster1, dtype= float).T ,columns=X_labels )
			axes = pd.tools.plotting.scatter_matrix(df)
			
			#plt.tight_layout()
			
			plt.savefig(filename+'Dataset_1_cluster_'+str(association_number)+'_scatter_matrix.pdf')
			cluster2 = [aOutData2[i] for i in iY]
			Y_labels = np.array([aOutName2[i] for i in iY])
			plt.figure()
			df = pd.DataFrame(np.array(cluster2, dtype= float).T ,columns=Y_labels )
			axes = pd.tools.plotting.scatter_matrix(df)
			#plt.tight_layout()
			plt.savefig(filename+'Dataset_2_cluster_'+str(association_number)+'_scatter_matrix.pdf')
			df1 = np.array(cluster1, dtype= float)
			df2 = np.array(cluster2, dtype= float)
			plt.figure()
			plt.scatter(halla.stats.pca(df1),halla.stats.pca(df2), alpha=0.5)
			plt.savefig(filename+'/association_'+str(association_number)+'.pdf')
			#plt.figure()
			plt.close("all")
		
		
		csvwc = csv.writer(args.costm , csv.excel_tab )
		csvwc.writerow( ['Level', "Dataset 1","Dataset 2" ] )
		for line in halla.hierarchy.reduce_tree_by_layer([H.meta_hypothesis_tree]):
			(level, clusters ) = line
			iX, iY = clusters[0], clusters[1]
			fP = line[1]
			#fP_adjust = line[2]
			aLineOut = map(str,[str(level),str(';'.join(aOutName1[i] for i in iX)),str(';'.join(aOutName2[i] for i in iY))])
			csvwc.writerow( aLineOut )
		print "R visualization!"
		from scipy.stats.stats import pearsonr
		X_labels = np.array([aOutName1[i] for i in associated_feature_X_indecies])
		Y_labels = np.array([aOutName2[i] for i in associated_feature_Y_indecies])
		cluster1 = [aOutData1[i] for i in associated_feature_X_indecies]	
		cluster2 = [aOutData2[i] for i in associated_feature_Y_indecies]
		df1 = np.array(cluster1, dtype= float)
		df2 = np.array(cluster2, dtype= float)
		p = np.zeros(shape=(len(associated_feature_X_indecies), len(associated_feature_Y_indecies)))
		for i in range(len(associated_feature_X_indecies)):
			for j in range(len(associated_feature_Y_indecies)):
				p[i][j] = pearsonr(df1[i],df2[j])[0]
		nmi = np.zeros(shape=(len(associated_feature_X_indecies), len(associated_feature_Y_indecies)))
		for i in range(len(associated_feature_X_indecies)):
			for j in range(len(associated_feature_Y_indecies)):
				nmi[i][j] = 1.0 - distance.NormalizedMutualInformation( halla.discretize(df1[i]),halla.discretize(df2[j]) ).get_distance()
		rpy2.robjects.numpy2ri.activate()
		ro.r('library("gplots")')
		ro.globalenv['nmi'] = nmi
		ro.globalenv['labRow'] = X_labels 
		ro.globalenv['labCol'] = Y_labels
		ro.r('pdf(file = "./output/NMI_heatmap.pdf")')
		ro.r('heatmap.2(nmi, labRow = labRow, labCol = labCol, col=redgreen(75), scale="row",  key=TRUE, symkey=FALSE, density.info="none", trace="none", cexRow=0.5)')
		ro.r('dev.off()')
		ro.globalenv['p'] = p
		ro.r('pdf(file = "./output/Pearson_heatmap.pdf")')
		ro.r('heatmap.2(p, , labRow = labRow, labCol = labCol, , col=redgreen(75), scale="column",  key=TRUE, symkey=FALSE, density.info="none", trace="none", cexRow=0.5)')
		ro.r('dev.off()')
		#set_default_mode(NO_CONVERSION)
		#rpy2.library("ALL")
		#hm = halla.plot.hclust2.Heatmap( p)#, cl.sdendrogram, cl.fdendrogram, snames, fnames, fnames_meta, args = args )
		#hm.draw()
		#halla.plot.heatmap(D = p, filename ='./output/pearson_heatmap')
		#halla.plot.heatmap(D = nmi, filename ='./output/nmi_heatmap')
	#_report(args, aOutData2, aOutName2, aOutType2, aOutHead2)	
if __name__ == '__main__':
	_main(  )

