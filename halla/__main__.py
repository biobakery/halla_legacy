## native python packages 

import itertools 

## structural packages 

from numpy import array 

import halla 
from halla.parser import Input, Output 
import csv 
import sys 
import re 
import os 
import pprint 

def _main():
	if len(sys.argv[1:]) > 1:
		strFile1, strFile2 = sys.argv[1:3]
	elif len(sys.argv[1:]) == 1:
		strFile1, strFile2 = sys.argv[1], sys.argv[1]

	
	aOut1, aOut2 = Input( strFile1, strFile2 )

	aOutData1, aOutName1, aOutType1, aOutHead1 = aOut1 
	aOutData2, aOutName2, aOutType2, aOutHead2 = aOut2 


	H = halla.HAllA( Data1, Data2 )

	#c_strOutputPath = "/home/ysupmoon/Dropbox/halla/output/"
	#H.set_directory( c_strOutputPath )

	def pr1():
		
		pOutHash = H.run_pr_test()
		csvw = csv.writer( sys.stdout , csv.excel_tab )
		astrHeaders = ["Var1", "Var2", "MID", "pPerm", "pPearson", "rPearson"]

		#Write the header
		csvw.writerow( astrHeaders )

		for k,v in pOutHash.items():
			iX, iY = k 
			csvw.writerow( [Name1[iX], Name2[iY]] + [v[j] for j in astrHeaders[2:]] )

		sys.stderr.write("Done!\n")
		#sys.stderr.write( str( pOutHash ) ) 

	def rev1():
		H.run_rev1_test() 

	def cake1():
		H.run_caketest()


"""
def halla( istm, ostm, dP, dPMI, iBootstrap ):

        pData = dataset.CDataset( datum.CDatum.mutual_information_distance )
        pData.open( istm )
        hashClusters = pData.hierarchy( dPMI )
        _halla_clusters( ostm, hashClusters, pData )
        _halla_test( ostm, pData, hashClusters, dP, iBootstrap )

argp = argparse.ArgumentParser( prog = "halla.py",
        description = """Hierarchical All-against-All significance association testing.""" )
argp.add_argument( "istm",              metavar = "input.txt",
        type = argparse.FileType( "r" ),        default = sys.stdin,    nargs = "?",
        help = "Tab-delimited text input file, one row per feature, one column per measurement" )
argp.add_argument( "-o",                dest = "ostm",                  metavar = "output.txt",
        type = argparse.FileType( "w" ),        default = sys.stdout,
        help = "Optional output file for association significance tests" )
argp.add_argument( "-p",                dest = "dP",                    metavar = "p_value",
        type = float,   default = 0.05,
        help = "P-value for overall significance tests" )
argp.add_argument( "-P",                dest = "dPMI",                  metavar = "p_mi",
        type = float,   default = 0.05,
        help = "P-value for permutation equivalence of MI clusters" )
argp.add_argument( "-b",                dest = "iBootstrap",    metavar = "bootstraps",
        type = int,             default = 100,
        help = "Number of bootstraps for significance testing" )
argp.add_argument( "-v",                dest = "iDebug",                metavar = "verbosity",
        type = int,             default = 10 - ( logging.WARNING / 10 ),
        help = "Debug logging level; increase for greater verbosity" )
"""
argp.add_argument( "-f",                dest = "fFlag",         action = "store_true",
        help = "A flag set to true if provided" )
argp.add_argument( "strString", metavar = "string",
        help = "A required free text string" )
"""
__doc__ = "::\n\n\t" + argp.format_help( ).replace( "\n", "\n\t" ) + __doc__

def _main( ):
        args = argp.parse_args( )

        lghn = logging.StreamHandler( sys.stderr )
        lghn.setFormatter( logging.Formatter( '%(asctime)s %(levelname)10s %(module)s.%(funcName)s@%(lineno)d %(message)s' ) )
        c_logrHAllA.addHandler( lghn )
        c_logrHAllA.setLevel( ( 10 - args.iDebug ) * 10 )

        halla( args.istm, args.ostm, args.dP, args.dPMI, args.iBootstrap )

if __name__ == "__main__":
        _main( )
"""

