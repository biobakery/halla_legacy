## native python packages 

import itertools 

## structural packages 

from numpy import array 

import halla 
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

	def _parser( strFile ):
		aOut = [] 
		csvr = csv.reader(open( strFile ), csv.excel_tab )

		astrHeaders = None 
		astrNames = []

		for line in csvr:
			def _dec( x ):
				return ( x.strip() if bool(x.strip()) else None )

			if not astrHeaders:
				astrHeaders = line 
			
			else: 
				strName = line[0]
				 
				line = line[1:]
				line = map( _dec , line )
				
				if all(line):

					astrNames.append( strName )
					try: 
						line = map(int, line) #is it explicitly categorical?  
					except ValueError:
						try:
							line = map(float, line) #is it continuous? 
						except ValueError:
							line = line #we are forced to conclude that it is implicitly categorical, with some lexical ordering 
					#sys.stderr.write( "\t".join( map(str,line)  ) + "\n" ) 
					aOut.append(line)
		
		Data = array( aOut )
		Name = array( astrNames )
		
		assert( len(Data) == len(Name) )
		#At this point, all empty data should have been thrown out 

		#Name, Data = Array[:,0][1:], Array[1:][:,1:]

		return Name, Data 

	Name1, Data1 = _parser( strFile1 )
	Name2, Data2 = _parser( strFile2 )

	#print Data1[0]
	#print Data2[0]

	CH = halla.HAllA( Data1, Data2 )

	#c_strOutputPath = "/home/ysupmoon/Dropbox/halla/output/"
	#CH.set_directory( c_strOutputPath )
	
	#CH.run_rev1_test() 

	CH.run_caketest()

	"""
	pOutHash = CH.run_pr_test()

	csvw = csv.writer( sys.stdout , csv.excel_tab )

	astrHeaders = ["Var1", "Var2", "MID", "pPerm", "pPearson", "rPearson"]

	#Write the header
	csvw.writerow( astrHeaders )

	for k,v in pOutHash.items():
		iX, iY = k 
		csvw.writerow( [Name1[iX], Name2[iY]] + [v[j] for j in astrHeaders[2:]] )

	sys.stderr.write("Done!\n")
	#sys.stderr.write( str( pOutHash ) ) 
	"""


_main() 