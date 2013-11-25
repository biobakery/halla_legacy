#!/usr/bin/env python 
'''
Parses input/output formats, 
manages transformations
'''

import csv 
import numpy as np 


class Input:
	"""
	
	Parser class for input 
	============================


	Handling Missing Values
	------------------------------

	Handling Data Types
	------------------------------

	* `CON` <- continous
	* `CAT` <- categorical
	* `BIN` <- binary 
	* `LEX` <- lexical 
	
	"""
	def __init__( self, strFileName1, strFileName2 = None, var_names = True, headers = False ):
		
		#Data types 
		self.continuous = "CON"
		self.categorical = "CAT"
		self.binary = "BIN"
		self.lexical = "LEX"

		#Boolean indicators 
		self.varNames = var_names 
		self.headers = headers

		#Initialize data structures 
		self.strFileName1 = strFileName1
		self.strFileName2 = strFileName1 if not strFileName2 else strFileName2 

		self.outData1 = None
		self.outData2 = None 

		self.outName1 = None 
		self.outName2 = None 

		self.outType1 = None
		self.outType2 = None 

		self.outHead1 = None
		self.outHead2 = None 

		self.csvr1 = csv.reader( open( self.strFileName1 ), csv.excel_tab )
		self.csvr2 = csv.reader( open( self.strFileName2 ), csv.excel_tab )
		
		self._load( ) 
		self._parse( )
		self._check( )
		self._out( )

	def _load( self ):
		self.outData1 = np.array( [x for x in self.csvr1] ) 
		self.outData2 = np.array( [x for x in self.csvr2] )

	def _parse( self ):
		@staticmethod 
		def __parse( pArray, bVar, bHeaders ):
			aOut = [] 
			aNames = []
			aTypes = []
			aHeaders = []
			
			## Parse headers and variable names 
			if bHeaders:
				aHeaders = pArray[0]
				pArray = pArray[1:]

			if bVar: 
				aNames = pArray[:,0]
				pArray = pArray[:,1:]

			## Parse data types, missing values, and whitespace 
			for i, line in enumerate( pArray ):
				line = map( lambda x: ( x.strip() if bool(x.strip()) else None ), line )
				if all(line):
					aOut.append(line)
					if not aNames:
						aNames.append( i )

					try: 
						line = map(int, line) #is it explicitly categorical?  
						aTypes.append("CAT")
					except ValueError:
						try:
							line = map(float, line) #is it continuous? 
							aTypes.append("CON")
						except ValueError:
							line = line #we are forced to conclude that it is implicitly categorical, with some lexical ordering 
							aTypes.append("LEX")

			return aOut, aNames, aTypes, aHeaders 

		self.outData1, self.outName1, self.outType1, self.outHead1 = _parse( self.outData1, self.varNames, self.headers )
		self.outData2, self.outName2, self.outType2, self.outHead2 = _parse( self.outData2, self.varNames, self.headers )

	def _check( self ):
		"""
		Make sure that the data are well-formed
		"""

		assert( len( self.outData1 ) == len( self.outType1 ) )
		assert( len( self.outData2 ) == len( self.outType2 ) )

		if self.outName1:
			assert( len( self.outData1 ) == len( self.outName1 ) )
		if self.outName2:
			assert( len( self.outData2 ) == len( self.outName2 ) )
		if self.outHead1:
			assert( len( self.outData1 ) == len( self.outHead1 ) )
		if self.outHead2:
			assert( len( self.outData2 ) == len( self.outHead2 ) )

	def _out( self ):
		return [(self.outData1, self.outName1, self.outType1, self.outHead1), (self.outData2, self.outName2, self.outType2, self.outHead2)] 


class Output:
	"""
	Parser class for output 
	============================


	Handling Missing Values
	------------------------------

	Handling Data Types
	------------------------------

	* `CON` <- continous
	* `CAT` <- categorical
	* `BIN` <- binary 
	* `LEX` <- lexical 
	"""
	def __init__( self ):
		pass 

