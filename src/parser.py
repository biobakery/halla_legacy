#!/usr/bin/env python 
'''
Parses input/output formats, 
manages transformations
'''

import csv 
import numpy as np 


class CInput:
	def __init__( self, strFileName ):
		self.strFileName = strFileName
		self.csvr = csv.reader( open( strFileName ), csv.excel_tab )
		self._load( ) 
	def _load( self ):
		self.inputData = np.array( [x for x in self.csvr] ) 

		
