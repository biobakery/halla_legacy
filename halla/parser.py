#!/usr/bin/env python 
'''
Parses input/output formats, 
manages transformations
'''

import csv
import re
import sys
from numpy import array

import numpy as np
import pandas as pd


class Input:
	"""
	
	Parser class for input 

	Handles missing values, data type transformations 

	* `CON` <- continous
	* `CAT` <- categorical
	* `BIN` <- binary 
	* `LEX` <- lexical 
	
	"""
	def __init__(self, strFileName1, strFileName2=None, var_names=True, headers=False):
		
		# Data types 
		self.continuous = "CON"
		self.categorical = "CAT"
		self.binary = "BIN"
		self.lexical = "LEX"

		# Boolean indicators 
		self.varNames = var_names 
		self.headers = headers

		# Initialize data structures 
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
		
		self._load()
		self._parse()
		self._check()

	def get(self):
		return [(self.outData1, self.outName1, self.outType1, self.outHead1), (self.outData2, self.outName2, self.outType2, self.outHead2)] 
		
	def _load(self):
		def __load(file):
			# Read in the file
			try:
				file_handle=open(file)
			except EnvironmentError:
				sys.exit("Error: Unable to read file: " + file)
				
			csvr = csv.reader(file_handle, csv.excel_tab)
			
			# Ignore comment lines in input file
			data=[]
			comments=[]
			for line in csvr:
				# Add comment to list
				if re.match("#",line[0]):
					comments.append(line)
				else:
					# First data line found
					data=[line]
					break
				
			# Check if last comment is header
			if comments:
				header=comments[-1]
				# if the same number of columns then last comment is header
				if len(header) == len(data[0]):
					data=[header,data[0]]
					
			# finish processing csv
			for line in csvr:
				data.append(line)
				
			# close csv file
			file_handle.close()
				
			return np.array(data)
		
		self.outData1 = __load(self.strFileName1)
		self.outData2 = __load(self.strFileName2)

	def _parse(self):
		def __parse(pArray, bVar, bHeaders):
 
			aOut = [] 
			aNames = []
			aTypes = []
			aHeaders = []
			
			# Parse header if indicated by user or "#"
			if bHeaders or re.match("#",pArray[0,0]):
				aHeaders = list(pArray[0,1:])
				pArray = pArray[1:]

			# Parse variable names
			if bVar: 
				aNames = list(pArray[:, 0])
				pArray = pArray[:, 1:]

			# # Parse data types, missing values, and whitespace
			from sklearn.preprocessing import Imputer
			imp = Imputer(missing_values='NaN', strategy='mean', axis=1)
			imp.fit(pArray)
			#Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)
			#line = [[np.nan, 2], [6, np.nan], [7, 6]]
			#print imp 
			missing_char =" "
			for i, line in enumerate(pArray):
				###########line = map( lambda x: ( x.strip() if bool(x.strip()) else None ), line )
				#*****************************************************************************************************
				# *   Modification by George Weingart  2014/03/20                                                     *
				# *   If the line is not full,  replace the Nones with nans                                           *
				#***************************************************************************************************** 
				
				line = map(lambda x: (x.strip(missing_char) if bool(x.strip(missing_char)) else np.nan), line)  ###### Convert missings to nans
				#line = df1 = pd.DataFrame(line)
				line = imp.transform(line)[0]
				#print line 
				if all(line):
					aOut.append(line)
					if not aNames:
						aNames.append(i)

					try: 
						line = map(int, line)  # is it explicitly categorical?  
						aTypes.append("CAT")
					except ValueError:
						try:
							line = map(float, line)  # is it continuous? 
							aTypes.append("CON")
						except ValueError:
							line = line  # we are forced to conclude that it is implicitly categorical, with some lexical ordering 
							aTypes.append("LEX")
				else:  # delete corresponding name from namespace 
					try:
						aNames.remove(aNames[i])
					except Exception:
						pass  

			return aOut, aNames, aTypes, aHeaders 

		self.outData1, self.outName1, self.outType1, self.outHead1 = __parse(self.outData1, self.varNames, self.headers)
		self.outData2, self.outName2, self.outType2, self.outHead2 = __parse(self.outData2, self.varNames, self.headers)

	def _check(self):
		"""
		Make sure that the data are well-formed
		"""
		assert(len(self.outData1[0]) == len(self.outData2[0]))
		assert(len(self.outData1) == len(self.outType1))
		assert(len(self.outData2) == len(self.outType2))

		if self.outName1:
			assert(len(self.outData1) == len(self.outName1))
		if self.outName2:
			assert(len(self.outData2) == len(self.outName2))
		if self.outHead1:
			assert(len(self.outData1[0]) == len(self.outHead1))
		if self.outHead2:
			assert(len(self.outData2[0]) == len(self.outHead2))
			
		# If sample names are included in headers in both files,
		# check that the samples are in the same order
		if self.outHead1 and self.outHead2:
			header1="\t".join(self.outHead1)
			header2="\t".join(self.outHead2)
			if not (header1.lower() == header2.lower()):
				print("Warning: The samples are not in the same order " + 
				    "in the two files. The program use the common samples between the two dataset based on headers")#+
				    #"." + " \n File1 header: " + header1 + "\n" +
				    #" File2 header: " + header2)
				try:
					df1 = pd.DataFrame(self.outData1, index = self.outName1, columns = self.outHead1)
				except:
					df1 = pd.DataFrame(self.outData1, index = self.outName1, columns = self.outHead1)
				try:
					df2 = pd.DataFrame(self.outData2, index = self.outName2, columns = self.outHead2)
				except:
					df2 = pd.DataFrame(self.outData2, index = self.outName2, columns = self.outHead2)
				#print df1.columns.isin(df2.columns)
				#print df2.columns.isin(df1.columns)
				df1 = df1.loc[: , df1.columns.isin(df2.columns)]
				df2 = df2.loc[: , df2.columns.isin(df1.columns)]
				
				# reorder df1 columns as the columns order of df2
				df1 = df1.loc[:, df2.columns]
				
				self.outData1 = df1.values
				self.outData2 = df2.values 
				#print self.outData1
				self.outName1 = list(df1.index.values) 
				self.outName2 = list(df2.index.values) 
				#print self.outName1
				#self.outType1 = int
				#self.outType2 = int 
		
				self.outHead1 = df1.columns.values
				self.outHead2 = df2.columns.values 
				#print df1
				#print df2
		

	
class Output:
	"""
	Parser class for output 
	
	In batch mode: takes data generated by HAllA object and transforms it to useable objects 

	"""
	def __init__(self):
		pass 

	def roc(self):
		pass

	def get_auc(self):
		pass 

	def plot(self):
		pass 







