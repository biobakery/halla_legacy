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
from pandas import *
from . import config
from . import distance
from . import store
from . import stats

def wrap_features(txt, width=40):
		'''helper function to wrap text for long labels'''
		import textwrap
		txt = str(txt).split("|")
		if len(txt)>1:
			txt = txt[len(txt)-2]+"_"+txt[len(txt)-1]
		else:
			txt = txt[0]
		return txt #'\n'.join(textwrap.wrap(txt, width))

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

		self.discretized_dataset1 = None
		self.discretized_dataset2 = None 
		self.orginal_dataset1 = None
		self.orginal_dataset2 = None 
		self.outName1 = None 
		self.outName2 = None 

		self.outType1 = None
		self.outType2 = None 

		self.outHead1 = None
		self.outHead2 = None 
		
		self._load()
		self._parse()
		self._filter_to_common_columns()
		if distance.c_hash_association_method_discretize[config.distance]:
			try: 
				self._remove_low_entropy_features()
			except:
				pass
		else:
			try: 
				self._remove_low_variant_features()
			except:
				pass
		if store.bypass_discretizing():
			try:
				self.orginal_dataset1= np.asarray(self.orginal_dataset1, dtype = float)
				self.orginal_dataset2= np.asarray(self.orginal_dataset2, dtype = float)
				self.discretized_dataset1 = self.orginal_dataset1
				self.discretized_dataset2 = self.orginal_dataset2
			except:
				sys.exit("--- Please check your data types and your similarity metric!")
		else:
		    print "Discretizing is started using: ", config.strDiscretizing, " style!"
		    self._discretize()
		
		
	def get(self):
		return [(self.discretized_dataset1, self.orginal_dataset1, self.outName1, self.outType1, self.outHead1), 
			(self.discretized_dataset2, self.orginal_dataset2, self.outName2, self.outType2, self.outHead2)] 
		
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
		
		self.orginal_dataset1 = __load(self.strFileName1)
		self.orginal_dataset2 = __load(self.strFileName2)
		
	
	def _discretize(self):
	    self.discretized_dataset1 = stats.discretize(self.orginal_dataset1, style = config.strDiscretizing)
	    self.discretized_dataset2 = stats.discretize(self.orginal_dataset2, style = config.strDiscretizing)
	   
	def _parse(self):
		def __parse(pArray, bVar, bHeaders):
 
			aOut = [] 
			aNames = []
			aTypes = []
			aHeaders = None
			
			# Parse header if indicated by user or "#"
			if bHeaders or re.match("#",pArray[0,0]):
				aHeaders = list(pArray[0,1:])
				pArray = pArray[1:]

			# Parse variable names
			if bVar: 
				aNames =  list(pArray[:, 0])
				aNames = map(str, aNames)
				aNames = map(wrap_features, aNames)
				pArray = pArray[:, 1:]

			# # Parse data types, missing values, and whitespace
			if config.missing_method:
				from sklearn.preprocessing import Imputer
				imp = Imputer(missing_values='NaN', strategy=config.missing_method, axis=1)
				imp.fit(pArray)
			#Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)
			#line = [[np.nan, 2], [6, np.nan], [7, 6]]
			#print imp 
 
			for i, line in enumerate(pArray):
				###########line = map( lambda x: ( x.strip() if bool(x.strip()) else None ), line )
				#*****************************************************************************************************
				# *   Modification by George Weingart  2014/03/20 # changed by Ali 2015/11/06                                                    *
				# *   If the line is not full,  replace the Nones with nans                                           *
				#***************************************************************************************************** 
				if config.missing_method is  None and not distance.c_hash_association_method_discretize[config.distance]:
					warn_message ="There is missing data in feature "+  aNames[i]+"!!! " + "Try --missing-method=method to choose your filling strategy. "
					line = map(lambda x: (x.strip(config.missing_char) if bool(x.strip(config.missing_char)) 
										else sys.exit(warn_message )), line)  ###### np.nan Convert missings to nans
				else:
					line = map(lambda x: (x.strip(config.missing_char) if bool(x.strip(config.missing_char)) else np.nan ), line)  ###### np.nan Convert missings to nans
					#line = df1 = pd.DataFrame(line)
					if not distance.c_hash_association_method_discretize[config.distance]:
						try:
							line = imp.transform(line)[0]
						except:
							print "there is an issue with filling missed data!"
					#print line 
				if all(val != config.missing_char for val in line):
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
						print aNames[i], " has an issue with filling missed data!"
						#aNames.remove(aNames[i])
					except Exception:
						pass  

			return aOut, aNames, aTypes, aHeaders 

		self.orginal_dataset1, self.outName1, self.outType1, self.outHead1 = __parse(self.orginal_dataset1, self.varNames, self.headers)
		self.orginal_dataset2, self.outName2, self.outType2, self.outHead2 = __parse(self.orginal_dataset2, self.varNames, self.headers)

	def _filter_to_common_columns(self):
		"""
		Make sure that the data are well-formed
		"""
		
		assert(len(self.orginal_dataset1) == len(self.outType1))
		assert(len(self.orginal_dataset2) == len(self.outType2))

		if self.outName1:
			assert(len(self.orginal_dataset1) == len(self.outName1))
		if self.outName2:
			assert(len(self.orginal_dataset2) == len(self.outName2))
		if self.outHead1:
			assert(len(self.orginal_dataset1[0]) == len(self.outHead1))
		if self.outHead2:
			assert(len(self.orginal_dataset2[0]) == len(self.outHead2))
			
		# If sample names are included in headers in both files,
		# check that the samples are in the same order
		if self.outHead1 and self.outHead2:
			header1="\t".join(self.outHead1)
			header2="\t".join(self.outHead2)
			if not (header1.lower() == header2.lower()):
				print("Warning: The samples are not in the same order " + 
				    "in the two files. The program uses the common samples between the two data sets based on headers")#+
				    #"." + " \n File1 header: " + header1 + "\n" +
				    #" File2 header: " + header2)
				try:
					df1 = pd.DataFrame(self.orginal_dataset1, index = self.outName1, columns = self.outHead1)
				except:
					df1 = pd.DataFrame(self.orginal_dataset1, index = self.outName1, columns = self.outHead1)
				try:
					df2 = pd.DataFrame(self.orginal_dataset2, index = self.outName2, columns = self.outHead2)
				except:
					df2 = pd.DataFrame(self.orginal_dataset2, index = self.outName2, columns = self.outHead2)
				#print df1.columns.isin(df2.columns)
				#print df2.columns.isin(df1.columns)
				df1 = df1.loc[: , df1.columns.isin(df2.columns)]
				df2 = df2.loc[: , df2.columns.isin(df1.columns)]
				
				# reorder df1 columns as the columns order of df2
				df1 = df1.loc[:, df2.columns]
				
				self.orginal_dataset1 = df1.values
				self.orginal_dataset2 = df2.values 
				#print self.orginal_dataset1
				self.outName1 = list(df1.index) 
				self.outName2 = list(df2.index) 
				#print self.outName1
				#print self.outName2
				#self.outType1 = int
				#self.outType2 = int 
		
				self.outHead1 = df1.columns
				self.outHead2 = df2.columns 
				#print self.outHead1
				#print df2
		assert(len(self.orginal_dataset1[0]) == len(self.orginal_dataset2[0]))
	def _remove_low_variant_features(self):
		try:
			df1 = pd.DataFrame(self.orginal_dataset1, index = self.outName1, columns = self.outHead1, dtype=float)
		except:
			df1 = pd.DataFrame(self.orginal_dataset1, index = self.outName1, columns = self.outHead1, dtype=float)
		try:
			df2 = pd.DataFrame(self.orginal_dataset2, index = self.outName2, columns = self.outHead2, dtype=float)
		except:
			df2 = pd.DataFrame(self.orginal_dataset2, index = self.outName2, columns = self.outHead2, dtype=float)
		#print df1.columns.isin(df2.columns)
		#print df2.columns.isin(df1.columns)
		#print df1.var(), np.var(df2, axis=1)
		l1_before =  len(df1.index)
		l2_before =  len(df2.index)
		df1 = df1[df1.var(axis=1) > config.min_var]
		df2 = df2[df2.var(axis=1) > config.min_var]
		
		l1_after = len(df1.index)
		l2_after = len(df2.index)
		if l1_before > l1_after:
			print "WARNING! %d features with variation equal or less than %d have been removed from the first dataset " % ((l1_before- l1_after), config.min_var)
			
		if l2_before > l2_after:
			print "WARNING! %d features with variation equal or less than %d have been removed from the second dataset " % ((l2_before- l2_after), config.min_var)
		# reorder df1 columns as the columns order of df2
		#df1 = df1.loc[:, df2.columns]
		
		self.orginal_dataset1 = df1.values
		self.orginal_dataset2 = df2.values 
		#print self.orginal_dataset1
		self.outName1 = list(df1.index) 
		self.outName2 = list(df2.index) 
		#print self.outName1
		#self.outType1 = int
		#self.outType2 = int 

		#self.outHead1 = df1.columns
		#self.outHead2 = df2.columns 
		#print self.outHead1
		#print df2
		assert(len(self.orginal_dataset1[0]) == len(self.orginal_dataset2[0]))
	def _remove_low_entropy_features(self):
		try:
			df1 = pd.DataFrame(self.discretized_dataset1, index = self.outName1, columns = self.outHead1)
			df1_org = pd.DataFrame(self.orginal_dataset1, index = self.outName1, columns = self.outHead1)
		except:
			df1 = pd.DataFrame(self.discretized_dataset1, index = self.outName1, columns = self.outHead1, dtype=float)
			df1_org = pd.DataFrame(self.orginal_dataset1, index = self.outName1, columns = self.outHead1)
		try:
			df2 = pd.DataFrame(self.discretized_dataset2, index = self.outName2, columns = self.outHead2, dtype=float)
			df2_org = pd.DataFrame(self.orginal_dataset2, index = self.outName2, columns = self.outHead2)
		except:
			df2 = pd.DataFrame(self.discretized_dataset2, index = self.outName2, columns = self.outHead2, dtype=float)
			df2_org = pd.DataFrame(self.orginal_dataset2, index = self.outName2, columns = self.outHead2)
		#print df1.columns.isin(df2.columns)
		#print df2.columns.isin(df1.columns)
		#print df1.var(), np.var(df2, axis=1)
		l1_before =  len(df1.index)
		l2_before =  len(df2.index)
		df1 = df1[len(set(df1)) > 0]
		df2 = df2[len(set(df2))> 0]
		
		l1_after = len(df1.index)
		l2_after = len(df2.index)
		if l1_before > l1_after:
			print "WARNING! %d features with variation equal or less than %d have been removed from the first dataset " % ((l1_before- l1_after), config.min_var)
			
		if l2_before > l2_after:
			print "WARNING! %d features with variation equal or less than %d have been removed from the second dataset " % ((l2_before- l2_after), config.min_var)
		# reorder df1 columns as the columns order of df2
		#df1 = df1.loc[:, df2.columns]
		
		self.discretized_dataset1 = df1.values
		self.discretized_dataset2 = df2.values 
		#print self.discretized_dataset1
		self.outName1 = list(df1.index) 
		self.outName2 = list(df2.index) 
		#print self.outName1
		#self.outType1 = int
		#self.outType2 = int 

		#self.outHead1 = df1.columns
		#self.outHead2 = df2.columns 
		#print self.outHead1
		#print df2
		assert(len(self.discretized_dataset1[0]) == len(self.discretized_dataset2[0]))	

	
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







