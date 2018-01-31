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
from . import HSIC

def wrap_features(txt, width=40):
		'''helper function to wrap text for long labels'''
		import textwrap
		txt = txt.replace('s__','').replace('g__','').replace('f__','').replace('o__','').replace('c__','').replace('p__','').replace('k__','')
		txt = str(txt).split("|")
		txt = [val for val in txt if len(val)>0 ]
		if len(txt)>1:
			txt = txt[len(txt)-2]+" "+txt[len(txt)-1]
		else:
			txt = txt[0]
		return txt #'\n'.join(textwrap.wrap(txt, width))
	
def substitute_special_characters(txt):
    txt = re.sub('[\n\;]', '_', txt).replace('__','_').replace('__','_').replace('_',' ') # replace('.','_')
    return txt
def load(file):
	# Read in the file
	try:
		import io
		file_handle=io.open(file, encoding='utf-8')
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
		print ("Discretizing is started using: %s style for filtering features with low entropy!" % config.strDiscretizing)
		self._discretize()
		self._remove_low_entropy_features()
		if len(self.outName1) <2 or len(self.outName1) <2:
			sys.exit("--- HAllA to continue needs at lease two features in each dataset!!!\n--- Please repeat the one feature or provide the -a AllA option in the command line to do pairwise alla-against-all test!!")
		store.smart_decisoin()
		if store.bypass_discretizing():
			try:
				self.orginal_dataset1= np.asarray(self.orginal_dataset1, dtype = float)
				self.orginal_dataset2= np.asarray(self.orginal_dataset2, dtype = float)
				self._transform_data()
				#self.discretized_dataset1 = self.orginal_dataset1
				#self.discretized_dataset2 = self.orginal_dataset2
			except:
				sys.exit("--- Please check your data types and your similarity metric!")
		self._check_for_semi_colon()
		
	def get(self):
		
		return [(self.discretized_dataset1, self.orginal_dataset1, self.outName1, self.outType1, self.outHead1), 
			(self.discretized_dataset2, self.orginal_dataset2, self.outName2, self.outType2, self.outHead2)] 
		
	def _load(self):
		self.orginal_dataset1 = load(self.strFileName1)
		self.orginal_dataset2 = load(self.strFileName2)
	def _check_for_semi_colon(self):
		# check the names of features that HAllA uses to make sure they don't have ; which
		# is special character to separate features in output files
		for i in range(len(self.outName1)):
			if ";"  in self.outName1[i]:
				print ("Feature names warning!") 
				print (self.outName1[i])
				sys.exit("In the first dataset, your feature (row) names contains ; which is the special character HAllA uses for separating features,\n \
				             Please replace it with another character such as _")
		for i in range(len(self.outName2)):
			if ";"  in self.outName2[i]:
				print ("Feature names warning!")
				print (self.outName2[i])
				sys.exit("In the second dataset, your feature (row) names contains ; which is the special character HAllA uses for separating features,\n \
				             Please replace it with another character such as _")

	def _discretize(self):
	    self.discretized_dataset1 = stats.discretize(self.orginal_dataset1, style = config.strDiscretizing, data_type = config.data_type[0])
	    self.discretized_dataset2 = stats.discretize(self.orginal_dataset2, style = config.strDiscretizing, data_type = config.data_type[1])
	   
	def _parse(self):
		def __parse(pArray, bVar, bHeaders):
 
			aOut = [] 
			aNames = []
			used_names = []
			aTypes = []
			aHeaders = None
			
			# Parse header if indicated by user or "#"
			if bHeaders or re.match("#",pArray[0,0]):
				aHeaders = list(pArray[0,1:])
				pArray = pArray[1:]

			# Parse variable names
			if bVar: 
				aNames =  list(pArray[:, 0])
				aNames = list(map(str, aNames))
				if config.format_feature_names:
					aNames = list(map(wrap_features, aNames))
					aNames = list(map(substitute_special_characters, aNames))
				pArray = pArray[:, 1:]

			# # Parse data types, missing values, and whitespace
			if config.missing_method:
				from sklearn.preprocessing import Imputer
				imp = Imputer(missing_values='nan', strategy=config.missing_method, axis=1)
				imp.fit(pArray)

			for i, line in enumerate(pArray):
				# *   If the line is not full,  replace the Nones with nans                                           *
				#***************************************************************************************************** 
				line = list(map(lambda x: 'nan' if x == config.missing_char else x, line))  ###### np.nan Convert missings to nans
				if all([val == 'nan' for val in line]):
					# if all values in a feature are missing values then skip the feature
					print ('All missing value in' , aNames[i])
					continue
				
				aOut.append(line)
				if not aNames:
					aNames.append(i)

				try:
					line = list(map(float, line))  # is it continuous? 
					# fill missing values
					if config.missing_method:
						try:
							line = imp.transform(line)[0]
						except:
							print (aNames[i], " has an issue with filling missed data!")
							pass
					else:
						line = list(map(float, line))
					#print "Continues data !"
					aTypes.append("CON")
				except ValueError:
					#print "Categorical data !"
					line = line  # we are forced to conclude that it is implicitly categorical, with some lexical ordering 
					aTypes.append("LEX")
			
				used_names.append(aNames[i]) 
			return aOut, used_names, aTypes, aHeaders 

		self.orginal_dataset1, self.outName1, self.outType1, self.outHead1 = __parse(self.orginal_dataset1, self.varNames, self.headers)
		self.orginal_dataset2, self.outName2, self.outType2, self.outHead2 = __parse(self.orginal_dataset2, self.varNames, self.headers)
		config.data_type[0] = self.outType1
		config.data_type[1] = self.outType2
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
			#print header1, header2
			#if not (header1.lower() == header2.lower()):
			#+
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
			
			l1_before =  len(df1.columns)
			l2_before =  len(df2.columns)

			# remove samples/columns with all NaN/missing values
			# First change missing value to np.NaN for pandas
			df1[df1=='nan'] =np.NAN
			df2[df2=='nan'] =np.NAN
			df1 = df1.dropna( axis=1, how='all')
			df2 = df2.dropna( axis=1, how='all')
			l1_after = len(df1.columns)
			l2_after = len(df2.columns)
			# replace np.NaN's with 'nan'
			df1[df1.isnull()] = 'nan'
			df2[df2.isnull()] = 'nan'
			
			if l1_before > l1_after:
				print ("--- %d samples/columns with all missing values have been removed from the first dataset " % (l1_before- l1_after))
				
			if l2_before > l2_after:
				print ("--- %d samples/columns with all missing values have been removed from the second dataset " % (l2_before- l2_after))
		
			
			# Keep common samples/columns between two data frame
			df1 = df1.loc[: , df1.columns.isin(df2.columns)]
			df2 = df2.loc[: , df2.columns.isin(df1.columns)]
		
			# reorder df1 columns as the columns order of df2
			
			df1 = df1.loc[:, df2.columns]
			
			self.orginal_dataset1 = df1.values
			self.orginal_dataset2 = df2.values 
			#print self.orginal_dataset1
			#print HSIC.HSIC_pval(df1.values,df2.values, p_method ='gamma', N_samp =1000)

			self.outName1 = list(df1.index) 
			self.outName2 = list(df2.index) 
			#print self.outName1
			#print self.outName2
			#self.outType1 = int
			#self.outType2 = int 
	
			#self.outHead1 = df1.columns
			#self.outHead2 = df2.columns 
			self.outHead1 = df1.columns
			self.outHead2 = df2.columns
			print("The program uses %s common samples between the two data sets based on headers")%(df1.shape[1])
		if len(self.orginal_dataset1[0]) != len(self.orginal_dataset2[0]):
			sys.exit("Have you provided --header option to use sample/column names for shared sample/columns.")
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
			print ("--- %d features with variation equal or less than %.3f have been removed from the first dataset " % (l1_before- l1_after, config.min_var))
			
		if l2_before > l2_after:
			print ("--- %d features with variation equal or less than %.3f have been removed from the second dataset " % (l2_before- l2_after, config.min_var))
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
		#print self.discretized_dataset1
		#print self.orginal_dataset1
		df1 = pd.DataFrame(self.discretized_dataset1, index = self.outName1, columns = self.outHead1)
		df1_org = pd.DataFrame(self.orginal_dataset1, index = self.outName1, columns = self.outHead1)
		
		df2 = pd.DataFrame(self.discretized_dataset2, index = self.outName2, columns = self.outHead2)
		df2_org = pd.DataFrame(self.orginal_dataset2, index = self.outName2, columns = self.outHead2)
		
		#print df1.columns.isin(df2.columns)
		#print df2.columns.isin(df1.columns)
		#print df1.var(), np.var(df2, axis=1)
		l1_before =  len(df1.index)
		l2_before =  len(df2.index)
		
		# filter for only features with entropy greater that the threshold
		temp_df1 = df1 
		df1 = df1[df1.apply(stats.get_enropy, 1) > config.entropy_threshold1]
		df1_org = df1_org[temp_df1.apply(stats.get_enropy, 1) > config.entropy_threshold1]
		
		temp_df2 = df2 
		df2 = df2[df2.apply(stats.get_enropy, 1) > config.entropy_threshold2]
		df2_org = df2_org[temp_df2.apply(stats.get_enropy, 1) > config.entropy_threshold2]
		
		l1_after = len(df1.index)
		l2_after = len(df2.index)
		if l1_before > l1_after:
			print ("--- %d features with entropy equal or less than %.3f have been removed from the first dataset " % ((l1_before- l1_after), config.entropy_threshold1))
			
		if l2_before > l2_after:
			print ("--- %d features with entropy equal or less than %.3f have been removed from the second dataset " % ((l2_before- l2_after), config.entropy_threshold2))
		# reorder df1 columns as the columns order of df2
		#df1 = df1.loc[:, df2.columns]
		
		self.discretized_dataset1 = df1.values
		self.orginal_dataset1 = df1_org.values
		
		self.discretized_dataset2 = df2.values
		self.orginal_dataset2 = df2_org.values 
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
		try:
			print ("--- %d features and %d samples are used from first dataset" % (l1_after, len(self.discretized_dataset1[0])))
		except IndexError:
			sys.exit("WARNING! No feature in the first dataset after filtering.")
		try:
			print ("--- %d features and %d samples are used from second dataset" % (l2_after, len(self.discretized_dataset2[0])))
		except IndexError:
			sys.exit("WARNING! No feature in the second dataset after filtering.")
			
		assert(len(self.discretized_dataset1[0]) == len(self.discretized_dataset2[0]))	
	
	def _transform_data(self):
		scale = config.transform_method
		#print(self.orginal_dataset1)
		self.orginal_dataset1 = stats.scale_data(self.orginal_dataset1, scale = scale)
		self.orginal_dataset2 = stats.scale_data(self.orginal_dataset2, scale = scale)
		#print(self.orginal_dataset1)





