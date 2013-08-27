import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as pl 
import pylab 
import numpy 
import csv 
import sys 
import scipy 
import scipy.stats 
from scipy.stats import scoreatpercentile as sap 
import itertools 
from pprint import pprint 

c_bPScatter = True  
c_bDataScatter = True     

c_iPercentPermLow, c_iPercentPermHigh = 5, 95
c_iPercentPearsonLow, c_iPercentPearsonHigh = 5, 95  

strFile = sys.argv[1]

csvr = csv.reader(open(strFile), csv.excel_tab)  

hashTable = {} 

## parse table 
astrHeaders = None 
for line in csvr:
	if not astrHeaders:
		astrHeaders = line 
		for x in line:
			hashTable[x] = [] 
		continue 
	else:
		for iItem, item in enumerate( line ):
			hashTable[astrHeaders[iItem]].append( item )

adPerm, adPearson, adPearsonr = map(float,hashTable["pPerm"]), map(float,hashTable["pPearson"]), map(float, hashTable["rPearson"] )

if c_bPScatter:
	pl.scatter( adPerm , adPearson ) 
	pl.xlabel("$p_{halla}$")
	pl.ylabel("$p_{pearson}$") 

	pl.savefig("pscatter.pdf")

if c_bDataScatter:
	
	aOut = [] 

	stMatch = set([]) 

	percentile_perm_low, percentile_perm_high = sap( adPerm, c_iPercentPermLow ), sap( adPerm, c_iPercentPermHigh )
	percentile_pearson_low, percentile_pearson_high  = sap( adPearson, c_iPercentPearsonLow ), sap( adPearson, c_iPercentPearsonHigh )

	astrVar1, astrVar2 = hashTable["Var1"], hashTable["Var2"]

	iCount = 0

	aOut.append(["Type", "Var1","Var2","pPerm", "pPearson", "rPearson"])

	for i,x in enumerate(adPerm):
		try: 
			y,z = adPearson[i], adPearsonr[i] 
			if ( x <= percentile_perm_low ) and ( y >= percentile_pearson_high ):
				
				aOut.append( ["Outlier:HAllA", astrVar1[i], astrVar2[i],x,y,z] )
			elif ( x >= percentile_perm_high ) and ( y <= percentile_pearson_low ):
				
				aOut.append( ["Outlier:Pearson", astrVar1[i], astrVar2[i],x,y,z] )

			elif ( x <= percentile_perm_low ) and ( y <= percentile_pearson_low ):
			
				aOut.append( ["Outlier:Both", astrVar1[i], astrVar2[i],x,y,z] )

			elif ( x >= percentile_perm_high ) and ( y >= percentile_pearson_high ):
			
				aOut.append( ["Outlier:None", astrVar1[i], astrVar2[i],x,y,z] )
		except IndexError:
			continue 
			#stMatch = stMatch | set([frozenset([astrVar1[i], astrVar2[i]])])



	for line in aOut:
		print "\t".join( map(str,line) ) 


