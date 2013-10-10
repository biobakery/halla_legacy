from matplotlib import pyplot as pl 
import pylab 
import numpy 
import csv 
import sys 

strFile = sys.argv[1]

csvr = csv.reader(open(strFile), csv.excel_tab)  
a = numpy.array( [l for l in csvr] ) 

#print list(a[:,0])
#print list(a[:,1])

strXlabel, strYlabel = a[0]

pl.scatter( map(float,a[:,0][1:]), map(float,a[:,1][1:]) ) 
pl.xlabel(strXlabel)
pl.ylabel(strYlabel) 

pl.show() 
