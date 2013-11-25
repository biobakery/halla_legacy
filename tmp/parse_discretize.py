#!/usr/bin/env python 

from datum import discretize 
import csv
import sys 
import numpy as np 

args = sys.argv[1:] if sys.argv[1:] else None 

if not args: 
	raise Exception("Usage: parse_discretize.py data.txt > data_parsed.txt")


csvr = csv.reader(open( args[0] ), csv.excel_tab )

sys.stderr.write("Loading Data ...\n")

data = np.array( [l for l in csvr] )

sys.stderr.write("Discretizing Data ...\n")
discretized_data = np.array( [ map( lambda x: "B"+str(x), line ) for line in discretize( data ) ] )

csvw = csv.writer( sys.stdout, csv.excel_tab ) 

iRow, iCol = discretized_data.shape

# Write Header 
csvw.writerow( ["Var" + str(i) for i in range(1,iRow+1)] )

for line in discretized_data.T:
	csvw.writerow( line ) 

sys.stderr.write("Done!\n") 
