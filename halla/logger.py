#!/usr/bin/env python

import logging
import pylab
import sys


c_logrHAllA	 = logging.getLogger("halla")

def _log_plot_histograms(aadValues, pFigure, strTitle=None, astrLegend=None):
	c_iN	 = 20

	figr = pylab.figure() if ((not pFigure) or isinstance(pFigure, str)) else pFigure
	for iValues, adValues in enumerate(aadValues):
		iN = min(c_iN, int((len(adValues) / 5.0) + 0.5))
		iBins, adBins, pPatches = pylab.hist(adValues, iN, normed=1, histtype="stepfilled")
		pylab.setp(pPatches, alpha=0.5)
	if strTitle:
		pylab.title(strTitle)
	if astrLegend:
		pylab.legend(astrLegend, loc="upper left")
	if isinstance(pFigure, str):
		pylab.savefig(pFigure)

def _log_plot_scatter(adX, adY, pFigure, strTitle=None, strX=None, strY=None):

	figr = pylab.figure() if ((not pFigure) or isinstance(pFigure, str)) else pFigure
	pylab.scatter(adX, adY)
	if strTitle:
		pylab.title(strTitle)
	if strX:
		pylab.xlabel(strX)
	if strY:
		pylab.ylabel(strY)
	if isinstance(pFigure, str):
		pylab.savefig(pFigure)

def write_table(data=None, name=None, rowheader=None, colheader=None, prefix = "label",  corner = None, delimiter= '\t'):
    
    '''
    wite a matrix of data in tab-delimated format file
    
    input:
    data: a 2 dimensioal array of data
    name: includes path and the name of file to save
    rowheader
    columnheader
    
    output:
    a file tabdelimated file 
    
    '''
    if data is None:
    	print ("Null input for writing table")
    	return
    f = open(name, 'w')
    # row numbers as header
    if colheader is None:
        if corner is None:
            f.write(delimiter)
        else:
            f.write(corner)
            f.write(delimiter)
        for i in range(len(data[0])):
            f.write("S"+str(i))
            if i < len(data[0]) - 1:
                f.write(delimiter)
        f.write('\n')
    elif len(colheader) == len(data[0]):
        if corner is None:
            f.write(delimiter)
        else:
            f.write(corner)
            f.write(delimiter)
        for i in range(len(data[0])):
            f.write(colheader[i])
            if i < len(data[0]) - 1:
                f.write(delimiter)
        f.write('\n')
    else:
        print("The label list in not matched with the data size")
        sys.exit()
        
    for i in range(len(data)):
        if rowheader is None :
            f.write(prefix+str(i))
            f.write(delimiter)
        else:
            f.write(rowheader[i])
            f.write(delimiter)
        for j in range(len(data[i])):
                f.write(str(data[i][j]))
                if j < len(data[i]) - 1:
                    f.write(delimiter)
        f.write('\n')
    f.close() 
    
def write_circos_table(data, name=None, rowheader=None, colheader=None, prefix = "label",  corner = None, delimiter= '\t'):
    
    '''
    wite a matrix of data in tab-delimated format file
    
    input:
    data: a 2 dimensioal array of data
    name: includes path and the name of file to save
    rowheader
    columnheader
    
    output:
    a file tabdelimated file 
    
    '''
    f = open(name, 'w')
    
    # write order header
    f.write("Data")
    f.write(delimiter)
    f.write("Data")
    f.write(delimiter)
    for i in range(len(data[0])):
            f.write(str(i+1))
            if i < len(data[0]) - 1:
                f.write(delimiter)
    f.write('\n')
    # column numbers as header
    f.write("Data")
    f.write(delimiter)
    if len(colheader) == 0:
        f.write("Data")
        f.write(delimiter)
        for i in range(len(data[0])):
            f.write(str(i))
            if i < len(data[0]) - 1:
                f.write(delimiter)
        f.write('\n')
    elif len(colheader) == len(data[0]):
        f.write("Data")
        f.write(delimiter)
        for i in range(len(data[0][:])):
            f.write(colheader[i])
            if i < len(data[0]) - 1:
                f.write(delimiter)
        f.write('\n')
    else:
        sys.err("The lable list in not matched with the data size")
        sys.exit()
    
    for i in range(len(data)):
        if len(rowheader) == 0:
            f.write(str(i+len(data[0])))
            f.write(prefix+str(i))
            f.write(delimiter)
        elif len(colheader) == len(data[0]):
            f.write(str(i+len(data[0])+1))
            f.write(delimiter)
            f.write(rowheader[i])
            f.write(delimiter)
        else:
            sys.err("The lable list in not matched with the data size")
            sys.exit()
        
        for j in range(len(data[i])):
                f.write(str(data[i][j]))
                if j < len(data[i]) - 1:
                    f.write(delimiter)
        f.write('\n')
    f.close() 