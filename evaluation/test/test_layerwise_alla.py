#!/usr/bin/env python 
# source: http://matplotlib.org/examples/pylab_examples/multipage_pdf.html

import strudel, halla 
import argparse
import datetime
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import sys 

#strFile = [sys.argv[1] if sys.argv[1:] else "test_layerwise_alla.pdf"]
def _main( iRow, iCol, strSpike ):
	s = strudel.Strudel()
	X = s.randmat( shape = (iRow,iCol) )
	Y,A = s.spike( X, strMethod = strSpike )

	h = halla.HAllA( X,Y )
	aA_emp = h.run( strMethod = "layerwise" )

	for i, A_emp in enumerate(aA_emp):
		tIn = (A.flatten(), 1.0 - A_emp.flatten())
		s.roc( tIn[0], tIn[1], strFile =  strSpike + "_" + "roc_" + str(iRow)+"x"+str(iCol)+ "_layer_" + str(i+1) + ".pdf" )
		s.alpha_fpr( tIn[0], tIn[1], strFile =  strSpike + "_" + "qq_" + str(iRow)+"x"+str(iCol)+ "_layer_" + str(i+1) + ".pdf" )


# Create the PdfPages object to which we will save the pages:
# The with statement makes sure that the PdfPages object is closed properly at
# the end of the block, even if an Exception occurs.
#with PdfPages( strFile ) as pdf:
   

	"""
    plt.figure(figsize=(3, 3))
    plt.plot(range(7), [3, 1, 4, 1, 5, 9, 2], 'r-o')
    plt.title('Page One')
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

    plt.rc('text', usetex=True)
    plt.figure(figsize=(8, 6))
    x = np.arange(0, 5, 0.1)
    plt.plot(x, np.sin(x), 'b-')
    plt.title('Page Two')
    pdf.savefig()
    plt.close()

    plt.rc('text', usetex=False)
    fig = plt.figure(figsize=(4, 5))
    plt.plot(x, x*x, 'ko')
    plt.title('Page Three')
    pdf.savefig(fig)  # or you can pass a Figure object to pdf.savefig
    plt.close()

    # We can also set the file's metadata via the PdfPages object:
    d = pdf.infodict()
    d['Title'] = 'Multipage PDF Example'
    d['Author'] = u'Jouni K. Sepp\xe4nen'
    d['Subject'] = 'How to create a multipage pdf file and set its metadata'
    d['Keywords'] = 'PdfPages multipage keywords author title subject'
    d['CreationDate'] = datetime.datetime(2009, 11, 13)
    d['ModDate'] = datetime.datetime.today()
    """

if __name__ == "__main__":
	argp = argparse.ArgumentParser( prog = "test_layerwise_alla.py",
	        description = "Test layerwise all-against-all in halla" )

	argp.add_argument( "--row",                dest = "iRow",             metavar = "num_rows",
	        type = int,   default = "20",
	        help = "Number of rows" )

	argp.add_argument( "--col",                dest = "iCol",             metavar = "num_cols",
	        type = int,   default = "20",
	        help = "Number of columns" )

	argp.add_argument( "--spike_method",                dest = "strSpike",             metavar = "spike_method",
	        type = str,   default = "parabola",
	        help = "Spike method: [linear, vee, sine, parabola, cubic, log, half_circle]" )

	args = argp.parse_args( ) 

	_main( args.iRow, args.iCol, args.strSpike )
