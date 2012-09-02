#!/usr/bin/env python
#######################################################################################
# This file is provided under the Creative Commons Attribution 3.0 license.
#
# You are free to share, copy, distribute, transmit, or adapt this work
# PROVIDED THAT you attribute the work to the authors listed below.
# For more information, please see the following web page:
# http://creativecommons.org/licenses/by/3.0/
#
# This file is a component of the SflE Scientific workFLow Environment for reproducible 
# research, authored by the Huttenhower lab at the Harvard School of Public Health
# (contact Curtis Huttenhower, chuttenh@hsph.harvard.edu).
#
# If you use this environment, the included scripts, or any related code in your work,
# please let us know, sign up for the SflE user's group (sfle-users@googlegroups.com),
# pass along any issues or feedback, and we'll let you know as soon as a formal citation
# is available.
#######################################################################################

"""
.. testsetup::

	from halla import *
"""

import argparse
import logging
import scipy.stats
import sys

import dataset
import datum

c_logrHAllA	= logging.getLogger( "halla" )

def _mi2distance( dMI ):

	return ( 1 / ( 1 + dMI ) )

def halla( istm, ostm, dPMI ):

	pData = dataset.CDataset( datum.CDatum.mutual_information_distance )
	pData.open( istm )
	hashClusters = pData.hierarchy( dPMI )
	
	#===============================================================================
	# Test leaf clusters
	#===============================================================================
	adMIs = [d for a in pData.m_aadPermutations for d in a]
	aaClusters = sorted( hashClusters.items( ) )
	for iOne, aOne in enumerate( aaClusters ):
		pOne = pData.get( aOne[1][0] )
		for aTwo in aaClusters[( iOne + 1 ):]:
			pTwo = pData.get( aTwo[1][0] )
			dMI = pOne.mutual_information( pTwo )
			dMIR = pOne.mutual_information_relative( pTwo )
			dP = scipy.stats.percentileofscore( adMIs, 1 - dMIR ) / 100
			c_logrHAllA.debug( "%s" % [pOne.m_strID, pTwo.m_strID, dMI, dMIR, dP] )

	return None

argp = argparse.ArgumentParser( prog = "halla.py",
	description = """Hierarchical All-against-All significance association testing.""" )
argp.add_argument( "istm",		metavar = "input.txt",
	type = argparse.FileType( "r" ),	default = sys.stdin,	nargs = "?",
	help = "Tab-delimited text input file, one row per feature, one column per measurement" )
argp.add_argument( "-o",		dest = "ostm",		metavar = "output.txt",
	type = argparse.FileType( "w" ),	default = sys.stdout,
	help = "Optional output file for association significance tests" )
argp.add_argument( "-p",		dest = "dPMI",		metavar = "p_mi",
	type = float,	default = 0.05,
	help = "P-value for permutation equivalence of MI clusters" )
argp.add_argument( "-v",		dest = "iDebug",	metavar = "verbosity",
	type = int,		default = 10 - ( logging.WARNING / 10 ),
	help = "Debug logging level; increase for greater verbosity" )
"""
argp.add_argument( "-f",		dest = "fFlag",		action = "store_true",
	help = "A flag set to true if provided" )
argp.add_argument( "strString",	metavar = "string",
	help = "A required free text string" )
"""
__doc__ = "::\n\n\t" + argp.format_help( ).replace( "\n", "\n\t" ) + __doc__

def _main( ):
	args = argp.parse_args( )

	lghn = logging.StreamHandler( sys.stderr )
	lghn.setFormatter( logging.Formatter( '%(asctime)s %(levelname)10s %(module)s.%(funcName)s@%(lineno)d %(message)s' ) )
	c_logrHAllA.addHandler( lghn )
	c_logrHAllA.setLevel( ( 10 - args.iDebug ) * 10 )

	halla( args.istm, args.ostm, args.dPMI )

if __name__ == "__main__":
	_main( )
