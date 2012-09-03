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
import numpy
import scipy.stats
import sys

import pylab

import dataset
import datum

c_logrHAllA	= logging.getLogger( "halla" )

def test1( pData, hashClusters ):

	#===============================================================================
	# Permutation test leaf clusters
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

def test2( pData, hashClusters, iBootstrap ):

	#===============================================================================
	# Bootstrap test leaf clusters
	#===============================================================================
	pData.bootstrap( iBootstrap )
	adTotal = pData.get_bootstrap( )
	dTotal = numpy.average( adTotal )
	aaClusters = sorted( hashClusters.items( ) )
	for iOne, aOne in enumerate( aaClusters ):
		iX = aOne[1][0]
		pOne = pData.get( iX )
		for aTwo in aaClusters[( iOne + 1 ):]:
			iY = aTwo[1][0]
			pTwo = pData.get( iY )
			dMID = pOne.mutual_information_distance( pTwo )
			dPOne = scipy.stats.percentileofscore( pData.get_permutation( iX, iY ), dMID ) / 100
			
			adBootstrap = pData.get_bootstrap( iX, iY )
			dMI = pOne.mutual_information( pTwo )

#			sys.stderr.write( "%s\n" % adBootstrap )
# too sensitive
#			dU, dPTwo = scipy.stats.mannwhitneyu( adTotal, adBootstrap )
#			dPTwo = 1 if ( numpy.average( adBootstrap ) > dTotal ) else ( dPTwo * 2 )
# too sensitive
#			dU, dPTwo = scipy.stats.ks_2samp( adTotal, adBootstrap )
			dU, dPTwo = scipy.stats.ttest_ind( adTotal, adBootstrap )
			if numpy.average( adBootstrap ) > dTotal:
				dPTwo = 1
# this works decently well, need to also test here whether the point estimate is significantly different than the permuted null

			c_logrHAllA.debug( "%s" % [pOne.m_strID, pTwo.m_strID, dMI, 1 - dMID, dU, dPTwo, dPOne] )

			if ( dPOne < 0.05 ) and ( dPTwo < ( 0.05 ) ):#/ ( len( aaClusters ) * ( len( aaClusters ) - 1 ) / 2 ) ) ):
				pylab.figure( )
				iN = 20
				iBins, adBins, pPatches = pylab.hist( adTotal, iN, normed = 1, histtype = "stepfilled" )
				pylab.setp( pPatches, "facecolor", "blue", "alpha", 0.5 )
				iBins, adBins, pPatches = pylab.hist( adBootstrap, iN, normed = 1, histtype = "stepfilled" )
				pylab.setp( pPatches, "facecolor", "red", "alpha", 0.5 )
				pylab.title( "%g (%g), p1=%g, p2=%g" % (dMI, 1 - dMID, dPOne, dPTwo) )
				pylab.savefig( "-".join( (s.split( "|" )[-1] for s in (pOne.m_strID, pTwo.m_strID)) ) + ".png" )

def halla( istm, ostm, dPMI, iBootstrap ):

	pData = dataset.CDataset( datum.CDatum.mutual_information_distance )
	pData.open( istm )
	hashClusters = pData.hierarchy( dPMI )

	test2( pData, hashClusters, iBootstrap )
	
	return None

argp = argparse.ArgumentParser( prog = "halla.py",
	description = """Hierarchical All-against-All significance association testing.""" )
argp.add_argument( "istm",		metavar = "input.txt",
	type = argparse.FileType( "r" ),	default = sys.stdin,	nargs = "?",
	help = "Tab-delimited text input file, one row per feature, one column per measurement" )
argp.add_argument( "-o",		dest = "ostm",			metavar = "output.txt",
	type = argparse.FileType( "w" ),	default = sys.stdout,
	help = "Optional output file for association significance tests" )
argp.add_argument( "-p",		dest = "dPMI",			metavar = "p_mi",
	type = float,	default = 0.05,
	help = "P-value for permutation equivalence of MI clusters" )
argp.add_argument( "-b",		dest = "iBootstrap",	metavar = "bootstraps",
	type = int,		default = 100,
	help = "Number of bootstraps for significance testing" )
argp.add_argument( "-v",		dest = "iDebug",		metavar = "verbosity",
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

	halla( args.istm, args.ostm, args.dPMI, args.iBootstrap )

if __name__ == "__main__":
	_main( )
