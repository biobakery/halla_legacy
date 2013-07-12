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

import dataset
import datum
import hlog #logging features
import htest # hypothesis testing module 

c_logrHAllA	= logging.getLogger( "halla" )

def _halla_clusters( ostm, hashClusters, pData ):
	
	fFirst = True
	for iCluster, aiCluster in hashClusters.items( ):
		if len( aiCluster ) > 1:
			if fFirst:
				fFirst = False
				ostm.write( "Clusters\n" )
			ostm.write( "%s\n" % "\t".join( pData.get( i ).m_strID for i in aiCluster ) )

def _halla_test( ostm, pData, hashClusters, dP, iBootstrap ):

	pData.bootstrap( iBootstrap )
	adTotal = pData.get_bootstrap( )
	dTotal = numpy.average( adTotal )
	aaClusters = sorted( hashClusters.items( ) )
	fFirst = True
#===============================================================================
# Should add hierarchical testing at this point to prevent all-against-all MHT.
# Specifically, each test is looking at two nodes in the pData hierarchy.  Each
# test should thus be:
#   dPPerm = something like
#     paired test of (real values) vs. (score-at-percentile-of-pvalue in permutations)
#       across all leaves in the node
#     except this can't be exactly right, since it doesn't reduce correctly in the
#       single-leaf case
#     what we want to test is "is at least one of these values different"
#       not sure if simes (min p-value) or similar would be appropriate
#   dPBoot = something like
#     t-test of (boot dists for all leaves in node) vs. (total boot dist)
#     this one's easy
# Then use the algorithm:
#   dDepth = 1.0 (full height)
#   while not done:
#     dDepth /= 2
#     hashNodes = clusters at dDepth
#     dPDepth = 1 - ( 1 - dP )**len( hashNodes )
#     for each node in hashNodes:
#       test dPPerm and dPBoot against dPDepth
#       if significant, recurse at half depth; if not, stop exploring branch
#===============================================================================
	for iOne, aOne in enumerate( aaClusters ):
		iX = aOne[1][0]
		c_logrHAllA.info( "Testing %d/%d" % (iOne, len( aaClusters )) )
		for iTwo in xrange( iOne + 1, len( aaClusters ) ):
			aTwo = aaClusters[iTwo]
			iY = aTwo[1][0]

			pTest = htest.CHTest( pData, iX, iY, adTotal, dTotal )
			dMID, dPPerm = pTest.test_permutation( )
			dU, dPBoot = pTest.test_bootstrap( )
			if all( ( d > dP ) for d in (dPPerm, dPBoot) ):
				continue
			if fFirst:
				fFirst = False
				pTest.save_header( ostm )
			pTest.save( ostm )
			
			if ( c_logrHAllA.level <= logging.DEBUG ) and \
				all( ( d <= dP ) for d in (dPPerm, dPBoot) ):
 					hlog._log_plot( pData, pTest, adTotal )

def halla( istm, ostm, dP, dPMI, iBootstrap ):

	pData = dataset.CDataset( datum.CDatum.mutual_information_distance )
	pData.open( istm )
	hashClusters = pData.hierarchy( dPMI )
	_halla_clusters( ostm, hashClusters, pData )
	_halla_test( ostm, pData, hashClusters, dP, iBootstrap )

argp = argparse.ArgumentParser( prog = "halla.py",
	description = """Hierarchical All-against-All significance association testing.""" )
argp.add_argument( "istm",		metavar = "input.txt",
	type = argparse.FileType( "r" ),	default = sys.stdin,	nargs = "?",
	help = "Tab-delimited text input file, one row per feature, one column per measurement" )
argp.add_argument( "-o",		dest = "ostm",			metavar = "output.txt",
	type = argparse.FileType( "w" ),	default = sys.stdout,
	help = "Optional output file for association significance tests" )
argp.add_argument( "-p",		dest = "dP",			metavar = "p_value",
	type = float,	default = 0.05,
	help = "P-value for overall significance tests" )
argp.add_argument( "-P",		dest = "dPMI",			metavar = "p_mi",
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

	halla( args.istm, args.ostm, args.dP, args.dPMI, args.iBootstrap )

if __name__ == "__main__":
	_main( )
