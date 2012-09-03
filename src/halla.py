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

class CTest:
	
	def __init__( self, pData, iOne, iTwo, adTotal = None ):
		
		self.m_pData = pData
		self.m_iOne, self.m_iTwo = iOne, iTwo
		self.m_pOne, self.m_pTwo = (pData.get( i ) for i in (self.m_iOne, self.m_iTwo))
		self.m_adTotal = adTotal if adTotal else self.m_pData.get_bootstrap( )
		self.m_dTotal = numpy.average( self.m_adTotal )
		self.m_dMI = self.m_dMID = self.m_dPPerm = self.m_dPBoot = None

	def test_permutation( self ):
		
		if self.m_dMID == None:
			self.m_dMID = self.m_pOne.mutual_information_distance( self.m_pTwo )
		if self.m_dPPerm == None:
			self.m_dPPerm = scipy.stats.percentileofscore( self.m_pData.get_permutation( self.m_iOne, self.m_iTwo ),
				self.m_dMID ) / 100

		return (self.m_dMID, self.m_dPPerm)

	def test_bootstrap( self ):

		if self.m_dPBoot == None:
			adBootstrap = self.m_pData.get_bootstrap( self.m_iOne, self.m_iTwo )
			dU, self.m_dPBoot = scipy.stats.ttest_ind( self.m_adTotal, adBootstrap )
			# Check this for sidedness
			if numpy.average( adBootstrap ) > self.m_dTotal:
				self.m_dPBoot = 1
		else:
			dU = None
			
		return (dU, self.m_dPBoot)

	def get_mutual_information( self ):
		
		if self.m_dMI == None:
			self.m_dMI = self.m_pOne.mutual_information( self.m_pTwo )
		return self.m_dMI
	
	def save_header( self, ostm ):

		ostm.write( "%s\n" % "\t".join( ("One", "Two", "MI", "MID", "Pperm", "Pboot", "P") ) )

	def save( self, ostm ):
		
		dMID, dPPerm = self.test_permutation( )
		dU, dPBoot = self.test_bootstrap( )
		ostm.write( "%s\n" % "\t".join( [self.m_pOne.m_strID, self.m_pTwo.m_strID] +
			[( "%g" % d ) for d in (self.get_mutual_information( ), dMID, dPPerm, dPBoot, max( dPPerm, dPBoot ))] ) )

def _log_plot_histograms( aadValues, strFile, strTitle = None ):
	c_iN	= 20

	pylab.figure( )
	for iValues, adValues in enumerate( aadValues ):
		iN = min( c_iN, int(( len( adValues ) / 5.0 ) + 0.5) )
		iBins, adBins, pPatches = pylab.hist( adValues, iN, normed = 1, histtype = "stepfilled" )
		pylab.setp( pPatches, alpha = 0.5 )
	if strTitle:
		pylab.title( strTitle )
	pylab.savefig( strFile )

def _halla_test( ostm, pData, hashClusters, dP, iBootstrap ):

	pData.bootstrap( iBootstrap )
	adTotal = pData.get_bootstrap( )
	dTotal = numpy.average( adTotal )
	aaClusters = sorted( hashClusters.items( ) )
	fFirst = True
	for iOne, aOne in enumerate( aaClusters ):
		iX = aOne[1][0]
		for aTwo in aaClusters[( iOne + 1 ):]:
			iY = aTwo[1][0]
			
			pTest = CTest( pData, iX, iY, adTotal )
			if fFirst:
				fFirst = False
				pTest.save_header( ostm )
			pTest.save( ostm )
			
			if c_logrHAllA.level <= logging.DEBUG:
				dMID, dPOne = pTest.test_permutation( )
				dU, dPTwo = pTest.test_bootstrap( )
 				if ( dPOne < dP ) and ( dPTwo < dP ):
 					pOne, pTwo = (pData.get( i ) for i in (iX, iY))
				 	_log_plot_histograms( [adTotal, pData.get_permutation( iX, iY ), pData.get_bootstrap( iX, iY )],
						"-".join( (s.split( "|" )[-1] for s in (pOne.m_strID, pTwo.m_strID)) ) + ".png",
						"%g (%g), p_perm=%g, p_boot=%g" % (dMID, pTest.get_mutual_information( ), dPOne, dPTwo) )

def halla( istm, ostm, dP, dPMI, iBootstrap ):

	pData = dataset.CDataset( datum.CDatum.mutual_information_distance )
	pData.open( istm )
	hashClusters = pData.hierarchy( dPMI )
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
