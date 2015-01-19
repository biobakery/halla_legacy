#!/usr/bin/env python

import logging
import sys

import pylab


c_logrHAllA	= logging.getLogger( "halla" )

def _log_plot_histograms( aadValues, pFigure, strTitle = None, astrLegend = None ):
	c_iN	= 20

	figr = pylab.figure( ) if ( ( not pFigure ) or isinstance( pFigure, str ) ) else pFigure
	for iValues, adValues in enumerate( aadValues ):
		iN = min( c_iN, int(( len( adValues ) / 5.0 ) + 0.5) )
		iBins, adBins, pPatches = pylab.hist( adValues, iN, normed = 1, histtype = "stepfilled" )
		pylab.setp( pPatches, alpha = 0.5 )
	if strTitle:
		pylab.title( strTitle )
	if astrLegend:
		pylab.legend( astrLegend, loc = "upper left" )
	if isinstance( pFigure, str ):
		pylab.savefig( pFigure )

def _log_plot_scatter( adX, adY, pFigure, strTitle = None, strX = None, strY = None ):

	figr = pylab.figure( ) if ( ( not pFigure ) or isinstance( pFigure, str ) ) else pFigure
	pylab.scatter( adX, adY )
	if strTitle:
		pylab.title( strTitle )
	if strX:
		pylab.xlabel( strX )
	if strY:
		pylab.ylabel( strY )
	if isinstance( pFigure, str ):
		pylab.savefig( pFigure )

def _log_plot( pData, pTest, adTotal ):
	
	# Convenience
	iOne, iTwo = pTest.m_iOne, pTest.m_iTwo
	pOne, pTwo = pTest.m_pOne, pTest.m_pTwo
	dMID, dPOne = pTest.test_permutation( )
	dU, dPTwo = pTest.test_bootstrap( )

	figr = pylab.figure( figsize = (12, 6) )
	pylab.subplot( 1, 2, 1 )
	# Fix me to plot categoricals as boxplots etc.
	if all( p.iscontinuous( ) for p in (pOne, pTwo) ):
		_log_plot_scatter( pOne.m_aDatum, pTwo.m_aDatum, figr, None, pOne.m_strID, pTwo.m_strID )
	pylab.subplot( 1, 2, 2 )
 	_log_plot_histograms( [adTotal, pData.get_permutation( iOne, iTwo ), pData.get_bootstrap( iOne, iTwo )],
		figr, "%g (%g), p_perm=%g, p_boot=%g" % (dMID, pTest.get_mutual_information( ), dPOne, dPTwo),
		["Total", "Perm", "Boot"] )
	figr.savefig( "-".join( (s.split( "|" )[-1] for s in (pOne.m_strID, pTwo.m_strID)) ) + ".png" )
