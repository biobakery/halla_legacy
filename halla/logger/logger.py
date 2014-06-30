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

	from hlog import *
"""

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
