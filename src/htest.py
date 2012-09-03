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

	from htest import *
"""

import logging
import numpy
import scipy.stats
import sys

c_logrHAllA	= logging.getLogger( "halla" )

def _ztest( adX, adY ):
	"""
	>>> def s( a ):
	... 	return [( "%g" % d ) for d in a]
	>>> s( _ztest( [0, 1, 2], [0, 1, 2] ) )
	['0', '0.5']

	>>> s( _ztest( [0, 1, 2], [0, 1, 0] ) )
	['-1.03528', '0.15027']

	>>> s( _ztest( [4, 5, 6, 4, 5, 6], [0, 1, 0, 1, 0, 1, 0, 1] ) )
	['-7.12168', '5.33105e-13']

	>>> s( _ztest( [0, 1, 0], [4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6] ) )
	['6.11296', '1']
	"""
	
	dMX, dMY = (numpy.mean( a ) for a in (adX, adY))
	dSX, dSY = (numpy.std( a ) for a in (adX, adY))
	dS = ( ( ( len( adX ) - 1 ) * dSX ) + ( ( len( adY ) - 1 ) * dSY ) ) / ( len( adX ) + len( adY ) - 2 )
	dZ = ( ( dMY - dMX ) / dS ) if dS else 0
	dP = scipy.stats.norm.cdf( dZ )
	
	return (dZ, dP)

class CHTest:
	
	def __init__( self, pData, iOne, iTwo, adTotal = None, dTotal = None ):
		
		self.m_pData = pData
		self.m_iOne, self.m_iTwo = iOne, iTwo
		self.m_pOne, self.m_pTwo = (pData.get( i ) for i in (self.m_iOne, self.m_iTwo))
		self.m_adTotal = adTotal if adTotal else self.m_pData.get_bootstrap( )
		self.m_dTotal = numpy.average( self.m_adTotal ) if ( dTotal == None ) else dTotal
		self.m_dMI = self.m_dMID = self.m_dPPerm = self.m_dPBoot = None

	def test_permutation( self ):
		
		if self.m_dMID == None:
			self.m_dMID = self.m_pOne.mutual_information_distance( self.m_pTwo )
		if self.m_dPPerm == None:
			# Z-test doesn't work here since MI isn't length-invariant
			# Guarantee at least one better score to adjust permutation p-value
			self.m_dPPerm = scipy.stats.percentileofscore( [self.m_dMID - 1] +
				self.m_pData.get_permutation( self.m_iOne, self.m_iTwo ), self.m_dMID ) / 100

		return (self.m_dMID, self.m_dPPerm)

	def test_bootstrap( self ):

		if self.m_dPBoot == None:
			adBootstrap = self.m_pData.get_bootstrap( self.m_iOne, self.m_iTwo )
			# Z-test doesn't work here for non-huge sample sizes
			dU, self.m_dPBoot = scipy.stats.ttest_ind( self.m_adTotal, adBootstrap )
			self.m_dPBoot /= 2
			if numpy.average( adBootstrap ) > self.m_dTotal:
				self.m_dPBoot = 1 - self.m_dPBoot
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
