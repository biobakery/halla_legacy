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

	from dataset import *
"""

import copy
import csv
import logging
import random
import scipy.cluster.hierarchy
import scipy.stats
import sys

import datum

c_logrHAllA	= logging.getLogger( "halla" )

class CDataset:

	def __init__( self, funcDist ):
		
		self.m_funcDist = funcDist
		self._dirty( )

	def _dirty( self ):
		
		self.m_aadBootstraps, self.m_aadPermutations, self.m_adY, self.m_aadZ = ([] for i in xrange( 4 ))
		self.m_hashData = {}

	def _pdist( self, apData, aiIndices = None ):
	
		adRet = []
		for iOne, pOne in enumerate( apData ):
			for iTwo in xrange( iOne + 1, len( apData ) ):
				pTwo = apData[iTwo]
#				sys.stderr.write( "%s\n" % [iOne, iTwo, aiIndices, self.m_funcDist( pOne, pTwo, aiIndices )] )
				adRet.append( self.m_funcDist( pOne, pTwo, aiIndices ) )
				
		return adRet
	
	def open( self, istm ):

		self._dirty( )
		self.m_astrCols = None
		self.m_apData = []
		for astrLine in csv.reader( istm, csv.excel_tab ):
			strID, astrData = astrLine[0], astrLine[1:]
			if self.m_astrCols:
				self.m_hashData[len( self.m_apData )] = pDatum = datum.CDatum( astrData, strID )
				self.m_apData.append( pDatum )
			else:
				self.m_astrCols = astrData

	def permute( self, iIterations ):

		c_logrHAllA.info( "Generating %s permutations" % iIterations )
		apPermute = copy.deepcopy( self.m_apData )
		for iIteration in xrange( iIterations ):
			c_logrHAllA.info( "Permutation %s/%s" % (iIteration, iIterations) )
			for pDatum in apPermute:
				pDatum.permute( )
			self.m_aadPermutations.append( self._pdist( apPermute ) )

	def bootstrap( self, iIterations ):

		c_logrHAllA.info( "Generating %s bootstraps" % iIterations )
		for iIteration in xrange( iIterations ):
			c_logrHAllA.info( "Bootstrap %s/%s" % (iIteration, iIterations) )
			aiBootstrap = [random.sample( xrange( len( self.m_astrCols ) ), 1 )[0] for i in xrange( len( self.m_apData ) )]
			self.m_aadBootstraps.append( self._pdist( self.m_apData, aiBootstrap ) )

	@staticmethod
	def _get_pdist( iN, aValues, iX, iY ):
		"""
		>>> CDataset._get_pdist( 3, [0, 1, 2], 0, 1 )
		0

		>>> CDataset._get_pdist( 3, [0, 1, 2], 0, 2 )
		1

		>>> CDataset._get_pdist( 3, [0, 1, 2], 2, 1 )
		2

		>>> CDataset._get_pdist( 4, [0, 1, 2, 3, 4, 5], 0, 3 )
		2

		>>> CDataset._get_pdist( 4, [0, 1, 2, 3, 4, 5], 2, 1 )
		3

		>>> CDataset._get_pdist( 4, [0, 1, 2, 3, 4, 5], 3, 1 )
		4

		>>> CDataset._get_pdist( 4, [0, 1, 2, 3, 4, 5], 3, 2 )
		5
		"""

		iA, iB = (min( iX, iY ), max( iX, iY ))
		iIndex = ( iA * ( iN - 1 ) ) - ( iA * ( iA - 1 ) / 2 ) + ( iB - iA - 1 )
		return aValues[iIndex]

	def _get_result( self, aadResults, iX, iY ):
		
		if iX == None:
			if iY == None:
				return [d for a in aadResults for d in a]
			iX, iY = iY, iX
		elif iY != None:
			if iY < iX:
				iX, iY = iY, iX
		
		adRet = []
		for adResult in aadResults:
			for iZ in ( xrange( len( self.m_apData ) ) if ( iY == None ) else (iY,) ):
				if iX != iZ:
					adRet.append( CDataset._get_pdist( len( self.m_apData ), adResult, iX, iZ ) )
				
		return adRet

	def get_bootstrap( self, iX = None, iY = None ):
		
		return self._get_result( self.m_aadBootstraps, iX, iY )

	def get_permutation( self, iX = None, iY = None ):
		
		return self._get_result( self.m_aadPermutations, iX, iY )

	def _permutations( self, dP ):
		
		iPermutations = int(1 / dP) + 1
		i = iPermutations - len( self.m_aadPermutations )
		if i > 0:
			self.permute( i )
		return self.m_aadPermutations

	def hierarchy( self, dP ):

		if not self.m_adY:
			self.m_adY = self._pdist( self.m_apData )
		if not self.m_aadZ:
			self.m_aadZ = scipy.cluster.hierarchy.linkage( self.m_adY, method = "complete" )
		
		adMins = [min( a ) for a in self._permutations( dP )]
		dT = scipy.stats.scoreatpercentile( adMins, dP * 100 )
		aiClusters = scipy.cluster.hierarchy.fcluster( self.m_aadZ, dT )
		hashRet = {}
		for iIndex, iCluster in enumerate( aiClusters ):
			hashRet.setdefault( iCluster, [] ).append( iIndex )
	
		if c_logrHAllA.level <= logging.INFO:
			for iCluster, aiCluster in hashRet.items( ):
				if len( aiCluster ) > 1:
					c_logrHAllA.info( "Cluster %s (%d): %s" % (iCluster, len( aiCluster ), [self.m_apData[i].m_strID for i in aiCluster]) )

		return hashRet

	def get( self, iDatum ):
		
		return self.m_apData[iDatum]

	def get_index( self, strDatum ):
		
		return self.m_hashData.get( strDatum )
