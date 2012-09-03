#!/usr/bin/env python
#############################################################################
# This file is provided under the Creative Commons Attribution 3.0 license.
#
# You are free to share, copy, distribute, transmit, or adapt this work
# PROVIDED THAT you attribute the work to the authors listed below.
# For more information, please see the following web page:
# http://creativecommons.org/licenses/by/3.0/
#
# This file is a component of HAllA, a Hierarchical All-against-All
# association testing method, authored by the Huttenhower lab at the Harvard
# School of Public Health (contact Curtis Huttenhower,
# chuttenh@hsph.harvard.edu).
#
# If you use this method or its code in your work, please cite the associated
# publication:
# ***
#############################################################################

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

#import time

import datum

c_logrHAllA	= logging.getLogger( "halla" )

class CDataset:

	def __init__( self, funcDist ):
		
		self.m_funcDist = funcDist
		self._dirty( )

	def _dirty( self ):
		
		self.m_aadBootstraps, self.m_aadPermutations, self.m_adY, self.m_aadZ = ([] for i in xrange( 4 ))
		self.m_hashData = {}

	def _pdist( self, apData, hashIndices = None ):

		aCache = [pDatum.cache( hashIndices ) for pDatum in apData]
		adRet = []
		for iOne, pOne in enumerate( apData ):
			for iTwo in xrange( iOne + 1, len( apData ) ):
				pTwo = apData[iTwo]
				adRet.append( self.m_funcDist( pOne, pTwo, hashIndices, aCache[iOne], aCache[iTwo] ) )
				
		return adRet
	
	def open( self, istm ):

		self._dirty( )
		self.m_astrCols = None
		self.m_apData = []
		for astrLine in csv.reader( istm, csv.excel_tab ):
			strID, astrData = astrLine[0], astrLine[1:]
			if self.m_astrCols:
				self.m_hashData[strID] = len( self.m_apData )
				self.m_apData.append( datum.CDatum( astrData, strID ) )
			else:
				self.m_astrCols = astrData

	def permute( self, iIterations ):

		c_logrHAllA.info( "Generating %s permutations" % iIterations )
		apPermute = copy.deepcopy( self.m_apData )
#		iTOne = 0
		for iIteration in xrange( iIterations ):
			c_logrHAllA.info( "Permutation %s/%s" % (iIteration, iIterations) )
			for pDatum in apPermute:
				pDatum.permute( )
#			iT = time.clock( )
			self.m_aadPermutations.append( self._pdist( apPermute ) )
#			iTOne += time.clock( ) - iT
#			sys.stderr.write( "%s\n" % [iIteration, iTOne] )

	def bootstrap( self, iIterations ):

		c_logrHAllA.info( "Generating %s bootstraps" % iIterations )
#		iTOne = 0
		for iIteration in xrange( iIterations ):
			c_logrHAllA.info( "Bootstrap %s/%s" % (iIteration, iIterations) )
			aiBootstrap = [random.sample( xrange( len( self.m_astrCols ) ), 1 )[0] for i in xrange( len( self.m_apData ) )]
			hashBootstrap = {}
			for i in aiBootstrap:
				hashBootstrap[i] = 1 + hashBootstrap.get( i, 0 )
#			iT = time.clock( )
			self.m_aadBootstraps.append( self._pdist( self.m_apData, hashBootstrap ) )
#			iTOne += time.clock( ) - iT
#			sys.stderr.write( "%s\n" % [iIteration, iTOne] )

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
		# Closed form solution of ( iB - iA - 1 ) + sum( iN - i - 1, i = 0..iB )
		# Counts up the first (iN-1) + (iN-2) + ... + (iN-iA-1) chunks, then adds the
		# offset of iB's difference from iA within the "triangle"
		iIndex = ( iA * ( iN - 1 ) ) - ( iA * ( iA - 1 ) / 2 ) + ( iB - iA - 1 )
		return aValues[iIndex]

# Should really fix this so it caches these correctly; right now it
# generates more at every request, rather than ensuring at least the
# requested number exist

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
		
		# Perform enough permutations to safely exceed the requested p-value
		iPermutations = int(2 / dP) + 1
		i = iPermutations - len( self.m_aadPermutations )
		if i > 0:
			self.permute( i )
		return self.m_aadPermutations

	def hierarchy( self, dP ):

		if not self.m_adY:
			self.m_adY = self._pdist( self.m_apData )
		if not self.m_aadZ:
			self.m_aadZ = scipy.cluster.hierarchy.linkage( self.m_adY, method = "complete" )

		# Initially cluster at a fairly strict threshold to group "indistinguishable" features		
		dT = scipy.stats.scoreatpercentile( [d for a in self._permutations( dP ) for d in a], dP * 100 )
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
