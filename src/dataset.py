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
import scipy.cluster.hierarchy
import scipy.stats

import datum

c_logrHAllA	= logging.getLogger( "halla" )

class CDataset:

	def __init__( self, funcDist ):
		
		self.m_funcDist = funcDist
		self._dirty( )

	def _dirty( self ):
		
		self.m_aadPermutations, self.m_adY, self.m_aadZ = ([] for i in xrange( 3 ))

	def _pdist( self, apData ):
	
		adRet = []
		for iOne, pOne in enumerate( apData ):
			for pTwo in apData[( iOne + 1 ):]:
				adRet.append( self.m_funcDist( pOne, pTwo ) )
				
		return adRet
	
	def open( self, istm ):

		self._dirty( )
		self.m_astrCols = None
		self.m_apData = []
		for astrLine in csv.reader( istm, csv.excel_tab ):
			strID, astrData = astrLine[0], astrLine[1:]
			if self.m_astrCols:
				self.m_apData.append( datum.CDatum( astrData, strID ) )
			else:
				self.m_astrCols = astrData

	def permute( self, iPermutations ):

		c_logrHAllA.info( "Generating %s permutations" % iPermutations )
		apPermute = copy.deepcopy( self.m_apData )
		adMins, adMIs = ([] for i in xrange( 2 ))
		for iPermutation in xrange( iPermutations ):
			c_logrHAllA.info( "Permutation %s/%s" % (iPermutation, iPermutations) )
			for pDatum in apPermute:
				pDatum.permute( )
			self.m_aadPermutations.append( self._pdist( apPermute ) )

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
