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

	from datum import *
"""

import copy
import logging
import math
import random
import sys

c_logrHAllA	= logging.getLogger( "halla" )

class CDatum:
	class EType:
		CATEGORICAL	= 0,
		CONTINUOUS	= 1,
	
	def __init__( self, astrDatum, strID = None ):
		
		self.m_strID = strID
		try:
			# Needs to be adapted to handle small-count integer values
			# e.g. [0, 1, 1, 0] probably shouldn't be continuous
			adDatum = [float(s) for s in astrDatum]
			self.m_eType = CDatum.EType.CONTINUOUS
			self.m_aDatum = adDatum
		except ValueError:
			self.m_eType = CDatum.EType.CATEGORICAL
			self.m_aDatum = astrDatum
		self._discretize( )
		
	def __repr__( self ):
		
		return ( "CDatum(%s, %s)" % (self.m_aDatum, self.m_strID) )
	
	def __copy__( self ):
		
		return CDatum( self.m_aDatum, self.m_strID )
	
	def __deepcopy__( self, hashMemo ):
		
		return CDatum( copy.deepcopy( self.m_aDatum, hashMemo ), self.m_strID )

	@staticmethod
	def _discretize_continuous( astrValues, iN = None ):
		"""
		>>> CDatum._discretize_continuous( [0] )
		[0]

		>>> CDatum._discretize_continuous( [0, 1] )
		[0, 0]

		>>> CDatum._discretize_continuous( [0, 1], 2 )
		[0, 1]

		>>> CDatum._discretize_continuous( [1, 0], 2 )
		[1, 0]

		>>> CDatum._discretize_continuous( [0.2, 0.1, 0.3], 3 )
		[1, 0, 2]

		>>> CDatum._discretize_continuous( [0.2, 0.1, 0.3], 1 )
		[0, 0, 0]

		>>> CDatum._discretize_continuous( [0.2, 0.1, 0.3], 2 )
		[0, 0, 1]

		>>> CDatum._discretize_continuous( [0.4, 0.2, 0.1, 0.3], 2 )
		[1, 0, 0, 1]

		>>> CDatum._discretize_continuous( [4, 0.2, 0.1, 0.3], 2 )
		[1, 0, 0, 1]

		>>> CDatum._discretize_continuous( [0.4, 0.2, 0.1, 0.3, 0.5] )
		[1, 0, 0, 0, 1]

		>>> CDatum._discretize_continuous( [0.4, 0.2, 0.1, 0.3, 0.5], 3 )
		[1, 0, 0, 1, 2]

		>>> CDatum._discretize_continuous( [0.4, 0.2, 0.6, 0.1, 0.3, 0.5] )
		[1, 0, 1, 0, 0, 1]

		>>> CDatum._discretize_continuous( [0.4, 0.2, 0.6, 0.1, 0.3, 0.5], 3 )
		[1, 0, 2, 0, 1, 2]

		>>> CDatum._discretize_continuous( [0.4, 0.2, 0.6, 0.1, 0.3, 0.5], 0 )
		[3, 1, 5, 0, 2, 4]

		>>> CDatum._discretize_continuous( [0.4, 0.2, 0.6, 0.1, 0.3, 0.5], 6 )
		[3, 1, 5, 0, 2, 4]

		>>> CDatum._discretize_continuous( [0.4, 0.2, 0.6, 0.1, 0.3, 0.5], 60 )
		[3, 1, 5, 0, 2, 4]

		>>> CDatum._discretize_continuous( [0, 0, 0, 0, 0, 0, 1, 2], 2 )
		[0, 0, 0, 0, 0, 0, 1, 1]

		>>> CDatum._discretize_continuous( [0, 0, 0, 0, 1, 2, 2, 2, 2, 3], 3 )
		[0, 0, 0, 0, 1, 1, 1, 1, 1, 2]

		>>> CDatum._discretize_continuous( [0.1, 0, 0, 0, 0, 0, 0, 0, 0] )
		[1, 0, 0, 0, 0, 0, 0, 0, 0]
		
		>>> CDatum._discretize_continuous( [0.992299, 1, 1, 0.999696, 0.999605, 0.663081, 0.978293, 0.987621, 0.997237, 0.999915, 0.984792, 0.998338, 0.999207, 0.98051, 0.997984, 0.999219, 0.579824, 0.998983, 0.720498, 1, 0.803619, 0.970992, 1, 0.952881, 0.999866, 0.997153, 0.014053, 0.998049, 0.977727, 0.971233, 0.995309, 0.0010376, 1, 0.989373, 0.989161, 0.91637, 1, 0.99977, 0.960816, 0.998025, 1, 0.998852, 0.960849, 0.957963, 0.998733, 0.999426, 0.876182, 0.998509, 0.988527, 0.998265, 0.943673] )
		[3, 6, 6, 5, 5, 0, 2, 2, 3, 5, 2, 4, 4, 2, 3, 5, 0, 4, 0, 6, 0, 1, 6, 1, 5, 3, 0, 3, 2, 1, 3, 0, 6, 3, 2, 0, 6, 5, 1, 3, 6, 4, 1, 1, 4, 5, 0, 4, 2, 4, 1]
		"""

		if iN == None:
			# Default to rounded sqrt(n) if no bin count requested
			iN = int(len( astrValues )**0.5 + 0.5)
		elif iN == 0:
			iN = len( astrValues )
		else:
			iN = min( iN, len( set(astrValues) ) )
			
		# This is still a bit buggy since ( [0, 0, 0, 1, 2, 2, 2, 2], 3 ) will exhibit suboptimal behavior
		aiIndices = sorted( range( len( astrValues ) ), cmp = lambda i, j: cmp( astrValues[i], astrValues[j] ) )
		astrRet = [None] * len( astrValues )
		iPrev = 0
		for i, iIndex in enumerate( aiIndices ):
			# If you're on a tie, you can't increase the bin number
			# Otherwise, increase by at most one
			iPrev = astrRet[iIndex] = iPrev if ( i and ( astrValues[iIndex] == astrValues[aiIndices[i - 1]] ) ) else \
				min( iPrev + 1, int(iN * i / float(len( astrValues ))) )
		
		return astrRet

	@staticmethod
	def _discretize_helper( astrBins ):
		"""
		>>> def s( h ):
		... 	return ( [(a[0], sorted( a[1] )) for a in sorted( h.items( ) )] if h else None )

		>>> s( CDatum._discretize_helper( [0, 1] ) )
		[(0, [0]), (1, [1])]

		>>> s( CDatum._discretize_helper( [0, 1, 0, 1] ) )
		[(0, [0, 2]), (1, [1, 3])]

		>>> s( CDatum._discretize_helper( [0, 1, 0, 1, 1, 1] ) )
		[(0, [0, 2]), (1, [1, 3, 4, 5])]

		>>> s( CDatum._discretize_helper( ["A", "B", "C", "A"] ) )
		[('A', [0, 3]), ('B', [1]), ('C', [2])]
		"""

		hashRet = {}
		for iIndex, strValue in enumerate( astrBins ):
			hashRet.setdefault( strValue, set() ).add( iIndex )

		return hashRet

	def _discretize( self ):
		
		self.m_astrBins = self.m_aDatum if ( self.m_eType == CDatum.EType.CATEGORICAL ) else \
			CDatum._discretize_continuous( self.m_aDatum )
		self.m_hashValues = CDatum._discretize_helper( self.m_astrBins )
				
	def mutual_information( self, pDatum, aiIndices = None ):
		"""
		>>> pOne = CDatum( ["A", "A", "B", "B"] )
		>>> pTwo = CDatum( ["B", "B", "A", "A"] )
		>>> pOne.mutual_information( pTwo )
		1.0

		>>> pThree = CDatum( ["A", "B", "A", "B"] )
		>>> pOne.mutual_information( pThree )
		0.0

		>>> pFour = CDatum( ["A", "B", "A", "A"] )
		>>> "%g" % pOne.mutual_information( pFour )
		'0.311278'

		>>> pFive = CDatum( [0.1, 0.2, 0.3, 0.4] )
		>>> "%g" % pFive.mutual_information( pFour )
		'0.311278'
		
		>>> pSix = CDatum( ["A", "B", "C", "A", "B", "C"] )
		>>> pSeven = CDatum( ["A", "A", "B", "A", "A", "B"] )
		>>> "%g" % pSix.mutual_information( pSeven )
		'0.918296'
		
		>>> pEight = CDatum( ["A", "A", "B", "B", "C", "C"] )
		>>> "%g" % pSix.mutual_information( pEight )
		'0.584963'

		>>> import random
		>>> ad = []
		>>> for i in xrange( 100 ):
		... 	pA, pB = (CDatum( [random.random( ) for j in xrange( 100 )] ) for k in xrange( 2 ))
		... 	ad.append( pA.mutual_information( pB ) )
		>>> ( min( ad ) >= 0 ) and ( max( ad ) <= 3.4 ) # log( 10, 2 ) = 3.32 and should be the max
		True

		>>> pSix = CDatum( ["A", "B", "C", "A", "B", "C", "D", "D"] )
		>>> pSeven = CDatum( ["A", "A", "B", "A", "A", "B", "B", "B"] )
		>>> "%g" % pSix.mutual_information( pSeven, range( 6 ) )
		'0.918296'
		
		>>> pEight = CDatum( [8.47E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] )
		>>> pNine = CDatum( [0.992299, 1, 1, 0.999696, 0.999605, 0.663081, 0.978293, 0.987621, 0.997237, 0.999915, 0.984792, 0.998338, 0.999207, 0.98051, 0.997984, 0.999219, 0.579824, 0.998983, 0.720498, 1, 0.803619, 0.970992, 1, 0.952881, 0.999866, 0.997153, 0.014053, 0.998049, 0.977727, 0.971233, 0.995309, 0.0010376, 1, 0.989373, 0.989161, 0.91637, 1, 0.99977, 0.960816, 0.998025, 1, 0.998852, 0.960849, 0.957963, 0.998733, 0.999426, 0.876182, 0.998509, 0.988527, 0.998265, 0.943673] )
		>>> "%g" % pEight.mutual_information( pNine )
		'0.053968'
		
		>>> "%g" % pEight.mutual_information( pNine, range( len( pEight.m_aDatum ) / 2 ) )
		'0.132097'
		
		>>> "%g" % pEight.mutual_information( pNine, range( len( pEight.m_aDatum ) / 4 ) )
		'0.24715'
		
		>>> "%g" % pEight.mutual_information( pNine, range( len( pEight.m_aDatum ) / 8 ) )
		'0.650022'
		"""
		
		setiIndices = set(aiIndices if aiIndices else xrange( len( self.m_aDatum ) ))
		dRet = 0
		for strY, setiY in self.m_hashValues.items( ):
			dYProbability = float(len( setiY & setiIndices )) / len( setiIndices )
			for strX, setiX in pDatum.m_hashValues.items( ):
				setiTmp = setiX & setiIndices
				dXProbability = float(len( setiTmp )) / len( setiIndices )
				setiXY = setiTmp & setiY
				dXY = float(len( setiXY )) / len( setiIndices )
				if dXY:
					dRet += dXY * math.log( dXY / dXProbability / dYProbability, 2 )
		
		return dRet
	
	def mutual_information_relative( self, pDatum, aiIndices = None ):
		"""
		>>> pOne = CDatum( ["A", "A", "B", "B"] )
		>>> pTwo = CDatum( ["B", "B", "A", "A"] )
		>>> pOne.mutual_information_relative( pTwo )
		1.0

		>>> pThree = CDatum( ["A", "B", "A", "B"] )
		>>> pOne.mutual_information_relative( pThree )
		0.0

		>>> pSix = CDatum( ["A", "B", "C", "A", "B", "C"] )
		>>> pSeven = CDatum( ["A", "A", "B", "A", "A", "B"] )
		>>> "%g" % pSix.mutual_information_relative( pSeven )
		'0.918296'
		
		>>> pEight = CDatum( ["A", "A", "B", "B", "C", "C"] )
		>>> "%g" % pSix.mutual_information_relative( pEight )
		'0.36907'
		
		>>> import random
		>>> ad = []
		>>> for i in xrange( 100 ):
		... 	pA, pB = (CDatum( [random.random( ) for j in xrange( 100 )] ) for k in xrange( 2 ))
		... 	ad.append( pA.mutual_information_relative( pB ) )
		>>> ( min( ad ) >= 0 ) and ( max( ad ) <= 1 )
		True
		
		>>> pEight = CDatum( [8.47E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] )
		>>> pNine = CDatum( [0.992299, 1, 1, 0.999696, 0.999605, 0.663081, 0.978293, 0.987621, 0.997237, 0.999915, 0.984792, 0.998338, 0.999207, 0.98051, 0.997984, 0.999219, 0.579824, 0.998983, 0.720498, 1, 0.803619, 0.970992, 1, 0.952881, 0.999866, 0.997153, 0.014053, 0.998049, 0.977727, 0.971233, 0.995309, 0.0010376, 1, 0.989373, 0.989161, 0.91637, 1, 0.99977, 0.960816, 0.998025, 1, 0.998852, 0.960849, 0.957963, 0.998733, 0.999426, 0.876182, 0.998509, 0.988527, 0.998265, 0.943673] )
		>>> "%g" % pEight.mutual_information( pNine )
		'0.053968'
		"""

		dMI = self.mutual_information( pDatum, aiIndices )
		if aiIndices:
			setiIndices = set(aiIndices)
			aiValues = []
			for p in (self, pDatum):
				iValues = 0
				for strValue, setiValue in p.m_hashValues.items( ):
					if setiIndices & setiValue:
						iValues += 1
				aiValues.append( iValues )
			iMin = min( aiValues )
		else:
			iMin = min( len( p.m_hashValues ) for p in (self, pDatum) )
		dRet = ( ( dMI / math.log( iMin, 2 ) ) if ( iMin > 1 ) else 0 )
		return dRet
	
	def mutual_information_distance( self, pDatum, aiIndices = None ):
		
		return ( 1 - self.mutual_information_relative( pDatum, aiIndices ) )
	
	def permute( self ):
		"""
		>>> pOne = CDatum( [0, 1, 2] )
		>>> pOne.permute( )
		>>> import re
		>>> mtch = re.search( r'^CDatum\(\[(.+)\], None\)$', "%s" % pOne )
		>>> sorted( mtch.group( 1 ).split( ", " ) )
		['0.0', '1.0', '2.0']
		"""

		aiIndices = random.sample( xrange( len( self.m_aDatum ) ), len( self.m_aDatum ) )
		self.m_astrBins, self.m_aDatum = ([a[i] for i in aiIndices] for a in (self.m_astrBins, self.m_aDatum))
		self.m_hashValues = CDatum._discretize_helper( self.m_astrBins )
