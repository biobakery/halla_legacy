"""
Wrappers for testing procedures, random data generation, etc 
"""

from itertools import compress

import matplotlib
from numpy import array
import numpy
from numpy.random import normal, multinomial
import scipy
from scipy.stats import percentileofscore

import halla.distance
import halla.stats


#================================================================================
# Base cases 
#================================================================================
def randmat( tShape = (10,10), pDist = numpy.random.normal ):
	"""
	Returns a tShape-dimensional matrix given by base distribution pDist 
	Order: Row, Col 
	"""	
	H = pDist #base measure 
	
	iRow, iCol = tShape

	assert( iRow != 0 and iCol !=0 ) 
	
	return array( [[H() for i in range(iCol)] for j in range(iRow)] )

def randmix( N, pDist, atParam, tPi ):
	"""
	Returns N copies drawn from a mixture distribution $H$ 
	Input: N <- number of components
		pDist <- pointer to base distribution H 
		atParam <- length $k$ parameters to distribution pDist, $\theta$  
		tPi <- length $k$ tuple (vector) to categorical rv Z_n 

	Output: N copies from mixture distribution $\sum_{k=1}^{K} \pi_k H(.| \theta )$ 
	""" 
	
	assert( len( atParam ) == len( tPi ) )
	
	aOut = [] 

	K = len( atParam ) 
	H = pDist 
	for n in range(N):
		# multinomial returns boolean vector 
		aParam = [x for x in compress( atParam, multinomial( 1, tPi ) )][0]
		aOut.append( H( *aParam ) )
	return aOut

#================================================================================
# Special cases 
#================================================================================

def uniformly_spaced_gaussian( N, K = 4, fD = 2.0, tPi = (0.25,0.25,0.25,0.25) ):
	"""
	Generate uniformly spaced Gaussian, with spacing fD in the mean.
	Constant 1.0 variance 

	"""
	return randmix( N, pDist = normal, atParam = [(m,1.0) for m in [fD*i for i in range(K)]], tPi = tPi )


#================================================================================
# Pipelines  
#================================================================================

def nov_27_2013():
	x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])
	y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0],
		[0.015625,0.125,0.421875,1.0]])
	dx = halla.stats.discretize( x, iN = None, method = None, aiSkip = [1,3] )
	dy = halla.stats.discretize( y, iN = None, method = None, aiSkip = [1] )
	from scipy.stats import spearmanr
	spearman = lambda x,y: spearmanr(x,y)[0]

	#Pearson
	lxpearson	= halla.hierarchy.linkage(x, metric="correlation")
	lypearson	= halla.hierarchy.linkage(y, metric="correlation")

	#Spearman 
	lxspearman	= halla.hierarchy.linkage(x, metric= spearman)
	lyspearman	= halla.hierarchy.linkage(y, metric= spearman)

	#MI
	lxmi		= halla.hierarchy.linkage(x, metric= halla.distance.mi)
	lymi		= halla.hierarchy.linkage(y, metric= halla.distance.mi)

	aL			= [(lxpearson,lypearson),(lxspearman,lyspearman), (lxmi, lymi)]
	aNames		= ["pearson", "spearman", "mi"]


	#### Plotting 
	from scipy.cluster.hierarchy import dendrogram
	for k, item in enumerate(aL):
		i,j = item
		i,j = numpy.abs(i), numpy.abs(j)
		#matplotlib.pyplot.figure(aNames[k])
		#matplotlib.pyplot.subplot(1,2,1)
		dendrogram(i)
		#matplotlib.pyplot.subplot(1,2,2)
		dendrogram(j)







"""
		.. plot::

		x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])
		y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0],[0.015625,0.125,0.421875,1.0]])
		dx = halla.stats.discretize( x, iN = None, method = None, aiSkip = [1,3] )
		dy = halla.stats.discretize( y, iN = None, method = None, aiSkip = [1] )

		spearman = lambda x,y: scipy.stats.spearmanr(x,y)[0]

		#Pearson
		lxpearson	= linkage(x, metric="correlation")
		lypearson	= linkage(y, metric="correlation")

		#Spearman 
		lxspearman	= linkage(x, metric= spearman)
		lyspearman	= linkage(y, metric= spearman)

		#MI
		lxmi		= linkage(dx, metric= mi)
		lymi		= linkage(dy, metric= mi)

		aL			= [(lxpearson,lypearson),(lxspearman,lyspearman), (lxmi, lymi)]
		aNames		= ["pearson", "spearman", "mi"]


		#### Plotting 

		for k, item in enumerate(aL):
			i,j = item
			i,j = numpy.abs(i), numpy.abs(j)
			matplotlib.pyplot.figure(aNames[k])
			matplotlib.pyplot.subplot(1,2,1)
			dendrogram(i)
			matplotlib.pyplot.subplot(1,2,2)
			dendrogram(j)
"""
