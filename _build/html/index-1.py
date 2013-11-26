import pylab 
import halla 
from numpy import array 
import scipy.stats
from scipy.cluster.hierarchy import dendrogram, linkage

x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])

#Pearson
lxpearson       = linkage(x, metric="correlation")

dendrogram(lxpearson)