from numpy import array 
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.cluster.hierarchy 
import halla

x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])
dx = halla.stats.discretize( x, iN = None, method = None, aiSkip = [1,3] )

lxmi = halla.hierarchy.hclust( dx, pdist_metric = halla.distance.norm_mid )

dendrogram(lxmi)