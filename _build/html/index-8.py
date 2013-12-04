from numpy import array 
from scipy.cluster.hierarchy import dendrogram 
import halla

y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0], [0.015625,0.125,0.421875,1.0]])
dy = halla.stats.discretize( y, iN = None, method = None, aiSkip = [1] )                

lymi = halla.hierarchy.hclust( dy, pdist_metric = halla.distance.norm_mid )

dendrogram( lymi )      