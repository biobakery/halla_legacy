from numpy import array 
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.cluster.hierarchy 
import halla

y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0], [0.015625,0.125,0.421875,1.0]])

lypearson = halla.hierarchy.hclust( y, pdist_metric = halla.distance.cord )

dendrogram(lypearson)