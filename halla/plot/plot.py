"""
Central namespace for plotting capacities in HAllA, 
including all graphics and 'data object to plot' transformations.
"""

#from pylab import plot, hist, scatter


import scipy
import pylab
import scipy.cluster.hierarchy as sch
#import pydot



class Plot:
    
    #Adopted from Ref: http://stackoverflow.com/questions/2982929/plotting-results-of-hierarchical-clustering-ontop-of-a-matrix-of-data-in-python
    
    @staticmethod
    def dendrogramHeatPlot(D):
        if len(D) == 0: 
            # Generate random features and distance matrix.
            print "The distance matrix is empty. The function generates a random matrix."
            x = scipy.rand(40)
            D = scipy.zeros([40,40])
            for i in range(40):
                for j in range(40):
                    D[i,j] = abs(x[i] - x[j])
        
        # Compute and plot first dendrogram.
        fig = pylab.figure(figsize=(8,8))
        ax1 = fig.add_axes([0.09,0.1,0.2,0.6])
        Y = sch.linkage(D, method='centroid')
        Z1 = sch.dendrogram(Y, orientation='right')
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # Compute and plot second dendrogram.
        ax2 = fig.add_axes([0.3,0.71,0.6,0.2])
        Y = sch.linkage(D, method='single')
        Z2 = sch.dendrogram(Y)
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        # Plot distance matrix.
        axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
        idx1 = Z1['leaves']
        idx2 = Z2['leaves']
        D = D[idx1,:]
        D = D[:,idx2]
        im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=pylab.cm.RdYlGn)
        axmatrix.set_xticks([])
        axmatrix.set_yticks([])
        
        # Plot colorbar.
        axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
        pylab.colorbar(im, cax=axcolor)
        fig.show()
        fig.savefig('dendrogram.png')
        '''
        axmatrix.set_xticks(range(40))
        axmatrix.set_xticklabels(idx1, minor=False)
        axmatrix.xaxis.set_label_position('bottom')
        axmatrix.xaxis.tick_bottom()
        
        pylab.xticks(rotation=-90, fontsize=4)
        
        axmatrix.set_yticks(range(40))
        axmatrix.set_yticklabels(idx2, minor=False)
        axmatrix.yaxis.set_label_position('right')
        axmatrix.yaxis.tick_right()
        
        #(0.5,0,0.5,1) adds an Axes on the right half of the figure. (0,0.5,1,0.5) adds an Axes on the top half of the figure.
        #Most people probably use add_subplot for its convenience. I like add_axes for its control.
        #To remove the border, use add_axes([left,bottom,width,height], frame_on=False)
        
        axcolor = fig.add_axes([0.94,0.1,0.02,0.6])
        
        '''
    '''@staticmethod
    def graphPlot():
        graph = pydot.Dot('graphname', graph_type='digraph') 
        subg = pydot.Subgraph('', rank='same') 
        subg.add_node(pydot.Node('a')) 
        graph.add_subgraph(subg) 
        subg.add_node(pydot.Node('b')) 
        subg.add_node(pydot.Node('c'))
        '''