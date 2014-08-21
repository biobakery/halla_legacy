"""
Central namespace for plotting capacities in HAllA, 
including all graphics and 'data object to plot' transformations.
"""

#from pylab import plot, hist, scatter



#import dot_parser
import pydot
#from pylab import *
def _main( ):
    plot_box()
def plot_box(data, alpha= .1 , figure_name='HAllA_Power_TypeI_Error', ylabel = None, labels = None ):
    
    import pylab as pl
    # multiple box plots on one figure
    
    pl.figure("HAllA vs. Other methods")
    ax = pl.axes()
    pl.hold(True)
    if len(labels) > 0:
        ax.set_xticklabels(labels)
    pl.xlabel('Methods')
    pl.ylabel(ylabel)
    pl.xlim([-0.05, 1.05])
    pl.ylim([-0.05, 1.05])
    pl.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
    #pl.hlines(1-alpha,0.0,2.5, color ='blue')
    if ylabel == 'type_I_error':
        pl.plot([.0, len(data)+.5], [alpha, alpha], 'k-', lw=1, color ='red')
    #hB, = pl.plot([1,1],'b-')
    hR, = pl.plot([1,1],'r-')
    pl.legend((hR,),('q cut-off',))
    #pl.legend((hB, hR),('???', '???'))
    #hB.set_visible(False)
    hR.set_visible(False)
    #savefig('box7')
    pl.savefig('/Users/rah/Documents/'+figure_name+'.pdf')
    pl.show()
    return;

def plot_roc(roc_info=None, figure_name = 'roc_plot_HAllA'):
    """
    =======================================
    Receiver Operating Characteristic (ROC)
    =======================================

    Parameters
    ------------
        roc_info : List of lists (method name, fpr, tpr) 

    Returns 
    -------------
        Plots ROC curves
        save as pdf and show  
    """

    print(__doc__)
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import svm, datasets
    from sklearn.metrics import roc_curve, auc
    from sklearn.cross_validation import train_test_split
    from sklearn.preprocessing import label_binarize
    from sklearn.multiclass import OneVsRestClassifier
    '''roc_info = [  
        ['HAllA',[.005, .1,.15,.2, .21, .22, .3, .35, .4,.41, .42,97], [.005,.35,.6,.65, .8, .85, .88, .89, .90,.93, .97, .999] ],
        ['AllA', [.005, .1,.15,.2, .21, .22, .3, .35, .4,.41, .42,97], [.005,.33,.5,.6, .7, .75, .8, .85, .88,.9, .93, .95] ]
        ]'''
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_name = ''
    roc_auc = dict()
    for i in range(len(roc_info)):
        #print ('Hi', (roc_info[i][1]))
        fpr[roc_info[i][0]] = roc_info[i][1]
        #print ((roc_info[i][1])[0])
        tpr[roc_info[i][0]] = roc_info[i][2]
        roc_auc[roc_info[i][0]] = auc(fpr[roc_info[i][0]], tpr[roc_info[i][0]])
        roc_name += '_'+roc_info[i][0] 
        
    # Plot ROC curve
    plt.figure()
    for i in range(len(roc_info)):
        plt.plot(fpr[roc_info[i][0]], tpr[roc_info[i][0]], label='ROC curve of class {0} (area = {1:0.2f})'
                                       ''.format(str(roc_info[i][0]), roc_auc[roc_info[i][0]]))   
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    #plt.savefig('./test/'+roc_name+'foo.pdf')
    plt.savefig(figure_name+'.pdf')
    plt.show()
    #return plt
    
def dendrogramHeatPlot(D):
    import scipy
    import pylab
    #import dot_parser
    import scipy.cluster.hierarchy as sch
    import pydot
    from numpy.matlib import rand
    from array import array
    #Adopted from Ref: http://stackoverflow.com/questions/2982929/plotting-results-of-hierarchical-clustering-ontop-of-a-matrix-of-data-in-python
    if len(D) == 0: 
        # Generate random features and distance matrix.
        print "The distance matrix is empty. The function generates a random matrix."
        x = scipy.rand(40)
        D = scipy.zeros([40,40])
        for i in range(40):
            for j in range(i,40):
                D[i,j] = abs(x[i] - x[j])
                D[j,i]=D[i,j]
    # Compute and plot first dendrogram.
    fig = pylab.figure(figsize=(8,8))
    ax1 = fig.add_axes([0.09,0.1,0.2,0.6])
    Y = sch.linkage(D, method='single')
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
def graphPlot():
    '''graph = pydot.Dot('graphname', graph_type='digraph') 
    subg = pydot.Subgraph('', rank='same') 
    subg.add_node(pydot.Node('a')) 
    graph.add_subgraph(subg) 
    subg.add_node(pydot.Node('b')) 
    subg.add_node(pydot.Node('c'))
    
    graph.write_png('example2_graph.png')
    '''
    # first you create a new graph, you do that with pydot.Dot()
    graph = pydot.Dot(graph_type='graph')
    
    for i in range(3):
        
        edge = pydot.Edge("root", "parent%d" % i)
        # and we obviosuly need to add the edge to our graph
        graph.add_edge(edge)
    
    # now let us add some vassals
    child_num = 0
    for i in range(3):
        for j in range(2):
            edge = pydot.Edge("parent%d" % i, "child%d" % child_num)
            graph.add_edge(edge)
            child_num += 1
    
    # ok, we are set, let's save our graph into a file
    graph.write_png('example1_graph.png')
    
    # and we are done!
if __name__ == '__main__':
    _main( )