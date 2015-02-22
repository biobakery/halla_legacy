"""
Central namespace for plotting capacities in HAllA, 
including all graphics and 'data object to plot' transformations.
"""

# from pylab import plot, hist, scatter



# import dot_parser
import pylab
import sys
import halla.distance
import scipy.cluster 
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import pdist
from matplotlib.pyplot import xlabel
# import pydot

def plot_box(data, alpha=.1 , figure_name='HAllA_Evaluation', xlabel = 'Methods', ylabel=None, labels=None):
    
    import pylab as pl
    import numpy as np
    # multiple box plots on one figure
    
    pl.figure("HAllA False Discovery Rate Controlling", dpi= 300, figsize=(10, 5))
    if ylabel == "FDR":
        pl.title("False Discovery Rate Controlling")
    if ylabel == "Recall":
        pl.title("Statistical Power")
        
    ax = pl.axes()
    pl.hold(True)
    if len(labels) > 0:
        ax.set_xticklabels(labels)
    pl.xlabel(xlabel)
    pl.xticks(range(len(labels)), labels, rotation=90, ha='right')
    pl.tight_layout()
    pl.ylabel(ylabel)
    pl.xlim([-0.05, 1.15])
    pl.ylim([-0.05, 1.15])
    bp = pl.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
    pl.scatter(np.repeat(np.arange(len(data)) + 1, len(data[0])), [item for sublist in data for item in sublist], marker='+', alpha=1)
    pl.setp(bp['boxes'], color='black')
    pl.setp(bp['whiskers'], color='blue')
    pl.setp(bp['fliers'], marker='+')
    # pl.plot(data)
    # pl.hlines(1-alpha,0.0,2.5, color ='blue')
    if ylabel == 'FDR':
        pl.plot([.0, len(data) + .5], [alpha, alpha], 'k-', lw=1, color='red')
    # hB, = pl.plot([1,1],'b-')
        hR, = pl.plot([1, 1], 'r-')
        pl.legend((hR,), ('q cut-off',))
    # pl.legend((hB, hR),('???', '???'))
    # hB.set_visible(False)
        hR.set_visible(False)
    # savefig('box7')
    
    pl.savefig(figure_name + '.pdf')
    pl.savefig(figure_name + '.png')
    #pl.show()
    pl.close()

def scatter_plot(x=None, y=None, alpha=.1, figure_name='Figure2', xlabel="Recall", ylabel="FDR", labels=None):
    import pylab as pl
    pl.figure("Recall vs. FDR", dpi= 300)
    pl.title("Recall vs. FDR")
    ax = pl.axes()
    pl.hold(True)
    # if len(labels) > 0:
    #    ax.set_xticklabels(labels)
    pl.xlabel(xlabel)
    # pl.xticks(range(len(labels)), labels, rotation=30, ha='right')
    pl.ylabel(ylabel)
    pl.xlim([-0.05, 1.15])
    pl.ylim([-0.05, 1.3])
    pl.tight_layout()
    pl.scatter(x, y , marker='o', alpha=.5)
    loc = True
    for i, txt in enumerate(labels):
        '''if loc :
            pos = "right"
            loc = False
        else:
            pos = "left"
            loc = True
        '''
        pl.annotate(txt, xy=(x[i], y[i]), xytext=(10, 10),
            textcoords='offset points', ha="right", va="bottom",
            bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.8', color='green'))
    # pl.plot(data)
    # pl.hlines(1-alpha,0.0,2.5, color ='blue')
    # if ylabel == 'Type I Error':
    # pl.plot([alpha, alpha], [-.05, 1.15], 'k-', lw=1, color ='red')
    
    pl.plot([-.05, 1.15], [alpha, alpha], 'k-', lw=1, color='red')
    # hB, = pl.plot([1,1],'b-')
    hR, = pl.plot([1, 1], 'r-')
    pl.legend((hR,), ('q cut-off',))
    # pl.legend((hB, hR),('???', '???'))
    # hB.set_visible(False)
    hR.set_visible(False)
    # savefig('box7')
    pl.savefig('Figure2.pdf')
    pl.savefig('Figure2.png')
    pl.show()
    return
    # fig, ax = pl.subplots()
    # ax.scatter(x, y)

    # for i, txt in enumerate(n):
     #   ax.annotate(txt, (z[i],y[i]))

def plot_roc(roc_info=None, figure_name='roc_plot_HAllA'):
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
        # print ('Hi', (roc_info[i][1]))
        fpr[roc_info[i][0]] = roc_info[i][1]
        # print ((roc_info[i][1])[0])
        tpr[roc_info[i][0]] = roc_info[i][2]
        roc_auc[roc_info[i][0]] = auc(fpr[roc_info[i][0]], tpr[roc_info[i][0]])
        roc_name += '_' + roc_info[i][0] 
        
    # Plot ROC curve
    plt.figure(dpi= 300)
    for i in range(len(roc_info)):
        plt.plot(fpr[roc_info[i][0]], tpr[roc_info[i][0]], label='{0} (area = {1:0.2f})'
                                       ''.format(str(roc_info[i][0]), roc_auc[roc_info[i][0]]))   
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    # plt.savefig('./test/'+roc_name+'foo.pdf')
    plt.savefig(figure_name + '.pdf')
    plt.show()
    # return plt
def heatmap_distance(pArray1, pArray2 = None, xlabels = None, filename='./hierarchical_heatmap', metric = "nmi", method = "single", ):
    import scipy
    import pylab
    import scipy.cluster.hierarchy as sch
    if pArray2 == None:
        pArray2 = pArray1
    pMetric = halla.distance.c_hash_metric[metric] 
    # # Remember, pMetric is a notion of _strength_, not _distance_ 
    # print str(pMetric)
    def pDistance(x, y):
        return  1.0 - pMetric(x, y)
    
    #D = pdist(pArray, metric=pDistance)
    D = scipy.zeros([len(pArray1), len(pArray2)])
    for i in range(len(pArray1)):
        for j in range(len(pArray2)):
            D[i][j] = pDistance(pArray1[i], pArray2[j])
    #print "Distance",D
    #plt.figure(figsize=(len(labels)/10.0 + 5.0, 5.0))
    Z = linkage(D, method = "single")
    
    # Compute and plot first dendrogram.
    fig = pylab.figure(dpi = 300,figsize=(len(pArray1)/3+2,len(pArray2)/3+2))
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
    im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu)
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])
    
    #axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
    
    axmatrix.set_xticks(range(len(pArray1)))
    axmatrix.set_xticklabels(idx1, minor=False)
    axmatrix.xaxis.set_label_position('bottom')
    axmatrix.xaxis.tick_bottom()
    
    pylab.xticks(rotation=-90, fontsize=10)
    
    axmatrix.set_yticks(range(len(pArray2)))
    axmatrix.set_yticklabels(idx2, minor=False)
    axmatrix.yaxis.set_label_position('right')
    axmatrix.yaxis.tick_right()
 
    # Plot colorbar.
    axcolor = fig.add_axes([0.94,0.1,0.02,0.6])
    pylab.colorbar(im, cax=axcolor)

    fig.savefig(filename + '.pdf')
        
def heatmap(pArray, xlabels = None, filename='./hierarchical_heatmap', metric = "nmi", method = "single", ):
    import scipy
    import pylab
    # import dot_parser
    import scipy.cluster.hierarchy as sch
    from numpy.matlib import rand
    from array import array
    # Adopted from Ref: http://stackoverflow.com/questions/2982929/plotting-results-of-hierarchical-clustering-ontop-of-a-matrix-of-data-in-python
    '''if len(D) == 0: 
        # Generate random features and distance matrix.
        print "The distance matrix is empty. The function generates a random matrix."
        x = scipy.rand(4)
        D = scipy.zeros([4, 4])
        for i in range(4):
            for j in range(i, 4):
                D[i, j] = abs(x[i] - x[j])
                D[j, i] = D[i, j]
     '''      
    pMetric = halla.distance.c_hash_metric[metric] 
    # # Remember, pMetric is a notion of _strength_, not _distance_ 
    # print str(pMetric)
    def pDistance(x, y):
        return  1.0 - pMetric(x, y)

    #D = pdist(pArray, metric=pDistance)
    # print "Distance",D
    #plt.figure(figsize=(len(labels)/10.0 + 5.0, 5.0))
    #Z = linkage(D, metric=pDistance)
    # Compute and plot first dendrogram.
    fig = pylab.figure(dpi= 300, figsize=(len(pArray[0])/3+2, len(pArray)/3+2))
    ax1 = fig.add_axes([0.09, 0.1, 0.2, 0.6], frame_on=True)
    Y1 = sch.linkage(pArray, metric=pDistance, method=method)
    Z1 = sch.dendrogram(Y1, orientation='right')#, labels= xlabels)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Compute and plot second dendrogram.
    ax2 = fig.add_axes([0.3, 0.71, 0.6, 0.2], frame_on=True)
    Y2 = sch.linkage(pArray.T)#, metric=pDistance, method=method)
    Z2 = sch.dendrogram(Y2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Plot distance matrix.
    axmatrix = fig.add_axes([0.3, 0.1, 0.6, 0.6])
    idx1 = Z1['leaves']
    idx2 = Z2['leaves']
    pArray = pArray[idx1, :]
    pArray = pArray[:, idx2]
    
    
    im = axmatrix.matshow(pArray, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu)
    
    
    axmatrix.set_xticks(range(len(idx2)))
    axmatrix.set_xticklabels(idx2, minor=False)
    axmatrix.xaxis.set_label_position('bottom')
    axmatrix.xaxis.tick_bottom()
    pylab.xticks(rotation=-90, fontsize=10)
    
    label1 = [xlabels[i] for i in idx1]
    axmatrix.set_yticks(range(len(idx1)))
    axmatrix.set_yticklabels(label1, minor=False)
    axmatrix.yaxis.set_label_position('right')
    axmatrix.yaxis.tick_right()
    pylab.xticks(rotation=0, fontsize=10)
    # Plot colorbar.
    #axcolor = fig.add_axes([0.91, 0.1, 0.02, 0.6])
    #pylab.colorbar(im, cax=axcolor)
    #fig.show()
    
    '''
    axmatrix.set_xticks(range(len(xlabels)))
    axmatrix.set_xticklabels(idx1, minor=False)
    axmatrix.xaxis.set_label_position('bottom')
    axmatrix.xaxis.tick_bottom()
    
    pylab.xticks(rotation=-90, fontsize=4)
    
    #axmatrix.set_yticks(range(40))
    axmatrix.set_yticklabels(idx2, minor=False)
    axmatrix.yaxis.set_label_position('right')
    axmatrix.yaxis.tick_right()
    
    #(0.5,0,0.5,1) adds an Axes on the right half of the figure. (0,0.5,1,0.5) adds an Axes on the top half of the figure.
    #Most people probably use add_subplot for its convenience. I like add_axes for its control.
    #To remove the border, use add_axes([left,bottom,width,height], frame_on=False)
    '''
    axcolor = fig.add_axes([0.94,0.1,0.02,0.6])
    pylab.colorbar(im, cax=axcolor)
    fig.savefig(filename + '.pdf')
    heatmap_distance(pArray, xlabels = xlabels, filename=filename+"_distance", metric = "nmi", method = "single", )
    return Y1