"""
Central namespace for plotting capacities in HAllA, 
including all graphics and 'data object to plot' transformations.
"""

# from pylab import plot, hist, scatter



# import dot_parser
import pylab
import sys
from . import distance
import scipy.cluster 
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import pdist
from matplotlib.pyplot import xlabel
# import pydot
import math
import pandas as pd

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

def scatter_plot(x=None, y=None, alpha=.1, file_name='Figure2', xlabel="Recall", ylabel="FDR", labels=None):
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
    #pl.show()
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
        roc_auc[roc_info[i][0]] = auc(fpr[roc_info[i][0]], tpr[roc_info[i][0]] )
        roc_name += '_' + roc_info[i][0] 
        
    # Plot ROC curve
    plt.figure(dpi= 300, figsize=(4, 4))
    for i in range(len(roc_info)):
        params = {'legend.fontsize': 6,
        'legend.linewidth': 2}
        plt.rcParams.update(params)
        plt.plot(fpr[roc_info[i][0]], tpr[roc_info[i][0]],  label='{0} (area = {1:0.2f})'
                                       ''.format(str(roc_info[i][0]), roc_auc[roc_info[i][0]]))   
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    # plt.savefig('./test/'+roc_name+'foo.pdf')
    plt.tight_layout()
    plt.savefig(figure_name + '.pdf')
    #plt.show()
    # return plt
def heatmap2(pArray1, pArray2 = None, xlabels = None, ylabels = None, filename='./hierarchical_heatmap2', metric = "nmi", method = "single", colLable = True, rowLabel = True, color_bar = False):
    import scipy
    import pylab
    import scipy.cluster.hierarchy as sch
    if len(pArray2) == 0:
        pArray2 = pArray1
        ylabels = xlabels
    pMetric = distance.c_hash_metric[metric] 
    # # Remember, pMetric is a notion of _strength_, not _distance_ 
    # print str(pMetric)
    def pDistance(x, y):
        return  1.0 - pMetric(x, y)
    
    #D = pdist(pArray, metric=pDistance)
    #print len(pArray1), len(pArray2)
    D = scipy.zeros([len(pArray1), len(pArray2)])
    for i in range(len(pArray1)):
        for j in range(len(pArray2)):
            D[i][j] = pDistance(pArray1[i], pArray2[j])
    # Compute and plot first dendrogram.
    fig = pylab.figure(dpi = 300,figsize=(math.ceil(len(pArray2)/5.0)+2, math.ceil(len(pArray1)/5.0)+2))
    ax1 = fig.add_axes([0.09,0.1,0.2,0.6])
    Y1 = sch.linkage(D, method = "single")
    if len(Y1) > 1:
        Z1 = sch.dendrogram(Y1, orientation='right')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Compute and plot second dendrogram.
    ax2 = fig.add_axes([0.3,0.71,0.6,0.2])
    Y2 = sch.linkage(D.T, method = "single")
    if len(Y2) > 1:
        Z2 = sch.dendrogram(Y2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Plot distance matrix.
    axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
    if len(Y1) > 1:
        idx1 = Z1['leaves']
    else:
        idx1 = [0]
    
    if len(Y2) > 1:
        idx2 = Z2['leaves']
    else:
        idx2 = [0]
    D = D[idx1,:]
    D = D[:,idx2]
    im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu)#YlGnBu #afmhot
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])
    if colLable:    
        if len(ylabels) == len(idx2):
            label2 = [ylabels[i] for i in idx2]
        else:
            label2 = idx2
        
        axmatrix.set_xticks(range(len(pArray2)))
        axmatrix.set_xticklabels(label2, minor=False)
        axmatrix.xaxis.set_label_position('bottom')
        axmatrix.xaxis.tick_bottom()
        pylab.xticks(rotation=-90, fontsize=6)
    if rowLabel:
        if len(xlabels) == len(idx1):
            label1 = [xlabels[i] for i in idx1]
        else:
            label1 = idx1
        axmatrix.set_yticks(range(len(pArray1)))
        axmatrix.set_yticklabels(label1, minor=False)
        axmatrix.yaxis.set_label_position('right')
        axmatrix.yaxis.tick_right()
        pylab.yticks(rotation=0, fontsize=6)
   
    # Plot colorbar.
    if color_bar:
        axcolor = fig.add_axes([0.94,0.1,0.02,0.6])
        pylab.colorbar(im, cax=axcolor)
    fig.savefig(filename + '.pdf')
    pylab.close()
        
def heatmap(Data, D=[], xlabels_order = [], xlabels = None, filename='./hierarchical_heatmap', metric = "nmi", method = "single", colLable = False, rowLabel = True, color_bar = False):
    import scipy
    import pylab
    # import dot_parser
    import scipy.cluster.hierarchy as sch
    from numpy.matlib import rand
    from array import array
    pArray =  Data
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
    pMetric = distance.c_hash_metric[metric] 
    # # Remember, pMetric is a notion of _strength_, not _distance_ 
    # print str(pMetric)
    def pDistance(x, y):
        return  1.0 - pMetric(x, y)

    #D = pdist(pArray, metric=pDistance)
    # print "Distance",D
    #plt.figure(figsize=(len(labels)/10.0 + 5.0, 5.0))
    #Z = linkage(D, metric=pDistance)
    # Compute and plot first dendrogram.
    fig = pylab.figure(dpi= 300, figsize=((math.ceil(len(pArray[0])/5.0)),(math.ceil(len(pArray)/5.0))))
    ax1 = fig.add_axes([0.09, 0.1, 0.2, 0.6], frame_on=True)
    if len(D) > 0:
        Y1 = sch.linkage(D, metric=pDistance, method=method)
    else:
        Y1 = sch.linkage(pArray, metric=pDistance, method=method)
    if len(Y1) > 1:
        Z1 = sch.dendrogram(Y1, orientation='right')#, labels= xlabels)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Compute and plot second dendrogram.
    if len(xlabels_order) == 0:
        ax2 = fig.add_axes([0.3, 0.71, 0.6, 0.2], frame_on=True)
        Y2 = sch.linkage(pArray.T)#, metric=pDistance, method=method)
        if len(Y2) > 1:
            Z2 = sch.dendrogram(Y2)
        ax2.set_xticks([])
        ax2.set_yticks([])
    else:
        Y2 = []
    
    # Plot distance matrix.
    axmatrix = fig.add_axes([0.3, 0.1, 0.6, 0.6])
    if len(Y1) > 1:
        idx1 = Z1['leaves']
    else:
        idx1 = [0]
        
    if len(Y2) > 1:
        idx2 = Z2['leaves']
    else:
        idx2 = [0]
    
    pArray = pArray[idx1, :]
    if len(xlabels_order) == 0:
        pArray = pArray[:, idx2]
        xlabels_order.extend(idx2)
    else:
        #pass
        pArray = pArray[:, xlabels_order]    
    im = axmatrix.matshow(pArray, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu)#YlGnBu
    if colLable:
        if len(ylabels) == len(idx2):
            label2 = [ylabels[i] for i in idx2]
        else:
            label2 = idx2
        axmatrix.set_xticks(range(len(idx2)))
        axmatrix.set_xticklabels(label2, minor=False)
        axmatrix.xaxis.set_label_position('bottom')
        axmatrix.xaxis.tick_bottom()
        pylab.xticks(rotation=90, fontsize=6)
    if rowLabel:
        if len(xlabels) == len(idx1):
            label1 = [xlabels[i] for i in idx1]
        else:
            label1 = idx1
        axmatrix.set_yticks(range(len(idx1)))
        axmatrix.set_yticklabels(label1, minor=False)
        axmatrix.yaxis.set_label_position('right')
        axmatrix.yaxis.tick_right()
        pylab.yticks(rotation=0, fontsize=6)
    if color_bar:   
        axcolor = fig.add_axes([0.94,0.1,0.02,0.6])
        pylab.colorbar(im, cax=axcolor)
    fig.savefig(filename + '.pdf')
    #heatmap2(pArray, xlabels = xlabels, filename=filename+"_distance", metric = "nmi", method = "single", )
    pylab.close()
    return Y1

import numpy as np
import matplotlib.pyplot as plt

def grouped_boxplots2(data, xlabels, file_name ="Grouped_Recall_FDR"):
    '''data = [[np.random.normal(i, 1, 30) for i in range(2)],
            [np.random.normal(i, 1.5, 30) for i in range(3)],
            [np.random.normal(i, 2, 30) for i in range(4)]]
    '''
    #ax = plt.axes()
    
    fig, ax = plt.subplots(dpi= 300, figsize=( len(data)/2+5, 5))# figsize=(4, 4)) 
    #plt.hold(True)
    #plt.xlim([-0.05, 1.15])
    #plt.ylim([-0.05, 1.15])
    groups = grouped_boxplots(data, ax, patch_artist=True, max_width=0.5, notch=0, sym='+', vert=1, whis=1.0)

    colors = ['lightgreen', 'bisque']#'lavender', 'lightblue',
    for item in groups:
        for color, patch in zip(colors, item['boxes']):
            patch.set(facecolor=color)

    proxy_artists = groups[-1]['boxes']
    ax.legend(proxy_artists, ['Recall', 'FDR'], loc='best')
    pylab.xticks(rotation=90, fontsize=10)
    #ax.xticks(range(len(labels)), labels, rotation=90, ha='right')
    #ax.tight_layout()
    if len(xlabels) > 0:
        ax.set_xticklabels(xlabels)
    #ax.xlabel('Method')
    #ax.xticks(range(len(xlabels)), xlabels, rotation=90, ha='right')
    
    
    ax.set(xlabel='Method', ylabel='', axisbelow=True, xticklabels=xlabels)
    #ax.plot([-.05, 5], [alpha, alpha], 'k-', lw=1, color='red')
    plt.tight_layout()
    #ax.grid(axis='y', ls='-', color='white', lw=2)
    #ax.patch.set(facecolor='0.95')
    plt.savefig(file_name+".pdf")
    #plt.show()
    plt.close()

def grouped_boxplots(data_groups, ax, max_width=0.95, pad=0.05, **kwargs):
    if ax is None:
        ax = plt.gca()
        
    max_group_size = max(len(item) for item in data_groups)
    total_padding = pad  * (max_group_size - 1)     
    width = (max_width - total_padding) / max_group_size
    kwargs['widths'] = width

    def positions(group, i):
        span = width * len(group) + pad * (len(group)-1)
        ends = (span - width) / 2
        x = np.linspace(-ends, ends, len(group))
        return x + i

    artists = []
    ends = 0
    for i, group in enumerate(data_groups, start=1):
        
        #if flag:
        pos = positions(group, i)
        artist = ax.boxplot(group, positions= pos, **kwargs)
        if i % 2 == 0:
            #print pos
            ax.bar( np.mean(pos)-(width+2*pad), 1 , zorder=0, color=".985", width=(width+2*pad)*2, edgecolor="none" )
            #'''width * len(group) + pad * (len(group) - 1)-width/2 -pad'''
            #plt.setp(artist, color ='red')
        #artist.patch.set(facecolor='0.1')
        else:
           ax.bar( np.mean(pos)-(width+2*pad), 1 , zorder=0, color="0.955", width=(width+2*pad)*2, edgecolor="none" ) 
           
        artist = ax.boxplot(group, positions=positions(group, i), **kwargs)
        #artist.patch.set(facecolor='0.95')
        artists.append(artist)
        flage = True

    ax.margins(0.05)
    ax.set(xticks=np.arange(len(data_groups)) + 1)
    ax.autoscale()
    return artists

def scatter_matrix(df, filename = None):
    plt.figure()
    axs = pd.tools.plotting.scatter_matrix(df, alpha = .5, range_padding = 0.2, figsize=(len(df.columns)*.6+3.5, len(df.columns)*.6+3.5))

    def wrap(txt, width=8):
        '''helper function to wrap text for long labels'''
        import textwrap
        return '\n'.join(textwrap.wrap(txt, width))
    
    for ax in axs[:,0]: # the left boundary
        #ax.grid('off', axis='both')
        ax.set_ylabel(wrap(ax.get_ylabel()), rotation=0, va='center', labelpad=20)
        #ax.set_yticks([])
    
    for ax in axs[-1,:]: # the lower boundary
        #ax.grid('off', axis='both')
        ax.set_xlabel(wrap(ax.get_xlabel()), rotation=90)
        #ax.set_xticks([])
    plt.tight_layout()
    plt.savefig(filename)

def confusion_matrix(X, Y, filename):
    from sklearn.metrics import confusion_matrix
    # Compute confusion matrix
    cm = confusion_matrix(y_true= Y, y_pred=X)
    
    #print(cm)
    
    # Show confusion matrix in a separate window
    plt.matshow(cm, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu)
    plt.title('Association between the representatives')
    cb = plt.colorbar()
    plt.xlabel('First representative from the first cluster')
    plt.ylabel('First representative from the second cluster')
    
    #plt.show()(y_test, y_pred)
    #labels = np.arange(0,len(X),1)
    #loc    = labels 
    #cb.set_ticks(loc)
    #cb.set_ticklabels(labels)
    plt.savefig(filename)
