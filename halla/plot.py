"""
Central namespace for plotting capacities in HAllA, 
including all graphics and 'data object to plot' transformations.
"""

# from pylab import plot, hist, scatter



import sys
import scipy
import pylab
from array import array
import math
import pandas as pd
import numpy as np
from numpy.matlib import rand
import matplotlib.pyplot as plt
import scipy.cluster 
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import pdist, squareform
import scipy.cluster.hierarchy as sch
import matplotlib
from matplotlib.pyplot import xlabel
from itertools import product

from . import config
from . import distance
from . import stats

#matplotlib.style.use('ggplot')
#matplotlib.use( "Agg" )
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
    ax.get_xaxis().set_tick_params(which='both', labelsize=8,top='off',  direction='out')
    ax.get_yaxis().set_tick_params(which='both', labelsize=8, right='off', direction='out')
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
    ax.get_xaxis().set_tick_params(which='both', labelsize=8,top='off',  direction='out')
    ax.get_yaxis().set_tick_params(which='both', labelsize=8, right='off', direction='out')
    pl.hold(True)
    # if len(labels) > 0:
    #    ax.set_xticklabels(labels)
    pl.xlabel(xlabel)
    # pl.xticks(range(len(labels)), labels, rotation=30, ha='right')
    pl.ylabel(ylabel)
    pl.xlim([-0.05, 1.15])
    pl.ylim([-0.05, 1.3])
    pl.tight_layout()
    ax.scatter(x, y , marker='o', alpha=.5)
    loc = True
    for i, txt in enumerate(labels):
        '''if loc :
            pos = "right"
            loc = False
        else:
            pos = "left"
            loc = True
        '''
        ax.annotate(txt, xy=(x[i], y[i]), xytext=(10, 10),
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

def plot_roc(roc_info=None, title = None, figure_name='roc_plot_HAllA', ax= None):
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
    #if fig is None:
    #    axe = plt.gca()
    #else:
    #axe = fig.axes[4]
    
    # Compute ROC curve and ROC area for each class
    labels_fontsize = 8
    ticks_fontsize = 6
    fpr = dict()
    tpr = dict()
    roc_info[0][0] = "AllA"
    roc_info[1][0] = "HAllA"
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
    
    #axe = plt.gca()
    #fig, axe = plt.subplots(figsize=(5, 5 ), dpi=300)#, sharex=False, sharey=False)
    #fig.set_size_inches(1, 1)
    #plt.figure(dpi= 300, figsize=(4, 4))
    for i in range(len(roc_info)):
        params = {'legend.fontsize': 8,
        'legend.fancybox': True}
        plt.rcParams.update(params)
        ax.plot(fpr[roc_info[i][0]], tpr[roc_info[i][0]],  label='{0} (area = {1:0.2f})'
                                       ''.format(str(roc_info[i][0]), roc_auc[roc_info[i][0]]))   
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower right")
    ax.set_ylabel('True Positive Rate', fontsize = labels_fontsize)
    ax.set_xlabel('False Positive Rate', fontsize = labels_fontsize)
    ax.get_xaxis().set_tick_params(which='both', labelsize=ticks_fontsize,top='off',  direction='out')
    ax.get_yaxis().set_tick_params(which='both', labelsize=ticks_fontsize, right='off', direction='out')
    ax.yaxis.set_label_position('left') 
    pylab.xticks(rotation=0)

    ax.set_title(title, fontsize=10, fontweight='bold', loc='left')
    # plt.savefig('./test/'+roc_name+'foo.pdf')
    #plt.tight_layout()
    #plt.savefig(figure_name + '.pdf')
    #plt.show()
    # return plt
    #fig.axes[1] = axe
    return ax
def heatmap2(pArray1, pArray2 = None, xlabels = None, ylabels = None, filename='./hierarchical_heatmap2', metric = "nmi", method = "single", colLable = True, rowLabel = True, color_bar = False, scale ='sqrt'):
    
    if len(pArray2) == 0:
        pArray2 = pArray1
        ylabels = xlabels
    pMetric = distance.c_hash_metric[metric] 
    # # Remember, pMetric is a notion of _strength_, not _distance_ 
    # print str(pMetric)
    def pDistance(x, y):
        return  1.0 - matha.fabs(pMetric(x, y))
    
    #D = pdist(pArray, metric=pDistance)
    #print len(pArray1), len(pArray2)
    D = scipy.zeros([len(pArray1), len(pArray2)])
    for i in range(len(pArray1)):
        for j in range(len(pArray2)):
            D[i][j] = pDistance(pArray1[i], pArray2[j])
    # Compute and plot first dendrogram.
    fig = pylab.figure(dpi = 300,figsize=(math.ceil(len(pArray2)/5.0)+2, math.ceil(len(pArray1)/5.0)+2))
    ax1 = fig.add_axes([0.09,0.1,0.2,0.6])
    ax1.get_xaxis().set_tick_params(which='both', labelsize=8,top='off',  direction='out')
    ax1.get_yaxis().set_tick_params(which='both', labelsize=8, right='off', direction='out')
    Y1 = sch.linkage(D, method = method)
    if len(Y1) > 1:
        Z1 = sch.dendrogram(Y1, orientation='right')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Compute and plot second dendrogram.
    ax2 = fig.add_axes([0.3,0.71,0.6,0.2])
    ax2.get_xaxis().set_tick_params(which='both', labelsize=8,top='off',  direction='out')
    ax2.get_yaxis().set_tick_params(which='both', labelsize=8, right='off', direction='out')
    Y2 = sch.linkage(D.T, method = method)
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
        axmatrix.get_xaxis().set_tick_params(which='both', labelsize=8,top='off',  direction='out')
        axmatrix.get_yaxis().set_tick_params(which='both', labelsize=8, right='off', direction='out')
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
        axmatrix.get_xaxis().set_tick_params(which='both', labelsize=8,top='off',  direction='out')
        axmatrix.get_yaxis().set_tick_params(which='both', labelsize=8, right='off', direction='out')
        pylab.yticks(rotation=0, fontsize=6)
   
    # Plot colorbar.
    if color_bar:
        axcolor = fig.add_axes([0.94,0.1,0.02,0.6])
        pylab.colorbar(im, cax=axcolor)
    #plt.tight_layout()
    fig.savefig(filename + '.pdf')
    pylab.close()
        
def heatmap(data_table, D=[], xlabels_order = [], xlabels = None, ylabels = [], filename='./hierarchical_heatmap', metric = config.similarity_method, method = "single", colLable = False, rowLabel = True, color_bar = True, sortCol = True, dataset_number = None, scale  ='sqrt'):
    # Adopted from Ref: http://stackoverflow.com/questions/2982929/plotting-results-of-hierarchical-clustering-ontop-of-a-matrix-of-data-in-python

    if not data_table is None:
        fig = pylab.figure(dpi= 300, figsize=((math.ceil(len(data_table[0])/5.0)+6),(math.ceil(len(data_table)/5.0)+6)))
    else:
        fig = pylab.figure(dpi= 300, figsize=((math.ceil(len(D)/5.0)+6.0),(math.ceil(len(D)/5.0))+6.0))

        
    ax1 = fig.add_axes([0.09, 0.1, 0.2, 0.6], frame_on=True)
    ax1.get_xaxis().set_tick_params(which='both', labelsize=8,top='off',  direction='out')
    ax1.get_yaxis().set_tick_params(which='both', labelsize=8, right='off', direction='out')
    if len(D) > 0:
        Y1 = sch.linkage(D, method=method)
    else:
        Y1 = sch.linkage(data_table, metric=distance.pDistance, method=method)
    if len(Y1) > 1:
        Z1 = sch.dendrogram(Y1, orientation='left')#, labels= xlabels)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Compute and plot second dendrogram.
    if len(xlabels_order) == 0:
        ax2 = fig.add_axes([0.3, 0.71, 0.6, 0.2], frame_on=True)
        ax2.get_xaxis().set_tick_params(which='both', labelsize=8,top='off',  direction='out')
        ax2.get_yaxis().set_tick_params(which='both', labelsize=8, right='off', direction='out')
        Y2 = []
        if not data_table is None:
            Y2 = sch.linkage(data_table.T, metric=distance.pDistance, method=method)
        if len(Y2) > 1:
            Z2 = sch.dendrogram(Y2)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.get_xaxis().set_tick_params(which='both', labelsize=8,top='off',  direction='out')
        ax2.get_yaxis().set_tick_params(which='both', labelsize=8, right='off', direction='out')
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
        if len(D) > 0:
            idx2 = idx1
        else:            
            idx2 = [0]
    if not data_table is None:
        data_table = data_table[idx1, :]
        if sortCol:
            if len(xlabels_order) == 0 :
                data_table = data_table[:, idx2]
                xlabels_order.extend(idx2)
            else:
                #pass
                
                data_table = data_table[:, xlabels_order]
                
    else:
        D = D.iloc[idx1, idx1]
    myColor =  pylab.cm.YlOrBr
    if distance.c_hash_association_method_discretize[config.similarity_method]:
        myColor = pylab.cm.YlGnBu   
    else:
        myColor = pylab.cm.RdBu_r
    if not data_table is None:
        scaled_values = stats.scale_data(data_table, scale = scale)
    else:
        myColor = pylab.cm.pink
        scaled_values = D#stats.scale_data(D, scale = scale)
    im = axmatrix.matshow(scaled_values, aspect='auto', origin='lower', cmap=myColor)#YlGnBu
    if colLable:
        if len(ylabels) == len(idx2):
            label2 = [ylabels[i] for i in idx2]
        else:
            label2 = idx2
        axmatrix.set_xticks(range(len(idx2)))
        axmatrix.set_xticklabels(label2, minor=False)
        axmatrix.xaxis.set_label_position('bottom')
        axmatrix.xaxis.tick_bottom()
        axmatrix.get_xaxis().set_tick_params(which='both', labelsize=8,top='off',  direction='out')
        axmatrix.get_yaxis().set_tick_params(which='both', labelsize=8, right='off', direction='out')
    else:
        axmatrix.set_xticks([])
        axmatrix.set_xticklabels([])
        axmatrix.get_xaxis().set_tick_params(which='both', top='off')
        axmatrix.get_xaxis().set_tick_params(which='both', bottom='off')
        axmatrix.get_yaxis().set_tick_params(which='both', right='off')
        axmatrix.xaxis.set_label_position('bottom')
        axmatrix.xaxis.tick_bottom()
        
        
        #pylab.xticks(rotation=90, fontsize=6)
    if rowLabel:
        if len(xlabels) == len(idx1):
            label1 = [xlabels[i] for i in idx1]
        else:
            label1 = idx1
        axmatrix.yaxis.set_label_position('right')
        axmatrix.set_yticklabels(label1, minor=False)
        axmatrix.set_yticks(range(len(idx1)))
        axmatrix.get_xaxis().set_tick_params(which='both', labelsize=8,top='off',  direction='out')
        axmatrix.get_yaxis().set_tick_params(which='both', labelsize=8, right='off', direction='out')
        axmatrix.yaxis.tick_right()
        #pylab.yticks(rotation=0, fontsize=6)
    if color_bar:
        l = 0.2
        b = 0.71
        w = 0.02
        h = 0.2
        rect = l,b,w,h
        axcolor = fig.add_axes(rect)
        #axcolor = fig.add_axes([0.94,0.1,0.02,0.6])
        fig.colorbar(im, cax=axcolor, label=str(config.similarity_method).upper()+" ("+scale+")" )
        #pylab.colorbar(ax=axmatrix) 
        #axmatrix.get_figure().colorbar(im, ax=axmatrix)
    #plt.tight_layout()
        
    fig.savefig(filename + '.pdf')
    #heatmap2(data_table, xlabels = xlabels, filename=filename+"_distance", metric = "nmi", method = "single", )
    pylab.close()
    return Y1


def grouped_boxplots2(data, title, threshold_line = 0, xlabels = [], ylabel = "Recall/FDR" , xlable_rotation = 10,  file_name ="Grouped_Recall_FDR", ax = None):
    '''data = [[np.random.normal(i, 1, 30) for i in range(2)],
            [np.random.normal(i, 1.5, 30) for i in range(3)],
            [np.random.normal(i, 2, 30) for i in range(4)]]
    '''
    #ax = plt.axes()
    labels_fontsize = 8
    ticks_fontsize = 6
    fig = None
    if ax == None:
       fig, ax = plt.subplots(dpi= 300, figsize=( len(data)/2+5, 5))# figsize=(4, 4)) 
    #plt.hold(True)
    #plt.xlim([-0.05, 1.15])
    #plt.ylim([-0.05, 1.15])
    groups = grouped_boxplots(data, ax, patch_artist=True, max_width=0.5, notch=0, sym='+', vert=1, whis=1.0)

    colors = ['darkgreen', 'darkgoldenrod']#'lightgreen', 'bisque']#'lavender', 'lightblue',
    for item in groups:
        for color, patch in zip(colors, item['boxes']):
            patch.set(facecolor=color, alpha=0.5)

    proxy_artists = groups[-1]['boxes']
    if "FPR" in ylabel:
        ax.legend(proxy_artists, ['Recall', 'FPR'], loc='best', fontsize = labels_fontsize)
    else:
        ax.legend(proxy_artists, ['Recall', 'FDR'], loc='best', fontsize = labels_fontsize)
    ax.get_xaxis().set_tick_params(which='both', labelsize=ticks_fontsize,top='off',  direction='out')
    ax.get_yaxis().set_tick_params(which='both', labelsize=ticks_fontsize, right='off', direction='out')
    #ax.xticks(range(len(labels)), labels, rotation=90, ha='right')
    #ax.tight_layout()
    if len(xlabels) > 0:
        ax.set_xticklabels(xlabels, rotation =xlable_rotation, fontsize = 8)
   
    ax.set_title(title, fontsize=10, fontweight='bold', loc='left')
    #ax.set(xlabel='Method', ylabel='Recall/FDR', axisbelow=True, xticklabels=xlabels)
    ax.set(axisbelow=True)
    #ax.set_xlabel('Method', fontsize = 10)
    ax.set_ylabel(ylabel, fontsize = labels_fontsize)
    #pylab.xticks(rotation=45)

    #ax.plot([-.05, 5], [.1, .1], 'k-', lw=1, color='red')
    if threshold_line !=0:
        ax.axhline(y = .1, linewidth=.5, color='r', alpha= 1)
    #
    #ax.grid(axis='y', ls='-', color='white', lw=2)
    #ax.patch.set(facecolor='0.95')
    if fig:
        plt.tight_layout()
        plt.savefig(file_name+".pdf")
        plt.savefig(file_name+".png")
    #plt.show()
    #plt.close()
    return ax
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
        #artist = ax.boxplot(group, positions= pos, **kwargs)
        if i % 2 == 0:
            #print pos
            ax.bar( np.mean(pos)-(width+2*pad), 1 , zorder=0, color=".985", width=(width+2*pad)*2, edgecolor="none" )
            #'''width * len(group) + pad * (len(group) - 1)-width/2 -pad'''
            #plt.setp(artist, color ='red')
        #artist.patch.set(facecolor='0.1')
        else:
           ax.bar( np.mean(pos)-(width+2*pad), 1 , zorder=0, color="0.955", width=(width+2*pad)*2, edgecolor="none" ) 
        ax.get_xaxis().set_tick_params(which='both', labelsize=8,top='off',  direction='out')
        ax.get_yaxis().set_tick_params(which='both', labelsize=8, right='off', direction='out')
           
        artist = ax.boxplot(group, positions=positions(group, i), **kwargs)
        #artist.patch.set(facecolor='0.95')
        set_box_color(artist, color = 'red')
        artists.append(artist)
        flage = True

    ax.margins(0.05)
    ax.set(xticks=np.arange(len(data_groups)) + 1)
    
    ax.autoscale()
    return artists
def set_box_color(bp, color):
    plt.setp(bp['boxes'], color='gray')
    plt.setp(bp['whiskers'], color='gray')
    plt.setp(bp['caps'], color='gray')
    plt.setp(bp['medians'], color='red')
    plt.setp(bp['fliers'], color='gray')
    #plt.setp(bp, linewidth='.5')
def scatter_matrix(df, x_size = 0, filename = None, ):
    plt.figure(figsize=(len(df.columns)*.8+5, len(df.columns)*.8+5))
    color = 'darkgreen'
    #if x_size>0:
    #    colors = ['green'  if i< x_size and j < x_size else \
    #              'yellow' if i >= x_size and j >= x_size else 'black' for i,j in product(range(len(df.columns)),range(len(df.columns)))]
    #    
    axs = pd.tools.plotting.scatter_matrix(df, alpha = .1, s =50, c = 'white',\
                                           hist_kwds={'color':['white']},\
                                           range_padding = .2, grid=False, figsize=(len(df.columns)*.7+5, len(df.columns)*.7+5)) # diagonal='kde', grid=False,
    #color scatters
    for i in range(len(axs[:,0])):
        for j in range(len(axs[-1,:])):
            if x_size>0:
                color = 'maroon'  if i< x_size and j < x_size else \
                      'darkblue' if i >= x_size and j >= x_size else 'black'
            if i!= j:
                axs[i,j].scatter(df[df.columns[j]], df[df.columns[i]], c = color, s =50, alpha = .5)
            else:
                axs[i,j].hist(df[df.columns[j]], color = 'darkslategrey')
            #if i>j:
            #     axs[i,j].visible(False)
                
        
    plt.subplots_adjust(wspace=.005, hspace=.005)
    def wrap(txt, width=20):
        '''helper function to wrap text for long labels'''
        import textwrap
        #txt = txt.split("|")
        #txt = txt[len(txt)-2]+"_"+txt[len(txt)-1]
        return '\n'.join(textwrap.wrap(txt, width))
    
    for ax in axs[:,0]: # the left boundary
        ax.grid('off', axis='both')
        ax.set_ylabel(wrap(ax.get_ylabel()), rotation=0, va='center', ha = 'center', labelpad=30, fontsize = 10)#, fontweight='bold')
        ax.get_xaxis().set_tick_params(which='both', labelsize=8,top='off',  direction='out')
        ax.get_yaxis().set_tick_params(which='both', labelsize=8, right='off', direction='out')
        #ax.set_yticks([])
        #ax.set_color("gray")
    
    for ax in axs[-1,:]: # the lower boundary
        ax.grid('off', axis='both')
        ax.set_xlabel(wrap(ax.get_xlabel()), fontsize = 10, rotation=45, va='center', ha = 'left',labelpad=30 )#, fontweight='bold'
        #ax.set_xticks([])
        ax.get_xaxis().set_tick_params(which='both', labelsize=8,top='off',  direction='out')
        ax.get_yaxis().set_tick_params(which='both', labelsize=8, right='off', direction='out')
        #ax.set_color('yellow')
    #Change label rotation
    #[s.xaxis.label.set_rotation(45) for s in axs.reshape(-1)]
    #[s.yaxis.label.set_rotation(0) for s in axs.reshape(-1)]
    
    #May need to offset label when rotating to prevent overlap of figure
    #[s.get_yaxis().set_label_coords(-0.3,0.5) for s in axs.reshape(-1)]
    
    #Hide all ticks
    #[s.set_xticks(()) for s in axs.reshape(-1)]
    #[s.set_yticks(()) for s in axs.reshape(-1)]
    plt.tight_layout()
    plt.subplots_adjust(wspace=.01, hspace=.01)
    plt.savefig(filename)

def confusion_matrix(X, Y, filename):
    from sklearn.metrics import confusion_matrix
    # Compute confusion matrix
    ig, ax = plt.subplots(figsize=(6,6))
    cm = confusion_matrix(y_true= Y, y_pred=X)
  
    # Show confusion matrix in a separate window
    pcm = ax.matshow(cm, aspect='auto', origin='lower', cmap=pylab.cm.YlOrBr)#YlGnBu
    
    #pylab.colorbar(ax=ax) 
    ax.set_xlabel('First representative from the first cluster', fontsize = 8)
    ax.set_ylabel('First representative from the second cluster', fontsize = 8)
    ax.set_title('Association between the representatives', fontsize=10, fontweight='bold')
    ax.get_figure().colorbar(pcm, ax=ax) 
    ax.get_xaxis().set_tick_params(which='both', labelsize=8,top='off',  direction='out')
    ax.get_xaxis().set_ticks_position('bottom')
    ax.get_yaxis().set_tick_params(which='both', labelsize=8, right='off', direction='out')
    
    plt.savefig(filename)
def scatter_plot(X, Y, filename = 'scatter'):
    fig = plt.figure(figsize=(4.5, 4.5))
    # Create an Axes object.
    ax = fig.add_subplot(1,1,1) # one row, one column, first plot
    plt.rc('xtick', labelsize=6) 
    plt.rc('ytick', labelsize=6) 
    ax.set_xlabel("Representative of the First Cluster", fontsize = 8)
    ax.set_ylabel("Representative of the Second Cluster", fontsize = 8)
    ax.set_title('Association between the representatives', fontsize=10, fontweight='bold')
    ax.get_xaxis().set_tick_params(which='both', labelsize=8,top='off',  direction='out')
    ax.get_yaxis().set_tick_params(which='both', labelsize=8, right='off', direction='out')
    ax.scatter( X, Y, alpha=0.5, s =120, c='darkgreen')
    fig.tight_layout()
    fig.savefig(filename + '.pdf')

