#!/usr/bin/env python

"""
Grand scatter plot of HAllA associations
===============================================
Author: Gholamali Rahnavard (gholamali.rahnavard@gmail.com)
"""

import os
import sys
import argparse
import csv
import getpass
import pandas as pd
try:
    csv.field_size_limit(sys.maxsize)
except:
    # for some Windows platforms
    csv.field_size_limit(2147483647)

#import matplotlib.pyplot as plt
import matplotlib 
#matplotlib.style.use('ggplot')
matplotlib.use( "Agg" )
import matplotlib.pyplot as plt
#import matplotlib as matplotlib
#matplotlib.use( "Agg" )
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib import font_manager
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("error")
    try:
        font_file = font_manager.findfont(font_manager.FontProperties(family='Arial'))
        matplotlib.rcParams["font.family"] = "Arial"
    except UserWarning:
        pass
from . import hallagram 
from . import plot, config
from plot import scatter_matrix

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( "association_number",
                         help="Association number to be plotted",
                         type=int )
    parser.add_argument( "--input",
                         help="HAllA output directory",
                         default='./' )
    parser.add_argument( "--outfile",
                         default=None, help="output file name" )
    return parser.parse_args()

def load_table(args):
    try: 
        associations = hallagram.load_associations( str(args.input)+'/associations.txt', largest=None, strongest=None, orderby = 'similarity' )
        df1 = pd.read_csv(str(args.input)+'/X_dataset.txt', sep='\t', header=0, index_col =0)
        df2 = pd.read_csv(str(args.input)+'/Y_dataset.txt', sep='\t', header=0, index_col =0)
    except ImportError:
        sys.exit("Input Error for plotting points file!") 
    association_number = 1
    sim_rank , row_items, col_items, sig, _, _ = associations[args.association_number]   
    two_clusters = pd.concat([df1.loc[row_items], df2.loc[col_items]], axis=0, ignore_index=True)
    two_labels = row_items + col_items
    df_all = pd.DataFrame(np.array(two_clusters, dtype= float).T ,columns=np.array(two_labels))
    df_all_rank = df_all.rank()
    if args.outfile:
        plot.scatter_matrix(df_all, x_size = len(row_items),filename = args.outfile)
        plot.scatter_matrix(df_all_rank, x_size = len(row_items),filename ='ranked_'+args.outfile)
    else:
        plot.scatter_matrix(df_all_rank, x_size = len(row_items),filename ='Scatter_association_' + str(args.association_number) + '_rank.pdf')
        plot.scatter_matrix(df_all, x_size = len(row_items),filename ='Scatter_association_' + str(args.association_number) + '.pdf')

# ---------------------------------------------------------------
# main
# ---------------------------------------------------------------
def main( ):
    args = get_args()
    load_table(args)

if __name__ == "__main__":
    main( )
