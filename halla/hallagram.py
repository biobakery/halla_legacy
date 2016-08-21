#!/usr/bin/env python

"""
Grand summary of HAllA output as heatmap figure
===============================================
Author: Eric Franzosa (eric.franzosa@gmail.com)
"""

import os
import sys
import argparse
import csv
import getpass
csv.field_size_limit(sys.maxsize)

#import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.style.use('ggplot')
mpl.use( "Agg" )
import matplotlib.pyplot as plt
#import matplotlib as mpl
#mpl.use( "Agg" )
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np

# ---------------------------------------------------------------
# constants / config
# ---------------------------------------------------------------

c_unit_h        = 0.35
c_unit_w        = 0.35
c_min_height    = 10 * c_unit_h + 2
c_min_width     = 20 * c_unit_w + 2
c_label_scale   = 18
c_label_shift   = -0.005
c_line_width    = 1.5
c_char_pad      = 0.1
c_label_aspect  = 0.7
c_small_text    = 12
c_large_text    = 16
c_giant_text    = 20
c_outline_width = 3
c_grid_color    = "0.9"
c_cbarspan      = 2
c_simstep       = 0.1

mpl.rcParams['xtick.major.pad'] = '10'
mpl.rcParams['ytick.major.pad'] = '10'

# ---------------------------------------------------------------
# classes
# ---------------------------------------------------------------

class Table:
    """ """   
    def __init__( self, path ):
        self.colheads = None
        self.rowheads = []
        self.data = []
        with open( path ) as fh:
            for row in csv.reader( fh, dialect="excel-tab" ):
                rowhead, values = row[0], row[1:]
                if self.colheads is None:
                    self.colheads = values
                else:
                    self.rowheads.append( rowhead )
                    self.data.append( map( float, values ) )
        self.data = np.array( self.data )
        self.update()
        
    def update( self ):
        self.nrows, self.ncols = self.data.shape
        self.rowmap = {}
        self.colmap = {}
        for i, h in zip( range( self.nrows ), self.rowheads ):
            self.rowmap[h] = i
        for i, h in zip( range( self.ncols ), self.colheads ):
            self.colmap[h] = i
        assert self.nrows == len( self.rowheads ) == len( self.rowmap ), "row dim failure"
        assert self.ncols == len( self.colheads ) == len( self.colmap ), "col dim failure"

# ---------------------------------------------------------------
# utilities
# ---------------------------------------------------------------

def reorder( array, order ):
    return [array[i] for i in order]

def get_args( ):
    parser = argparse.ArgumentParser()
    parser.add_argument( "simtable",
                         help="table of pairwise similarity scores" )
    parser.add_argument( "tree",
                         help="hypothesis tree (for getting feature order)" )
    parser.add_argument( "associations",
                         help="HAllA associations" )
    parser.add_argument( "--strongest",
                         default=None, type=int, help="isolate the N strongest associations" )
    parser.add_argument( "--largest",
                         default=None, type=int, help="isolate the N largest associations" )
    parser.add_argument( "--mask",
                         action="store_true", help="mask feature pairs not in associations" )
    parser.add_argument( "--cmap",
                         default="RdBu_r", help="matplotlib color map" )
    parser.add_argument( "--axlabels", nargs=2,
                         default=["1st dataset", "2nd dataset"], help="axis labels" )
    parser.add_argument( "--outfile",
                         default="hallagram.pdf", help="output file name" )
    parser.add_argument( "--similarity",
                         default="Pairwise Similarity", \
                         help="Similarity metric has been used for similarity measurement" )
    parser.add_argument( "--orderby",
                         default="similarity", \
                         help="Order the significant association by similarity, pvalue, or qvalue" )
    
    return parser.parse_args()

def get_order( path ):
    with open( path ) as fh:
        for row in csv.reader( fh, dialect="excel-tab" ):
            if row[0] == "0":
                row_order = row[1].split( ";" )
                col_order = row[2].split( ";" )
                break
    return [row_order, col_order]

def load_order_table( p_table, p_tree, associations ):
    allowed_rowheads = {k for items in associations for k in items[1]}
    allowed_colheads = {k for items in associations for k in items[2]}
    simtable = Table( p_table )
    row_order, col_order = get_order( p_tree )
    # reorder the rows
    row_order = [simtable.rowmap[k] for k in row_order if k in simtable.rowmap and k in allowed_rowheads]
    simtable.rowheads = reorder( simtable.rowheads, row_order )
    simtable.data = simtable.data[row_order, :]
    simtable.update( )
    # reorder the cols
    col_order = [simtable.colmap[k] for k in col_order if k in simtable.colmap and k in allowed_colheads]
    simtable.colheads = reorder( simtable.colheads, col_order )
    simtable.data = simtable.data[:, col_order]
    simtable.update( )
    return simtable

def load_associations( path, largest=None, strongest=None, orderby = 'similarity' ):
    pairs = []
    dic_order = {'p-value':3, 'q-value':4, 'similarity':5}
    with open( path ) as fh:
        for row in csv.reader( fh, dialect="excel-tab" ):
            if "Association" not in row[0]:
                pairs.append( [row[0], row[1].split( ";" ), row[3].split( ";" ), float( row[5] ), float( row[6] ),  float( row[7] )] )
    if largest is not None and strongest is not None:
        sys.exit( "Can only specify one of LARGEST and STRONGEST" )
    elif largest is not None:
        pairs = sorted( pairs, key=lambda row: len( row[1] ) * len( row[2] ), reverse=True )
        pairs = pairs[0:min( len( pairs ), largest )]
    elif strongest is not None:
        # not reversed, smaller p = stronger assoc
        pairs = sorted( pairs, key=lambda row: row[dic_order[orderby]], reverse=True )
        pairs = pairs[0:min( len( pairs ), strongest )]
    return pairs

def mask_table( simtable, associations ):
    allowed = {}
    for number, row_items, col_items, sig, _, _ in associations:
        for r in row_items:
            for c in col_items:
                ri = simtable.rowmap[r]
                ci = simtable.colmap[c]
                allowed[(ri, ci)] = 1
    for ri in range( simtable.nrows ):
        for ci in range( simtable.ncols ):
            if (ri, ci) not in allowed:
                simtable.data[ri][ci] = np.nan
    simtable.data = np.ma.masked_where( np.isnan( simtable.data ), simtable.data )

# ---------------------------------------------------------------
# main plotting function
# ---------------------------------------------------------------
    
def plot( simtable, associations, cmap, mask, axlabels, outfile, similarity ):
    # reverse roworder of simtable to match plotting
    order = range( simtable.nrows )[::-1]
    simtable.rowheads = reorder( simtable.rowheads, order )
    simtable.data = simtable.data[order, :]
    simtable.update( )
    # scale of the data
    tmin = np.min( simtable.data )
    tmax = np.max( simtable.data )
    crit = 0
    while crit < max( abs( tmin ), tmax ):
        crit += c_simstep
    vmin = 0 if tmin >= 0 else -crit
    vmax = crit
    # masking
    if mask:
        mask_table( simtable, associations )
    # begin plotting
    fig = plt.figure()
    width = max( c_unit_w * simtable.ncols, c_min_width )
    width += c_char_pad * max( map( len, simtable.rowheads ) )
    height = max( c_unit_h * simtable.nrows, c_min_height )
    height += c_char_pad * max( map( len, simtable.colheads ) )
    fig.set_size_inches( width, height )
    span = simtable.ncols
    cbarspan = c_cbarspan
    ax = plt.subplot2grid( ( 1, span ), ( 0, cbarspan ), rowspan=1, colspan=span-cbarspan )
    ax_cbar = plt.subplot2grid( ( 1, span ), ( 0, 0 ), rowspan=1, colspan=cbarspan )
    ax.yaxis.tick_right( )
    ax.yaxis.set_label_position("right")
    ax.set_yticks( [0.5+i for i in range( simtable.nrows )] )
    ax.set_xticks( [0.5+i for i in range( simtable.ncols )] )
    ax.set_yticklabels( simtable.rowheads, size=c_large_text )
    ax.set_xticklabels( simtable.colheads, rotation=90, rotation_mode="anchor", ha="right", va="center", size=c_large_text )
    ax.xaxis.set_ticks_position( 'none' ) 
    ax.yaxis.set_ticks_position( 'none' ) 
    ax.set_ylim( 0, len( simtable.rowheads ) )
    ax.set_xlim( 0, len( simtable.colheads ) )
    ax.set_ylabel( axlabels[0], size=c_giant_text )
    ax.set_xlabel( axlabels[1], size=c_giant_text )
    # if masking, draw a light grid to help with orientation
    if mask:
        kwargs = {"zorder":0, "color":c_grid_color}
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        for y in ax.get_yticks():
            ax.add_line( plt.Line2D( [xmin, xmax], [y, y], **kwargs ) )
        for x in ax.get_xticks():
            ax.add_line( plt.Line2D( [x, x], [ymin, ymax], **kwargs ) )
    # main heatmap
    heatmap = ax.pcolormesh( simtable.data, cmap=cmap, vmin=vmin, vmax=vmax )
    # craziness for getting cbar on the left with left-facing ticks
    norm = mpl.colors.Normalize( vmin=vmin, vmax=vmax )
    cbar = mpl.colorbar.ColorbarBase( 
        ax_cbar,
        norm=norm,
        cmap=cmap,
        orientation="vertical",
    )
    cbar.set_ticks( [] )
    twin_ax = plt.twinx( ax=cbar.ax )
    twin_ax.yaxis.set_label_position("left")
    twin_ax.yaxis.tick_left()
    [tick.set_size( c_large_text ) for tick in twin_ax.get_yticklabels()]
    twin_ax.set_ylim( vmin, vmax )
    if similarity != "Pairwise Similarity":
        similarity = "Pairwise "+similarity
    twin_ax.set_ylabel( similarity, size=c_giant_text, fontsize=10 )
    ticks = [vmin]
    while ticks[-1] < vmax:
        ticks.append( ticks[-1] + c_simstep )
    twin_ax.set_yticks( ticks )
    # add associations
    for number, row_items, col_items, sig, _, _ in associations:
        row_items = row_items[::-1]
        y1 = simtable.rowmap[row_items[0]]
        y2 = simtable.rowmap[row_items[-1]]
        x1 = simtable.colmap[col_items[0]]
        x2 = simtable.colmap[col_items[-1]]
        delx = abs( x2 - x1 ) + 1 
        dely = abs( y2 - y1 ) + 1
        # box
        ax.add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1 + 1,
                y2 - y1 + 1,
                facecolor="none",
                linewidth=c_line_width,
                clip_on=False,
            )
        )
        # label
        text = str( number )
        size = c_label_scale * min( delx, dely )
        size /= 1 if len( text ) == 1 else c_label_aspect * len( text )
        size = int( size )
        text = ax.text(
            np.mean( [x1, x2] )+.75+c_label_shift*size if (len(row_items)%2 != 0 and len(row_items)>1 and len(col_items) >1)  else\
            np.mean( [x1, x2] )+.5+c_label_shift*size if len(row_items) == 1 else np.mean( [x1, x2] )+0.5+c_label_shift*size ,
            np.mean( [y1, y2] )+0.5+c_label_shift*size,
            text,
            size=size,
            color="white",
            ha="center",
            va="center",
            weight="bold",
        )
        text.set_path_effects( [
            path_effects.Stroke( linewidth=c_outline_width, foreground='black'),
            path_effects.Normal(),
        ] )
    # craziness for hiding the border
    plt.setp( [child for child in ax.get_children() if isinstance( child, mpl.spines.Spine )], visible=False )
    plt.tight_layout()
    plt.savefig( outfile )

# ---------------------------------------------------------------
# main
# ---------------------------------------------------------------
    
def main( ):
    args = get_args( )
    associations = load_associations(
        args.associations,
        largest=args.largest,
        strongest=args.strongest, orderby = args.orderby)
    simtable = load_order_table( args.simtable, args.tree, associations )
    plot(
        simtable,
        associations,
        cmap=args.cmap,
        mask=args.mask,
        axlabels=args.axlabels,
        outfile=args.outfile,
        similarity=args.similarity
    )

if __name__ == "__main__":
    main( )
