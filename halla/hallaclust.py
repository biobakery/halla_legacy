#!/usr/bin/env python

"""
HAllA's Clustering using hierarchical clustering and Silhouette score 
To Run: 
$ ./hallaclust.py -i <distance_martix.txt> -o <clustering_output.txt>

"""

import argparse
import sys
import tempfile
import os
import shutil
import re
import pandas as pd
try:
    from . import hierarchy
except ImportError:
    sys.exit("CRITICAL ERROR: Unable to find the hierarchy module." + 
        " Please check your halla install.")
from . import config
from . import parser

def parse_arguments(args):
    """ 
    Parse the arguments from the user
    """
    
    parser = argparse.ArgumentParser(
        description= "HAllA's Clustering using hierarchical clustering and Silhouette score.\n",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "-v","--verbose", 
        help="additional output is printed\n", 
        action="store_true",
        default=False)
    parser.add_argument(
        "-i","--input",
        help="the input file D*N, Rows: D features and columns: N samples \n",
        required=False)
    parser.add_argument(
        "-d","--distance_matrix",
        help="the distance matrix file D*D (if input file is not provided), Rows: D features and columns: N samples \n",
        required=False)
    parser.add_argument(
        "-o","--output",
        help="the output directory\n",
        required=True)
    parser.add_argument(
        "-m", "--similarity_method",
        help="similarity measurement {default spearman, options: spearman, nmi, ami, dmic, mic, pearson, dcor}")
    parser.add_argument(
        "-n", "--estimated_number_of_clusters",
        type=int,
        help="estimated number of clusters")
    parser.add_argument(
        "-c","--linkage_method", 
        default= 'single',
        help="linkage clustering method method {default = single, options average, complete\n")

    return parser.parse_args()


def main( ):
    # Parse arguments from command line
    args=parse_arguments(sys.argv)
    config.similarity_method = args.similarity_method
    output_dir= args.output+"/"
    #os.path.dirname(args.output)+"/"
    
    df_distance = pd.read_table(args.distance_matrix, header=0, index_col =0)
    
    clusters = hierarchy.resoltion_hclust(distance_matrix=df_distance, number_of_estimated_clusters = args.estimated_number_of_clusters , linkage_method = args.linkage_method)
    #args.output+"hallaclust.txt", 'w'
    
    # write the results into outpute
    if os.path.isdir(output_dir):
        try:
            shutil.rmtree(output_dir)
        except EnvironmentError:
            sys.exit("Unable to remove directory: "+output_dir)
    
    # create new directory
    try:
        os.mkdir(output_dir)
    except EnvironmentError:
        sys.exit("Unable to create directory: "+output_dir)
    f = open(output_dir+"/hallaclust.txt", 'w')
    print "There are %s clusters" %(len(clusters))
    for i in range(len(clusters)):
        f.write("cluster"+str(i+1)+"\t")
        features = clusters[i].pre_order(lambda x: x.id)
        feature_names = [df_distance.index[val] for val in features]
        print feature_names
        for item in feature_names:
            f.write("%s " % item)
        
        f.write("\n")
    print "Output is written in " + args.output+"/hallaclust.txt"

        
if __name__ == "__main__":
    main( )