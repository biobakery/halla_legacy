#**HAllA: Hierarchical All-against-All association testing **#

[TOC]

## Description ##

Gholamali Rahnavard, Yo Sup Moon, Curtis Huttenhower

Google Group
 halla-users: https://groups.google.com/forum/#!forum/halla-users

License
 MIT License

URL
 http://huttenhower.sph.harvard.edu/halla

Citation
 Gholamali Rahnavard, Yo Sup Moon, Curtis Huttenhower, "Retrieving Signal from Noise in Big Data: An Information-Theoretic Approach to Hierarchical Exploratory Data Analysis" (In Preparation)

--------------------------------------------

HAllA (pronounced [challah](http://en.wikipedia.org/wiki/Challah)) is an
end-to-end statistical method for Hierarchical All-against-All discovery of
significant relationships among data features with high power.  HAllA is robust
to data type, operating both on continuous and categorical values, and works well
both on homogeneous datasets (where all measurements are of the same type, e.g.
gene expression microarrays) and on heterogeneous data (containing measurements
with different units or types, e.g. patient clinical metadata).  Finally, it is
also aware of multiple input, multiple output problems, in which data might
contain of two (or more) distinct subsets sharing an index (e.g. clinical metadata,
genotypes, microarrays, and microbiomes all drawn from the same subjects).  In
all of these cases, HAllA will identify which pairs of features (genes,
microbes, loci, etc.) share statistically significant co-variation, without
getting tripped up by high-dimensionality.

In short, HAllA is like testing for correlation among all pairs of variables
in a high-dimensional dataset, but without tripping over multiple hypothesis
testing, the problem of figuring out what "correlation" means for different
units or scales, or differentiating between predictor/input or response/output
variables.  It's your one-stop shop for statistical significance!

If you use this tool, the included scripts, or any related code in your work,
please let us know, sign up for the HAllA Users Group (halla-users@googlegroups.com), and pass along any issues or feedback.

# Requirements #

## Operating System ##
	* Supported 
		* Ubuntu Linux (>= 12.04) 
		* Mac OS X (>= 10.7)

	* Untested  
		* Windows (>= XP) 


## Softwares ##
```
* Python (>= 2.7)
* Numpy (>= 1.7.1)
* Scipy (>= 0.12) 
* Scikit-learn (>=0.13)
* matplotlib
```
** Optional for plotting results by using --plotting_results option in command line **
```
* Pandas (>=0.15.2)
```
# Installation #

## Downloading HAllA ##
HAllA can be downloaded in two ways:

* [Download](https://bitbucket.org/biobakery/halla/downloads) a compressed set of files.
* Create a clone of the repository on your computer with the command: 
	
	``hg clone https://bitbucket.org/biobakery/halla ``

Note: Creating a clone of the repository requires [Mercurial](http://mercurial.selenic.com/) to be installed. Once the repository has been cloned upgrading to the latest release of HAllA is simple. Just type ``hg pull -u`` from within the repository which will download the latest release. Then follow the install steps to update your install.

## Install HAllA  ##

How to install HAllA if you have admin priviledges (ie root):

1. From the HAllA directory run : 

        $ python setup.py install 


How to install HAllA if you do not have admin privileges (ie root):

1. Select a directory where you have write permissions to install HAllA ($DIR)
1. Create the directory where the HAllA executables will be installed : `` $ mkdir -p $DIR/bin/ ``
1. Add the bin directory to your path : `` $ export PATH=$PATH:$DIR/bin/ ``
1. Create the directory where the HAllA libraries will be installed : `` $ mkdir -p $DIR/lib/ ``
1. Add the lib directory to your pythonpath : `` $ export PYTHONPATH=$PYTHONPATH:$DIR/lib/ ``
1. From the HAllA directory run :  

        $ python setup.py install --install-scripts $DIR/bin/ --install-lib $DIR/lib/ 

NOTE: The changes to the paths made if you do not have admin priviledges will only be in effect while your shell is open. 
Closing your shell or opening a new shell will not include these changes to the paths. If you are running in a bash shell 
(you can check this by typing: `` $ ps -p $$ `` ), add the two export statements to your bashrc file located at ~/.bashrc . 
Then run: `` $ source ~/.bashrc `` so the updates are made to your current shell. Now every time you open a new shell
these changes to the paths will always be included.

# How to Run #

## Basic usage ##

Type the command:

`` halla -X $DATASET1 -Y DATASET2 --output $OUTPUT_DIR``

HAllA by default takes two tab-delimited text files as an input, where in each file, each row describes feature (data/metadata) and each column represents an instance. In other words, input `X` is a `D x N` matrix where `D` is the number of dimensions in each instance of the data and `N` is the number of instances (samples). The "edges" of the matrix should contain labels of the data, if desired. The following is an example input ::

	+---------+---------+---------+--------+
	|         | Sample1 | Sample2 | Sample3|
	+---------+---------+---------+--------+
	| DataX1 | 0       | 1       | 2      |
	+---------+---------+---------+--------+ 
	| DataX2 | 1.5     | 100.2   | -30.7  |
	+---------+---------+---------+--------+

	+---------+---------+---------+--------+
	|         | Sample1 | Sample2 | Sample3|
	+---------+---------+---------+--------+
	| DataY1  | 0       | 1       | 2      |
	+---------+---------+---------+--------+ 
	| DataY2  | 1.5     | 100.2   | -30.7  |
	+---------+---------+---------+--------+

Note: the input files have the same samples(columns) but features(rows) could be different. 

### Output ###
-----------------------------------------------

HAllA by default prints a tab-delimited text file as output ::

	+----------+----------+-------+------+------+
	| One      | Two      | NMI   | Pperm| Padjust|
	+ ---------+----------+-------+------+------+
	| X_CLUSTER| Y_CLUSTER| 0.64  | 0.02 | 0.008|
	+----------+----------+-------+------+------+  	

`NMIS` stands for "normalized mutual information", which is an information-theoretic measure of association between two random variables. `Pperm` and `Padjust` corresponds to the p-values of the permutation tests and adjusted p-value from Benjamini-Hochberg-Yekutieli approach used to assess the statistical significance of the mutual information distance (i.e. lower p-values signify that the association between two variables 
is not likely to be caused by the noise in the data).  

$OUTPUT_DIR = the output directory

**Three output files will be created:**

1. $OUTPUT_DIR/associations.tsv
1. $OUTPUT_DIR/all_association_results_one_by_one.tsv
1. $OUTPUT_DIR/all_compared_clusters_hypotheses_tree.tsv

** Optional outputs if --plotting_results is used in command line:**

1. $OUTPUT_DIR/associationsN/ (for all associations from 0..N if there is any)
	1. association_1.pdf
	1. Dataset_1_cluster_N_scatter_matrix.pdf
	1. Dataset_2_cluster_N_scatter_matrix.pdf
1. $OUTPUT_DIR/Pearson_heatmap.pdf
1. $OUTPUT_DIR/NMI_heatmap.pdf

## Demo runs ##

The input folder contains four demo input files. These files are tab-delimitated files.

To run the HAllA demo type the command:

`` halla -X examples/X_syntheticData.txt -Y examples/Y_syntheticData.txt -o $OUTPUT_DIR ``

$OUTPUT_DIR is the output directory

## Output files ##

HAllA produces three output files which by default are tab-delimited text.

### Significant Discovered Associations ###
```
Association Number	Clusters First Dataset	Cluster Similarity Score (NMI)	Explained Variance by the First PC of the cluster	 	Clusters Second Dataset	Cluster Similarity Score (NMI)	Explained Variance by the First PC of the cluster	 	nominal-pvalue	adjusted-pvalue	SImilarity score between Clusters (NMI)
1	6;7;8	0.294140401	0.647012305	 	6;7;8	0.249909049	0.642530249	 	0.000999001	0.011111111	0.259588486
2	0;1;2	0.286179622	0.653736652	 	2;0;1	0.259510749	0.541880507	 	0.002997003	0.022222222	0.238560794
3	3;4;5	0.243583717	0.64574729	 	3;4;5	0.250180471	0.61356624	 	0.022977023	0.033333333	0.222516454 
```
### All Association Results One-by-One ###
```
./input/X_syntheticData.txt	./input/Y_syntheticData.txt	nominal-pvalue	adjusted-pvalue
0	0	0.002997003	0.022222222
0	1	0.002997003	0.022222222
0	2	0.002997003	0.022222222
1	0	0.002997003	0.022222222
1	1	0.002997003	0.022222222
1	2	0.002997003	0.022222222
2	0	0.002997003	0.022222222
2	1	0.002997003	0.022222222
2	2	0.002997003	0.022222222
3	3	0.022977023	0.033333333
3	4	0.022977023	0.033333333
3	5	0.022977023	0.033333333
4	3	0.022977023	0.033333333
4	4	0.022977023	0.033333333
4	5	0.022977023	0.033333333
5	3	0.022977023	0.033333333
5	4	0.022977023	0.033333333
5	5	0.022977023	0.033333333
6	6	0.000999001	0.011111111
6	7	0.000999001	0.011111111
6	8	0.000999001	0.011111111
7	6	0.000999001	0.011111111
7	7	0.000999001	0.011111111
7	8	0.000999001	0.011111111
8	6	0.000999001	0.011111111
8	7	0.000999001	0.011111111
8	8	0.000999001	0.011111111
```
### All Compared Clusters from the Hypotheses Tree ###
```
Level	Dataset 1	Dataset 2
0	0;1;2;6;7;8;3;4;5	3;4;5;2;0;1;6;7;8
1	0;1;2	3;4;5
1	0;1;2	6;7;8
1	0;1;2	2;0;1
1	3;4;5	3;4;5
1	3;4;5	6;7;8
1	3;4;5	2;0;1
1	6;7;8	3;4;5
1	6;7;8	6;7;8
1	6;7;8	2;0;1
```
# Complete option list #
```
usage: halla [-h] -X <input_dataset_1.txt> [-Y <input_dataset_2.txt>] -o
             <output> [-q <0.1>] [-s <0.01>] [--descending] [-f {BHF,BHL,BHA}]
             [-i <1000>] [-m {nmi,ami,pearson}]
             [--decomposition {pca,cca,kpca,pls}] [-a {BH,FDR,Bonferroni,BHY}]
             [-t {permutation}] [-v] [--plotting-results]
             [--bypass-discretizing] [--header]

Hierarchical All-against-All significance association testing

optional arguments:
  -h, --help            show this help message and exit
  -X <input_dataset_1.txt>
                        first file: Tab-delimited text input file, one row per feature, one column per measurement
                        [REQUIRED]
  -Y <input_dataset_2.txt>
                        second file: Tab-delimited text input file, one row per feature, one column per measurement
                        [default = the first file (-X)]
  -o <output>, --output <output>
                        directory to write output files
                        [REQUIRED]
  -q <0.1>, --q-value <0.1>
                        q-value for overall significance tests (cut-off for false discovery rate)
                        [default = 0.1]
  -s <0.01>, --similarity-threshold <0.01>
                        threshold for similarity to count a cluster as one unit and not consider sub-clusters as sub-unit
                        [default = 0.01]
  --descending          hierarchical descending
                        [default = all-against-all]
  -f {BHF,BHL,BHA}, --fdr {BHF,BHL,BHA}
                        function to maximize statistical power and control false discovery rate
                        [default = BHF]
  -i <1000>, --iterations <1000>
                        iterations for nonparametric significance testing (permutation test)
                        [default = 1000]
  -m {nmi,ami,pearson}, --metric {nmi,ami,pearson}
                        metric to be used for similarity measurement
                        [default = nmi]
  --decomposition {pca,cca,kpca,pls}
                        approach for reducing dimensions (or decomposition) 
                        [default = pca]
  -a {BH,FDR,Bonferroni,BHY}, --adjusting {BH,FDR,Bonferroni,BHY}
                        approach for calculating adjusted p-value 
                        [default = BH]
  -t {permutation}, --test {permutation}
                        approach for association test
                        [default = permutation]
  -v, --verbose         additional output is printed
  --plotting-results    plot the results
  --bypass-discretizing
                        bypass the discretizing step
  --header              the input files contain a header line
```
# Frequently Asked Questions #

Please see all FAQ at the [ halla-users google group]( https://groups.google.com/forum/#!forum/halla-users).
