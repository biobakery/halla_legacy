#HAllA: Hierarchical All-against-All association testing #
HAllA is an acronym for Hierarchical All-against-All association testing, and is designed as a command-line tool to find associations in high-dimensional, heterogeneous datasets. 

*If you use the HUMAnN2 software, please cite our manuscript:*

Gholamali Rahnavard, Yo Sup Moon, George Weingart, Lauren J. McIver, Eric A. Franzosa, Levi Waldron, Curtis Huttenhower, "Retrieving Signal from Noise in Big Data: An Information-Theoretic Approach to Hierarchical Exploratory Data Analysis" (In Preparation) 

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

For additional information, please see the [HAllA User Manual](http://huttenhower.sph.harvard.edu/halla/manual).

If you use this tool, the included scripts, or any related code in your work,
please let us know, sign up for the HAllA Users Group [HAllA Users Google Group](https://groups.google.com/forum/#!forum/halla-users), and pass along any issues or feedback.

License
 MIT License
--------------------------------------------

## Contents ##
* [Features](#markdown-header-features)
* [Overview](#markdown-header-workflow)
* [Requirements](#markdown-header-requirements)
    * [Operating System](#markdown-header-operating-system)
    * [Software](#markdown-header-software)
* [Installation](#markdown-header-installation)
    * [Downloading HAllA](#markdown-header-downloading-halla)
    * [Installing HAllA](#markdown-header-installing-halla)
* [How to Run](#markdown-header-how-to-run)
    * [Basic usage](#markdown-header-basic-usage)
    * [Output files](#markdown-header-output-files)

        1. [Significant Discovered Associations File](#markdown-header-1-significant-discovered-associations-file)
        2. [All Association Results One-by-One File](#markdown-header-2-all-association-results-one-by-one-file)
        3. [All Compared Clusters from the Hypotheses Tree File](#markdown-header-3-all-compared-clusters-from-the-hypotheses-tree-file)
        4. [Optional Plotting Results Files](#markdown-header-4-optional-plotting-results-files)

* [Demo runs](#markdown-header-demo-runs)
* [Complete option list](#markdown-header-complete-option-list)
* [Frequently Asked Questions](#markdown-header-frequently-asked-questions)

## Features ##


In short, HAllA is like testing for correlation among all pairs of variables
in a high-dimensional dataset, but without tripping over multiple hypothesis
testing, the problem of figuring out what "correlation" means for different
units or scales, or differentiating between predictor/input or response/output
variables.  It's your one-stop shop for statistical significance!
Its advantages include:   

1. Generality: HAllA can handle datasets of mixed data types: categorical, binary, continuous, lexical (text strings with or without inherent order)

2. Efficiency: Rather than checking all possible possible associations, HAllA prioritizes computation such that only statistically promising candidate variables are tested in detail.

3. Reliability: HAllA utilizes hierarchical false discovery correction to limit false discoveries and loss of statistical power attributed to multiple hypothesis testing. 

4. Extensibility


## Overview ##

![](http://huttenhower.sph.harvard.edu//sites/default/files/Figure1_0.png)


## Requirements ##

### Operating System ###
```
* Linux 
* Mac OS X (>= 10.7.4) 
```

### Software ###
```
* Python (>= 2.7)
* Numpy (>= 1.7.1)
* Scipy (>= 0.12) 
* Scikit-learn (>=0.13)
* matplotlib
```
** Optional for plotting results by using the --plotting_results option **
```
* Pandas (>=0.15.2)
```
* R and FactoMineR package
```
## Installation ##

### Downloading HAllA ###
You can download the latest HAllA release or the development version.

Option 1: Latest Release (Recommended)

* [Download](https://bitbucket.org/biobakery/halla/downloads/biobakery-halla-0.5.0.tar) and unpack the latest release of HAllA.

Option 2: Development Version

* Create a clone of the repository: 
	
	``hg clone https://bitbucket.org/biobakery/halla ``

	Note: Creating a clone of the repository requires [Mercurial](http://mercurial.selenic.com/) to be installed. 

### Installing HAllA  ###

1. Move to the HAllA directory : ``$ cd HAllA_PATH ``
1. Install HAllA: ``$ python setup.py install``

Note: If you do not have write permissions to '/usr/lib/', then add the option "--user" to the install command. This will install the python package into subdirectories of '~/.local'.

## How to Run ##

### Basic usage ###

Type the command:

`` halla -X $DATASET1 -Y DATASET2 --output $OUTPUT_DIR``

HAllA by default takes two tab-delimited text files as an input, where in each file, each row describes feature (data/metadata) and each column represents an instance. In other words, input `X` is a `D x N` matrix where `D` is the number of dimensions in each instance of the data and `N` is the number of instances (samples). The "edges" of the matrix should contain labels of the data, if desired. 

The following is an example input:


Feature | Sample1 | Sample2 | Sample3
------------- | ------------- | ------------- | -------------
DataX1 | 0       | 1       | 2 
DataX2 | 1.5     | 100.2   | -30.7



Feature | Sample1 | Sample2 | Sample3
------------- | ------------- | ------------- | -------------
DataY1  | 0       | 1       | 2 
DataY2  | 1.5     | 100.2   | -30.7


Note: the input files have the same samples(columns) but features(rows) could be different. 

## Output files ##
-----------------------------------------------

HAllA by default prints a tab-delimited text file as output:



One      | Two      | NMI   | Pperm| Padjust
------------- | ------------- | ------------- | ------------- | -------------
X_CLUSTER| Y_CLUSTER| 0.64  | 0.02 | 0.008



`NMIS` stands for "normalized mutual information", which is an information-theoretic measure of association between two random variables. `Pperm` and `Padjust` corresponds to the p-values of the permutation tests and adjusted p-value from Benjamini-Hochberg-Yekutieli approach used to assess the statistical significance of the mutual information distance (i.e. lower p-values signify that the association between two variables 
is not likely to be caused by the noise in the data).  

$OUTPUT_DIR = the output directory

** Three tab-delimited output files will be created: **

### 1. Significant Discovered Associations File ###

File name: $OUTPUT_DIR/associations.tsv

```
Association Number	Clusters First Dataset	Cluster Similarity Score (NMI)	Explained Variance by the First PC of the cluster	 	Clusters Second Dataset	Cluster Similarity Score (NMI)	Explained Variance by the First PC of the cluster	 	nominal-pvalue	adjusted-pvalue	SImilarity score between Clusters (NMI)
1	6;7;8	0.294140401	0.647012305	 	6;7;8	0.249909049	0.642530249	 	0.000999001	0.011111111	0.259588486
2	0;1;2	0.286179622	0.653736652	 	2;0;1	0.259510749	0.541880507	 	0.002997003	0.022222222	0.238560794
3	3;4;5	0.243583717	0.64574729	 	3;4;5	0.250180471	0.61356624	 	0.022977023	0.033333333	0.222516454 
```

### 2. All Association Results One-by-One File ###

File name: $OUTPUT_DIR/all_association_results_one_by_one.tsv

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

### 3. All Compared Clusters from the Hypotheses Tree File ###

File name:  $OUTPUT_DIR/all_compared_clusters_hypotheses_tree.tsv

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

### 4. Optional Plotting Results Files ###

If the option "--plotting-results" is provided, the following files will also be written:

1. $OUTPUT_DIR/associationsN/ (for all associations from 0..N if there is any)
	1. association_1.pdf
	2. Dataset_1_cluster_N_scatter_matrix.pdf
	3. Dataset_2_cluster_N_scatter_matrix.pdf
2. $OUTPUT_DIR/Pearson_heatmap.pdf
3. $OUTPUT_DIR/NMI_heatmap.pdf


## Demo runs ##

The input folder contains two demo input files. These files are tab-delimitated files.

To run the HAllA demo type the command:

`` halla -X examples/X_syntheticData.txt -Y examples/Y_syntheticData.txt -o $OUTPUT_DIR ``

$OUTPUT_DIR is the output directory

## Complete option list ##
```
usage: halla [-h] -X <input_dataset_1.txt> [-Y <input_dataset_2.txt>] -o
             <output> [-q <0.1>] [-s <0.01>] [--descending] [-f {BHF,BHL,BHA}]
             [-i <1000>] [-m {nmi,ami,pearson}]
             [--decomposition {pca,cca,kpca,pls}]
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
  -t {permutation}, --test {permutation}
                        approach for association test
                        [default = permutation]
  -v, --verbose         additional output is printed
  --plotting-results    plot the results
  --bypass-discretizing
                        bypass the discretizing step
  --header              the input files contain a header line
```
## Frequently Asked Questions ##

Please see all FAQ at the [ halla-users google group]( https://groups.google.com/forum/#!forum/halla-users).