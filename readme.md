
# HAllA Legacy User Manual #

Hierarchical All-against-All association testing (HAllA) is a method for general purpose and well-powered association discovery in high-dimensional heterogeneous datasets. 

----

**The HAllA manuscript has been submitted!**

**Citation:**

Gholamali Rahnavard, Eric A. Franzosa, Lauren J. McIver, Emma Schwager, Jason Lloyd-Price, George Weingart, Yo Sup Moon, Xochitl C. Morgan, Levi Waldron, Curtis Huttenhower, *High-sensitivity pattern discovery in large multi'omic datasets.* [huttenhower.sph.harvard.edu/halla_legacy](http://huttenhower.sph.harvard.edu/halla)

**For installation and a quick demo, read the [HAllA Legacy Tutorial](https://github.com/biobakery/biobakery/wiki/halla_legacy)**

# ( **For additional information, read the [HAllA Legacy Tutorial](https://github.com/biobakery/biobakery/wiki/halla_legacy)** )

HAllA combines hierarchical nonparametric hypothesis testing with false discovery rate correction to enable high-sensitivity discovery of linear and non-linear associations in high-dimensional datasets (which may be categorical, continuous, or mixed). HAllA operates by 1) discretizing data to a unified representation, 2) hierarchically clustering paired high-dimensional datasets, 3) applying dimensionality reduction to boost power and potentially improve signal-to-noise ratio, and 4) iteratively testing associations between blocks of progressively more related features. 

## Contents ##

* [Features](#markdown-header-features)
* [Overview workflow](#markdown-header-overview-workflow)
* [Requirements](#markdown-header-requirements)
    * [Software](#markdown-header-software)
    * [Other](#markdown-header-other)
* [Initial Installation](#markdown-header-initial-installation)
    1. [Install HAllA](#markdown-header-1-install-halla)
    2. [Test the install](#markdown-header-2-test-the-install)
    3. [Try out a demo run](#markdown-header-3-try-out-a-demo-run)
* [How to run](#markdown-header-how-to-run)
    * [Basic usage](#markdown-header-basic-usage)
    * [Setting for association types](#markdown-header-setting-for-association-types)
    * [Demo runs](#markdown-header-demo-runs)
* [Output files](#markdown-header-output-files)
    1. [Associations file](#markdown-header-1-associations-file)
    2. [Similarity table](#markdown-header-2-similarity-table)
    3. [Hypothesis tree](#markdown-header-3-hypothesis-tree)
* [Result plots](#markdown-header-result-plots)
    1. [First dataset heatmap](#markdown-header-1-first-dataset-heatmap)
    2. [Second dataset heatmap](#markdown-header-2-second-dataset-heatmap)
    3. [Associations hallagram](#markdown-header-3-associations-hallagram)
    4. [Diagnostics scatter or confusion matrix plot](#markdown-header-4-diagnostics-scatter-or-confusion-matrix-plot)
* [Configuration](#markdown-header-configuration)
* [Tutorials](#markdown-header-tutorials)
    * [Adjust for covariates](#markdown-header-adjust-for-covariates)
    * [Selecting a desired false discovery rate](#markdown-header-selecting-a-desired-false-discovery-rate)
    * [Selecting a similarity measurement ](#markdown-header-selecting-a-similarity-measurement)
    * [Selecting a decomposition method ](#markdown-header-selecting-a-decomposition-method)
    * [Pairwise association testsing or AllA](#markdown-header-pairwise-association-testsing-or-alla)
    * [Filtering features from input by minimum entropy](#markdown-header-filtering-features-from-input-by-minimum-entropy)
* [Tools](#markdown-header-tools)
    * [halladata: synthetic paired dataset generator](#markdown-header-halladata-a-tool-for-synthetic-data)
    * [hallagram](#markdown-header-hallagram-a-tool-for-visualization)
    * [hallascatter](#markdown-header-hallascatter-a-tool-for-visualization)
    * [HAllA Python API](#markdown-header-halla-python-api)
    * [Nonparametric p value](#markdown-header-nonparametric-p-value)
* [FAQs](#markdown-header-faqs)
* [Complete option list](#markdown-header-complete-option-list)

## Features ##

1. Generality: HAllA can handle datasets of mixed data types: categorical, binary, continuous, lexical (text strings with or without inherent order)

2. Efficiency: Rather than checking all possible associations, HAllA prioritizes computation such that only statistically promising candidate variables are tested in detail.

3. Reliability: HAllA utilizes hierarchical false discovery correction to limit false discoveries and loss of statistical power attributed to multiple hypothesis testing.

4. Extensibility: HAllA is extensible to use different methods so measurement in its steps.
Similarity measurement it has the following metrics implemented: normalized mutual information (NMI), adjusted mutual information (AMI), mutual information (MI), maximum information coefficient (MIC), discretized mutual information as default, Spearman correlation, Pearson correlation, and distance correlation (DCOR).
Dimension reduction, decomposition, method uses medoid of clusters as default, and also has principal component analysis (PCA), independent component analysis (ICA), multiple corresponding analysis (MCA), centroid of clusters, partial least square (PLS), canonical component analysis (CCA), kernel principal component analysis (KPCA) implemented as options.

5. False discovery rate correction (FDR) methods are included: Benjamini–Hochberg(BH) as default, Benjamini–Yekutieli (BY), and Bonferroni.

6. A simple user interface (single command driven flow)

    * The user only needs to provide a paired dataset


## Overview workflow ##

![](http://huttenhower.sph.harvard.edu/sites/default/files/public/fig1.png)

*   File Types: tab-delimited text file with columns headers as samples or no sample names with the same order samples without head and row names as features
 
## Requirements ##

### Software ###

1. [Python](http://www.python.org/) (version >= 2.7 or >= 3.4)
2. [Numpy](http://http://www.numpy.org/) (version >= 1.9.2) (automatically installed)
3. [Scipy](https://www.scipy.org/) (version >= 0.17.1) (automatically installed)
4. [Matplotlib](http://http://matplotlib.org/) (version >= 1.5.1) (automatically installed)
5. [Scikit-learn](http://http://scikit-learn.org/stable/) (version >= 0.14.1) (automatically installed)
6. [pandas](https://http://pandas.pydata.org/pandas-docs/stable/) (version >= 0.18.1) (automatically installed)

### Other ###

1. Memory depends on input size mainly the number of features in each dataset 
2. Runtime depends on input size mainly the number of features in each dataset and similarity score that has been chosen
3. Operating system (Linux, Mac, or Windows)

## Initial Installation ##


### 1. Install HAllA ###

#### Installing with pip ####

1. Install HAllA
    * `` $ pip install halla ``
    * This command will automatically install HAllA and its dependencies.
    * To overwrite existing installs of dependencies use "-U" to force update them. 
    * To use the existing version of dependencies use "--no-dependencies." 
    * If you do not have write permissions to '/usr/lib/,' then add the option "--user" to the HAllA install command. Using this option will install the python package into subdirectories of '~/.local' on Linux. Please note when using the "--user" install option on some platforms, you might need to add '~/.local/bin/' to your $PATH as default might not include it. You will know if it needs to be added if you see the following message `HAllA: command not found` when trying to run HAllA after installing with the "--user" option.
    * If you use Windows operating system you can install it with administrator permission easily (please open a terminal with administrator permission and the rest is the same). 
    * If you have both Python 2 and Python 3 on your machine then use pip3 for Python 3.


#### Installing from source ####

1. Download HAllA
You can download the latest HAllA release or the development version. The source contains example files. If installing with pip, it is optional first to download the HAllA source.

Option 1: Latest Release (Recommended)

* [halla.tar.gz](https://pypi.python.org/pypi/halla) and unpack the latest release of HAllA.

Option 2: Development Version

* Create a clone of the repository: 
    
	``$ git clone https://github.com/biobakery/halla.git ``

	Note: Creating a clone of the repository requires [Git](https://git-scm.com/) to be installed. Once the clone is created, you can always update to the latest version of the repository with `` $ git pull ``.


2. Move to the HAllA directory

    * ``$ cd $HAllA_PATH `` 

3. Install HAllA

    * ``$ python setup.py install.''
    *  This command will automatically install HAllA and its dependencies.
    * To overwrite existing installs of dependencies us "-U" to force update them. 
    * If you do not have write permissions to '/usr/lib/,' then add the option "--user" to the HAllA install command. This will install the python package into subdirectories of '~/.local' on Linux. Please note when using the "--user" install option on some platforms, you might need to add '~/.local/bin/' to your $PATH as it might not be included by default. You will know if it needs to be added if you see the following message `HAllA: command not found` when trying to run HAllA after installing with the "--user" option.


### 2. Test the install ###

1. Test out the install with unit and functional tests
     * `` $ halla_test``

### 3. Try out a demo run ###

**Option 1: **HAllA uses **Spearman** as similarity metric by default for continuous data.

**Option 2: **HAllA uses **NMI** as similarity metric by default for mixed (categorical, continuous, and binary) data.

Users can override the default by providing other similarity metric implemented in HAllA using `-m`.

With HAllA installed you can try out a demo run using two sample synthetic datasets. 
``$ halla -X examples/X_16_100.txt examples/Y_16_100.txt -o $OUTPUT_DIR --hallagram --diagnostics-plot``

The output from this demo run will be written to the folder $OUTPUT_DIR.

## Installation Update ##

If you have already installed HAllA, using the [Initial Installation](#markdown-header-initial-installation) steps, and would like to upgrade your installed version to the latest version, please do:

`sudo -H pip install halla --upgrade --no-deps` or
`pip install halla --upgrade --no-deps`

This command upgrades HAllA to the latest version and ignores updating HAllA's dependencies. 

## How to run ##

### Basic usage ###

```
$ halla -X $DATASET1 -Y $DATASET2 --output $OUTPUT_DIR --diagnostics-plot -m spearman
```
* If not provided by `-m spearman`, As all the features are continuous data, HAllA uses Spearman coefficient, as the default similarity metric for continuous data.*

$DATASET1 and $DATASET2 = two input files that have the following format:

1.  tab-delimited text file (txt or tsv format)
2.  features are rows with mandatory row names 
3.  samples are columns with optional columns names. If the columns are the same order and the same size between two datasets then column name is not required but recommended. Otherwise, the file should contain column names in the first row and start with `#` or user should provide option `--header` in the command line. 

$OUTPUT_DIR = the output directory

`--hallagram` is an option for visualizing the results as a hallagram
`--diagnostics-plot` is an option to generate plots for each association
`-m spearman` is an option to use spearman as similarity measurement as our datasets contain continuous data and we look for monotonic relationships in this case. 

**Output files will be created:**

1. $OUTPUT_DIR/assocaitions.txt
	* the list of discovered associations 
2. $OUTPUT_DIR/assoction_N 
	* a list of plots each association where N is from 1 to number of discovered associations 
3. $OUTPUT_DIR/similarity_table.txt
        * as a matrix format file that contains similarity between individual features among two data sets.
5. $OUTPUT_DIR/hypothesis_tree.txt 
        * contains clusters that have been tested at different levels in the hypothesis tree.
6. $OUTPUT_DIR/hallagram.pdf 
        * a plot for a summary of associations.
7. $OUTPUT_DIR/peformance.txt 
        * includes the configuration that has been used (for reproducibility) and steps runtime.
8. $OUTPUT_DIR/X_dataest.txt 
        * first dataset that has been used after being processed.
9. $OUTPUT_DIR/Y_dataest.txt 
        * second dataset that has been used after being processed.
10. $OUTPUT_DIR/circus_table.txt 
        * input for Circus tool for visualization.
11. $OUTPUT_DIR/all_association_results_one_by_one.txt 
        * list of associations in individually paired features with p-value and q-value.	
12. $OUTPUT_DIR/hierarchical_heatmap.pdf 
        * HAllA produces two heatmaps on original datasets after parsing them(filtering features with low entropy or removing noncommon samples between two datasets).	

### Setting for association types ###

HAllA by default uses:

* Spearman correlation for continuous data (appropriate metric monotonic and linear associations) and medoid for clusters decomposition.

* Normalized mutual information (NMI) for mixed (categorical, continuous, and binary) data (appropriate metric any type of association) and medoid for clusters decomposition.

| Association type     | Data type  | Similarity metric | Decomposition    |
|----------------------|------------|-------------------|------------------|
| Any                  | Any        | NMI               | Medoid, MCA      |
| Linear or monotonic   | Continuous | Spearman          | Medoid, PCA, MCA |
| Parabola (quadratic) | Continuous | NMI, dCor         | Medoid, MCA      |
| L shape              | Any        | NMI               | Medoid, MCA      |
| Step pattern         | Any        | NMI               | Medoid, MCA      | 

### Demo runs ###

To run the demo:

`` $ halla -X examples/X_linear0_32_100.txt -Y examples/Y_linear0_32_100.txt -m spearman --output OUTPUT --diagnostics-plot ``


$OUTPUT_DIR is the output directory


## Output files ##

When HAllA is completed, three main output files will be created:

### 1. Associations file ###

``` 
| association_rank | cluster1                | cluster1_similarity_score | cluster2                | cluster2_similarity_score | pvalue   | qvalue      | similarity_score_between_clusters |
|------------------|-------------------------|---------------------------|-------------------------|---------------------------|----------|-------------|-----------------------------------|
| 1                | X30;X31                 | 0.738949895               | Y30;Y31                 | 0.562388239               | 3.33E-37 | 2.81E-34    | -0.900426043                      |
| 2                | X7;X10;X11;X9;X6;X8     | 0.521149715               | Y7;Y10;Y11;Y8;Y6;Y9     | 0.478449445               | 6.91E-32 | 2.92E-29    | -0.870183018                      |
| 3                | X16;X17;X15;X13;X12;X14 | 0.466724272               | Y16;Y13;Y17;Y15;Y12;Y14 | 0.400633663               | 2.94E-31 | 8.28E-29    | -0.866006601                      |
| 4                | X1;X3;X2;X4;X0;X5       | 0.567457546               | Y3;Y1;Y5;Y2;Y0;Y4       | 0.458731473               | 1.33E-28 | 2.81E-26    | -0.846672667                      |
| 5                | X28;X27;X26;X25;X24;X29 | 0.502168617               | Y27;Y28;Y25;Y26;Y24;Y29 | 0.414425443               | 4.91E-26 | 8.30E-24    | -0.825058506                      |
| 6                | X22;X20;X23;X19;X18;X21 | 0.511786379               | Y20;Y21;Y18;Y22;Y19;Y23 | 0.415246325               | 3.39E-23 | 4.77E-21    | -0.797119712                      |
| 7                | X0;X5                   | 0.781482148               | Y20                     | 1                         | 9.12E-05 | 0.011005714 | 0.381206121                       |
```

*   File name: `` $OUTPUT_DIR/associations.txt ``
*   This file details the associations. Features are grouped in clusters that participated in an association with another cluster.
*    **```association_rank```**: associations are sorted by high similarity score and low p-values.
*    **```cluster1```**: has one or more homogenous features from the first dataset that participate in the association.
*    **```cluster1_similarity_score```**: this value is corresponding to `1 - condensed distance` of the cluster in the hierarchy of the first dataset.
*    **```cluster2```**: has one or more homogenous features from the second dataset that participate in the association.
*    **```cluster2_similarity_score```**: this value is corresponding to `1 - condensed distance` of the cluster in the hierarchy of the second dataset.
*    **```pvalue```**: p-value from Benjamini-Hochbergapproach used to assess the statistical significance of the mutual information distance.
*    **```qvalue```**: q value calculates after BH correction for each test.
*    **```similarity_score_between_clusters```**: is the similarity score of the representatives (medoids) of two clusters in the association.

## Output files ##
1. [First dataset heatmap](#markdown-header-1-first-dataset-heatmap)
    2. [Second dataset heatmap](#markdown-header-2-second-dataset-heatmap)
    3. [Associations hallagram](#markdown-header-3-associations-hallagram)
    4. [Diagnostics scatter or confusion matrix plot](#markdown-header-4-diagnostics-plot)

### 1. First dataset heatmap ###
![](http://huttenhower.sph.harvard.edu/sites/default/files/public/hierarchical_heatmap_spearman_1.png =15x)

### 2. Second dataset heatmap ###
![](http://huttenhower.sph.harvard.edu/sites/default/files/public/hierarchical_heatmap_spearman_2.png =15x)

### 3. Associations hallagram ###
![](http://huttenhower.sph.harvard.edu/sites/default/files/public/hallagram_strongest_7.png =20x)

*   File name: `` $OUTPUT_DIR/hallagram.pdf ``
*   This file has a visualized representative of results in a heatmap. Rows are the features from the first dataset that participated in at least on association and the orders comes from their order in `linkage hierarchical cluster` with `average method`. Columns are the features from the second dataset that participated in at least on association and the orders comes from their order in `linkage hierarchical cluster` with `average method`. 
*   Each cell color represents the pairwise similarity between individual features.
*   Number on each block represents significant association numbers which are based on `similarity score` descending order (largest first) with p-value ascending order in a case of the same `similarity score`.

### 4. Diagnostics scatter or confusion matrix plot ###
![](http://huttenhower.sph.harvard.edu/sites/default/files/public/Scatter_association1.png =20x)

*   If option `` --diagnostics-plot`` is provided with ``halla`` command line then for each association, a set of plots will be produced at the end of HAllA's process.
*   File name: `` $OUTPUT_DIR/diagnostics_plot/association_1/Scatter_association1.pdf ``
*   This file has a visualized representative of Association 1 in a heatmap. X's are closer of features from a cluster in the first dataset that is significantly associated with a cluster of features, Ys, in the second dataset. The scatter plot shows how the association looks like within cluster and between initial features. 


## Configuration ##

HAllA produces a performance file to store user configuration settings. This configuration file is automatically created in the output directory.

```sh
$ vi performance.txt
HAllA version:	0.7.5
Decomposition method: 	medoid
Similarity method: 	spearman
Hierarchical linkage method: 	average
q: FDR cut-off : 	0.1
FDR adjusting method : 	bh
FDR using : 	level
Applied stop condition : 	False
Discretizing method : 	equal-area
Permutation function: 	none
Seed number: 	0
Number of permutations iterations for estimating pvalues: 	1000
Minimum entropy for filtering threshold : 	0.5

Number of association cluster-by-cluster:	7
Number of association feature-by-feature: 	186

Hierarchical clustering time	0:00:11.115361
Level-by-level hypothesis testing	0:00:02.486063
number of performed permutation tests: 	845
Summary statistics time	0.0033469200134277344
Plotting results time	0:02:21.775510
Total execution time	0:02:35.402486
```

## Tutorials ##

### Adjust for covariates ###

HAllA can be used to test the relationship between metadata (e.g. age and gender) and data (e.g. microbial species abundance and immune cell counts). In this case, related (covaried) metadata cluster together. In circumstances that two datasets are tested such as microbiome vs. metabolites, the effect of covariates (e.g. age, gender, and batch effect)  from both datasets such as (microbial species and metabolites) should be regressed out. Users should adjust for covariates. Here we provide two examples of R programming that how to adjust for a variable.

* **Adjust for age**: let's regress out the age effect from microbial species or metabolites:
 
```
#!python

lmer(microbe ~ age, microbial_abundance_data = table)
lmer(metabolite ~ age, metabolites_data = table)
```

* **Adjust for time**: this type of adjustment with groups structure involving has more complexity for adjusting and we recommend to read Winkler et al. Neuroimage. 2014 entitled *Permutation inference for the general linear model.* A simple code for this case would be: assume we have microbial samples from the same subject in several time-points a linear mixed-effects model is fit using the R lme4 package to each microbial species or metabolites of the form:


```
#!python

lmer(microbe ~ 1 + (1 | subject) + time, microbial_abundance_data = table)
```


### Selecting a desired false discovery rate ###

HAllA by default use 0.1 as the target false discovery rate. Users can change it to the desired value, for example, 0.05 or 0.25 by using `-q 0.05`.

### Selecting a similarity measurement ###

HAllA’s implementation and hypothesis testing scheme are highly general, allowing them to be used with a wide variety of similar measures. For similarity measurement option we recommend: 1) Spearman coefficient for continues data, 2)(default for HAllA) normalized mutual information (NMI) for mixed data (continuous, categorical, and binary data), and 3) discretized maximum information coefficient (DMIC) for complicated associations types such as sine waves in continuous data.  Similarity measures are implemented in the current version of HAllA that user can use as options are: Spearman coefficient, discretized normalized mutual information, discretized adjusted mutual information, discretized maximal information coefficient, Pearson correlation, distance correlation (dCor).  
``-m spearman`` for example change the default similarity to Spearman coefficient as similarity measurement, and it automatically bypasses discretizing step. For available similarity metrics, please look at HAllA options using ``halla -h``.

### Selecting a decomposition method ###

HAllA uses medoid of clusters as a representative to test the relation between clusters. A user can use other options using ``-d`` with other decomposition methods such as PCA, ICA, MCA. For example, ``-d pca`` will use the first principal component of a cluster as its representative.   

### Pairwise association testsing or AllA ###

A user can choose AllA as a naive pairwise testing approach using ``-a AllA`` option in the command line where the default is ``-a HAllA`` which uses the hierarchical approach.

### Filtering features from input by minimum entropy ###

HAllA by default removes features with low entropy (<.5) to reduce the unnecessary number tests. A user can use different threshold using option ``-e $THRESHOLD``. $THRESHOLD by default is .5.

## Tools ##


### hallagram: a tool for visualization ###

HAllA includes tools to be used with results.


`` $ cd $OUTPUT_DIR ``

`` $ hallagram  $hallagram similarity_table.txt hypotheses_tree.txt associations.txt hallagram.pdf ``

*   $TABLE = gene/pathway table (tsv or biom format)
*   $OUTPUT_DIR = the directory to write new gene/pathway tables (one per sample, in biom format if input is biom format)
```
usage: hallagram [-h] [--strongest STRONGEST] [--largest LARGEST] [--mask]
                 [--cmap CMAP] [--axlabels AXLABELS AXLABELS]
                 [--outfile OUTFILE] [--similarity SIMILARITY]
                 [--orderby ORDERBY]
                 simtable tree associations

positional arguments:
  simtable              table of pairwise similarity scores
  tree                  hypothesis tree (for getting feature order)
  associations          HAllA associations

optional arguments:
  -h, --help            show this help message and exit
  --strongest STRONGEST
                        isolate the N strongest associations
  --largest LARGEST     isolate the N largest associations
  --mask                mask feature pairs not in associations
  --cmap CMAP           matplotlib color map
  --axlabels AXLABELS AXLABELS
                        axis labels
  --outfile OUTFILE     output file name
  --similarity SIMILARITY
                        Similarity metric has been used for similarity
                        measurement
  --orderby ORDERBY     Order the significant association by similarity,
                        pvalue, or qvalue
```

### hallascatter: a tool for visualization ###
HAllA provides a script `hallascatter` to make a scatter matrix of between all features participate in an association.

`` $ hallascatter 1 --input ./ --outfile scatter_1.pdf``

```
usage: hallascatter [-h] [--input INPUT] [--outfile OUTFILE]
                    association_number

positional arguments:
  association_number  Association number to be plotted

optional arguments:
  -h, --help          show this help message and exit
  --input INPUT       HAllA output directory
  --outfile OUTFILE   output file name
```

### halladata: a tool for synthetic data ###
halladata generates paired datasets with various properties including: the size (number of features (rows) and samples (columns)),  the number of blocks (clusters within each dataset, the structure of clusters, the type of associations between features,  distribution of data (normal and uniform), the structure of clustering with each dataset,  the strongness of association between cluster among datasets define by noise between associated blocks, and the strongness of similarity between features within clusters defined by noise within blocks.

Here are two examples to generate paired datasets with the associations between them and HAllA runs. 

`halladata -f 32 -n 100 -a line -d uniform -s balanced -o halla_data_f32_n100_line`

The outputs will be located in `halla_data_f32_s100_line` directory and include a paired datasets: `X_line_32_100.txt`    `Y_line_32_100.txt` and `A_line_32_100.txt` association between them. A's rows are features in X dataset, and A's columns are features in Y dataset and for each cell in A zero means no significant association and 1 mean significant association. 
To run halla use on this synthetic data use:

`halla -X halla_data_f32_n100_line/X_line_32_100.txt -Y halla_data_f32_n100_line/Y_line_32_100.txt -o halla_output_f32_n100_line_spearman`
As all features in datasets are continuous halla uses Spearman coefficient as the similarity metric. One can specify a different similarity metric.  For example, try the same dataset with Normalized Mutual Information:
`halla -X halla_data_f32_n100_line/X_line_32_100.txt -Y halla_data_f32_n100_line/Y_line_32_100.txt -o halla_output_f32_n100_line_nmi -m nmi`

For mixed data (categorical, continuse data) HAllA automatically uses NMI as simialrity metric.
Let's generate some mixed data:

`halladata -f 32 -n 100 -a mixed -d uniform -s balanced -o halla_data_f32_n100_mixed`

Run HAllA od the data:

`halla -X halla_data_f32_n100_mixed/X_mixed_32_100.txt -Y halla_data_f32_n100_mixed/Y_mixed_32_100.txt -o halla_output_f32_n100_mixed`

If you try mixed data, HAllA provides a warning and ends as Spearman does NOT work with noncontinuous data. 

```
usage: halladata [-h] [-v] [-f FEATURES] [-n SAMPLES] [-a ASSOCIATION]
                 [-d DISTRIBUTION] [-b NOISE_BETWEEN] [-w NOISE_WITHIN] -o
                 OUTPUT [-s STRUCTURE]

HAllA synthetic data generator to produce paired data sets with association among their features.

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         additional output is printed
  -f FEATURES, --features FEATURES
                        number of features in the input file D*N, Rows: D features and columns: N samples 
  -n SAMPLES, --samples SAMPLES
                        number of samples in the input file D*N, Rows: D features and columns: N samples 
  -a ASSOCIATION, --association ASSOCIATION
                        association type [sine, parabola, log, line, L, step, happy_face, default =parabola] 
  -d DISTRIBUTION, --distribution DISTRIBUTION
                        Distribution [normal, uniform, default =uniform] 
  -b NOISE_BETWEEN, --noise-between NOISE_BETWEEN
                        number of samples in the input file D*N, Rows: D features and columns: N samples 
  -w NOISE_WITHIN, --noise-within NOISE_WITHIN
                        number of samples in the input file D*N, Rows: D features and columns: N samples 
  -o OUTPUT, --output OUTPUT
                        the output directory
  -s STRUCTURE, --structure STRUCTURE
                        structure [balanced, imbalanced, default =balanced] 
```

### HAllA Python API ###
HAllA function along with command line can be called from other programs using Python API we provide an example is demonstrated here to show how to import and use `hallatest` function :

```
#!python

from halla.halla import hallatest

def main():
    hallatest(X='/path/to/first/datase/X.txt',\
               Y= '/path/to/second/datase/Y.txt',\
               output_dir='/path/to/halla/output/halla_output_demo')

if __name__ == "__main__":
    main( )  
```



### Nonparametric p-value ###
We have implemented both empirical cumulative distribution function (ECDF) and fast and accurate approach, generalized Pareto distribution (GPD) by Knijnenburg et al. 2009, permutation test. The function can be imported to other python programs :

```
#!python

from halla.stats import permutation_test_pvalue 
import numpy as np

def main():
    
    # Generate a list of random values for first vector
    np.random.seed(0)
    x_rand = np.random.rand(1,10)[0]   
    
    # Generate a list of random values for second vector
    # set the numpy seed for different random values from the first set
    np.random.seed(1)
    y_rand = np.random.rand(1, 10)[0]
    
    # Calculate pvalue using empirical cumulative distribution function (ECDF) 
    p_random_ecdf = permutation_test_pvalue(X  = x_rand, Y = y_rand, similarity_method = 'spearman',permutation_func = 'ecdf')
    p_perfect_ecdf = permutation_test_pvalue(X  = x_rand, Y = x_rand, similarity_method = 'spearman', permutation_func = 'ecdf')
    print ("ECDF P-value for random data: %s, ECDF P-value for perfect correlation data: %s")%(p_random_ecdf, p_perfect_ecdf)
    
    # Calculate pvalue using our implementation in HAllA for generalized Pareto distribution (GPD) approach proposed by Knijnenburg et al. 2009 
    p_random_gpd = permutation_test_pvalue(X  = x_rand, Y = y_rand, similarity_method = 'spearman',permutation_func = 'gpd')
    p_perfect_gpd = permutation_test_pvalue(X  = x_rand, Y = x_rand, similarity_method = 'spearman', permutation_func = 'gpd')
    print ("GPD P-value for random data: %s, GPD P-value for perfect correlation data: %s")%(p_random_gpd, p_perfect_gpd)

if __name__ == "__main__":
    main( ) 
```
The parameters that can be provided to the permutation test for calculating p-value are:

* `iterations`: the number permutation for the test (i.e. 1000)
* `permutation_func` can be either 'ECDF' or 'GPD'
* `similarity_method` a similarity metric supported by HAllA (check what are the choices by 'halla -h')
* `seed` if -1 each run seeds a random value, 0 uses the same seed any place does permutation.



##### Complete halla option list #####
```
usage: halla [-h] [--version] -X <input_dataset_1.txt>
             [-Y <input_dataset_2.txt>] -o <output> [-q <.1>]
             [-p {ecdf,gpd,none}] [-a {HAllA,AllA}] [-i <1000>]
             [-m {nmi,ami,mic,dmic,dcor,pearson,spearman}]
             [-d {none,mca,pca,ica,cca,kpca,pls,medoid}]
             [--fdr {bh,by,bonferroni,no_adjusting}] [-v VERBOSE]
             [--diagnostics-plot] [--discretizing {equal-area,hclust,none}]
             [--linkage {single,average,complete,weighted}]
             [--apply-stop-condition] [--generate-one-null-samples] [--header]
             [--format-feature-names] [--nproc <1>] [--nbin <None>] [-s SEED]
             [-e ENTROPY_THRESHOLD] [-e1 ENTROPY_THRESHOLD1]
             [-e2 ENTROPY_THRESHOLD2] [--missing-char MISSING_CHAR]
             [--missing-method {mean,median,most_frequent}]
             [--missing-data-category] [--write-hypothesis-tree]
             [-t {log,sqrt,arcsin,arcsinh,}]

HAllA: Hierarchical All-against-All significance association testing

optional arguments:
  -h, --help            show this help message and exit
  --version             show program`s version number and exit
  -X <input_dataset_1.txt>
                        first file: Tab-delimited text input file, one row per feature, one column per measurement
                        [REQUIRED]
  -Y <input_dataset_2.txt>
                        second file: Tab-delimited text input file, one row per feature, one column per measurement
                        [default = the first file (-X)]
  -o <output>, --output <output>
                        directory to write output files
                        [REQUIRED]
  -q <.1>, --q-value <.1>
                        q-value for overall significance tests (cut-off for false discovery rate)
                        [default = 0.1]
  -p {ecdf,gpd,none}, --permutation {ecdf,gpd,none}
                        permutation function 
                        [default = none for Spearman and Pearson and gpd for other]
  -a {HAllA,AllA}, --descending {HAllA,AllA}
                        descending approach
                        [default = HAllA for hierarchical all-against-all]
  -i <1000>, --iterations <1000>
                        iterations for nonparametric significance testing (permutation test)
                        [default = 1000]
  -m {nmi,ami,mic,dmic,dcor,pearson,spearman}, --metric {nmi,ami,mic,dmic,dcor,pearson,spearman}
                        metric to be used for similarity measurement
                        [default = nmi]
  -d {none,mca,pca,ica,cca,kpca,pls,medoid}, --decomposition {none,mca,pca,ica,cca,kpca,pls,medoid}
                        approach for reducing dimensions (or decomposition)
                        [default = medoid]
  --fdr {bh,by,bonferroni,no_adjusting}
                        approach for FDR correction
                        [default = bh]
  -v VERBOSE, --verbose VERBOSE
                        additional output is printed
  --diagnostics-plot    Diagnostics plot for associations 
  --discretizing {equal-area,hclust,none}
                        approach for discretizing continuous data
                        [default = equal-area]
  --linkage {single,average,complete,weighted}
                        The method to be used in hierarchical linkage clustering.
  --apply-stop-condition
                        stops when two clusters are too far from each other
  --generate-one-null-samples, --fast
                        Use one null distribution for permutation test
  --header              the input files contain a header line
  --format-feature-names
                        Replaces special characters and for OTUs separated  by | uses the known end of a clade
  --nproc <1>           the number of processing units available
                        [default = 1]
  --nbin <None>         the number of bins for discretizing 
                        [default = None]
  -s SEED, --seed SEED  a seed number to make the random permutation reproducible
                        [default = 0,and -1 for random number]
  -e ENTROPY_THRESHOLD, --entropy ENTROPY_THRESHOLD
                        Minimum entropy threshold to filter features with low information
                        [default = 0.5]
  -e1 ENTROPY_THRESHOLD1, --entropy1 ENTROPY_THRESHOLD1
                        Minimum entropy threshold for the first dataset 
                        [default = None]
  -e2 ENTROPY_THRESHOLD2, --entropy2 ENTROPY_THRESHOLD2
                        Minimum entropy threshold for the second dataset 
                        [default = None]
  --missing-char MISSING_CHAR
                        defines missing characters
                        [default = '']
  --missing-method {mean,median,most_frequent}
                        defines missing strategy to fill missing data.
                        For categorical data puts all missing data in one new category.
  --missing-data-category
                        To count the missing data as a category
  --write-hypothesis-tree
                        To write levels of hypothesis tree in the file
  -t {log,sqrt,arcsin,arcsinh,}, --transform {log,sqrt,arcsin,arcsinh,}
                        data transformation method 
                        [default = '' 
```
