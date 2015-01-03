.. HAllA: Hierarchical All-against All documentation master file, created by
   sphinx-quickstart on Mon Nov 25 13:27:23 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


=============================================================
HAllA: Hierarchical All-against-All association testing 
=============================================================

-------------------------------------------------------
Version 0.0.1
-------------------------------------------------------

Authors
 Gholamali Rahnavard, Yo Sup Moon, Curtis Huttenhower

Google Group
 halla-users: https://groups.google.com/forum/#!forum/halla-users

License
 MIT License

URL
 http://huttenhower.sph.harvard.edu/halla

Citation
 Yo Sup Moon, George Weingart, Curtis Huttenhower, "Retrieving Signal from Noise in Big Data: An Information-Theoretic Approach to Hierarchical Exploratory Data Analysis" (In Preparation)

.. toctree::
   :maxdepth: 2


Chapter 0 Getting Started
============================================ 

Operating System  
--------------------------------------------

* Supported 
	* Ubuntu Linux (>= 12.04) 
	* Mac OS X (>= 10.7)

* Untested  
	* Windows (>= XP) 

Dependencies 
--------------------------------------------

* Required 
	* Python (>= 2.7)
	* Numpy (>= 1.7.1)
	* Scipy (>= 0.12) 
	* Scikit-learn (>=0.13)  
	* sampledoc-master
All the required packages are included in Anaconda which is available at https://store.continuum.io/cshop/anaconda/

* Recommended Tools for documentation 
	* Docutils
	* itex2MML

Getting HAllA
--------------------------------------------

HAllA can be downloaded from its bitbucket repository: http://bitbucket.org/chuttenh/halla.

Directory Structure -- What you get 
---------------------------------------------

bin/testdata (source: Put into bin any scripts you’ve written that use your towelstuff package and which you think would be useful for your users. If you don’t have any, then remove the bin directory.)



Chapter 1 Basics 
============================================

Introduction
--------------------------------------------

`HAllA`_ (pronounced "`challah <http://en.wikipedia.org/wiki/Challah>`_") is an
end-to-end statistical method for Hierarchical All-against-All discovery of
significant relationships among data features with high power.  `HAllA`_ is robust
to data type, operating both on continuous and categorical values, and works well
both on homogeneous datasets (where all measurements are of the same type, e.g.
gene expression microarrays) and on heterogeneous data (containing measurements
with different units or types, e.g. patient clinical metadata).  Finally, it is
also aware of multiple input, multiple output problems, in which data might
contain of two (or more) distinct subsets sharing an index (e.g. clinical metadata,
genotypes, microarrays, and microbiomes all drawn from the same subjects).  In
all of these cases, `HAllA`_ will identify which pairs of features (genes,
microbes, loci, etc.) share statistically significant co-variation, without
getting tripped up by high-dimensionality.

In short, `HAllA`_ is like testing for correlation among all pairs of variables
in a high-dimensional dataset, but without tripping over multiple hypothesis
testing, the problem of figuring out what "correlation" means for different
units or scales, or differentiating between predictor/input or response/output
variables.  It's your one-stop shop for statistical signficance!

If you use this tool, the included scripts, or any related code in your work,
please let us know, sign up for the `HAllA users group`_
(`halla-users@googlegroups.com`_), and pass along any issues or feedback.

Contents
========

Input 
----------------------------------------------

HAlLA by default takes a tab-delimited text file as an input, where each row describes feature (data/metadata) and each column represents an instance. In other words, input `X` is a `D x N` matrix where `D` is the number of dimensions in each instance of the data and `N` is the number of instances (samples). The "edges" of the matrix should contain labels of the data, if desired. The following is an example input ::

	+-------+---------+---------+--------+
	|       | Sample1 | Sample2 | Sample3|
	+-------+---------+---------+--------+
	| Data1 | 0       | 1       | 2      |
	+-------+---------+---------+--------+ 
	| Data2 | 1.5     | 100.2   | -30.7  |
	+-------+---------+---------+--------+

Note: as the inputs datasets have the same samples, the input files must not have sample headers
Output 
-----------------------------------------------

HAllA by default prints a tab-delimited text file as output ::

	+------+------+-------+------+------+
	| One  | Two  | MID   | Pperm| Pboot|
	+------+------+-------+------+------+
	| Data1| Data2| 0.64  | 0.02 | 0.008|
	+------+------+-------+------+------+  	

`MID` stands for "mutual information distance", which is an information-theoretic measure of association between two random variables. `Pperm` and `Pboot` corresponds to the p-values of the permutation and bootstrap tests used to assess the statistical significance of the mutual information distance (i.e. lower p-values signify that the association between two variables 
is not likely to be caused by the noise in the data).  


Advanced 
------------------------------------------------

The following is a list of all available arguments that can be passed into halla:: 

	usage: halla_cli.py -X inputfile1.tx -Y inputfile2.txt -q p-value -b output_for_associations.txt -c out_for_clusters.txt [-h] [-o output.txt] [-v verbosity] 

	Hierarchical All-against-All significance association testing.

	positional arguments:
	  input.txt      Tab-delimited text input file, one row per feature, one
			 column per measurement

	optional arguments:
	  -h, --help     show this help message and exit
	  -o output.txt  Optional output file for association significance tests
	  -p p_value     P-value for overall significance tests
	  -P p_mi        P-value for permutation equivalence of MI clusters
	  -b bootstraps  Number of bootstraps for significance testing
	  -v verbosity   Debug logging level; increase for greater verbosity
         -x  method that want to use, the default is HAllA-PCA-NMI

Mini-tutorial
---------------------------------------------------

Suppose you have two tab-delimited file containing the datasets you wish to run halla on. We will call this files `input1.txt` and `input2.txt`. We will call the output file `associations.txt`. In the root directory of halla, one can type::
	
	$ python halla_cli.py -X input1.txt -Y input2.txt -q .2 -x "HAllA-PCA-NMI" -b output.txt

To obtain the output in `out.txt`. 
	

Frequently Asked Questions 
==============================================

NB: Direct all questions to the halla-users google group. 


Functions
==============================================

.. automodule:: halla 
	:members:

.. automodule:: halla.stats
	:members:

.. automodule:: halla.distance
	:members:

.. automodule:: halla.parser
	:members:

.. automodule:: halla.logger
	:members:

.. automodule:: halla.test 
	:members:

.. automodule:: halla.hierarchy
	:members:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`