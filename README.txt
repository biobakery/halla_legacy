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
 Yo Sup Moon, Curtis Huttenhower

Google Group
 halla-users: https://groups.google.com/forum/#!forum/halla-users

License
 MIT License

URL
 http://huttenhower.sph.harvard.edu/halla

Citation
 Yo Sup Moon, Curtis Huttenhower, "Retrieving Signal from Noise in Big Data: An Information-Theoretic Approach to Hierarchical Exploratory Data Analysis" (In Preparation)

.. toctree::
   :maxdepth: 2


Chapter 0 Getting Started
============================================ 

Operating System  
--------------------------------------------

* Supported 
	* Ubuntu Linux (>= 12.04) 
	* Mac OS X (>= 10.7)

* Unsupported 
	* Windows (>= XP) 

Dependencies 
--------------------------------------------

* Required
	* Python (>= 2.7)
	* Numpy (>= 1.7.1)
	* Scipy (>= 0.12) 
	* Scikit-learn (>=0.13)  
	* rpy (>=2.0)
	* sampledoc-master

* Recommended Tools for documentation 
	* Docutils
	* itex2MML


Getting HAllA
--------------------------------------------

HAllA can be downloaded from its bitbucket repository: http://bitbucket.org/chuttenh/halla.


Chapter 1 Basics 
============================================

Introduction
--------------------------------------------

HAllA: is a programmatic tool for performing multiple association testing between two or more heterogeneous datasets, each containing a mixture of discrete, binary, or continuous data. HAllA is a robust and efficient alternative to traditional all-against-all association testing of variables. Its robustness relies on the usage of mutual information-based measures to calculate the degree to which two variables are related. Mutual-information is well-suited to serve as an all-purpose measure since it is well-behaved even when comparing two variables of different data types. Its efficiency relies on a hierarchical clustering scheme to reduce the number of tests necessary to discover interesting associations in datasets that contain potentially millions of genotypic and phenotypic data. In a traditional all-against-all association-testing scheme, the number of pairwise tests scale quadratically with the number of features in the data (O(N^2)). The sheer number of association tests dramatically reduces the power of standard hypothesis tests to discover relationships among variables. We introduce a hierarchical hypothesis-testing scheme to perform tiered testing on clusters of data to reduce computational time for comparisons. Hierarchical false discovery rate correction is implemented to curb discoveries of associations due to noise in the data. 

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

	usage: halla.py [-h] [-o output.txt] [-p p_value] [-P p_mi] [-b bootstraps] [-v verbosity] [input.txt]

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

Mini-tutorial
---------------------------------------------------

Suppose you have a tab-delimited file containing the dataset you wish to run halla on. We will call this file `in.txt`. We will call the output file `out.txt`. In the root directory of halla, one can type::
	
	$ python halla.py in.txt > out.txt 

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


License
==============================================

This software is licensed under the MIT license.

Copyright (c) 2013 Yo Sup Moon, Curtis Huttenhower

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


