.. include:: global.rst

.. toctree::
	:hidden:

	global

=====
HAllA
=====

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

.. toctree::
	:maxdepth: 2
	
	components

.. automodule:: halla
	:members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
