#HAllA: Hierarchical All-against-All association testing #
HAllA is an acronym for Hierarchical All-against-All association testing and is designed as a command-line tool to find associations in high-dimensional, heterogeneous datasets. 

**If you use the HAllA software, please cite our manuscript:**

Gholamali Rahnavard, Eric A. Franzosa, Yo Sup Moon,  Lauren J. McIver, Emma Schwager, George Weingart, Xochitl C. Morgan, Levi Waldron, Curtis Huttenhower, **"High-sensitivity pattern discovery in high-dimensional heterogeneous datasets"** (Submitted) 

HAllA is an end-to-end statistical method for Hierarchical All-against-All discovery of significant relationships among data features with high power.  HAllA is robust to data type, operating both on continuous and categorical values, and works well both on homogeneous datasets (where all measurements are of the same type, e.g., gene expression microarrays) and on heterogeneous data (containing measurements with different units or types, e.g., patient clinical metadata).  Finally, it is also aware of multiple inputs, multiple output problems, in which data might contain two (or more) distinct subsets sharing an index (e.g., clinical metadata,
genotypes, microarrays, and microbiomes all drawn from the same subjects).  In all of these cases, HAllA will identify which pairs of features (genes, microbes, loci, etc.) share statistically significant co-variation, without getting tripped up by high-dimensionality.

For additional information, please see the [HAllA User Manual](http://huttenhower.sph.harvard.edu/halla/manual).

For a quick demo and installation, please see the [HAllA Toturial](https://bitbucket.org/biobakery/biobakery/wiki/halla).

If you use this tool, the included scripts, or any related code in your work,
please let us know, sign up for the HAllA Users Group [HAllA Users Google Group](https://groups.google.com/forum/#!forum/halla-users), and pass along any issues or feedback.
