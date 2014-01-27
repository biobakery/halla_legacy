=================================
notes for halla organization 
=================================
* halla - main script that launches process. mains class and also executable 
	* inherits from (1) distance, (2) hierarchy, (3) parser, (4) stats, (5) plots, (6) test


* distance - contains all notions of distance in nicely abstracted, instantiable objects 

* hierarchy - contains hiearchical clustering functions, general clustering

* logger - determines log behavior, non-implemented for now

* parser - contains all input/output parsing ability. You can launch this as an executable as well. 
    + output wrapper for output generation  
	+ basically anything that is reasonable to have as a multi-purpose parser tool wrapper for the lab. 
	+ basic executable wrangling inheritted from stats 
	+ inherit plotting capacities from plot 

* stats - contains abstracted statistics procedures 
	+ frequentist hypothesis testing 
	+ fdr correction 
	+ inference techniques 
	+ MCMC procedures  
	+ wrangling: dimensionality reduction, discretization, density estimation

* plot - unified namespace for plotting and graphics 
	+ Plan to support matplotlib and d3 

* test - (mostly) executable script for unit-testing halla
	+ dummy data generation inheritted from stats 

-------------------
FLOW 
-------------------

* parser -> halla -> parser 


