Collage
=======

Collage is a gene prioritization approach that relies on data fusion by
collective matrix factorization. Its aim is to identify most promising
candidate genes that may be associated with the phenotype of interest.

This repository contains supplementary code and data for *Gene prioritization 
by compressive data fusion and chaining* by Zitnik et al. 


About Collage
-------------

In everyday life, we make decisions by considering all the available information, 
and often find that inclusion of even seemingly circumstantial evidence provides an advantage. 

Collage can prioritize genes from a large collection of heterogeneous data. It considers data 
sets of various association levels with the prediction task, utilizes collective matrix 
factorization to compress the data, and chaining to relate different object types contained in 
a data compendium. Collage prioritizes genes based on their similarity to several seed genes. 

We tested Collage by prioritizing bacterial response genes in Dictyostelium. Using 4 seed genes 
and 14 data sets, Collage proposed 8 candidate genes that were readily validated as necessary 
for the response of Dictyostelium to Gram-negative bacteria. 


System requirements
-------------------

Running this package will require a machine with 8 GB of RAM. Collective
matrix factorization is a compute intensive procedure. Much of the
analysis we mention in the paper was performed in parallel on a compute 
cluster.


Dependencies
------------
The required dependencies to run the software are `Numpy >= 1.8` and `SciPy >= 0.10`.

This code was last tested using Python 2.7.6.


Usage
-----

[collage.py](collage.py) - Demonstrates usage of Collage on data
    used to identify bacterial response genes in Dictyostelium.
    
See also [scikit-fusion](http://github.com/marinkaz/scikit-fusion), our module
for data fusion using collective latent factor models. 

   
Citing
------

    @article{Zitnik2015,
      title     = {Gene prioritization by compressive data fusion and chaining},
      author    = {{\v{Z}}itnik, Marinka and Nam, Edward A and Dinh, Christopher and
                    Kuspa, Adam and Shaulsky, Gad and Zupan, Bla{\v{z}}},
      journal   = {PLoS Computational Biology},
      volume    = {},
      number    = {},
      pages     = {},
      year      = {2015},
      publisher = {PLOS}
    }
    
    
License
-------
Collage is licensed under the GPLv2.
