# Expected Information Maximization
Code for "Expected Information Maximization: Using the I-Projection for Mixture Density Estimation" - published at ICLR 2020, see https://openreview.net/forum?id=ByglLlHFDS

Update 23.04.2020: 
We decided to remove the C++ parts from the code and provide a full python implementation to increase 
understandability and usability of the implementation. Note that this implementation is a bit slower than the original,
especially the component update which is no longer parallelized. The original implementation with the C++ parts can 
still be found in the branch "cppEIM". We used this older version for all experiments reported in the paper.   


### Code Structure

- data: Functionality to generate/read the required data and providing it in an unified format
- distributions: implementation of GMMs and the Deep Mixture of Expert Model (using tensorflow)
- itps: implements the individual updates for components and coefficients using the (slightly modified) MORE algorithm
- EIM: Actual Implementation of EIM for GMMs and Mixtures of Experts and the density ratio estimator
- recoding: utility for recording results and real time plotting
- util: miscellaneous utility functionality 
- scripts to run experiments

### Requirements Python 
Tested with Python 3.6.9. A list of all required packages, including version numbers can be found in req.txt and can
be installed with
 
```pip install -r req.txt```



