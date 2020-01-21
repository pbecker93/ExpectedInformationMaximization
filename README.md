# Expected Information Maximization
Code for "Expected Information Maximization: Using the I-Projection for Mixture Density Estimation" - published at ICLR 2020, see https://openreview.net/forum?id=ByglLlHFDS


### Code Structure
Parts of the Code, especially for the Gaussian Mixture Model case are written in C++ for performance reasons. The main
loop is in python and  interfaces are provided using pybind11. An installation guide for the C++ part can be found below.

Python part (EIM)
- data: Functionality to generate/read the required data and providing it in an unified format
- distributions: implementation of the Deep Mixture of Expert Model (using tensorflow)
- EIM: Actual Implementation of EIM for GMMs and Mixtures of Experts and the density ratio estimator
- recoding: utility for recording results and real time plotting
- util: miscellaneous utility functionality 
- scripts to run experiments



C++ part (CppEIM)
- distributions: implementation of the GMM model
- itps: implements the individual updates for components and coefficients using the (slightly modified) MORE algorithm
- model_learner: implementation of the M-Step for GMMs
- regression: Simple regression module for the surrogates required by MORE
  

### Requirements Python 
Tested with Python 3.6.8. A list of all required packages, including version numbers can be found in req.txt and can
be installed with
 
```pip install -r req.txt```

### Installing the C++ part

#####Install required packages and libraries 
The following libraries are required: gcc, openmp, gfortran, openblas, lapack cmake. Install them using your package manager

(On Ubuntu: run

```sudo apt-get install gcc gfortran libopenblas-dev liblapack-dev cmake ```

) 
     
##### Install NLOPT (tested with version 2.6)
Follow the download and installation instruction here: https://nlopt.readthedocs.io/en/latest/
You do not need to install using sudo but can put the library anywhere.

Change the 'include_directories' and 'link_directories' statements in CppEIM/CMakeLists.txt such that they point to the
NLOPT headers and libraries. 

##### Install Armadillo (tested with version 9.8000)

Download Armadillo (http://arma.sourceforge.net/download.html) unpack and run ./configure . You do not need to build 
Armadillo

Change the 'include_directories' and statement in CppEIM/CMakeLists.txt such that it points to the
armadillo headers. 


##### Download pybind11 (tested with version 2.4)

Download from here https://github.com/pybind/pybind11/releases, unpack, rename to pybind11, and place the pybind11 
folder under CppEIM (you can place it somewhere else but then need to adapt the CMakeLists.txt such that pybind11 is found)

##### Install CppEIM package 
go to CppEIM and run 

```sudo python3 setup.py install``` or ```python3 setup.py install --user``

