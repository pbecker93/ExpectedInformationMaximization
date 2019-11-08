cmake_minimum_required(VERSION 2.8.12)
project(CppEIM)

add_definitions(-std=c++11)

# required libraries
find_package(OpenMP REQUIRED)
find_library(gfortran REQUIRED)
find_library(openblas REQUIRED)
find_library(lapack REQUIRED)
### FOR INSTALLATION CHANGE STUFF HERE! ###################################
#Add pybind11 subfolder (under CppEIM) or change command if pybind11 is installed on system
add_subdirectory(pybind11)

#Adapt these so that armadillo and nlopt are found
include_directories(~/nfs/eim_lib/armadillo/include ~/nfs/eim_lib/nlopt/include)
link_directories(~/nfs/eim_lib/nlopt/lib)
### END OF STUFF THE NEEDS TO BE CHANGED ##################################

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

include_directories(include)

#Distributions
add_library(distributionlib SHARED
        src/distributions/Categorical.cpp
        src/distributions/Gaussian.cpp
        src/distributions/GMM.cpp)
target_link_libraries(distributionlib openblas lapack gfortran nlopt)

#Regression
add_library(regressionlib SHARED
        src/regression/Regression.cpp)
target_link_libraries(regressionlib openblas lapack gfortran)

# information theoretic policy search
add_library(itpslib SHARED
        src/itps/MoreGaussian.cpp
        src/itps/RepsCategorical.cpp)
target_link_libraries(itpslib openblas lapack gfortran nlopt)

#ModelLearner
add_library(modellearnerlib SHARED
        src/model_learner/GMMLearner.cpp)

target_link_libraries(modellearnerlib openblas lapack gfortran nlopt)

#Python Bindings
pybind11_add_module(cpp_distributions src/distributions/PyDistributions.cpp)
target_link_libraries(cpp_distributions PUBLIC openblas lapack distributionlib)

pybind11_add_module(cpp_regression src/regression/PyRegression.cpp)
target_link_libraries(cpp_regression PUBLIC openblas lapack regressionlib)

pybind11_add_module(cpp_model_learner src/model_learner/PyModelLearner.cpp)
target_link_libraries(cpp_model_learner PUBLIC openblas lapack modellearnerlib regressionlib distributionlib itpslib)


