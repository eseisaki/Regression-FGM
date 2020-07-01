# Linear-Regression-FGM

## Abstract
We introduce  *Functional Geometric Monitoring* a substantial theoretical
 and practical improvement on the core ideas of Geometric Monitoring. 
 Instead of a binary constraint, each site is provided with a complex, non-linear 
 function, which, applied to its local summary vector, projects to it to a
  real number.
 
 ___
 
 ## Simulator
 In order to evaluate this distributed algorithm we used Distributed-Data
 -Stream-Simulator, which is implemented in python.
 
 ## Requirements

 Implemented with python: version 3.6 (or higher)


 Some not built-in python modules are required. To install run:

    conda env create --file envname.yml
 
 ## How to Run
 In order to run an example algorithm , simply run:
 
    python main.py

 ## How to Test
 For testing pytest module was used.To run all test do the following:

    pytest -v
    
 In order to run a specific group of tests you have to run the following:

    pytest -k {keyword} -v
 
 ## Documentation
 
 For the documentation Sphinx 2.3.1. software was used.

 To create documentation just run:

    cd docs
    make clean html

 Then open the index.html with a browser.
