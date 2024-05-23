# [The BDCA for Value-at-Risk Constrained Portfolio Optimization](https://arxiv.org/abs/2402.09194)

This repository contains the python code that has been used to create the content in the paper 

[The Boosted Difference of Convex Functions Algorithm for Value-at-Risk Constrained Portfolio Optimization](https://arxiv.org/abs/2402.09194)

that is available as preprint on ArXiv and currently under review.

## Description of Modules
    
    All modules have a header at the beginning of the file that describes the purpose of the module in more detail and 
    shows a table of content. In this section we just give a brief overview over the modules in this repository.

    1. bdca.py
    In this module the Difference-of-Convex Functions Algorithm (DCA) is defined as a class. Similarly, the Boosted 
    Difference-of-Convex Functions Algorithm (BDCA) is defined as a class that inherits the DCA class.

    2. data_sets.py
    In this script all real-world data sets are loaded that will be relevant for the numerical experiments. For all
    data sets basic plausibility checks are performed.

    3. helper_funcs.py
    This module contains several helper functions that mostly support the functions defined in var_dc.py.
    
    4. var_dc.py
    In this script all functions are defined that are relevant for the Value-at-Risk constrained Portfolio Optimization.

    5. xprmnt_1.py
    This module contains the code that has been used to create the content of Section 5.1 (Experiment 1) in the 
    preprint of the paper.

    6. xprmnt_2.py
    This script contains the code that has been used to create the content of Section 5.2 (Experiment 2) in the 
    preprint of the paper. 

    7. xprmnt_funcs.py
    The code in this script mostly supports the content in xprmnt_1.py and xprmnt_2.py, but also contains functions 
    that are relevant to create the content of graphics and tables in the preprint.

## Notes
    
    The code has been developed under Python 3.10.13. The file requirements.txt contains information about all packages 
    with their respective versions.