#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2

"""
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
--
-- Second Experiment of Portfolio Selection Case Study.
--
-- Description: In this script the code of the second experiment of the case study is defined. In the framework of the
--              experiment a fixed cost approach was implemented to compare the results of DCA and the BDCA.
--              Note that the simulation has been performed in batches using the IRIDIS High Performance Computing
--              Facility  of the University of Southampton and therefore batch variables have to be defined in a
--              separate batch file. If one wants to run the code on a PC, the code has to be adjusted accordingly.
--
-- Content:     0. Set-up
--                  0.0 Required Libraries
--                  0.1 Batch Variables
--                  0.2 Logging Set-up
--              1. Numerical Experiment
--                  1.0 Get Data Set
--                  1.1 Results File (Path)
--                  1.2 Experiment Set-Up
--                  1.3 Run Experiment
--              2. Publisher's Imprint
--
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
Version  Date        Author    Major Changes
1.0      2023-03-01  MLT       Initialization
1.1      2024-05-22  MLT       Smaller corrections of comments
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
"""
########################################################################################################################
# 0. Set-ups
########################################################################################################################
######################################################################################################
# 0.0 Required Libraries
######################################################################################################

# Generally required
import sys
import os

# Algorithms and Grid Search
from xprmnt_funcs import fixed_cost_search

# Data sets
from data_sets import *

# Parallel computing
from multiprocessing import Pool
from itertools import product

######################################################################################################
# 0.1 Batch Variables
######################################################################################################

# VaR constraint [0.958, ..., 0.968]
input_var = float(sys.argv[1])

# Input data set [dj, ff49, ftse100, nasdaq100]
input_data = sys.argv[2]

# Start repetition [1, ..., 499]
input_brep = sys.argv[3]

# End repetition [2, ..., 500]
input_erep = sys.argv[4]

######################################################################################################
# 0.2 Logging Set-up
######################################################################################################

# Define logger name
logger = logging.getLogger(__name__)

# Set current logging level
logger.setLevel(logging.INFO)

# Define the general format of the logs
formatter = logging.Formatter(fmt='%(asctime)s[%(levelname)s] %(name)s.%(funcName)s: %(message)s',
                              datefmt='%m/%d/%Y %I:%M:%S %p')

# Create logging file
file_handler = logging.FileHandler("log_file.log")

# Add file to logger
logger.addHandler(file_handler)

# Set the format of the file
file_handler.setFormatter(formatter)

# Add logs to the console
# stream_handler = logging.StreamHandler()

# Add it to the logger
# logger.addHandler(stream_handler)

########################################################################################################################
# 1. Numerical Experiment
########################################################################################################################
######################################################################################################
# 1.0 Get Data Set
######################################################################################################

if input_data == "dj":
    data_path = "Dow_Jones"
    data_set = dow_jones_data
elif input_data == "ff49":
    data_path = "FF49_Industries"
    data_set = ff49_data
elif input_data == "ftse100":
    data_path = "FTSE100"
    data_set = ftse100_data
else:
    data_path = "NASDAQ100"
    data_set = nasdaq100_data

######################################################################################################
# 1.1 Results File (Path)
######################################################################################################

# Different components of file path
path_1 = f"Simulation/{data_path}/sim_{data_path}_{input_var}_fixed_cost_approach_"
path_2 = f"brep_{input_brep}_erep_{input_erep}.csv"

# Combine components to file path
results_file = path_1 + path_2

# Create file in folder if it does not exist so far
if not os.path.isfile(results_file):
    with open(results_file, "w") as f:
        f.write("data,algorithm,repetition,cpu_time,var,alpha_var,decs,stop_mecha,tau,alpha_diri,alpha_bdca,"
                "beta,iter,max_iter,cnt_ill_cond,cnt_line_search,cnt_non_descent_dc,avg_lambda_k,median_lambda_k,"
                "max_lambda_k,min_lambda_k,func_v,var_model,objective,sum_weights,min_weight,max_weight,median_weight,"
                "avg_weight,nonzero_weights,cnt_weights\n")

######################################################################################################
# 1.2 Experiment Set-Up
######################################################################################################


def main():
    # Define all components for the grid search
    file = [results_file]

    # Define beginning of file name based on the data set
    data_name = [data_path]

    # VaR constraint[0.958, 0.96, 0.962, 0.964, 0.965, 0.966]
    my_var = [input_var]

    # Alpha-Level of the VaR constraint
    alpha_var = [0.05]

    # Penalty parameter
    tau = [90]

    # Dirichlet Settings (1 -> 0.001)
    alpha_d1 = [0.001]

    # BDCA iterations
    max_iter_bdca = [1000]

    # Maximum number of iterations for the DCA
    max_iter_dca = [10000]

    # BDCA parameters
    alpha_bdca = [0.001]
    beta = [0.5]

    # Repetitions [1, ..., 100]
    rep = np.arange(int(input_brep), int(input_erep) + 1)

    # Data set [dow_jones_data, ff49_data, ftse100_data, nasdaq100_data]
    return_dat = [data_set]

    # Scenario Probabilities (if None, all scenarios have equal probabilities)
    probs = [None]

    # Number of decimals for stopping criterion (tolerance)
    decs = [7]

    # Run 40 jobs in parallel to perform experiment
    with Pool(40) as pool:
        result = pool.starmap(fixed_cost_search, product(file, data_name, my_var, alpha_var, tau,
                                                         alpha_d1, alpha_bdca, beta, rep, return_dat,
                                                         probs, decs, max_iter_bdca, max_iter_dca))

    return result


######################################################################################################
# 1.3 Run Experiment
######################################################################################################

if __name__ == "__main__":
    result = main()

########################################################################################################################
# 2. Publisher's Imprint
########################################################################################################################

__author__ = "Marah-Lisanne Thormann"
__credits__ = ["Phan Vuong", "Alain Zemkoho"]
__version__ = "1.1"
__email__ = "m.-l.thormann@soton.ac.uk"

########################################################################################################################
########################################################################################################################

