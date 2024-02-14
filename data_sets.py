#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2

"""
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
--
-- Real World Data Sets for Portfolio Selection.
--
-- Description: In this script all real world data sets are loaded that will be relevant for the numerical experiments.
--              Note that the data has been prepared and provided by Bruni et al. (2016). The paper can be found based
--              on the following link: https://doi.org/10.1016/j.dib.2016.06.031.
--
-- Content:     0. Set-up
--                  0.0 Required Libraries
--                  0.1 Logging Set-up
--              2. Dow Jones
--              3. FF49 Industries
--              4. FTSE 100
--              5. NASDAQ 100
--              6. NASDAQ Comp
--              7. Publisher's Imprint
--
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
Version  Date        Author    Major Changes
1.0      2023-02-02  MLT       Initialization
1.1      2023-02-09  MLT       Removal of NASDAQ Comp due to implausible values
1.2      2023-03-22  MLT       Finalized documentation
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
import pandas as pd
import numpy as np
import logging

######################################################################################################
# 0.1 Logging Set-up
######################################################################################################

# Define logger name
logger_data = logging.getLogger(__name__)

# Set current logging level
logger_data.setLevel(logging.INFO)

# Define the general format of the logs
formatter = logging.Formatter(fmt='%(asctime)s[%(levelname)s] %(name)s.%(funcName)s: %(message)s',
                              datefmt='%m/%d/%Y %I:%M:%S %p')

# Create logging file
file_handler = logging.FileHandler("log_file.log")

# Add file to logger
logger_data.addHandler(file_handler)

# Set the format of the file
file_handler.setFormatter(formatter)

# Add logs to the console
stream_handler = logging.StreamHandler()

# Add it to the logger
logger_data.addHandler(stream_handler)

########################################################################################################################
# 1. Dow Jones
########################################################################################################################

# Read in data set
dow_jones_data = 1 + pd.read_excel("Datasets/DowJones/DowJones.xlsx", header=None)

# Some additional logging information
logger_data.info('Rows of Dow Jones Data: {}'.format(dow_jones_data.shape[0]))
logger_data.info('Columns of Dow Jones Data: {}'.format(dow_jones_data.shape[1]))

# Check data for missing values
if dow_jones_data.isnull().values.any():
    logger_data.warning('Dow Jones data contains missing values.')

# Check for unlogical values
if np.any(dow_jones_data < 0):
    logger_data.warning('Dow Jones data contains values smaller than -1.')
if np.any(dow_jones_data > 2.5):
    logger_data.warning('Dow Jones data contains values bigger than 1.')

# Transform to numpy array
dow_jones_data = dow_jones_data.to_numpy()

########################################################################################################################
# 2. FF49 Industries
########################################################################################################################

# Read in data set
ff49_data = 1 + pd.read_excel("Datasets/FF49Industries/FF49Industries.xlsx", header=None)

# Some additional logging information
logger_data.info('Rows of FF49 Industries Data: {}'.format(ff49_data.shape[0]))
logger_data.info('Columns of FF49 Industries Data: {}'.format(ff49_data.shape[1]))

# Check data for missing values
if ff49_data.isnull().values.any():
    logger_data.warning('FF49 Industries data contains missing values.')

# Check for unlogical values
if np.any(ff49_data < 0):
    logger_data.warning('FF49 Industries data contains values smaller than -1.')
if np.any(ff49_data > 2.5):
    logger_data.warning('FF49 Industries data contains values bigger than 1.')

# Transform to numpy array
ff49_data = ff49_data.to_numpy()

########################################################################################################################
# 3. FTSE 100
########################################################################################################################

# Read in data set
ftse100_data = 1 + pd.read_excel("Datasets/FTSE100/FTSE100.xlsx", header=None)

# Some additional logging information
logger_data.info('Rows of FTSE 100 Data: {}'.format(ftse100_data.shape[0]))
logger_data.info('Columns of FTSE 100 Data: {}'.format(ftse100_data.shape[1]))

# Check data for missing values
if ftse100_data.isnull().values.any():
    logger_data.warning('FTSE 100 data contains missing values.')

# Check for unlogical values
if np.any(ftse100_data < 0):
    logger_data.warning('FTSE 100 data contains values smaller than -1.')
if np.any(ftse100_data > 2.5):
    logger_data.warning('FTSE 100 data contains values bigger than 1.')

# Transform to numpy array
ftse100_data = ftse100_data.to_numpy()

########################################################################################################################
# 4. NASDAQ 100
########################################################################################################################

# Read in data set
nasdaq100_data = 1 + pd.read_excel("Datasets/NASDAQ100/NASDAQ100.xlsx", header=None)

# Some additional logging information
logger_data.info('Rows of NASDAQ 100 Data: {}'.format(nasdaq100_data.shape[0]))
logger_data.info('Columns of NASDAQ 100 Data: {}'.format(nasdaq100_data.shape[1]))

# Check data for missing values
if nasdaq100_data.isnull().values.any():
    logger_data.warning('NASDAQ 100 data contains missing values.')

# Check for unlogical values
if np.any(nasdaq100_data < 0):
    logger_data.warning('NASDAQ 100 data contains values smaller than -1.')
if np.any(nasdaq100_data > 2.5):
    logger_data.warning('NASDAQ 100 data contains values bigger than 1.')


# Transform to numpy array
nasdaq100_data = nasdaq100_data.to_numpy()

########################################################################################################################
# 5. NASDAQ Comp
########################################################################################################################

# Read in data set
# nasdaq_comp_data = 1 + pd.read_excel("Datasets/NASDAQComp/NASDAQComp.xlsx", header=None)
#
# # Some additional logging information
# logger_data.info('Rows of NASDAQ Comp Data: {}'.format(nasdaq_comp_data.shape[0]))
# logger_data.info('Columns of NASDAQ Comp Data: {}'.format(nasdaq_comp_data.shape[1]))
#
# # Check data for missing values
# if nasdaq_comp_data.isnull().values.any():
#     logger_data.warning('NASDAQ Comp data contains missing values.')
#
# # Check for unlogical values
# if np.any(nasdaq_comp_data < 0):
#     logger_data.warning('NASDAQ Comp data contains values smaller than -1.')
#
# if np.any(nasdaq_comp_data > 2.5):
#     logger_data.warning('NASDAQ Comp data contains values bigger than 1.')
#
# # Transform to numpy array
# nasdaq_comp_data = nasdaq_comp_data.to_numpy()

# Note: This data set is not used as there are some implausible values.

########################################################################################################################
# 6. Publisher's Imprint
########################################################################################################################

__author__ = "Marah-Lisanne Thormann"
__credits__ = ["Phan Vuong", "Alain Zemkoho", "Renato Bruni", "Francesco Cesarone", "Andrea Scozzari", "Fabio Tardella"]
__version__ = "1.2"
__maintainer__ = "Marah-Lisanne Thormann"
__email__ = "m.-l.thormann@soton.ac.uk"

########################################################################################################################
########################################################################################################################

