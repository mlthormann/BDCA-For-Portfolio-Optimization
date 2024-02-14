#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2

"""
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
--
-- Helper Functions for Value-at-Risk (VaR) Constrained Optimization Using the Boosted Difference of Convex Algorithm.
--
-- Description: In this script several helper functions are defined that are used to solve a VaR constrained portfolio
--              optimization problem with the Boosted Difference of Convex Algorithm (BDCA).
--
-- Content:     0. Set-up
--                  0.0 Required Libraries
--                  0.1 Logging Set-up
--              1. Helper Functions
--                  1.0 get_func_args
--                  1.1 create_args_dict
--                  1.2 check_profits
--                  1.3 check_probs
--                  1.4 check_weights
--                  1.5 check_returns
--                  1.6 check_alpha
--                  1.7 check_var_alpha
--                  1.8 check_cvar_alpha
--                  1.9 check_epsilon
--                  1.10 check_tau
--                  1.11 create_probs
--                  1.12 get_gamma
--                  1.13 constraint_eq
--              2. Publisher's Imprint
--
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
Version  Date        Author    Major Changes
1.0      2023-01-13  MLT       Initialization
1.1      2023-02-11  MLT       Updated logging, docstring & comments
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
import numpy as np
import types
import logging

# Functions
import inspect

######################################################################################################
# 0.1 Logging Set-up
######################################################################################################

# Define logger name
logger_help_func = logging.getLogger(__name__)

# Set current logging level
logger_help_func.setLevel(logging.INFO)

# Define the general format of the logs
formatter = logging.Formatter(fmt='%(asctime)s[%(levelname)s] %(name)s.%(funcName)s: %(message)s',
                              datefmt='%m/%d/%Y %I:%M:%S %p')

# Create logging file
file_handler = logging.FileHandler("log_file.log")

# Add file to logger
logger_help_func.addHandler(file_handler)

# Set the format of the file
file_handler.setFormatter(formatter)

# Add logs to the console
stream_handler = logging.StreamHandler()

# Add it to the logger
logger_help_func.addHandler(stream_handler)

########################################################################################################################
# 1. Helper Functions
########################################################################################################################
######################################################################################################
# 1.0 get_func_args
######################################################################################################


def get_func_args(func: callable) -> list:
    """
    Outputs all argument names of a function.

    :param func: A function from which the input names are required.
    :return: A list with all function argument names as single items.
    """
    # ------------------------------------------------------------------------------------------------------------------

    # Check if function input is valid
    if not isinstance(func, types.FunctionType):
        logger_help_func.error('1.0 - create_func_args - input_check - func: {}'.format(type(func)))
        raise TypeError("The provided input for the argument 'func' is not a function.")

    # ------------------------------------------------------------------------------------------------------------------

    return inspect.getfullargspec(func).args


######################################################################################################
# 1.1 create_args_dict
######################################################################################################


def create_args_dict(func: callable) -> dict:
    """
    Creates a dictionary having keys that correspond to the function arguments except the first.

    :param func: A function from which the input names are required.
    :return: A dictionary that contains all function argument names after the first one as keys.
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Check function inputs

    debug_path = "1.1 - create_func_args - "

    # Func
    if not isinstance(func, types.FunctionType):
        logger_help_func.error(debug_path + 'input_checks - func: {}'.format(type(func)))
        raise TypeError("The provided input for the argument 'func' is not a function.")

    # ------------------------------------------------------------------------------------------------------------------
    # Input transformations

    # Get all function arguments after the first one
    func_args = get_func_args(func)[1:]

    # Check number of input arguments
    if len(func_args) == 0:
        logger_help_func.error(debug_path + 'input_checks - func - # of input arguments: {}'.format(len(func_args)))
        raise ValueError("The provided 'func' does not have more than one input argument.")
    else:
        logger_help_func.debug(debug_path + 'input_checks - len(func_args): {}'.format(len(func_args)))

    # ------------------------------------------------------------------------------------------------------------------
    # Compute output

    # Create a dictionary to store the function arguments as keys
    args_dict = dict(zip(func_args, np.repeat(None, len(func_args))))

    return args_dict


######################################################################################################
# 1.2 check_profits
######################################################################################################


def check_profits(profits: np.ndarray) -> None:
    """
    Checks if the provided profits input fulfills all formal requirements.

    :param profits: An array that contains all profits of the portfolio.
    :return: None.
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Check function inputs

    debug_path = "1.2 - check_profits - "

    if not isinstance(profits, np.ndarray):
        logger_help_func.error(debug_path + 'input_check - profits: {}'.format(type(profits)))
        raise TypeError("The provided input for the argument 'profits' is not a np.ndarray.")
    elif len(profits) == 0:
        logger_help_func.error(debug_path + 'input_check - len(profits): {}'.format(len(profits)))
        raise ValueError("The provided input for the argument 'profits' does not contain any values.")
    elif profits.dtype != float:
        logger_help_func.error(debug_path + 'input_check - profits.dtype: {}'.format(profits.dtype))
        raise TypeError("The provided values for the argument 'profits' are not floats.")
    elif np.min(profits) < 0:
        logger_help_func.error(debug_path + 'input_check - np.min(profits): {}'.format(np.min(profits)))
        raise ValueError("The minimum value of 'profits' is smaller than zero.")

    # ------------------------------------------------------------------------------------------------------------------
    # Debugging
    if logger_help_func.isEnabledFor(logging.DEBUG):
        logger_help_func.debug(debug_path + 'np.min(profits): {0:.6f}'.format(np.min(profits)))
        logger_help_func.debug(debug_path + 'np.max(profits): {0:.6f}'.format(np.max(profits)))
        logger_help_func.debug(debug_path + 'np.mean(profits): {0:.6f}'.format(np.round(profits)))
        logger_help_func.debug(debug_path + 'np.median(profits): {0:.6f}'.format(np.median(profits)))

    # ------------------------------------------------------------------------------------------------------------------
    # Compute output

    return None


######################################################################################################
# 1.3 check_probs
######################################################################################################


def check_probs(probs: np.ndarray, profits: np.ndarray) -> None:
    """
    Checks if the provided probs input fulfills all formal requirements.

    :param probs:  An array that contains the scenario probabilities according to the profits.
    :param profits: An array that contains all profits of the portfolio.
    :return: None.
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Check function inputs

    debug_path = "1.3 - check_probs - "

    if not isinstance(probs, np.ndarray) and probs is not None:
        logger_help_func.error(debug_path + 'input_check - probs: {}'.format(type(probs)))
        raise TypeError("The provided input for the argument 'probs' is not a np.ndarray or None.")
    elif isinstance(probs, np.ndarray):
        if len(probs) != len(profits):
            logger_help_func.error(debug_path + 'input_check - probs - len(probs): {}'.format(len(probs),
                                   'len(profits): {}'.format(len(profits))))
            raise ValueError("The provided input for 'probs' has not the same length as 'profits'.")
        elif probs.dtype != float:
            logger_help_func.error(debug_path + 'input_check - probs.dtype: {}'.format(probs.dtype))
            raise TypeError("The provided values for the argument 'probs' are not floats.")
        elif np.round(np.sum(probs), 10) != 1:
            logger_help_func.error(debug_path + 'input_check - probs - np.sum(probs): {}'.format(np.sum(probs)))
            raise ValueError("The values of 'probs' do not sum up to one.")
        elif np.min(probs) < 0 or np.max(probs) > 1:
            logger_help_func.error(debug_path + 'input_check - probs - min: {}'.format(np.min(probs) +
                                   ' - max: {}'.format(np.max(probs))))
            raise ValueError("The values of 'probs' are not in the interval [0, 1].")

    # ------------------------------------------------------------------------------------------------------------------
    # Debugging

    if probs is not None and logger_help_func.isEnabledFor(logging.DEBUG):
        logger_help_func.debug(debug_path + 'np.min(probs): {0:.6f}'.format(np.min(probs)))
        logger_help_func.debug(debug_path + 'np.max(probs): {0:.6f}'.format(np.max(probs)))
        logger_help_func.debug(debug_path + 'np.mean(probs): {0:.6f}'.format(np.mean(probs)))
        logger_help_func.debug(debug_path + 'np.median(probs): {0:.6f}'.format(np.median(probs)))

    # ------------------------------------------------------------------------------------------------------------------
    # Compute output

    return None


######################################################################################################
# 1.4 check_weights
######################################################################################################


def check_weights(weights: np.ndarray) -> None:
    """
    Checks if the provided weights fulfills all formal requirements.

    :param weights: An array that contains the asset weights.
    :return: None.
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Check function inputs

    debug_path = "1.4 - check_weights - "

    # Weights
    if not isinstance(weights, np.ndarray):
        logger_help_func.error(debug_path + 'input_check - weights: '.format(type(weights)))
        raise TypeError("The provided input for the argument 'weights' is not a np.ndarray.")
    elif len(weights) == 0:
        logger_help_func.error(debug_path + 'input_check - len(weights): {}'.format(len(weights)))
        raise ValueError("The provided input for the argument 'weights' does not contain any values.")
    elif weights.dtype != float:
        logger_help_func.error(debug_path + 'input_check - weights.dtype: {}'.format(weights.dtype))
        raise TypeError("The provided values for the argument 'weights' are not floats.")
    elif np.round(np.sum(weights), 6) != 1:
        logger_help_func.warning(debug_path + 'input_check - weights - np.sum(weights): {0:.6f}'.format(
                                 np.sum(weights)) + " - np.min(weights): ".format(np.min(weights)) +
                                 " - np.max(weights): ".format(np.max(weights)))
    elif np.round(np.min(weights), 6) < 0 or np.round(np.max(weights), 6) > 1:
        logger_help_func.warning(debug_path + 'input_check - weights - min: '.format(np.min(weights)) +
                                 ' - max: '.format(np.max(weights)))

    # ------------------------------------------------------------------------------------------------------------------
    # Debugging
    if logger_help_func.isEnabledFor(logging.DEBUG):
        logger_help_func.debug(debug_path + 'np.min(weights): {0:.6f}'.format(np.min(weights)))
        logger_help_func.debug(debug_path + 'np.max(weights): {0:.6f}'.format(np.max(weights)))
        logger_help_func.debug(debug_path + 'np.mean(weights): {0:.6f}'.format(np.mean(weights)))
        logger_help_func.debug(debug_path + 'np.median(weights): {0:.6f}'.format(np.median(weights)))

    # ------------------------------------------------------------------------------------------------------------------
    # Compute output

    return None


######################################################################################################
# 1.5 check_returns
######################################################################################################


def check_returns(returns: np.ndarray, weights: np.ndarray) -> None:
    """
    Checks if the provided returns fulfills all formal requirements.

    :param returns: An array that contains the asset returns.
    :param weights: An array that contains the asset weights.
    :return: None.
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Check function inputs

    debug_path = "1.5 - check_returns - "

    if not isinstance(returns, np.ndarray):
        logger_help_func.error(debug_path + 'input_check - returns: {}'.format(type(returns)))
        raise TypeError("The provided input for the argument 'returns' is not a np.ndarray.")
    elif not np.any(returns):
        logger_help_func.error(debug_path + 'input_check - returns: {}'.format(returns.shape))
        raise ValueError("The provided input for the argument 'returns' is empty.")
    elif returns.shape[1] != len(weights):
        logger_help_func.error(debug_path + 'input_check - returns.shape[1]: {}'.format(returns.shape[1],
                               ' - len(weights): {}'.format(len(weights))))
        raise ValueError("The shape of 'returns' does not match the shape of 'weights'.")
    elif returns.dtype != float:
        logger_help_func.error(debug_path + 'input_check - returns.dtype: {}'.format(returns.dtype))
        raise TypeError("The provided values for the argument 'returns' are not floats.")

    # ------------------------------------------------------------------------------------------------------------------
    # Debugging
    if logger_help_func.isEnabledFor(logging.DEBUG):
        logger_help_func.debug(debug_path + 'np.min(returns): {0:.6f}'.format(np.min(returns)))
        logger_help_func.debug(debug_path + 'np.max(returns): {0:.6f}'.format(np.max(returns)))
        logger_help_func.debug(debug_path + 'np.mean(returns): {0:.6f}'.format(np.mean(returns)))
        logger_help_func.debug(debug_path + 'np.median(returns): {0:.6f}'.format(np.median(returns)))

    # ------------------------------------------------------------------------------------------------------------------
    # Compute output

    return None


######################################################################################################
# 1.6 check_alpha
######################################################################################################


def check_alpha(alpha: float) -> None:
    """
    Checks if the provided alpha input fulfills all formal requirements.

    :param alpha: A float in the interval (0, 1) that corresponds to the alpha level of the Value-at-Risk.
    :return: None.
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Check function inputs

    debug_path = "1.6 - check_alpha - "

    if not isinstance(alpha, float):
        logger_help_func.error(debug_path + 'input_check - alpha: {}'.format(type(alpha)))
        raise TypeError("The provided input for the argument 'alpha' is not a float.")
    elif alpha <= 0 or alpha >= 1:
        logger_help_func.error(debug_path + 'input_check - alpha - min-max: {}'.format(alpha))
        raise ValueError("The provided input for the argument 'alpha' is not in the interval (0, 1).")

    # ------------------------------------------------------------------------------------------------------------------
    # Debugging

    if logger_help_func.isEnabledFor(logging.DEBUG):
        logger_help_func.debug(debug_path + 'alpha: {0:.6f}'.format(alpha))

    # ------------------------------------------------------------------------------------------------------------------
    # Compute output

    return None


######################################################################################################
# 1.7 check_var_alpha
######################################################################################################


def check_var_alpha(var_alpha: float) -> None:
    """
    Checks if the provided VaR at level alpha fulfills all formal requirements.

    :param var_alpha: A non-negative float that corresponds to the VaR at level alpha.
    :return: None.
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Check function inputs

    debug_path = "1.7 - check_var_alpha - "

    if not isinstance(var_alpha, float):
        logger_help_func.error(debug_path + 'var_alpha: {}'.format(var_alpha))
        raise TypeError('The computed VaR at level alpha is not a float.')
    elif var_alpha < 0:
        logger_help_func.error(debug_path + 'var_alpha - negative: {}'.format(var_alpha))
        raise ValueError('The computed VaR at level alpha is negative.')

    # ------------------------------------------------------------------------------------------------------------------
    # Debugging

    if logger_help_func.isEnabledFor(logging.DEBUG):
        logger_help_func.debug(debug_path + 'var_alpha: {0:.6f}'.format(var_alpha))

    # ------------------------------------------------------------------------------------------------------------------
    # Compute output

    return None


######################################################################################################
# 1.8 check_cvar_alpha
######################################################################################################


def check_cvar_alpha(cvar_alpha: float) -> None:
    """
    Checks if the provided CVaR at level alpha fulfills all formal requirements.

    :param cvar_alpha: A non-negative float that corresponds to the VaR at level alpha.
    :return: None.
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Check function inputs

    debug_path = "1.8 - check_cvar_alpha - "

    if not isinstance(cvar_alpha, float):
        logger_help_func.error(debug_path + 'cvar_alpha: {}'.format(cvar_alpha))
        raise TypeError('The computed CVaR at level alpha is not a float.')
    elif cvar_alpha < 0:
        logger_help_func.error(debug_path + 'cvar_alpha - negative: {}'.format(cvar_alpha))
        raise ValueError('The computed CVaR at level alpha is negative.')

    # ------------------------------------------------------------------------------------------------------------------
    # Debugging

    if logger_help_func.isEnabledFor(logging.DEBUG):
        logger_help_func.debug(debug_path + 'cvar_alpha: {0:.6f}'.format(cvar_alpha))

    # ------------------------------------------------------------------------------------------------------------------
    # Compute output

    return None


######################################################################################################
# 1.9 check_epsilon
######################################################################################################


def check_epsilon(epsilon: float) -> None:
    """
    Checks if the provided epsilon fulfills all formal requirements.

    :param epsilon: A non-negative float that corresponds to the epsilon parameter of the CVaR at level alpha.
    :return: None.
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Check function inputs

    debug_path = "1.9 - check_epsilon - "

    if not isinstance(epsilon, float):
        logger_help_func.error(debug_path + 'epsilon: {}'.format(epsilon))
        raise TypeError('The computed epsilon is not a float.')
    elif epsilon < 0:
        logger_help_func.error(debug_path + 'epsilon - negative: {}'.format(epsilon))
        raise ValueError('The computed epsilon is negative.')

    # ------------------------------------------------------------------------------------------------------------------
    # Debugging

    if logger_help_func.isEnabledFor(logging.DEBUG):
        logger_help_func.debug(debug_path + 'epsilon: {0:.6f}'.format(epsilon))

    # ------------------------------------------------------------------------------------------------------------------
    # Compute output

    return None


######################################################################################################
# 1.10 check_tau
######################################################################################################


def check_tau(tau: int) -> None:
    """
    Checks if the provided tau fulfills all formal requirements.

    :param tau: A non-negative number that corresponds to the penalty parameter of the VaR constraint.
    :return: None.
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Check function inputs

    debug_path = "1.10 - check_tau - "

    if not isinstance(tau, int):
        logger_help_func.error(debug_path + 'tau: {}'.format(tau))
        raise TypeError('The computed tau is not a number.')
    elif tau < 0:
        logger_help_func.error(debug_path + 'tau - negative: {}'.format(tau))
        raise ValueError('The computed tau is negative.')

    # ------------------------------------------------------------------------------------------------------------------
    # Debugging

    if logger_help_func.isEnabledFor(logging.DEBUG):
        logger_help_func.debug(debug_path + 'tau: {0:.6f}'.format(tau))

    # ------------------------------------------------------------------------------------------------------------------
    # Compute output

    return None


######################################################################################################
# 1.11 create_probs
######################################################################################################


def create_probs(probs: np.ndarray, profits: np.ndarray, input_checks: bool = True) -> np.ndarray:
    """
    Creates equal scenario probabilities for the case they were not provided.

    :param probs: An array that contains the scenario probabilities according to the profits.
    :param profits: An array that contains all profits of the portfolio.
    :param input_checks: A boolean that indicates if the input check functions should be applied.
    :return: An array containing the scenario probabilities.
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Check function inputs

    if input_checks:
        # Profits
        check_profits(profits)
        # Probs
        check_probs(probs, profits)

    # ------------------------------------------------------------------------------------------------------------------
    # Compute output

    # Create equal scenario probabilities if no probabilities are provided
    if probs is None:
        # No. of scenarios
        s = profits.shape[0]
        probs = np.repeat(1 / s, repeats=s).reshape((s, 1))

    # Debugging
    if logger_help_func.isEnabledFor(logging.DEBUG):
        debug_path = "1.11 - create_probs - "
        logger_help_func.debug(debug_path + 'np.min(probs): {0:.6f}'.format(np.min(probs)))
        logger_help_func.debug(debug_path + 'np.max(probs): {0:.6f}'.format(np.max(probs)))
        logger_help_func.debug(debug_path + 'np.mean(probs): {0:.6f}'.format(np.mean(probs)))
        logger_help_func.debug(debug_path + 'np.median(probs): {0:.6f}'.format(np.median(probs)))

    return probs


######################################################################################################
# 1.12 get_gamma
######################################################################################################


def get_gamma(profits: np.ndarray, alpha: float, probs: np.ndarray = None, input_checks: bool = True) -> float:
    """
    Calculates the gamma parameter as the minimum of 0.5 * epsilon and 0.01 * the smallest scenario probability.

    :param profits: An array that contains all profits of the portfolio.
    :param alpha: A float in the interval (0, 1) that corresponds to the alpha level of the Value-at-Risk.
    :param probs: An array that contains the scenario probabilities according to the profits.
    :param input_checks: A boolean that indicates if the input check functions should be applied.
    :return: A float that corresponds to the gamma parameter.
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Check function inputs

    if input_checks:
        # Profit
        check_profits(profits)
        # Alpha
        check_alpha(alpha)
        # Probs
        check_probs(probs, profits)

    # ------------------------------------------------------------------------------------------------------------------
    # Input transformations

    # Get index from smallest to the highest profit
    sorted_profits_index = np.argsort(profits, axis=0)

    # Create equal scenario probabilities if no probabilities are provided
    probs = create_probs(probs, profits)

    # Sort the given scenario probabilities according to the sorted profits
    sorted_probs = probs[sorted_profits_index]

    # Compute the cumulative probabilities
    cum_probs = np.cumsum(sorted_probs)

    # Compute epsilon based on the scenario that is on the boundary
    epsilon = alpha - np.max(cum_probs[cum_probs < alpha])

    debug_path = "1.12 - get_gamma - "

    # Debugging
    if logger_help_func.isEnabledFor(logging.DEBUG):
        logger_help_func.debug(debug_path + 'np.min(cum_probs): {0:.6f}'.format(np.min(cum_probs)))
        logger_help_func.debug(debug_path + 'np.max(cum_probs): {0:.6f}'.format(np.max(cum_probs)))
        logger_help_func.debug(debug_path + 'len(cum_probs[cum_probs < alpha]): {}'.format(
                               len(cum_probs[cum_probs < alpha])))
        logger_help_func.debug(debug_path + 'np.argmax(cum_probs[cum_probs < alpha]): {}'.format(
                               np.argmax(cum_probs[cum_probs < alpha])))
        logger_help_func.debug(debug_path + 'epsilon: {0:.6f}'.format(epsilon))
        logger_help_func.debug(debug_path + 'sorted_probs.min(): {0:.6f}'.format(sorted_probs.min()))

    # ------------------------------------------------------------------------------------------------------------------
    # Compute output

    gamma = np.minimum(sorted_probs.min() * 0.01, epsilon * 0.5)

    # Check properties
    if gamma <= 0 or gamma >= epsilon:
        logger_help_func.error(debug_path + 'gamma - out-of-range: {}'.format(gamma))
        raise ValueError('The computed gamma is not in the interval (0, epsilon).')
    elif not isinstance(gamma, float):
        logger_help_func.error(debug_path + 'gamma - type(gamma): {}'.format(type(gamma)))
        raise TypeError('The computed gamma is not a float.')

    # Debugging
    if logger_help_func.isEnabledFor(logging.DEBUG):
        logger_help_func.debug(debug_path + 'gamma: {0:.6f}'.format(gamma))

    return gamma


######################################################################################################
# 1.13 constraint_eq
######################################################################################################


def constraint_eq(w):
    """
    Defines the required equality constraint for the weight vector.

    :param w: An array that contains the asset weights.
    :return: A constraint that ensures that all weights sum up to one.
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Compute output

    a = np.ones(w.shape)
    b = 1
    constraint_val = np.matmul(a, w.T) - b

    return constraint_val


########################################################################################################################
# 2. Publisher's Imprint
########################################################################################################################

__author__ = "Marah-Lisanne Thormann"
__credits__ = ["Phan Vuong", "Alain Zemkoho"]
__version__ = "1.2"
__maintainer__ = "Marah-Lisanne Thormann"
__email__ = "m.-l.thormann@soton.ac.uk"

########################################################################################################################
########################################################################################################################
