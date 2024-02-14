#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2

"""
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
--
-- Value-at-Risk (VaR) as a Difference of Convex (DC) Functions.
--
-- Description: In this script several functions are defined that are used to represent the Value-at-Risk (VaR) at level
--              alpha as a difference of two convex functions. In this framework, the functions are defined based on
--              the content of Wozabal et al. (2010) and Wozabal et al. (2012). The papers can be found based on the
--              following links: https://doi.org/10.1080/02331931003700731 & https://doi.org/10.1007/s00291-010-0225-0.
--
-- Content:     0. Set-up
--                  0.0 Required Libraries
--                  0.1 Logging Set-up
--              1. Functions for VaR as DC Representation
--                  1.0 profit
--                  1.1 var
--                  1.2 cvar
--                  1.3 g_w
--                  1.4 h_w
--                  1.5 dh_w
--              2. Publisher's Imprint
--
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
Version  Date        Author    Major Changes
1.0      2023-01-13  MLT       Initialization
1.1      2023-02-09  MLT       Adjusted doc string and comments
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
from helper_funcs import *

######################################################################################################
# 0.1 Logging Set-up
######################################################################################################

# Define logger name
logger_var_dc = logging.getLogger(__name__)

# Set current logging level
logger_var_dc.setLevel(logging.INFO)

# Define the general format of the logs
formatter = logging.Formatter(fmt='%(asctime)s[%(levelname)s] %(name)s.%(funcName)s: %(message)s',
                              datefmt='%m/%d/%Y %I:%M:%S %p')
# Create logging file
file_handler = logging.FileHandler("log_file.log")

# Add file to logger
logger_var_dc.addHandler(file_handler)

# Set the format of the file
file_handler.setFormatter(formatter)

# Add logs to the console
stream_handler = logging.StreamHandler()

# Add it to the logger
logger_var_dc.addHandler(stream_handler)

########################################################################################################################
# 1. Functions for VaR as DC Representation
########################################################################################################################
######################################################################################################
# 1.0 profit
######################################################################################################


def profit(weights: np.ndarray, returns: np.ndarray, input_checks: bool = True) -> np.ndarray:
    """
    Calculates the profit of a portfolio for given asset weights and asset returns.

    :param weights: An array that contains the asset weights.
    :param returns: An array that contains the asset returns.
    :param input_checks: A boolean that indicates if the input check functions should be applied.
    :return: An array that contains the profits of the portfolio.
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Check function inputs

    if input_checks:
        # Weights
        check_weights(weights)
        # Returns
        check_returns(returns, weights)

    # ------------------------------------------------------------------------------------------------------------------
    # Compute output

    profits = returns @ weights

    # Debugging
    if logger_var_dc.isEnabledFor(logging.DEBUG):
        debug_path = "1.0 - profit -"
        logger_var_dc.debug(debug_path + 'np.min(profits): {0:.6f}'.format(np.min(profits)))
        logger_var_dc.debug(debug_path + 'np.max(profits): {0:.6f}'.format(np.max(profits)))
        logger_var_dc.debug(debug_path + 'np.mean(profits): {0:.6f}'.format(np.mean(profits)))
        logger_var_dc.debug(debug_path + 'np.median(profits): {0:.6f}'.format(np.median(profits)))

    return profits


######################################################################################################
# 1.1 var
######################################################################################################


def var(profits: np.ndarray, alpha: float, probs: np.ndarray = None, input_checks: bool = True) -> float:
    """
    Calculates the Value-At-Risk (VaR) of a portfolio for a given level alpha.

    :param profits: An array that contains all profits of the portfolio.
    :param alpha: A float in the interval (0, 1) that corresponds to the alpha level of the Value-at-Risk.
    :param probs: An array that contains the scenario probabilities according to the profits.
    :param input_checks: A boolean that indicates if the input check functions should be applied.
    :return: A float that corresponds to the VaR at level alpha.
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Check function inputs

    if input_checks:
        # Profits
        check_profits(profits)
        # Alpha
        check_alpha(alpha)
        # Probs
        check_probs(probs, profits)

    # ------------------------------------------------------------------------------------------------------------------
    # Input transformations

    # Get index from smallest to highest profit
    sorted_profits_index = np.argsort(profits, axis=0)

    # Sort the profits from smallest to largest
    sorted_profits = profits[sorted_profits_index]

    # Create equal scenario probabilities if no probabilities are provided
    probs = create_probs(probs, profits)

    # Sort the given scenario probabilities according to the sorted profits
    sorted_probs = probs[sorted_profits_index]

    # Compute the cumulative probabilities
    cum_probs = np.cumsum(sorted_probs)

    # Debugging
    debug_path = "1.1 - var - "
    if logger_var_dc.isEnabledFor(logging.DEBUG):
        logger_var_dc.debug(debug_path + 'np.min(cum_probs): {0:.6f}'.format(np.min(cum_probs)))
        logger_var_dc.debug(debug_path + 'np.max(cum_probs): {0:.6f}'.format(np.max(cum_probs)))
        logger_var_dc.debug(debug_path + 'len(cum_probs[cum_probs < alpha]): {}'.format(
                            len(cum_probs[cum_probs < alpha])))
        logger_var_dc.debug(debug_path + 'np.argmax(cum_probs[cum_probs < alpha]): {}'.format(
                            np.argmax(cum_probs[cum_probs < alpha])))

    # ------------------------------------------------------------------------------------------------------------------
    # Compute output

    # Compute VaR at level alpha as the scenario that is on the boundary
    var_alpha = sorted_profits[np.argmax(cum_probs[cum_probs < alpha]) + 1]

    # Check properties
    # check_var_alpha(var_alpha)

    # Debugging
    if logger_var_dc.isEnabledFor(logging.DEBUG):
        logger_var_dc.debug(debug_path + 'var_alpha: {0:.6f}'.format(var_alpha))

    return var_alpha


######################################################################################################
# 1.2 cvar
######################################################################################################


def cvar(profits: np.ndarray, alpha: float, probs: np.ndarray = None, input_checks: bool = True) -> float:
    """
    Calculates the Conditional-Value-At-Risk (CVaR) of a portfolio for a given level alpha.

    :param profits: An array that contains all profits of the portfolio.
    :param alpha: A float in the interval (0, 1) that corresponds to the alpha level of the Value-at-Risk.
    :param probs: An array that contains the scenario probabilities according to the profits.
    :param input_checks: A boolean that indicates if the input check functions should be applied.
    :return: A float that corresponds to the VaR at level alpha.
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Check function inputs

    if input_checks:
        # Profits
        check_profits(profits)
        # Alpha
        check_alpha(alpha)
        # Probs
        check_probs(probs, profits)

    # ------------------------------------------------------------------------------------------------------------------
    # Input transformations

    # Get index from smallest to highest profit
    sorted_profits_index = np.argsort(profits, axis=0)

    # Sort the profits from smallest to largest
    sorted_profits = profits[sorted_profits_index]

    # Create equal scenario probabilities if no probabilities are provided
    probs = create_probs(probs, profits)

    # Sort the given scenario probabilities according to the sorted profits
    sorted_probs = probs[sorted_profits_index]

    # Compute the cumulative probabilities
    cum_probs = np.cumsum(sorted_probs)

    # Compute epsilon based on the scenario that is on the boundary
    epsilon = alpha - np.max(cum_probs[cum_probs < alpha])

    # Check properties
    check_epsilon(epsilon)
    print(epsilon)

    # Compute VaR
    var_alpha = sorted_profits[np.argmax(cum_probs[cum_probs < alpha]) + 1]

    # Check properties
    # check_var_alpha(var_alpha)

    # Debugging
    debug_path = "1.2 - cvar - "
    if logger_var_dc.isEnabledFor(logging.DEBUG):
        logger_var_dc.debug(debug_path + 'epsilon: {0:.6f}'.format(epsilon))
        logger_var_dc.debug(debug_path + 'var_alpha: {0:.6f}'.format(var_alpha))
        logger_var_dc.debug(debug_path + 'np.min(cum_probs): {0:.6f}'.format(np.min(cum_probs)))
        logger_var_dc.debug(debug_path + 'np.max(cum_probs): {0:.6f}'.format(np.max(cum_probs)))
        logger_var_dc.debug(debug_path + 'len(cum_probs[cum_probs < alpha]): {}'.format(
                            len(cum_probs[cum_probs < alpha])))
        logger_var_dc.debug(debug_path + 'np.argmax(cum_probs[cum_probs < alpha]): '.format(
                            np.argmax(cum_probs[cum_probs < alpha])))

    # ------------------------------------------------------------------------------------------------------------------
    # Compute output

    # Compute a weighted sum of the profits and divide it by alpha
    sum_weighted_profits = (1 / alpha) * sorted_probs[cum_probs < alpha].T @ sorted_profits[cum_probs < alpha]

    # Weight the VaR with epsilon and divide it by alpha
    last_scenario = (1 / alpha) * epsilon * var_alpha

    # Compute CVaR at level alpha
    cvar_alpha = (sum_weighted_profits + last_scenario)[0]

    # Check properties
    # check_cvar_alpha(cvar_alpha)

    # Debugging
    if logger_var_dc.isEnabledFor(logging.DEBUG):
        logger_var_dc.debug(debug_path + 'sum_weighted_profits: {0:.6f}'.format(sum_weighted_profits))
        logger_var_dc.debug(debug_path + 'last_scenario: {0:.6f}'.format(last_scenario))
        logger_var_dc.debug(debug_path + 'cvar_alpha: {0:.6f}'.format(cvar_alpha))

    return cvar_alpha


######################################################################################################
# 1.3 g_w
######################################################################################################


def g_w(weights: np.ndarray, returns: np.ndarray, alpha: float, tau: int,
        var_threshold: float, probs: np.ndarray = None) -> float:
    """
    Calculates the output of the left convex function of the VaR as difference of convex formulation.

    :param weights: An array that contains the asset weights.
    :param returns: An array that contains the asset returns.
    :param alpha: A float in the interval (0, 1) that corresponds to the alpha level of the Value-at-Risk.
    :param tau: A non-negative integer that corresponds to the penalty parameter of the VaR constraint.
    :param var_threshold: A non-negative float that corresponds to the VaR at level alpha constraint threshold.
    :param probs: An array that contains the scenario probabilities according to the profits.
    :return: A float that corresponds to the value of the left convex function of the difference of convex.
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Check function inputs

    # Weights
    check_weights(weights)

    # Returns
    check_returns(returns, weights)

    # Profits
    profits = profit(weights, returns, input_checks=False)
    check_profits(profits)

    # Alpha
    check_alpha(alpha)

    # Tau
    check_tau(tau)

    # Var_threshold
    check_var_alpha(var_threshold)

    # Probs
    check_probs(probs, profits)

    # ------------------------------------------------------------------------------------------------------------------
    # Input transformations

    # Create equal scenario probabilities if no probabilities are provided
    probs = create_probs(probs, profits, input_checks=False)

    # Compute the gamma parameter
    gamma = get_gamma(profits, alpha, probs, input_checks=False)

    # Debugging
    debug_path = "1.3 - g_w - "
    logger_var_dc.debug(debug_path + 'gamma: {0:.6f}'.format(gamma))

    # ------------------------------------------------------------------------------------------------------------------
    # Compute output

    part_1 = -1 * probs.T @ profits

    part_2 = tau * np.maximum(-1 * (alpha / gamma) * cvar(profits, alpha, probs, input_checks=False) + var_threshold,
                              -1 * ((alpha - gamma) / gamma) * cvar(profits, alpha - gamma, probs, input_checks=False)
                              )

    result = (part_1 + part_2)[0]

    # Check properties
    if not isinstance(result, float):
        logger_var_dc.error(debug_path + 'result - type(result): {}'.format(type(result)))
        raise TypeError('The computed result is not a float.')

    # Debugging
    if logger_var_dc.isEnabledFor(logging.DEBUG):
        logger_var_dc.debug(debug_path + 'result: {0:.6f}'.format(result))
        logger_var_dc.debug(debug_path + 'part_1: {0:.6f}'.format(part_1))
        logger_var_dc.debug(debug_path + 'part_2: {0:.6f}'.format(part_2))

    return result


######################################################################################################
# 1.4 h_w
######################################################################################################


def h_w(weights: np.ndarray, returns: np.ndarray, alpha: float, tau: int, probs: np.ndarray = None) -> float:
    """
    Calculates the output of the right convex function of the VaR as difference of convex formulation.

    :param weights: An array that contains the asset weights.
    :param returns: An array that contains the asset returns.
    :param alpha: A float in the interval (0, 1) that corresponds to the alpha level of the Value-at-Risk.
    :param tau: A non-negative integer that corresponds to the penalty parameter of the VaR constraint.
    :param probs: An array that contains the scenario probabilities according to the profits.
    :return: A float that corresponds to the value of the left convex function of the difference of convex.
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Check function inputs

    # Weights
    check_weights(weights)

    # Returns
    check_returns(returns, weights)

    # Profits
    profits = profit(weights, returns, input_checks=False)
    check_profits(profits)

    # Alpha
    check_alpha(alpha)

    # Tau
    check_tau(tau)

    # Probs
    check_probs(probs, profits)

    # ------------------------------------------------------------------------------------------------------------------
    # Input transformations

    # Create equal scenario probabilities if no probabilities are provided
    create_probs(probs, profits, input_checks=False)

    # Compute the gamma parameter
    gamma = get_gamma(profits, alpha, probs, input_checks=False)

    # Check properties
    check_alpha(alpha - gamma)

    # Debugging
    debug_path = "1.4 - h_w - "
    logger_var_dc.debug(debug_path + 'gamma: {0:.6f}'.format(gamma))

    # ------------------------------------------------------------------------------------------------------------------
    # Compute output

    result = -1 * tau * ((alpha - gamma) / gamma) * cvar(profits, alpha - gamma, probs, input_checks=False)

    # Check properties
    if not isinstance(result, float):
        logger_var_dc.error(debug_path + 'result - type(result): {}'.format(type(result)))
        raise TypeError('The computed result is not a float.')

    # Debugging
    if logger_var_dc.isEnabledFor(logging.DEBUG):
        logger_var_dc.debug(debug_path + 'result: {0:.6f}'.format(result))

    return result


######################################################################################################
# 1.5 dh_w
######################################################################################################


def dh_w(weights: np.ndarray, returns: np.ndarray, alpha: float, tau: int, probs: np.ndarray = None) -> np.ndarray:
    """
    Calculates the (sub)gradient of the right convex function of the VaR as difference of convex formulation.

    :param weights: An array that contains the asset weights.
    :param returns: An array that contains the asset returns.
    :param alpha: A float in the interval (0, 1) that corresponds to the alpha level of the Value-at-Risk.
    :param tau: A non-negative integer that corresponds to the penalty parameter of the VaR constraint.
    :param probs: An array that contains the scenario probabilities according to the profits.
    :return: A float that corresponds to the value of the left convex function of the difference of convex.
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Check function inputs

    # Weights
    check_weights(weights)

    # Returns
    check_returns(returns, weights)

    # Profits
    profits = profit(weights, returns)
    check_profits(profits)

    # Alpha
    check_alpha(alpha)

    # Tau
    check_tau(tau)

    # Probs
    check_probs(probs, profits)

    # ------------------------------------------------------------------------------------------------------------------
    # Input transformations

    # Get index from smallest to highest profit
    sorted_profits_index = np.argsort(profits, axis=0)

    # Sort profits
    sorted_profits = profits[sorted_profits_index]

    # Sort the single returns according to the sorted profits
    sorted_returns = returns[sorted_profits_index, :].reshape((returns.shape[0], returns.shape[1]))

    # Create equal scenario probabilities if no probabilities are provided
    probs = create_probs(probs, profits, input_checks=False)

    # Sort the given scenario probabilities according to the sorted profits
    sorted_probs = probs[sorted_profits_index]

    # Computes the cumulative probabilities
    cum_probs = np.cumsum(sorted_probs)

    # Computes the remainder probability
    epsilon = alpha - np.max(cum_probs[cum_probs < alpha])

    # Check properties
    check_epsilon(epsilon)

    # Get gamma
    gamma = get_gamma(profits, alpha, probs, input_checks=False)

    # Debugging
    debug_path = "1.5 - dh_w - "
    if logger_var_dc.isEnabledFor(logging.DEBUG):
        logger_var_dc.debug(debug_path + 'epsilon: {0:.6f}'.format(epsilon))
        logger_var_dc.debug(debug_path + 'gamma: {0:.6f}'.format(gamma))

    # ------------------------------------------------------------------------------------------------------------------
    # Compute output

    # Compute VaR at level alpha as the scenario that is on the boundary
    var_alpha = sorted_profits[np.argmax(cum_probs[cum_probs < alpha]) + 1]

    # Check properties
    check_var_alpha(var_alpha)

    # Get the indices of all profits that are equivalent to var_alpha
    equal_indices = np.where(sorted_profits == var_alpha)

    # Compute subgradient if gradient of h_w does not exist, otherwise take the gradient
    if len(equal_indices) > 1:

        logger_var_dc.warning(debug_path + 'Gradient does not exist, subgradient is calculated.')

        # Compute random permutation of indices without replacement
        random_vec = np.random.choice(equal_indices, len(equal_indices), replace=False)

        # Index of first equal value to the left side of the VaR scenario
        s_1 = equal_indices[0]

        # Add all indices to the left of the equal values
        new_sorted_index = np.append(np.arange(s_1), random_vec)

        # Sort the returns and the probabilities according to the new ordering
        new_sorted_returns = sorted_returns[new_sorted_index, :]
        new_sorted_probs = sorted_probs[new_sorted_index]

        # Compute new cumulative probabilities
        new_cum_probs = np.cumsum(new_sorted_probs)

        # Get the index of new VaR scenario
        new_last_ret_index = np.argmax(new_cum_probs[new_cum_probs < alpha]) + 1

        # Get the returns of the VaR scenario
        new_var_alpha = new_sorted_returns[new_last_ret_index, :]

        # Weighted returns of scenarios to the left of the equal values
        part_1 = new_cum_probs[:s_1].T @ new_sorted_returns[:s_1, :]

        # Weighted returns of newly ordered scenarios
        part_2 = new_cum_probs[s_1:new_last_ret_index].T @ new_sorted_returns[s_1:new_last_ret_index, :]

        # Weighted returns of the VaR scenario
        part_3 = (alpha - gamma - new_cum_probs[new_last_ret_index - 1]) * new_var_alpha

        # Compute different components for the subgradient
        result = (-1) * (tau / gamma) * (part_1 + part_2 + part_3)

    else:
        # Sum of weighted returns
        weighted_returns = sorted_probs[cum_probs < alpha].T @ sorted_returns[cum_probs < alpha, :]

        # Get boundary scenario
        last_returns = sorted_returns[np.argmax(cum_probs[cum_probs < alpha]) + 1, :]

        # Compute the gradient by adding the different components
        result = -1 * (tau / gamma) * weighted_returns - (tau / gamma) * last_returns * (epsilon - gamma)

    # Check properties
    if not isinstance(result, np.ndarray):
        logger_var_dc.error(debug_path + 'result - type(result): {}'.format(type(result)))
        raise TypeError('The computed result is not a np.ndarray.')
    elif result.shape[1] != len(weights):
        logger_var_dc.error(debug_path + 'result - len(result): {} - len(weights): {}'.format(len(result),
                                                                                              len(weights)))
        raise TypeError("The computed 'result' has not the same length as the 'weights'.")

    return result


########################################################################################################################
# 2. Publisher's Imprint
########################################################################################################################

__author__ = "Marah-Lisanne Thormann"
__credits__ = ["Phan Vuong", "Alain Zemkoho", "David Wozabal", "Ronald Hochreiter", "Georg Ch. Pflug"]
__version__ = "1.2"
__maintainer__ = "Marah-Lisanne Thormann"
__email__ = "m.-l.thormann@soton.ac.uk"

########################################################################################################################
########################################################################################################################
