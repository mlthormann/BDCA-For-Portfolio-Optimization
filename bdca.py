#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2

"""
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
--
-- The Boosted Difference of Convex Algorithm (BDCA).
--
-- Description: In this script the Boosted Difference of Convex Algorithm (BDCA) will be defined as an extension of the
--              Difference of Convex Algorithm (DCA). The objects are defined based on the content of Tao and El
--              Bernoussi (1988), Aragon Artacho et al. (2018), Aragon Artacho and Vuong (2020) and Aragon Artacho et
--              al. (2022). The papers can be found based on the following links:
--              https://doi.org/10.1007/978-3-0348-9297-1_18, https://doi.org/10.1007/s10107-017-1180-1,
--              https://doi.org/10.1137/18M123339X, https://arxiv.org/abs/1908.01138.
--
--
-- Content:     0. Set-up
--                  0.0 Required Libraries
--                  0.1 Logging Set-up
--              2. DCA Object
--              3. BDCA Object
--              4. Publisher's Imprint
--
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
Version  Date        Author    Major Changes
1.0      2023-01-13  MLT       Initialization
1.1      2023-02-09  MLT       Smaller adjustments and more comments
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
import matplotlib.pyplot as plt
from var_dc import *

# Creation of (B)DCA Object
from scipy.optimize import minimize
from scipy.optimize import Bounds

######################################################################################################
# 0.1 Logging Set-up
######################################################################################################

# Define logger name
logger_bdca = logging.getLogger(__name__)

# Set current logging level
logger_bdca.setLevel(logging.INFO)

# Define the general format of the logs
formatter = logging.Formatter(fmt='%(asctime)s[%(levelname)s] %(name)s.%(funcName)s: %(message)s',
                              datefmt='%m/%d/%Y %I:%M:%S %p')

# Create logging file
file_handler = logging.FileHandler("log_file.log")

# Add file to logger
logger_bdca.addHandler(file_handler)

# Set the format of the file
file_handler.setFormatter(formatter)

# Add logs to the console
stream_handler = logging.StreamHandler()

# Add it to the logger
logger_bdca.addHandler(stream_handler)

########################################################################################################################
# 1. DCA Object
########################################################################################################################


class dca:
    """
    The Difference-of-Convex Algorithm (DCA) initialization.

    :param g_x: The left convex function of the DC objective function.
    :param h_x: The right convex function of the DC objective function.
    :param dh_x: The linear approximation of the right convex function of the DC objective function.
    :return: None.
    """
    def __init__(self, g_x: callable, h_x: callable, dh_x: callable) -> None:
        self.g_x = g_x
        self.g_x_params = create_args_dict(g_x)
        self.h_x = h_x
        self.h_x_params = create_args_dict(h_x)
        self.dh_x = dh_x
        self.dh_x_params = create_args_dict(dh_x)
        self.x_k = np.nan
        self.k = np.nan
        self.track_f_proxy = np.nan
        self.track_f = np.nan
        self.ill_cond_cnt = 0

    def update_key_values(self, func_args_dict: dict, args_dict: dict, input_checks: bool = True) -> None:
        """
        Updates the input values for a given dictionary of functions arguments.

        :param func_args_dict: A dictionary that contains all function arguments as keys.
        :param args_dict: A dictionary that contains updated function arguments.
        :param input_checks: A boolean that indicates if the input check functions should be applied.
        :return: None.
        """
        # --------------------------------------------------------------------------------------------------------------
        # Check function inputs

        if input_checks:
            # Func_args_dict
            if not isinstance(func_args_dict, dict):
                logger_bdca.error('1.0 - dca - update_key_values - input_check - func_args_dict: {}'.format(
                                  type(func_args_dict)))
                raise TypeError("The provided input for the argument 'func_args_dict' is not a dictionary.")
            # Args_dict
            if not isinstance(args_dict, dict):
                logger_bdca.error('1.0 - dca - update_key_values - input_check - args_dict: {}'.format(type(args_dict)))
                raise TypeError("The provided input for the argument 'args_dict' is not a dictionary.")

        # --------------------------------------------------------------------------------------------------------------
        # Compute output

        # Iterate over dictionary keys to update values
        for key in func_args_dict:
            try:
                func_args_dict.update({key: args_dict[key]})
            except KeyError:
                continue

        return None

    def f_x(self, x: np.ndarray, args_dict: dict, input_checks: bool = True) -> float:
        """
        Computes the difference-of-convex objective function value.

        :param x: An array that contains the function input values.
        :param args_dict: A dictionary that contains all required function inputs for g(x) and h(x).
        :param input_checks: A boolean that indicates if the input check functions should be applied.
        :return: A float that corresponds to the objective function value.
        """
        # --------------------------------------------------------------------------------------------------------------
        # Check function inputs
        
        if input_checks:
            debug_path = "1.0 - dca - f_x - input_check - "
            # X
            if not isinstance(x, np.ndarray):
                logger_bdca.error(debug_path + 'type(x): {}'.format(type(x)))
                raise TypeError("The provided input for the argument 'x' is not a np.ndarray.")
    
            # Args_dict
            if not isinstance(args_dict, dict):
                logger_bdca.error(debug_path + 'args_dict: {}'.format(type(args_dict)))
                raise TypeError("The provided input for the argument 'args_dict' is not a dictionary.")

        # --------------------------------------------------------------------------------------------------------------
        # Input transformations

        self.update_key_values(self.g_x_params, args_dict)
        self.update_key_values(self.h_x_params, args_dict)

        # --------------------------------------------------------------------------------------------------------------
        # Compute output

        # Debugging
        if logger_var_dc.isEnabledFor(logging.DEBUG):
            logger_bdca.debug('1.0 - dca - f(x): {}'.format(self.g_x(x, **self.g_x_params) -
                                                            self.h_x(x, **self.h_x_params)))

        return self.g_x(x, **self.g_x_params) - self.h_x(x, **self.h_x_params)

    def f_x_proxy(self, x: np.ndarray, x_0: np.ndarray, args_dict: dict, input_checks: bool = True) -> float:
        """
        Computes the approximated difference-of-convex objective function value.

        :param x: An array that contains the function input values.
        :param x_0: An array that contains the function input values from the previous iteration.
        :param args_dict: A dictionary that contains all required function inputs for g(x) and h(x).
        :param input_checks: A boolean that indicates if the input check functions should be applied.
        :return: A float that corresponds to the approximated objective function value.
        """
        # --------------------------------------------------------------------------------------------------------------
        # Check function inputs
        
        if input_checks:
            debug_path = "1.0 - dca - f_x_proxy - input_check - "
            # X
            if not isinstance(x, np.ndarray):
                logger_bdca.error(debug_path + 'type(x): {}'.format(type(x)))
                raise TypeError("The provided input for the argument 'x' is not a np.ndarray.")
    
            # X_0
            if not isinstance(x_0, np.ndarray):
                logger_bdca.error(debug_path + 'type(x_0): {}'.format(type(x_0)))
                raise TypeError("The provided input for the argument 'x_0' is not a np.ndarray.")
            elif x.shape != x_0.shape:
                logger_bdca.error(debug_path + 'x.shape != x_0.shape: {}, {}'.format(x.shape, x_0.shape))
                raise TypeError("The shapes of 'x' and 'x_0' are not the same.")
    
            # Args_dict
            if not isinstance(args_dict, dict):
                logger_bdca.error(debug_path + 'args_dict: {}'.format(type(args_dict)))
                raise TypeError("The provided input for the argument 'args_dict' is not a dictionary.")

        # --------------------------------------------------------------------------------------------------------------
        # Input transformations

        self.update_key_values(self.g_x_params, args_dict)
        self.update_key_values(self.dh_x_params, args_dict)

        # --------------------------------------------------------------------------------------------------------------
        # Compute output

        # Debugging
        if logger_var_dc.isEnabledFor(logging.DEBUG):
            logger_bdca.debug('1.0 - dca - f_proxy(x): {}'.format(
                              self.g_x(x, **self.g_x_params) - np.dot(self.dh_x(x_0, **self.dh_x_params), x)))

        return self.g_x(x, **self.g_x_params) - np.dot(self.dh_x(x_0, **self.dh_x_params), x)

    def objective_function_value_plot(self) -> plt.plot:
        """
        Plots the development of the objective function value with respect to the iterations.

        :return: A figure that shows the objective function value on the y-axis and the iterations on the x-axis.
        """
        # --------------------------------------------------------------------------------------------------------------
        # Check function inputs

        if len(self.track_f) < 2:
            raise ValueError("Not enough function values to compute convergence plot.")

        # --------------------------------------------------------------------------------------------------------------
        # Compute output

        # Set up figure
        fig, ax = plt.subplots()
        # Add objective function value plot
        ax.plot(self.track_f)
        # Adjust y-label
        ax.set_ylabel(r"$\phi$(w_{k})")
        # Adjust x-label
        ax.set_xlabel("Iteration")

        return fig

    def get_convergence_constant(self) -> np.ndarray:
        """
        Computes the convergence constant q of the equation: ||f(x_{k + 1}) - f(x^*)|| <= q ||f(x_{k}) - f(x^*)||
        at each iteration. It is assumed that the last objective function value corresponds to the optimal value.

        :return: A figure that shows the constant q on the y-axis and the iterations on the x-axis.
        """
        # --------------------------------------------------------------------------------------------------------------
        # Check function inputs

        if len(self.track_f) < 2:
            raise ValueError("Not enough function values to compute convergence plot.")

        # --------------------------------------------------------------------------------------------------------------
        # Input transformations

        # Sequence that excludes the last value in the objective function vector: ||f(x_{k}) - f(x^*)||
        lower_part = np.abs(self.track_f - self.track_f[-1])[:(len(self.track_f) - 1)]
        # Sequence that excludes the first value in the objective function vector: ||f(x_{k + 1}) - f(x^*)||
        upper_part = np.abs(self.track_f - self.track_f[-1])[1:]

        # Transform values where we would divide by 0
        upper_part[lower_part == 0] = 0
        lower_part[lower_part == 0] = 1000

        # --------------------------------------------------------------------------------------------------------------
        # Compute output

        # q = ||f(x_{k + 1}) - f(x^*)|| / ||f(x_{k}) - f(x^*)||
        constant_seq = np.divide(upper_part, lower_part)

        return constant_seq

    def linear_convergence_plot(self) -> plt.plot:
        """
        Plots the development of the convergence constant q with respect to the iterations.

        :return:A figure that shows the constant q on the y-axis and the iterations on the x-axis.
        """
        # --------------------------------------------------------------------------------------------------------------
        # Check function inputs
        
        # --------------------------------------------------------------------------------------------------------------
        # Input transformations

        # Get sequence of convergence constants
        constant_seq = self.get_convergence_constant()

        # Get maximal q
        max_q = np.round(np.max(constant_seq), 6)

        # --------------------------------------------------------------------------------------------------------------
        # Compute output

        fig, ax = plt.subplots()
        # Add sequence of convergence constants
        ax.plot(constant_seq)
        # Adjust y-label
        ax.set_ylabel("q")
        # Adjust x-label
        ax.set_xlabel("Iteration")
        # Add horizontal line at 1
        ax.axhline(1, color='r')
        # Add text with the maximal convergence constant
        ax.text(0.1, 0.1, " max q: " + str(max_q))

        return fig

    def dca_optimizer(self,
                      x_0: np.ndarray,
                      k_max: int,
                      args_dict: dict,
                      stop_crit: str,
                      decs: int = 7,
                      bdca_value: float = None,
                      verbose: bool = False, **kwargs) -> None:
        """
        Performs the DC optimization, i.e. the minimization of the objective function for given inputs.

        :param x_0: An array that contains the starting values for the objective function.
        :param k_max: A non-negative integer that corresponds to the maximal number of iterations.
        :param args_dict: A dictionary that contains all relevant function inputs for f(x).
        :param stop_crit: A string that indicates the stopping criteria.
        :param decs: An integer that indicates the relevant decimals for the "func" and "vec" stopping criteria.
        :param bdca_value: A float that corresponds to the objective function value that the BDCA found.
        :param verbose: A boolean that indicates if the iteration number should be printed to the console.
        :param kwargs: Keyword arguments that are relevant for the optimization algorithm.
        :return: None. 
        """
        # --------------------------------------------------------------------------------------------------------------
        # Check function inputs

        debug_path = "1.0 - dca - dca_optimizer - "
        
        # X_0
        if not isinstance(x_0, np.ndarray):
            logger_bdca.error(debug_path + 'input_check - type(x_0): {}'.format(type(x_0)))
            raise TypeError("The provided input for the argument 'x_0' is not a np.ndarray.")
        
        # K_max
        if not isinstance(k_max, int):
            logger_bdca.error(debug_path + 'input_check - type(k_max): {}'.format(type(k_max)))
            raise TypeError("The provided input for the argument 'k_max' is not a integer.")
        elif k_max < 1:
            logger_bdca.error(debug_path + 'input_check - x_0 < 1: {}'.format(x_0))
            raise ValueError("The provided input for the argument 'k_max' is smaller than 1.")
        
        # Args_dict
        if not isinstance(args_dict, dict):
            logger_bdca.error(debug_path + 'input_check - args_dict: {}'.format(type(args_dict)))
            raise TypeError("The provided input for the argument 'args_dict' is not a dictionary.")
        
        # Stop_crit
        if not isinstance(stop_crit, str):
            logger_bdca.error(debug_path + 'input_check - stop_crit: {}'.format(type(stop_crit)))
            raise TypeError("The provided input for the argument 'stop_crit' is not a string.")
        elif stop_crit not in ['iter', 'func_abs', "func_rel", 'vec_abs', "vec_rel", "bdca"]:
            logger_bdca.error(debug_path + 'input_check - stop_crit: {}'.format(stop_crit))
            raise ValueError("The provided input 'stop_crit' is not valid. The options are: 'iter', 'func' and 'vec'.")
        
        # Decs
        if not isinstance(decs, int):
            logger_bdca.error(debug_path + 'input_check - type(decs): {}'.format(type(decs)))
            raise TypeError("The provided input for the argument 'decs is not a integer.")
        elif k_max < 0:
            logger_bdca.error(debug_path + 'input_check - decs < 0: {}'.format(decs))
            raise ValueError("The provided input for the argument 'decs' is smaller than 0.")
        
        # Verbose
        if not isinstance(verbose, bool):
            logger_bdca.error(debug_path + 'input_check - type(verbose): {}'.format(type(verbose)))
            raise TypeError("The provided input for the argument 'verbose' is not a boolean.")
            
        # --------------------------------------------------------------------------------------------------------------
        # Compute output
        
        # Current iteration
        k = 0
        # Initial input vectors
        x_k = x_0
        cond = True
        
        # Compute initial (approximated) objective function value and store it
        self.track_f_proxy = np.array(self.f_x_proxy(x_k, x_k, args_dict))
        self.track_f = np.array(self.f_x(x_k, args_dict))

        # Debugging
        if logger_bdca.isEnabledFor(logging.DEBUG):
            logger_bdca.debug(debug_path + 'initial f(x): {}'.format(self.track_f[k]))
            logger_bdca.debug(debug_path + 'initial f_proxy(x): {}'.format(self.track_f_proxy[k]))

        while cond:

            # ----------------------------------------------------------------------------------------------------------
            # Solve convex subproblem

            # DCA step
            res = minimize(self.f_x_proxy,
                           method='SLSQP',
                           x0=x_k,
                           args=(x_k, args_dict),
                           constraints=({'type': 'eq', 'fun': constraint_eq}),
                           bounds=Bounds(np.repeat(0, len(x_k)), np.repeat(1, len(x_k))),
                           **kwargs
                           )
            
            # Update iteration
            k += 1

            # ----------------------------------------------------------------------------------------------------------
            # Check new solution

            # Update the track of the (approximated) objective function value
            self.track_f_proxy = np.append(self.track_f_proxy, self.f_x_proxy(res.x, x_k, args_dict))
            self.track_f = np.append(self.track_f, self.f_x(res.x, args_dict))

            # Debugging
            if logger_bdca.isEnabledFor(logging.DEBUG):
                logger_bdca.debug(debug_path + 'iteration: {}'.format(k))
                logger_bdca.debug(debug_path + 'min(res.x): {}'.format(np.min(res.x)))
                logger_bdca.debug(debug_path + 'max(res.x): {}'.format(np.max(res.x)))
                logger_bdca.debug(debug_path + 'mean(res.x): {}'.format(np.mean(res.x)))
                logger_bdca.debug(debug_path + 'median(res.x): {}'.format(np.median(res.x)))
                logger_bdca.debug(debug_path + 'sum(res.x): {}'.format(np.sum(res.x)))
                logger_bdca.debug(debug_path + 'current f_proxy(x): {}'.format(self.track_f_proxy[k]))
                logger_bdca.debug(debug_path + 'current f(x): {}'.format(self.track_f[k]))

            # Check if current solution is feasible
            curr_solu = res.x
            n_feasible_solu = np.round(np.sum(curr_solu), decs) != 1 or any(0 > curr_solu) or any(curr_solu > 1)

            # Check for ill conditioning, this means that:
            #       - the weights do not sum up to one
            #       - the old objective function value is smaller than the new one
            if n_feasible_solu or np.round(self.track_f[k] - self.track_f[k - 1], decs) > 0:
                self.ill_cond_cnt += 1

            # ----------------------------------------------------------------------------------------------------------
            # Check stopping criterion

            # Based on iterations
            if stop_crit == "iter":
                stop_crit_value = k
                cond = k < k_max
            # Based on the absolute difference between old and new objective function value
            elif stop_crit == "func_abs":
                stop_crit_value = np.round(self.track_f[k] - self.track_f[k - 1], decs)
                cond = (stop_crit_value != 0 or n_feasible_solu) and k < k_max
            # Based on the relative difference between old and new objective function value
            elif stop_crit == "func_rel":
                stop_crit_value = np.abs(np.round((self.track_f[k] - self.track_f[k - 1]) / self.track_f[k - 1], decs))
                cond = (stop_crit_value < (10**(-decs)) or n_feasible_solu) and k < k_max
            # Based on the absolute difference between old and new solution
            elif stop_crit == "vec_abs":
                stop_crit_value = np.round(res.x - x_k, decs)
                cond = (np.all(stop_crit_value == 0) or n_feasible_solu) and k < k_max
            # Based on the relative difference between old and new solution
            elif stop_crit == "vec_rel":
                stop_crit_value = np.abs(np.round((res.x - x_k) / (x_k + 10**(-(decs + 5))), decs))
                cond = (np.all(stop_crit_value < (10**(-decs))) or n_feasible_solu) and k < k_max
            elif stop_crit == "bdca":
                stop_crit_value = self.track_f[k] > bdca_value
                cond = (stop_crit_value or n_feasible_solu) and k < k_max
            else:
                return logger_bdca.error("Please specify a valid 'stop_crit'.")
                
            # Debugging
            if logger_bdca.isEnabledFor(logging.DEBUG):
                logger_bdca.debug(debug_path + 'stop_crit - value: {}'.format(stop_crit_value))
            
            # Update solution
            x_k = res.x

        if verbose:
            print(k)

        # If the maximum iteration is reached print a warning
        if k == k_max and stop_crit != "iter" and stop_crit != "bdca":
            logger_bdca.warning("The maximum number of iterations has been reached."
                                "The algorithm likely did not converge.")

        # Update internal variables
        self.k = k
        self.x_k = x_k

        return None


########################################################################################################################
# 2. BDCA Object
########################################################################################################################


class bdca(dca):
    """
    The Boosted Difference-of-Convex Algorithm (BDCA) initialization.

    :param g_x: The left convex function of the DC objective function.
    :param h_x: The right convex function of the DC objective function.
    :param dh_x: The linear approximation of the right convex function of the DC objective function.
    :return: None.
    """
    def __init__(self, g_x, h_x, dh_x) -> None:
        # Set bdca up as extension of dca
        dca.__init__(self, g_x, h_x, dh_x)
        # Counter for how many times the line search is performed
        self.cnt_line_search = 0
        # Tracker for all lambda_k's
        self.track_lambdas = np.array([])
        # Counter for how many times the line search was not a descent direction
        self.cnt_non_descent_dc = 0

    def bdca_optimizer(self,
                       x_0: np.ndarray,
                       k_max: int,
                       args_dict: dict,
                       stop_crit: str,
                       alpha_2: float = 0.5,
                       lambda_bar: float = None,
                       beta: float = 0.1,
                       decs: int = 10,
                       verbose: bool = False, **kwargs) -> None:
        """
        Performs the BDC optimization, i.e. minimization of the objective function for given inputs.

        :param x_0: An array that contains the starting values for the objective function.
        :param k_max: A non-negative integer that corresponds to the maximal number of iterations.
        :param args_dict: A dictionary that contains all relevant function inputs for f(x).
        :param stop_crit: A string that indicates the stopping criteria.
        :param alpha_2: A non-negative float that is used as constant in the backtracking procedure.
        :param lambda_bar: A float that serves as starting point for lambda_k of the backtracking procedure.
        :param beta: A float in the interval (0, 1) that scales down lambda_k in the backtracking procedure.
        :param decs: An integer that indicates the relevant decimals for the "func" and "vec" stopping criteria.
        :param verbose: A boolean that indicates if the iteration number should be printed to the console.
        :param kwargs: Keyword arguments that are relevant for the optimization algorithm.
        :return: None.
        """
        # --------------------------------------------------------------------------------------------------------------
        # Check function inputs

        # Set up debug path
        debug_path = "2.0 - bdca - bdca_optimizer - "

        # X_0
        if not isinstance(x_0, np.ndarray):
            logger_bdca.error(debug_path + 'input_check - type(x_0): {}'.format(type(x_0)))
            raise TypeError("The provided input for the argument 'x_0' is not a np.ndarray.")

        # K_max
        if not isinstance(k_max, int):
            logger_bdca.error(debug_path + 'input_check - type(k_max): {}'.format(type(k_max)))
            raise TypeError("The provided input for the argument 'k_max' is not a integer.")
        elif k_max < 1:
            logger_bdca.error(debug_path + 'input_check - x_0 < 1: {}'.format(x_0))
            raise ValueError("The provided input for the argument 'k_max' is smaller than 1.")

        # Args_dict
        if not isinstance(args_dict, dict):
            logger_bdca.error(debug_path + 'input_check - args_dict: {}'.format(type(args_dict)))
            raise TypeError("The provided input for the argument 'args_dict' is not a dictionary.")

        # Stop_crit
        if not isinstance(stop_crit, str):
            logger_bdca.error(debug_path + 'input_check - stop_crit: {}'.format(type(stop_crit)))
            raise TypeError("The provided input for the argument 'stop_crit' is not a string.")
        elif stop_crit not in ['iter', 'func_abs', "func_rel", 'vec_abs', "vec_rel"]:
            logger_bdca.error(debug_path + 'input_check - stop_crit: {}'.format(stop_crit))
            raise ValueError("The provided input 'stop_crit' is not valid. The options are: 'iter', 'func' and 'vec'.")
        
        # Alpha_2
        if not (isinstance(alpha_2, float) or isinstance(alpha_2, int)):
            logger_bdca.error(debug_path + 'input_check - type(alpha_2): {}'.format(type(alpha_2)))
            raise TypeError("The provided input for the argument 'alpha_2' is not a number.")
        elif alpha_2 <= 0:
            logger_bdca.error(debug_path + 'input_check - alpha_2 <= 0: {}'.format(alpha_2))
            raise ValueError("The provided input for the argument 'alpha_2' has to be bigger than 0.")
        
        # Lambda_bar
        if not (isinstance(lambda_bar, float) or isinstance(lambda_bar, int) or lambda_bar is None):
            logger_bdca.error(debug_path + 'input_check - type(lambda_bar): {}'.format(type(lambda_bar)))
            raise TypeError("The provided input for the argument 'lambda_bar' is not a number.")

        if lambda_bar is not None:
            if lambda_bar <= 0:
                logger_bdca.error(debug_path + 'input_check - lambda_bar <= 0: {}'.format(lambda_bar))
                raise ValueError("The provided input for the argument 'lambda_bar' has to be bigger than 0.")
        
        # Beta
        if not isinstance(beta, float):
            logger_bdca.error(debug_path + 'input_check - type(beta): {}'.format(type(beta)))
            raise TypeError("The provided input for the argument 'beta' is not a float.")
        elif beta <= 0 or beta >= 1:
            logger_bdca.error(debug_path + 'input_check - beta <= 0 or beta >= 1: {}'.format(beta))
            raise ValueError("The provided input for the argument 'beta' is not in the interval (0, 1).")

        # Decs
        if not isinstance(decs, int):
            logger_bdca.error(debug_path + 'input_check - type(decs): {}'.format(type(decs)))
            raise TypeError("The provided input for the argument 'decs is not a integer.")
        elif k_max < 0:
            logger_bdca.error(debug_path + 'input_check - decs < 0: {}'.format(decs))
            raise ValueError("The provided input for the argument 'decs' is smaller than 0.")

        # Verbose
        if not isinstance(verbose, bool):
            logger_bdca.error(debug_path + 'input_check - type(verbose): {}'.format(type(verbose)))
            raise TypeError("The provided input for the argument 'verbose' is not a boolean.")
        
        # --------------------------------------------------------------------------------------------------------------
        # Compute output

        # Current iteration
        k = 0

        # Initial input vector
        x_k = x_0

        # Initialize while loop condition
        cond = True

        # Compute initial (approximated) objective function value and store it
        self.track_f_proxy = np.array(self.f_x_proxy(x_k, x_k, args_dict))
        self.track_f = np.array([self.f_x(x_k, args_dict)])

        # Debugging
        if logger_bdca.isEnabledFor(logging.DEBUG):
            logger_bdca.debug(debug_path + 'initial f(x): {}'.format(self.track_f[k]))
            logger_bdca.debug(debug_path + 'initial f_proxy(x): {}'.format(self.track_f_proxy[k]))

        while cond:

            # ----------------------------------------------------------------------------------------------------------
            # Solve convex subproblem

            # Perform DCA step
            res = minimize(self.f_x_proxy,
                           method='SLSQP',
                           x0=x_k,
                           args=(x_k, args_dict),
                           constraints=({'type': 'eq', 'fun': constraint_eq}),
                           bounds=Bounds(np.repeat(0, len(x_k)), np.repeat(1, len(x_k))),
                           **kwargs
                           )

            # Update iteration
            k += 1

            # ----------------------------------------------------------------------------------------------------------
            # Perform line search

            # Update descent direction (difference between new and old solution)
            d_k = res.x - x_k            
            
            # Debugging
            if logger_bdca.isEnabledFor(logging.DEBUG):
                logger_bdca.debug(debug_path + 'iteration: {}'.format(k))
                logger_bdca.debug(debug_path + 'min(res.x): {}'.format(np.min(res.x)))
                logger_bdca.debug(debug_path + 'max(res.x): {}'.format(np.max(res.x)))
                logger_bdca.debug(debug_path + 'mean(res.x): {}'.format(np.mean(res.x)))
                logger_bdca.debug(debug_path + 'median(res.x): {}'.format(np.median(res.x)))
                logger_bdca.debug(debug_path + 'sum(res.x): {}'.format(np.sum(res.x)))
                logger_bdca.debug(debug_path + 'min(d_k): {}'.format(np.min(d_k)))
                logger_bdca.debug(debug_path + 'max(d_k): {}'.format(np.max(d_k)))
                logger_bdca.debug(debug_path + 'mean(d_k): {}'.format(np.mean(d_k)))
                logger_bdca.debug(debug_path + 'median(d_k): {}'.format(np.median(d_k)))
                logger_bdca.debug(debug_path + 'sum(d_k): {}'.format(np.sum(d_k)))
                logger_bdca.debug(debug_path + 'current f_proxy(x): {}'.format(self.track_f_proxy[k]))
                logger_bdca.debug(debug_path + 'current f(x): {}'.format(self.track_f[k]))

            # Check if we get a feasible direction, i.e d_k is not pointing into a direction out of the feasible set,
            # and at least one value has changed between old and new solution.
            # This would not be the case if:
            #       - At least one old value was non-zero and the corresponding new value is zero
            #       - The whole weight is on one value of the new solution
            #       - d_k only contains zeros
            if np.all(res.x[x_k != 0] != 0) and np.all(res.x != 1) and any(d_k != 0):

                # Update the counter for performing the line search
                self.cnt_line_search += 1
                
                # Compute adaptive lambda_k or set it to a fixed value lambda_bar
                if lambda_bar is None:
                    # Derive maximum lambda_k based on solving two equations
                    # 0 = res.x + lambda_k * d_k  <->  lambda_k = - res.x / d_k
                    lambda_0 = - res.x[d_k != 0] / d_k[d_k != 0]
                    # 1 = res.x + lambda_k * d_k  <->  lambda_k = (1 - res.x) / d_k
                    lambda_1 = (1 - res.x[d_k != 0]) / d_k[d_k != 0]
                    # Take only positive lambda_0's and lambda_1's and take the minimum
                    lambda_k = np.min((lambda_0 > lambda_1) * lambda_0 + (lambda_1 > lambda_0) * lambda_1)
                else:
                    lambda_k = lambda_bar

                # Check if we start the line search with a feasible solution
                # Especially relevant for non-adaptive lambda_k
                cond_overall = any(res.x + lambda_k * d_k < 0) or any(res.x + lambda_k * d_k > 1)
                
                # Debugging
                if logger_bdca.isEnabledFor(logging.DEBUG):
                    logger_bdca.debug(debug_path + 'feasible direction: {}'.format(True))
                    logger_bdca.debug(debug_path + 'lambda_k: {}'.format(lambda_k))
                    logger_bdca.debug(debug_path + 'cond_overall: {}'.format(cond_overall))
                
                # Check if current solution is non-feasible, scale down lambda_k
                while cond_overall:
                    lambda_k *= beta
                    cond_overall = any(res.x + lambda_k * d_k < 0) or any(res.x + lambda_k * d_k > 1)

                # Initialize left side of Armijo type condition
                cond_left = self.f_x(res.x + lambda_k * d_k, args_dict)

                # Initialize right side of Armijo type condition
                cond_right = self.f_x(res.x, args_dict) - alpha_2 * np.square(lambda_k) * np.square(np.linalg.norm(d_k))

                # Debugging
                if logger_bdca.isEnabledFor(logging.DEBUG):
                    logger_bdca.debug(debug_path + 'feasible lambda_k: {}'.format(lambda_k))
                    logger_bdca.debug(debug_path + 'cond_left: {}'.format(cond_left))
                    logger_bdca.debug(debug_path + 'cond_right: {}'.format(cond_right))
                    logger_bdca.debug(debug_path + 'cond_left > cond_right: {}'.format(cond_left > cond_right))
                
                # Check Armijo type condition and scale down lambda_k until condition is fulfilled
                while cond_left > cond_right or cond_overall:
                    # Scale down lambda_k
                    lambda_k *= beta
                    # Update left and right part of Armijo type condition
                    cond_left = self.f_x(res.x + lambda_k * d_k, args_dict)
                    cond_right = self.f_x(res.x, args_dict) - \
                                 alpha_2 * np.square(lambda_k) * np.square(np.linalg.norm(d_k))
                    # Check if solution is still feasible
                    cond_overall = any(res.x + lambda_k * d_k < 0) or any(res.x + lambda_k * d_k > 1)

            else:
                lambda_k = 0

            # Debugging
            if logger_bdca.isEnabledFor(logging.DEBUG):
                logger_bdca.debug(debug_path + 'final lambda_k: {}'.format(lambda_k))
                logger_bdca.debug(debug_path + 'final f(x): {}'.format(self.f_x(res.x + lambda_k * d_k, args_dict)))

            # ----------------------------------------------------------------------------------------------------------
            # Check line search result

            # Check if d_k was a descent direction
            if self.f_x(res.x, args_dict) < self.f_x(res.x + lambda_k * d_k, args_dict):
                # Update the counter for non descent directions
                self.cnt_non_descent_dc += 1
                # Reset counter for line search
                self.cnt_line_search -= 1
                # Set lambda to zero as d_k was a non-descent direction
                lambda_k = 0

            # ----------------------------------------------------------------------------------------------------------
            # Check new solution

            # Save current solution
            curr_solu = res.x + lambda_k * d_k

            # Add new (approximated) function value and current lambda_k to the tracking vectors
            self.track_f_proxy = np.append(self.track_f_proxy, self.f_x_proxy(curr_solu, x_k, args_dict))
            self.track_f = np.append(self.track_f, self.f_x(curr_solu, args_dict))
            self.track_lambdas = np.append(self.track_lambdas, lambda_k)

            # Debugging
            if logger_bdca.isEnabledFor(logging.DEBUG):
                logger_bdca.debug(debug_path + 'min(x_k): {}'.format(np.min(curr_solu)))
                logger_bdca.debug(debug_path + 'max(x_k): {}'.format(np.max(curr_solu)))
                logger_bdca.debug(debug_path + 'mean(x_k): {}'.format(np.mean(curr_solu)))
                logger_bdca.debug(debug_path + 'median(x_k): {}'.format(np.median(curr_solu)))
                logger_bdca.debug(debug_path + 'sum(x_k): {}'.format(np.sum(curr_solu)))
                logger_bdca.debug(debug_path + 'current f_proxy(x): {}'.format(self.track_f_proxy[k]))
                logger_bdca.debug(debug_path + 'current f(x): {}'.format(self.track_f[k]))

            # Check if current solution is feasible
            n_feasible_solu = (np.round(np.sum(curr_solu), decs) != 1) or any(0 > curr_solu) or any(curr_solu > 1)

            # Check for ill conditioning, this means that:
            #       - the weights do not sum up to one
            #       - the old objective function value is smaller than the new one
            if n_feasible_solu or np.round(self.track_f[k] - self.track_f[k - 1], decs) > 0:
                self.ill_cond_cnt += 1

            # ----------------------------------------------------------------------------------------------------------
            # Check stopping criterion

            # Based on iterations
            if stop_crit == "iter":
                stop_crit_value = k
                cond = k < k_max
            # Based on the absolute difference between old and new objective function value
            elif stop_crit == "func_abs":
                stop_crit_value = np.round(self.track_f[k] - self.track_f[k - 1], decs)
                cond = (stop_crit_value != 0 or n_feasible_solu) and k < k_max
            # Based on the relative difference between old and new objective function value
            elif stop_crit == "func_rel":
                stop_crit_value = np.abs(np.round((self.track_f[k] - self.track_f[k - 1]) / self.track_f[k - 1], decs))
                cond = (stop_crit_value < (10**(-decs)) or n_feasible_solu) and k < k_max
            # Based on the absolute difference between old and new solution
            elif stop_crit == "vec_abs":
                stop_crit_value = np.round(curr_solu - x_k, decs)
                cond = (np.all(stop_crit_value == 0) or n_feasible_solu) and k < k_max
            # Based on the relative difference between old and new solution
            elif stop_crit == "vec_rel":
                stop_crit_value = np.abs(np.round((curr_solu - x_k) / (x_k + 10**(-(decs + 5))), decs))
                cond = (np.all(stop_crit_value < (10**(-decs))) or n_feasible_solu) and k < k_max
            else:
                return logger_bdca.error("Please specify a valid 'stop_crit'.")
            
            # Update x_k with the current solution
            x_k = curr_solu
            
            # Debugging
            if logger_bdca.isEnabledFor(logging.DEBUG):
                logger_bdca.debug(debug_path + 'stop_crit - value: {}'.format(stop_crit_value))

        # If the maximum iteration is reached print a warning
        if k == k_max and stop_crit != "iter":
            logger_bdca.warning("The maximum number of iterations has been reached. "
                                "The algorithm likely did not converge.")
            
        if verbose:
            print(k)

        # Update internal variables
        self.k = k
        self.x_k = x_k

        return None


########################################################################################################################
# 4. Publisher's Imprint
########################################################################################################################

__author__ = "Marah-Lisanne Thormann"
__copyright__ = "Copyright 2023, The VaR-BDCA Project"
__credits__ = ["Phan Vuong", "Alain Zemkoho"]
__version__ = "1.2"
__maintainer__ = "Marah-Lisanne Thormann"
__email__ = "m.-l.thormann@soton.ac.uk"
__status__ = "Production"

########################################################################################################################
########################################################################################################################
