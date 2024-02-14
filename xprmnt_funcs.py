#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2

"""
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
--
-- Functions with respect to the distinct Experiments.
--
-- Description: In this script all functions are defined that are used in the framework of the experiments or to
--              (graphically) summarize the experimental results.
--
-- Content:     0. Set-up
--                  0.0 Required Libraries
--                  0.1 Logging Set-up
--              1. Adaptive Lambda
--                  1.0 get_adaptive_lambda
--              2. Linear Convergence
--                  2.0 get_convergence_constants
--              3. Bootstrapping
--                  3.0 get_bootstrap_ci
--                  3.1 get_mean_bootstrap_ci
--                  3.2 get_median_bootstrap_ci
--              4. Experiment 1
--                  4.0 grid_search
--              5. Experiment 2
--                  3.0 fixed_cost_search
--              6. Data Quality Checks
--                  6.0 check_column_dtype
--                  6.1 check_column_missings
--                  6.2 check_column_min
--                  6.3 check_column_max
--                  6.4 check_column_distinct
--                  6.5 check_column_ndistinct
--                  6.6 check_column_ndistinct_cnt
--                  6.7 check_sim_results
--              7. Publisher's Imprint
--
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
Version  Date        Author    Major Changes
1.0      2023-01-23  MLT       Initialization
1.1      2023-09-23  MLT       Finalized documentation
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
"""
########################################################################################################################
# 0. Set-ups
########################################################################################################################
######################################################################################################
# 0.0 Required Libraries
######################################################################################################

# To track CPU time
from time import process_time

# Algorithms
from bdca import *

# Bootstrapping
from sklearn.utils import resample

######################################################################################################
# 0.1 Logging Set-up
######################################################################################################

# Define logger name
logger_exp_funcs = logging.getLogger(__name__)

# Set current logging level
logger_exp_funcs.setLevel(logging.INFO)

# Define the general format of the logs
formatter = logging.Formatter(fmt='%(asctime)s[%(levelname)s] %(name)s.%(funcName)s: %(message)s',
                              datefmt='%m/%d/%Y %I:%M:%S %p')

# Create logging file
file_handler = logging.FileHandler("log_file.log")

# Add file to logger
logger_exp_funcs.addHandler(file_handler)

# Set the format of the file
file_handler.setFormatter(formatter)

# Add logs to the console
stream_handler = logging.StreamHandler()

# Add it to the logger
logger_exp_funcs.addHandler(stream_handler)

########################################################################################################################
# 1. Adaptive Lambda
########################################################################################################################
######################################################################################################
# 1.0 get_adaptive_lambda
######################################################################################################


def get_adaptive_lambda(data: pd.DataFrame, lambdas: list) -> dict:
    """
    Runs the BDCA for the given data set per distinct lambda and outputs the objective function values.

    :param data: A np.ndarray that contains the return data of different assets.
    :param lambdas: A list containing all values for lambda_bar.
    :return: A dictionary containing the sequences of the objective function per lambda.
    """

    # Dictionary to save the simulation results
    results = {}

    # Iterate over lambdas
    for lamb in lambdas:

        # Create initial weights
        x_k = np.repeat(1 / data.shape[1], data.shape[1])

        # Set up remaining function arguments
        data_kwargs = {"returns": data, "alpha": 0.05, "tau": 50, "var_threshold": 0.952}

        # Initialize BDCA
        my_optimizer = bdca(g_w, h_w, dh_w)

        # Run optimizer
        my_optimizer.bdca_optimizer(x_0=x_k,
                                    k_max=60,
                                    args_dict=data_kwargs,
                                    stop_crit="iter",
                                    alpha_2=0.001,
                                    beta=0.5,
                                    decs=7,
                                    lambda_bar=lamb)

        # Save result of current repetition in dictionary
        if lamb is None:
            results["Adaptive"] = my_optimizer.track_f
        else:
            results[lamb] = my_optimizer.track_f

    return results


########################################################################################################################
# 2. Linear Convergence
########################################################################################################################
######################################################################################################
# 2.0 get_convergence_constants
######################################################################################################

def get_convergence_constants(data: np.ndarray, rep: int) -> dict:
    """
    This function draws random starting values and runs the BDCA for a specified amount of repetitions. Afterwards, it
    outputs the sequences of the convergence constant q based on rearranging the following equation:

        || f(x_{k+1}) - f(x^*) || <= q || f(x_{k}) - f(x^*) ||

    :param data: A np.ndarray that contains the return data of different assets.
    :param rep: An non-negative integer corresponding to the number of repetitions.
    :return: A dictionary containing the sequences of the convergence constant q.
    """
    # ------------------------------------------------------------------------------------------------------------------

    # Dictionary to save the simulation results
    results = {}

    # Loop over the different repetitions
    for i in range(rep):
        # Draw initial weights
        x_k = np.random.dirichlet(np.ones(data.shape[1]) / 0.001, size=1).reshape((data.shape[1],))

        # Set up remaining function arguments
        data_kwargs = {"returns": data, "alpha": 0.05, "tau": 20, "var_threshold": 0.958}

        # Initialize BDCA
        my_optimizer = bdca(g_w, h_w, dh_w)

        # Run optimizer
        my_optimizer.bdca_optimizer(x_0=x_k, k_max=50, args_dict=data_kwargs, stop_crit="iter",
                                    alpha_2=0.001, beta=0.5, decs=7)

        # Save result of current repetition in dictionary
        results[i] = my_optimizer.get_convergence_constant()

    return results

########################################################################################################################
# 3. Bootstrapping
########################################################################################################################
######################################################################################################
# 3.0 get_bootstrap_ci
######################################################################################################


def get_bootstrap_ci(col_name: str,
                     feature: str,
                     data: pd.DataFrame,
                     stat_f: callable,
                     alpha: float,
                     rep: int) -> pd.DataFrame:
    """
    Computes bootstrap confidence intervals (CIs) for the stat_f and the given alpha level per unique value of col_name.

    :param col_name: A string indicating the column for grouping the data by the distinct column values.
    :param feature: A string indicating the column that should be used for the bootstrapping
    :param data: A pd.DataFrame that contains at least the col_name and feature as columns.
    :param stat_f: A callable statistical function that summarizes the bootstramp sample.
    :param alpha: A float in the interval [0, 1] indicating the alpha level of the intervals.
    :param rep: An integer for the amount of bootstrap samples.
    :return: A pd.DataFrame containing the lower, mid and upper value of the CI.
    """
    # ------------------------------------------------------------------------------------------------------------------

    results = pd.DataFrame(index=np.unique(data.loc[:, col_name]),
                           columns=["LB", "MD", "UB"])

    for col_value in np.unique(data.loc[:, col_name]):
        data_sub = data.loc[data.loc[:, col_name] == col_value, :]
        bs_samples = np.zeros(rep)
        for i in range(rep):
            bs_samples[i] = stat_f(resample(data_sub[feature], replace=True, n_samples=len(data_sub[feature])))

        bs_samples = np.sort(bs_samples)
        results.loc[col_value, :] = [bs_samples[int(np.floor(alpha * rep)) - 1],
                                     stat_f(data_sub[feature]),
                                     bs_samples[int(np.ceil((1 - alpha) * rep)) - 1]]

    return results

######################################################################################################
# 3.1 get_mean_bootstrap_ci
######################################################################################################


def get_mean_bootstrap_ci(data: pd.Series,
                          alpha: float = 0.025 / 80,
                          rep: int = 100000) -> list:
    """
    Computes the bootstrap confidence intervals (CIs) for the mean and the given alpha level.

    :param data: A pd.Series containing the values that have to be resampled.
    :param alpha: A float in the interval [0, 1] indicating the alpha level of the intervals.
    :param rep: An integer for the amount of bootstrap samples.
    :return: A list containing the lower and upper value of the CI.
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Bootstrapping

    # Get vector with zero entries
    bs_samples = np.zeros(rep)

    # Resampling for the repetitions
    for i in range(rep):
        # Compute mean of bootstrap sample
        bs_samples[i] = np.mean(resample(data, replace=True, n_samples=len(data)))

    # Sort the samples from smallest to highest
    bs_samples = np.sort(bs_samples)

    # ------------------------------------------------------------------------------------------------------------------
    # Compute output

    return [bs_samples[int(alpha*rep) - 1], bs_samples[int((1 - alpha) * rep) - 1]]

######################################################################################################
# 3.2 get_median_bootstrap_ci
######################################################################################################


def get_median_bootstrap_ci(data: pd.Series,
                            alpha: float = 0.025 / 80,
                            rep: int = 100000) -> list:
    """
    Computes the bootstrap confidence intervals (CIs) for the median and the given alpha level.

    :param data: A pd.Series containing the values that have to be resampled.
    :param alpha: A float in the interval [0, 1] indicating the alpha level of the intervals.
    :param rep: An integer for the amount of bootstrap samples.
    :return: A list containing the lower and upper value of the CI.
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Bootstrapping

    # Get vector with zero entries
    bs_samples = np.zeros(rep)

    # Resampling for the repetitions
    for i in range(rep):
        # Compute median of bootstrap sample
        bs_samples[i] = np.median(resample(data, replace=True, n_samples=len(data)))

    # Sort the samples from smallest to highest
    bs_samples = np.sort(bs_samples)

    # ------------------------------------------------------------------------------------------------------------------
    # Compute output

    return [bs_samples[int(alpha*rep) - 1], bs_samples[int((1 - alpha) * rep) - 1]]

########################################################################################################################
# 4. Experiment 1
########################################################################################################################
######################################################################################################
# 4.0 grid_search
######################################################################################################


def grid_search(file: str, data_name: str, optimizer: object, my_var: float, alpha_var: float, tau: int,
                alpha_d1: float, alpha_bdca: float, beta: float, rep: int, stop_mecha: str, data: np.ndarray,
                probs: np.ndarray, decs: int, max_iter: int) -> pd.DataFrame:
    """
    This function is used to perform a grid search in the first experiment.

    :param file: A string that corresponds to the file path where the results should be saved.
    :param data_name: A string corresponding to the name of the data set.
    :param optimizer: An object that contains the optimization algorithm.
    :param my_var: A float corresponding to the VaR constraint.
    :param alpha_var: A float corresponding to the alpha level of the VaR.
    :param tau: An integer which represents the penalty parameter.
    :param alpha_d1: A float that is used for construction the initialization weights.
    :param alpha_bdca: A float that corresponds to the bar{alpha} parameter of the BDCA.
    :param beta: A float that corresponds to the beta parameter of the BDCA.
    :param rep: An integer that corresponds to the repetition number of the experiment.
    :param stop_mecha: A string that corresponds to the stopping condition.
    :param data: An np.ndarray that contains the data set.
    :param probs: An np.ndarray that contains the scenario probabilities.
    :param decs: An integer indicating the decimals to round the results
    :param max_iter: An integer that corresponds to the maximum iterations.
    :return: A pd.DataFrame containing the simulation results.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Setup

    # Set seed to ensure reproducibility of repetition
    np.random.seed(rep)

    # Draw random initial weights
    x_k = np.random.dirichlet(np.ones(data.shape[1]) / alpha_d1, size=1).reshape((data.shape[1],))

    # Set up input arguments for dc functions
    my_kwargs = {"returns": data,
                 "alpha": alpha_var,
                 "tau": tau,
                 "var_threshold": my_var,
                 "probs": probs}

    # ------------------------------------------------------------------------------------------------------------------
    # Choose optimizer

    if optimizer == "DCA":
        # Initialize DCA
        my_optimizer = dca(g_w, h_w, dh_w)
        # Start tracking CPU time
        time_start = process_time()
        # Perform optimization
        my_optimizer.dca_optimizer(x_k, max_iter, my_kwargs, stop_mecha, decs=decs)
        # Stop tracking cpu time
        time_stop = process_time()

        # Set BDCA specific parameters to NaN
        cnt_line_search = np.nan
        cnt_non_descent_dc = np.nan
        min_lambda_k = np.nan
        max_lambda_k = np.nan
        median_lambda_k = np.nan
        avg_lambda_k = np.nan
    else:
        # Initialize BDCA
        my_optimizer = bdca(g_w, h_w, dh_w)
        # Start tracking CPU time
        time_start = process_time()
        # Perform optimization
        my_optimizer.bdca_optimizer(x_k, max_iter, my_kwargs, stop_crit=stop_mecha,
                                    alpha_2=alpha_bdca, beta=beta, decs=decs)
        # Stop tracking cpu time
        time_stop = process_time()

        # Save BDCA specific parameter values
        cnt_line_search = my_optimizer.cnt_line_search
        avg_lambda_k = np.round(np.mean(my_optimizer.track_lambdas), decs)
        median_lambda_k = np.round(np.median(my_optimizer.track_lambdas), decs)
        max_lambda_k = np.round(np.max(my_optimizer.track_lambdas), decs)
        min_lambda_k = np.round(np.min(my_optimizer.track_lambdas), decs)
        cnt_non_descent_dc = my_optimizer.cnt_non_descent_dc

    # Create a dictionary to summarize the results
    results = {"Rep": rep,
               "Algo": optimizer,
               "Comp_Time": np.round(time_stop - time_start, decs),
               "VaR": my_var,
               "alpha_var": alpha_var,
               "decs": decs,
               "stop_mecha": stop_mecha,
               "tau": tau,
               "alpha_d1": alpha_d1,
               "iter": my_optimizer.k,
               "max_iter": max_iter,
               "cnt_ill_cond": my_optimizer.ill_cond_cnt,
               "cnt_line_search": cnt_line_search,
               "cnt_non_descent_dc": cnt_non_descent_dc,
               "alpha_bdca": alpha_bdca,
               "beta": beta,
               "avg_lambda_k": avg_lambda_k,
               "median_lambda_k": median_lambda_k,
               "max_lambda_k": max_lambda_k,
               "min_lambda_k": min_lambda_k,
               "func_v": np.round(my_optimizer.track_f[-1], decs),
               "var_model": np.round(var(profit(my_optimizer.x_k, data), alpha_var), decs),
               "objective": np.round(np.sum(profit(my_optimizer.x_k, data)), decs),
               "sum_weights": np.round(np.sum(my_optimizer.x_k), decs),
               "min_weight": np.round(np.min(my_optimizer.x_k), decs),
               "max_weight": np.round(np.max(my_optimizer.x_k), decs),
               "median_weight": np.round(np.median(my_optimizer.x_k), decs),
               "avg_weight": np.round(np.mean(my_optimizer.x_k), decs),
               "nonzero_weights": np.sum((np.round(my_optimizer.x_k, decs) != 0) * 1),
               "cnt_weights": len(my_optimizer.x_k)
               }

    # Add results to final table
    sim_results = pd.DataFrame(results, index=[0])

    # ------------------------------------------------------------------------------------------------------------------
    # Save results in file

    with open(file, "a") as f:
        f.write(f"{data_name},{results['Algo']},{results['Rep']},{results['Comp_Time']},{results['VaR']},"
                f"{results['alpha_var']},{results['decs']},{results['stop_mecha']},{results['tau']},"
                f"{results['alpha_d1']},{results['alpha_bdca']},{results['beta']},{results['iter']},"
                f"{results['max_iter']},{results['cnt_ill_cond']},{results['cnt_line_search']},"
                f"{results['cnt_non_descent_dc']},{results['avg_lambda_k']},{results['median_lambda_k']},"
                f"{results['max_lambda_k']},{results['min_lambda_k']},{results['func_v']},{results['var_model']},"
                f"{results['objective']},{results['sum_weights']},{results['min_weight']},{results['max_weight']},"
                f"{results['median_weight']},{results['avg_weight']},{results['nonzero_weights']},"
                f"{results['cnt_weights']}\n")

    return sim_results

########################################################################################################################
# 5. Experiment 2
########################################################################################################################


def fixed_cost_search(file: str, data_name: str, my_var: float, alpha_var: float, tau: int, alpha_d1: float,
                      alpha_bdca: float, beta: float, rep: int, data: np.ndarray, probs: np.ndarray, decs: int,
                      max_iter_bdca: int, max_iter_dca: int) -> None:
    """

    :param file: A string that corresponds to the file path where the results should be saved.
    :param data_name: A string corresponding to the name of the data set.
    :param my_var: A float corresponding to the VaR constraint.
    :param alpha_var: A float corresponding to the alpha level of the VaR.
    :param tau: An integer which represents the penalty parameter.
    :param alpha_d1: A float that is used for construction the initialization weights.
    :param alpha_bdca: A float that corresponds to the bar{alpha} parameter of the BDCA.
    :param beta: A float that corresponds to the beta parameter of the BDCA.
    :param rep: An integer that corresponds to the repetition number of the experiment.
    :param data: An np.ndarray that contains the data set.
    :param probs: An np.ndarray that contains the scenario probabilities.
    :param decs: An integer indicating the decimals to round the results
    :param max_iter_bdca: An integer that corresponds to the maximum iterations of the BDCA.
    :param max_iter_dca: An integer that corresponds to the maximum iterations of the DCA.
    :return: None.
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Setup

    # Set seed to ensure reproducibility of repetition
    np.random.seed(rep)

    # Draw random initial weights
    x_k = np.random.dirichlet(np.ones(data.shape[1]) / alpha_d1, size=1).reshape((data.shape[1],))

    # Set up input arguments for dc functions
    my_kwargs = {"returns": data,
                 "alpha": alpha_var,
                 "tau": tau,
                 "var_threshold": my_var,
                 "probs": probs}

    # ------------------------------------------------------------------------------------------------------------------

    # Choose optimizer
    for optimizer in ["BDCA", "DCA"]:
        if optimizer == "BDCA":
            my_optimizer = bdca(g_w, h_w, dh_w)
            time_start = process_time()
            my_optimizer.bdca_optimizer(x_k, max_iter_bdca, my_kwargs, "iter", alpha_2=alpha_bdca, beta=beta, decs=decs)
            time_stop = process_time()
            cnt_line_search = my_optimizer.cnt_line_search
            avg_lambda_k = np.round(np.mean(my_optimizer.track_lambdas), decs)
            median_lambda_k = np.round(np.median(my_optimizer.track_lambdas), decs)
            max_lambda_k = np.round(np.max(my_optimizer.track_lambdas), decs)
            min_lambda_k = np.round(np.min(my_optimizer.track_lambdas), decs)
            cnt_non_descent_dc = my_optimizer.cnt_non_descent_dc
            stop_mecha = "iter"
            max_iter = max_iter_bdca
            bdca_value = np.round(my_optimizer.track_f[-1], decs)
        else:
            my_optimizer = dca(g_w, h_w, dh_w)
            time_start = process_time()
            my_optimizer.dca_optimizer(x_k, max_iter_dca, my_kwargs, "bdca", bdca_value=bdca_value, decs=decs)
            time_stop = process_time()
            stop_mecha = "bdca"
            max_iter = max_iter_dca
            cnt_line_search = np.nan
            cnt_non_descent_dc = np.nan
            min_lambda_k = np.nan
            max_lambda_k = np.nan
            median_lambda_k = np.nan
            avg_lambda_k = np.nan

        # Summarize results
        results = {"Rep": rep,
                   "Algo": optimizer,
                   "Comp_Time": np.round(time_stop - time_start, decs),
                   "VaR": my_var,
                   "alpha_var": alpha_var,
                   "decs": decs,
                   "stop_mecha": stop_mecha,
                   "tau": tau,
                   "alpha_d1": alpha_d1,
                   "iter": my_optimizer.k,
                   "max_iter": max_iter,
                   "cnt_ill_cond": my_optimizer.ill_cond_cnt,
                   "cnt_line_search": cnt_line_search,
                   "cnt_non_descent_dc": cnt_non_descent_dc,
                   "alpha_bdca": alpha_bdca,
                   "beta": beta,
                   "avg_lambda_k": avg_lambda_k,
                   "median_lambda_k": median_lambda_k,
                   "max_lambda_k": max_lambda_k,
                   "min_lambda_k": min_lambda_k,
                   "func_v": np.round(my_optimizer.track_f[-1], decs),
                   "var_model": np.round(var(profit(my_optimizer.x_k, data), alpha_var), decs),
                   "objective": np.round(np.sum(profit(my_optimizer.x_k, data)) / data.shape[0], decs),
                   "sum_weights": np.round(np.sum(my_optimizer.x_k), decs),
                   "min_weight": np.round(np.min(my_optimizer.x_k), decs),
                   "max_weight": np.round(np.max(my_optimizer.x_k), decs),
                   "median_weight": np.round(np.median(my_optimizer.x_k), decs),
                   "avg_weight": np.round(np.mean(my_optimizer.x_k), decs),
                   "nonzero_weights": np.sum((np.round(my_optimizer.x_k, decs) != 0) * 1),
                   "cnt_weights": len(my_optimizer.x_k)
                   }

        # --------------------------------------------------------------------------------------------------------------
        # Save output in file

        with open(file, "a") as f:
            f.write(f"{data_name},{results['Algo']},{results['Rep']},{results['Comp_Time']},{results['VaR']},"
                    f"{results['alpha_var']},{results['decs']},{results['stop_mecha']},{results['tau']},"
                    f"{results['alpha_d1']},{results['alpha_bdca']},{results['beta']},{results['iter']},"
                    f"{results['max_iter']},{results['cnt_ill_cond']},{results['cnt_line_search']},"
                    f"{results['cnt_non_descent_dc']},{results['avg_lambda_k']},{results['median_lambda_k']},"
                    f"{results['max_lambda_k']},{results['min_lambda_k']},{results['func_v']},{results['var_model']},"
                    f"{results['objective']},{results['sum_weights']},{results['min_weight']},{results['max_weight']},"
                    f"{results['median_weight']},{results['avg_weight']},{results['nonzero_weights']},"
                    f"{results['cnt_weights']}\n")

    return None

########################################################################################################################
# 6. Data Quality Checks
########################################################################################################################
######################################################################################################
# 6.0 check_column_dtype
######################################################################################################


def check_column_dtype(check_summary: pd.DataFrame,
                       id: int,
                       data: pd.DataFrame,
                       col_id: int,
                       data_type: callable) -> pd.DataFrame:
    """
    Checks if the col_id column of the data has the correct data type.

    :param check_summary: A pd.DataFrame containing the current test summary.
    :param id: An integer that corresponds to the test id.
    :param data: A pd.DataFrame that contains at least the id column.
    :param col_id: An integer indicating the column that should be checked.
    :param data_type: A callable that corresponds to the desired data type of the column.
    :return: A pd.DataFrame that corresponds to the check_summary including the result of the newest test.
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Compute check

    # Get current column name
    curr_col_name_str = data.columns[col_id] + "' column"

    # Check datatype
    check_10 = data.dtypes[col_id] != data_type

    # Create dataframe to store the results of the check
    results = pd.DataFrame({"ID": [id],
                            "Description": ["Datatype of '" + curr_col_name_str],
                            "Result": [check_10]})

    # ------------------------------------------------------------------------------------------------------------------
    # Add results to the check summary

    return pd.concat([check_summary, results])

######################################################################################################
# 6.1 check_column_missings
######################################################################################################


def check_column_missings(check_summary: pd.DataFrame,
                          id: int,
                          data: pd.DataFrame,
                          col_id: int,
                          m_type: int) -> pd.DataFrame:
    """
    Checks if the col_id column of the data has missing values.

    :param check_summary: A pd.DataFrame containing the current test summary.
    :param id: An integer that corresponds to the test id.
    :param data: A pd.DataFrame that contains at least the id column.
    :param col_id: An integer indicating the column that should be checked.
    :param m_type: An integer indicating if the DCA has no values for the column (Yes = 1, No = 0).
    :return: A pd.DataFrame that corresponds to the check_summary including the result of the newest test.
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Compute check

    # Get current column name
    curr_col_name_str = data.columns[col_id] + "' column"
    curr_col_name = data.columns[col_id]

    # Check for missing values
    if m_type == 1:
        check = data.iloc[:, col_id].isnull().any()
    else:
        check = data.loc[data.algorithm == "BDCA", curr_col_name].isnull().any()

    # Create dataframe to store the results of the check
    check = pd.DataFrame({"ID": [id],
                          "Description": ["Missing values in '" + curr_col_name_str],
                          "Result": [check]})

    # ------------------------------------------------------------------------------------------------------------------
    # Add results to the check summary

    return pd.concat([check_summary, check])

######################################################################################################
# 6.2 check_column_min
######################################################################################################


def check_column_min(check_summary: pd.DataFrame,
                     id: int,
                     data: pd.DataFrame,
                     col_id: int,
                     m_type: int,
                     min_v: float) -> pd.DataFrame:
    """
    Checks if the col_id column of the data has the correct minimum value.

    :param check_summary: A pd.DataFrame containing the current test summary.
    :param id: An integer that corresponds to the test id.
    :param data: A pd.DataFrame that contains at least the id column.
    :param col_id: An integer indicating the column that should be checked.
    :param m_type: An integer indicating if the DCA has no values for the column (Yes = 1, No = 0).
    :param min_v: A float corresponding to the expected minimum.
    :return: A pd.DataFrame that corresponds to the check_summary including the result of the newest test.
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Compute check

    # Get current column name
    curr_col_name_str = data.columns[col_id] + "' column"
    curr_col_name = data.columns[col_id]

    # Check for missing values
    if m_type == 1:
        check = np.min(data.loc[:, curr_col_name]) < min_v
    else:
        check = np.min(data.loc[data.algorithm == "BDCA", curr_col_name]) < min_v

    # Store result in dict
    check = {"ID": [id],
             "Description": ["Minimum of '" + curr_col_name_str],
             "Result": [check]}

    # ------------------------------------------------------------------------------------------------------------------
    # Add results to the check summary

    return pd.concat([check_summary, pd.DataFrame(check)])

######################################################################################################
# 6.3 check_column_max
######################################################################################################


def check_column_max(check_summary, id, data, col_id, m_type, max_v):
    """
    Checks if the col_id column of the data has the correct maximum value.

    :param check_summary: A pd.DataFrame containing the current test summary.
    :param id: An integer that corresponds to the test id.
    :param data: A pd.DataFrame that contains at least the id column.
    :param col_id: An integer indicating the column that should be checked.
    :param m_type: An integer indicating if the DCA has no values for the column (Yes = 1, No = 0).
    :param max_v: A float corresponding to the expected maximum.
    :return: A pd.DataFrame that corresponds to the check_summary including the result of the newest test.
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Compute check

    # Get current column name
    curr_col_name_str = data.columns[col_id] + "' column"
    curr_col_name = data.columns[col_id]

    # Check for missing values
    if m_type == 1:
        check = np.max(data.loc[:, curr_col_name]) > max_v
    else:
        check = np.max(data.loc[data.algorithm == "BDCA", curr_col_name]) > max_v

    # Store result in dict
    check = {"ID": [id],
             "Description": ["Maximum of '" + curr_col_name_str],
             "Result": [check]}

    # ------------------------------------------------------------------------------------------------------------------
    # Add results to the check summary

    check_summary = pd.concat([check_summary, pd.DataFrame(check)])

    return check_summary

######################################################################################################
# 6.4 check_column_distinct
######################################################################################################


def check_column_distinct(check_summary, id, data, col_id, m_type, distinct):
    """
    Checks if the col_id column of the data contains possible values.

    :param check_summary: A pd.DataFrame containing the current test summary.
    :param id: An integer that corresponds to the test id.
    :param data: A pd.DataFrame that contains at least the id column.
    :param col_id: An integer indicating the column that should be checked.
    :param m_type: An integer indicating if the DCA has no values for the column (Yes = 1, No = 0).
    :param distinct: A list containing all possible values for the column.
    :return: A pd.DataFrame that corresponds to the check_summary including the result of the newest test.
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Compute check

    # Get current column name
    curr_col_name_str = data.columns[col_id] + "' column"
    curr_col_name = data.columns[col_id]

    # Check for missing values
    if m_type == 1:
        check = np.sum(np.in1d(np.unique(data.loc[:, curr_col_name]), distinct, invert=True))
    else:
        check = np.sum(np.in1d(np.unique(data.loc[data.algorithm == "BDCA", curr_col_name]), distinct, invert=True))

    # Store result in dict
    check = {"ID": [id],
             "Description": ["Distinct Values of '" + curr_col_name_str],
             "Result": [check]}

    # ------------------------------------------------------------------------------------------------------------------
    # Add results to the check summary

    check_summary = pd.concat([check_summary, pd.DataFrame(check)])

    return check_summary

######################################################################################################
# 6.5 check_column_ndistinct
######################################################################################################


def check_column_ndistinct(check_summary, id, data, col_id, m_type, n_distinct):
    """
    Checks if the col_id column of the data contains all possible values.

    :param check_summary: A pd.DataFrame containing the current test summary.
    :param id: An integer that corresponds to the test id.
    :param data: A pd.DataFrame that contains at least the id column.
    :param col_id: An integer indicating the column that should be checked.
    :param m_type: An integer indicating if the DCA has no values for the column (Yes = 1, No = 0).
    :param n_distinct: A list containing all possible values for the column.
    :return: A pd.DataFrame that corresponds to the check_summary including the result of the newest test.
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Compute check

    # Get current column name
    curr_col_name_str = data.columns[col_id] + "' column"
    curr_col_name = data.columns[col_id]

    # Check for missing values
    if m_type == 1:
        check = len(np.unique(data.loc[:, curr_col_name])) != len(n_distinct)
    else:
        check = len(np.unique(data.loc[data.algorithm == "BDCA", curr_col_name])) != len(n_distinct)

    # Store result in dict
    check = {"ID": [id],
             "Description": ["# of Distinct Values of '" + curr_col_name_str],
             "Result": [check]}

    # ------------------------------------------------------------------------------------------------------------------
    # Add results to the check summary

    return pd.concat([check_summary, pd.DataFrame(check)])

######################################################################################################
# 6.6 check_column_ndistinct_cnt
######################################################################################################


def check_column_ndistinct_cnt(check_summary, id, data, col_id, m_type):
    """
    Checks if the col_id column of the data contains an equal amount of observations per distinct value.

    :param check_summary: A pd.DataFrame containing the current test summary.
    :param id: An integer that corresponds to the test id.
    :param data: A pd.DataFrame that contains at least the id column.
    :param col_id: An integer indicating the column that should be checked.
    :param m_type: An integer indicating if the DCA has no values for the column (Yes = 1, No = 0).
    :return: A pd.DataFrame that corresponds to the check_summary including the result of the newest test.
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Compute check

    # Get current column name
    curr_col_name_str = data.columns[col_id] + "' column"
    curr_col_name = data.columns[col_id]

    # Check for missing values
    if m_type == 1:
        check = data.groupby([curr_col_name]).agg(curr_col_name).count()
        check = len(np.unique(check)) != 1
    else:
        check = data.loc[data.algorithm == "BDCA", :].groupby([curr_col_name]).agg(curr_col_name).count()
        check = len(np.unique(check)) != 1

    # Store result in dict
    check = {"ID": [id],
             "Description": ["# of Distinct Values Count of '" + curr_col_name_str],
             "Result": [check]}

    # ------------------------------------------------------------------------------------------------------------------
    # Add results to the check summary

    return pd.concat([check_summary, pd.DataFrame(check)])

######################################################################################################
# 6.7 check_sim_results
######################################################################################################


def check_sim_results(sim_results: pd.DataFrame,
                      name: str,
                      n_assets: int,
                      stop_crit: list,
                      n_rep: int,
                      beta: list,
                      alpha_bdca: list,
                      tau: list,
                      n_vars: list,
                      iter: int,
                      alpha_diri: list) -> pd.DataFrame:
    """
    Data quality checks for outputs of the numerical experiments.

    :param sim_results: A pd.DataFrame containing the results of the numerical experiment.
    :param name: A string containing the name of the data set.
    :param n_assets: An integer that corresponds to the number of assets in the data set.
    :param stop_crit: A list containing the distinct stopping conditions.
    :param n_rep: An integer indicating how often the distinct experiments have been repeated.
    :param beta: A list containing the distinct beta values.
    :param alpha_bdca: A list containing the distinct Bar{alpha} values.
    :param tau: A list containing the distinct tau values.
    :param n_vars: A list containing the distinct VaR constraints.
    :param iter: An integer indicating the maximal possible number of iterations.
    :param alpha_diri: A list containing the distinct values for the Dirichelet setting.
    :return:
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Check function inputs

    # Set up debug_path
    debug_path = "3.1 - check_sim_results - "

    # Sim_results
    if not isinstance(sim_results, pd.DataFrame):
        logger_help_func.error(debug_path + 'type(sim_results): {}'.format(type(sim_results)))
        raise TypeError("The provided input for the argument 'sim_results' is not a pd.DataFrame.")
    elif sim_results.empty:
        logger_help_func.error(debug_path + 'sim_results.empty: {}'.format(sim_results.empty))
        raise ValueError("The provided input for the argument 'sim_results' is empty.")
    elif sim_results.shape[1] != 32 and sim_results.shape[1] != 33:
        logger_help_func.warning(debug_path + 'sim_results.shape[1]: {}'.format(sim_results.shape[1]))
        raise ValueError("The provided input for the argument 'sim_results' does not have 33 columns.")

    # ------------------------------------------------------------------------------------------------------------------
    # Compute output

    # Initialize column counter
    i = 0
    # Initialize test ID counter
    j = 0

    # Create empty data frame for the results
    check_summary = pd.DataFrame()

    # Iterate over the distinct columns of the results dataframe
    for col in [['data', object, 1, None, None, [name], 1, 1],
                ['algorithm', object, 1, None, None, ['DCA', 'BDCA'], 1, None],
                ['repetition', np.int64, 1, 0, n_rep, np.arange(1, n_rep + 1), 1, 1],
                ['cpu_time', np.float64, 1, 0, None, None, None, None],
                ['var', np.float64, 1, 0, n_vars[-1], n_vars, 1, 1],
                ['alpha_var', np.float64, 1, 0.05, 0.05, [0.05], 1, 1],
                ['decs', np.int64, 1, 0, 7, [7], 1, 1],
                ['stop_mecha', object, 1, None, None, stop_crit, 1, 1],
                ['tau', np.int64, 1, 0, 90, tau, 1, 1],
                ['alpha_diri', np.float64, 1, 0, 10, alpha_diri, 1, 1],
                ['alpha_bdca', np.float64, 2, 0, 0.01, alpha_bdca, 1, 1],
                ['beta', np.float64, 2, 0, 1, beta, 1, 1],
                ['iter', np.int64, 1, 0, iter, None, None, None],
                ['max_iter', np.int64, 1, 0, iter, None, None, None],
                ['cnt_ill_cond', np.int64, 1, 0, iter, None, None, None],
                ['cnt_line_search', np.float64, 2, 0, iter, None, None, None],
                ['cnt_non_descent_dc', np.float64, 2, 0, iter, None, None, None],
                ['avg_lambda_k', np.float64, 2, 0, None, None, None, None],
                ['median_lambda_k', np.float64, 2, 0, None, None, None, None],
                ['max_lambda_k', np.float64, 2, 0, None, None, None, None],
                ['min_lambda_k', np.float64, 2, 0, None, None, None, None],
                ['func_v', np.float64, 1, None, None, None, None, None],
                ['var_model', np.float64, 1, 0, 1.5, None, None, None],
                ['avg_return', np.float64, 1, 0, None, None, None, None],
                ['sum_weights', np.float64, 1, 0, 1.2, None, None, None],
                ['min_weight', np.float64, 1, 0, 1, None, None, None],
                ['max_weight', np.float64, 1, 0, 1, None, None, None],
                ['median_weight', np.float64, 1, 0, 1, None, None, None],
                ['avg_weight', np.float64, 1, 0, 1, None, None, None],
                ['nonzero_weights', np.int64, 1, 0, n_assets, None, None, None],
                ['cnt_weights', np.int64, 1, 0, n_assets, [n_assets], 1, 1],
                ['infeasible', np.int64, 1, 0, 1, [0, 1], 1, None]
                ]:

        # Check the data type of the column
        check_summary = check_column_dtype(check_summary, j, sim_results, i, col[1])
        j += 1

        # Check for missing values
        check_summary = check_column_missings(check_summary, j, sim_results, i, col[2])
        j += 1

        if col[3] is not None:
            # Check the observed minimum
            check_summary = check_column_min(check_summary, j, sim_results, i, col[2], col[3])
            j += 1
        if col[4] is not None:
            # Check the observed maximum
            check_summary = check_column_max(check_summary, j, sim_results, i, col[2], col[4])
            j += 1
        if col[5] is not None:
            # Check if possible distinct values have been observed
            check_summary = check_column_distinct(check_summary, j, sim_results, i, col[2], col[5])
            j += 1
        if col[6] is not None:
            # Check if all distinct values have been observed
            check_summary = check_column_ndistinct(check_summary, j, sim_results, i, col[2], col[5])
            j += 1
        if col[7] is not None:
            # Check if the distinct values have been equally observed
            check_summary = check_column_ndistinct_cnt(check_summary, j, sim_results, i, col[2])
            j += 1

        # Increase column counter
        i += 1

    return check_summary


########################################################################################################################
# 7. Publisher's Imprint
########################################################################################################################

__author__ = "Marah-Lisanne Thormann"
__credits__ = ["Phan Vuong", "Alain Zemkoho"]
__version__ = "1.1"
__maintainer__ = "Marah-Lisanne Thormann"
__email__ = "m.-l.thormann@soton.ac.uk"

########################################################################################################################
########################################################################################################################
