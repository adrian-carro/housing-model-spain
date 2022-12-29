# -*- coding: utf-8 -*-
"""
Class to estimate values for the model parameters using the simulation output resulting from Launcher.py.

@author: Adrian Carro
"""

import numpy as np
from scipy.fftpack import fft
import platform
from pathlib import Path
import os
import re
import pandas as pd
import glob


def get_simulation_moments(_simulation_output_file, _n_skip):
    # Read results into pandas data frame, skipping first n_skip lines after the header
    _results = pd.read_csv(_simulation_output_file, delimiter=";", skipinitialspace=True,
                           skiprows=range(1, _n_skip + 1), usecols={"nRenting", "nOwnerOccupier", "Sale nSales",
                                                                    "nActiveBTL"})
    return np.vstack((np.array([_results.mean()]).T, np.array([_results.std()]).T))


def get_all_simulation_moments(_simulation_output_folder, _n_skip):
    # First, read results from core indicator files and temporarily store them in lists
    housing_transactions_mean = []
    housing_transactions_std = []
    # TODO: Paths need to be made compatible with Linux also
    with open(_simulation_output_folder + "\coreIndicator-housingTransactions.csv") as f:
        for _line in f:
            housing_transactions_mean.append(np.mean([int(element) for element in _line.split(";")[_n_skip + 1:]]))
            housing_transactions_std.append(np.std([int(element) for element in _line.split(";")[_n_skip + 1:]]))
    price_to_income_mean = []
    price_to_income_std = []
    with open(_simulation_output_folder + "\coreIndicator-priceToIncome.csv") as f:
        for _line in f:
            price_to_income_mean.append(np.mean([float(element) for element in _line.split(";")[_n_skip + 1:]]))
            price_to_income_std.append(np.std([float(element) for element in _line.split(";")[_n_skip + 1:]]))
    rental_yield_mean = []
    with open(_simulation_output_folder + "\coreIndicator-rentalYield.csv") as f:
        for _line in f:
            rental_yield_mean.append(np.mean([float(element) for element in _line.split(";")[_n_skip + 1:]]))
    spread_mean = []
    with open(_simulation_output_folder + "\coreIndicator-interestRateSpread.csv") as f:
        for _line in f:
            spread_mean.append(np.mean([float(element) for element in _line.split(";")[_n_skip + 1:]]))
    
    # Then, read results from general output files and create a list of vectors of moments
    _all_simulation_moments = []
    for j, _simulation_output_file in enumerate(glob.glob("{}/Output-run*.csv".format(_simulation_output_folder))):
        _results = pd.read_csv(_simulation_output_file, delimiter=";", skipinitialspace=True,
                               skiprows=range(1, _n_skip + 1), usecols={"nRenting", "nOwnerOccupier", "nActiveBTL",
                                                                        "TotalPopulation", "Sale HPI", "Rental HPI"})
        _results["fHomeOwner"] = (_results["nOwnerOccupier"] + _results["nActiveBTL"]) / _results["TotalPopulation"]
        _results["fRenting"] = _results["nRenting"] / _results["TotalPopulation"]
        _results["fActiveBTL"] = _results["nActiveBTL"] / _results["TotalPopulation"]
        _all_simulation_moments.append(np.vstack([_results["Sale HPI"].mean(), _results["Sale HPI"].std(),
                                                  get_period(_results["Sale HPI"]), _results["Rental HPI"].mean(),
                                                  _results["fHomeOwner"].mean(), _results["fRenting"].mean(),
                                                  _results["fActiveBTL"].mean(), housing_transactions_mean[j],
                                                  housing_transactions_std[j], price_to_income_mean[j],
                                                  price_to_income_std[j], rental_yield_mean[j], spread_mean[j]]))
#        break
    return _all_simulation_moments


def get_period(_time_series):
        n = len(_time_series)  # Number of sample points
        fast_fourier_transform_hpi = (1.0 / n) * np.abs(fft(_time_series)[1:int(n / 2)])
        frequency_domain = np.linspace(0.0, 1.0, n)[1:int(n / 2)]  # This assumes sample spacing of 1
        return 1 / frequency_domain[fast_fourier_transform_hpi.argmax()]


def get_jacobian_of_error_vector(criterion_df):
    print("WARNING: Using an ad-hoc Jacobian matrix computed at the center of each parameter interval instead of at "
          "the chosen parameter values!")
    # Attempt at computing errors for the moments analysed at the middle of each range
    _jac_err = np.zeros((6, 2))
    _jac_err[:, 0] = \
        ((criterion_df.loc[(criterion_df["BTL_CHOICE_INTENSITY"] == 70.0)
                           & (criterion_df["SENSITIVITY_RENT_OR_PURCHASE"] == 0.0003), "Error Vector"].values[0] -
          criterion_df.loc[(criterion_df["BTL_CHOICE_INTENSITY"] == 30.0)
                           & (criterion_df["SENSITIVITY_RENT_OR_PURCHASE"] == 0.0003), "Error Vector"].values[0])
         / (2 * 20.0)).flatten()
    _jac_err[:, 1] = \
        ((criterion_df.loc[(criterion_df["BTL_CHOICE_INTENSITY"] == 50.0)
                           & (criterion_df["SENSITIVITY_RENT_OR_PURCHASE"] == 0.0005), "Error Vector"].values[0] -
          criterion_df.loc[(criterion_df["BTL_CHOICE_INTENSITY"] == 50.0)
                           & (criterion_df["SENSITIVITY_RENT_OR_PURCHASE"] == 0.0001), "Error Vector"].values[0])
         / (2 * 0.0002)).flatten()
    return _jac_err


def get_gamma(v, n_simulations, _error_matrix):
    _error_matrix = np.matrix(_error_matrix)
    sum_over_simulations = np.zeros((len(_error_matrix), len(_error_matrix)))
    for j in range(v, n_simulations):
        sum_over_simulations += _error_matrix[:, j] @ _error_matrix[:, j - v].T
    return (1 / n_simulations) * sum_over_simulations


def get_newey_west_weights(n_moments, n_simulations, band_width, _error_matrix):
    add = np.zeros((n_moments, n_moments))
    for v in range(1, band_width + 1):
        add += (1 - v / (band_width + 1)) * (get_gamma(v, n_simulations, _error_matrix)
                                             + get_gamma(v, n_simulations, _error_matrix).T)
    return np.linalg.inv(get_gamma(0, n_simulations, _error_matrix) + add)


if __name__ == "__main__":
    # Control parameters
    parameters = ["MARKET_AVERAGE_PRICE_DECAY", "BTL_PROBABILITY_MULTIPLIER", "BTL_CHOICE_INTENSITY",
                  "PSYCHOLOGICAL_COST_OF_RENTING", "SENSITIVITY_RENT_OR_PURCHASE"]
    experimentName = "Output-Calibration-2020-11-17"
    n_skip = 500  # Number of initial time steps to skip as burnout period
    # Target moments
    HPI_MEAN = 1.0
    HPI_STD = 0.3424
    HPI_PERIOD = 201.0
    RPI_MEAN = 1.0
    SHARE_HOUSEHOLDS_OWNING = 0.65
    SHARE_HOUSEHOLDS_RENTING = 0.17
    SHARE_HOUSEHOLDS_ACTIVE_BTL = 0.07526
    HOUSING_TRANSACTIONS_MEAN = 73564.17
    HOUSING_TRANSACTIONS_STD = 12723.93
    HOUSE_PRICE_TO_HOUSEHOLD_DISPOSABLE_INCOME_RATIO_MEAN = 3.81
    HOUSE_PRICE_TO_HOUSEHOLD_DISPOSABLE_INCOME_RATIO_STD = 0.11
    RENTAL_YIELD_MEAN = 5.10
    RESIDENTIAL_MORTGAGES_SPREAD_MEAN = 2.9985

    # Addresses (ADD HERE CORRECT PATHS)
    if platform.system() == "Windows":
        generalRoot = ""
        maxColWidth = -1
    else:
        generalRoot = ""
        maxColWidth = None
    rootExperiment = r"{}/results/{}".format(generalRoot, experimentName)
    
    # Create sorted list of files to read
    sortedFolders = sorted([f.path for f in os.scandir(Path("{}/Results/".format(rootExperiment))) if f.is_dir()])

    # Create vector of data moments from inputs given
    data_moments = np.array([[HPI_MEAN, HPI_STD, HPI_PERIOD, RPI_MEAN, SHARE_HOUSEHOLDS_OWNING,
                              SHARE_HOUSEHOLDS_RENTING, SHARE_HOUSEHOLDS_ACTIVE_BTL,
                              HOUSING_TRANSACTIONS_MEAN, HOUSING_TRANSACTIONS_STD,
                              HOUSE_PRICE_TO_HOUSEHOLD_DISPOSABLE_INCOME_RATIO_MEAN,
                              HOUSE_PRICE_TO_HOUSEHOLD_DISPOSABLE_INCOME_RATIO_STD,
                              RENTAL_YIELD_MEAN, RESIDENTIAL_MORTGAGES_SPREAD_MEAN]]).T
    nMoments = len(data_moments)

    # Define the weighting matrix for the different moment errors
    w_hat = np.eye(nMoments)
    w_hat[0][0] = 5    # HPI_MEAN
    w_hat[1][1] = 10   # HPI_STD
    w_hat[2][2] = 10   # HPI_PERIOD
    w_hat[3][3] = 1    # RPI_MEAN
    w_hat[4][4] = 1    # SHARE_HOUSEHOLDS_OWNING
    w_hat[5][5] = 2    # SHARE_HOUSEHOLDS_RENTING
    w_hat[6][6] = 2    # SHARE_ACTIVE_BTL
    w_hat[7][7] = 0    # HOUSING_TRANSACTIONS_MEAN
    w_hat[8][8] = 0    # HOUSING_TRANSACTIONS_STD
    w_hat[9][9] = 0    # HOUSE_PRICE_TO_HOUSEHOLD_DISPOSABLE_INCOME_RATIO_MEAN
    w_hat[10][10] = 0  # HOUSE_PRICE_TO_HOUSEHOLD_DISPOSABLE_INCOME_RATIO_STD
    w_hat[11][11] = 5  # RENTAL_YIELD_MEAN
    w_hat[12][12] = 1  # RESIDENTIAL_MORTGAGES_SPREAD_MEAN

    # Run through results folder for each process
    criterionListDict = []
    nSimulations = None
    rex = re.compile(r"\d+$")
    i = 0
    for processFolder in sortedFolders[0:]:
        i += 1
        if i % 1000 == 0:
            print(i)
        # if i == 5:
        #     break
        # Find process number
        if rex.search(processFolder):
            processNumber = rex.search(processFolder).group(0)
        else:
            print("Unable to find process number for folder {}".format(processFolder))
            processNumber = None
            exit()
        # Find process parameter values
        parameterValues = dict()
        with open("{}/config{}.properties".format(processFolder, processNumber), "r") as f:
            for line in f:
                if len(line.split()) > 0 and line.split()[0] in parameters:
                    parameterValues[line.split()[0]] = float(line.split()[2])

        # if not (parameterValues["MARKET_AVERAGE_PRICE_DECAY"] == 0.5
        #         and parameterValues["BTL_PROBABILITY_MULTIPLIER"] == 1.68
        #         and parameterValues["BTL_CHOICE_INTENSITY"] == 31.62
        #         and parameterValues["PSYCHOLOGICAL_COST_OF_RENTING"] == 0.1
        #         and parameterValues["SENSITIVITY_RENT_OR_PURCHASE"] == 0.0003162):
        # if not (parameterValues["MARKET_AVERAGE_PRICE_DECAY"] == 0.7
        #         and parameterValues["BTL_PROBABILITY_MULTIPLIER"] == 1.78
        #         and parameterValues["BTL_CHOICE_INTENSITY"] == 316.2
        #         and parameterValues["PSYCHOLOGICAL_COST_OF_RENTING"] == 0.1
        #         and parameterValues["SENSITIVITY_RENT_OR_PURCHASE"] == 0.03162):
        #     continue
        
        # For each simulation within this process, and thus with these parameter values, read results and
        # the compute corresponding simulation moments vector
        all_simulation_moments = get_all_simulation_moments(processFolder, n_skip)
        nSimulations = len(all_simulation_moments)

        # Compute the error matrix, where each (r, s) element is the contribution of the sth simulated moment to the rth
        # moment error...
        error_matrix = ((np.column_stack(all_simulation_moments) - np.ones((nMoments, nSimulations)) * data_moments) /
                        (np.ones((nMoments, nSimulations)) * data_moments))
        # ...the corresponding error vector
        error_vector = (1 / nSimulations) * error_matrix @ np.ones((nSimulations, 1))
        # ...and the corresponding criterion value (weighted sum of squared moment errors)
        criterion_value = error_vector.T @ w_hat @ error_vector
        
        #######################################################################
        # print(parameterValues)
        # print(criterion_value)
        # for i, name in enumerate(["HPI_MEAN", "HPI_STD", "HPI_PERIOD", "RPI_MEAN", "SHARE_HOUSEHOLDS_OWNING",
        #                           "SHARE_HOUSEHOLDS_RENTING", "SHARE_HOUSEHOLDS_ACTIVE_BTL",
        #                           "HOUSING_TRANSACTIONS_MEAN", "HOUSING_TRANSACTIONS_STD",
        #                           "HOUSE_PRICE_TO_HOUSEHOLD_DISPOSABLE_INCOME_RATIO_MEAN",
        #                           "HOUSE_PRICE_TO_HOUSEHOLD_DISPOSABLE_INCOME_RATIO_STD",
        #                           "RENTAL_YIELD_MEAN", "RESIDENTIAL_MORTGAGES_SPREAD_MEAN"]):
        #     print("{} ---> {}, {}, {}".format(name, np.mean(all_simulation_moments, axis=0)[i][0], data_moments[i][0],
        #                                       error_vector[i][0]))
        # exit()
        #######################################################################
        
        # Finally, add this criterion value to the parameter values dictionary and append it to the list of dicts
        parameterValues["Criterion Value (Identity)"] = criterion_value[0][0]
        parameterValues["Error Vector"] = error_vector
        parameterValues["Error Matrix"] = error_matrix
        criterionListDict.append(parameterValues)

    # Create a DataFrame from the criterion list of dicts
    criterionDF = pd.DataFrame(criterionListDict, columns=criterionListDict[0].keys())
    # plot_criterion_surface(criterionDF)
    min_row = criterionDF.loc[criterionDF["Criterion Value (Identity)"].idxmin(axis=0)]
    print("Minimum Criterion Value ({:.6f}) reached for ".format(min_row["Criterion Value (Identity)"])
          + ", ".join(["{} = {}".format(min_row.index[i], min_row[min_row.index[i]]) for i in range(5)]))

    # TWO-STEP VARIANCE-COVARIANCE ESTIMATOR FOR THE WEIGHTING MATRIX
    # Re-estimate the weighting matrix at the chosen model parameter values using the inverse of the variance-covariance
    # matrix of the moment vector (note that this uses the most recent nSimulations value). Note that re-estimating this
    # matrix at a different set of parameter values will lead to that set being chosen as minimising the criterion
    # function, as this re-estimation and weighting is thought as exploring the surroundings of the specific parameter
    # set chosen. Thus, it makes sense to stick to the specific set chosen in the first step (with an identity matrix)
    w_hat_2 = np.linalg.inv((1 / nSimulations) * min_row["Error Matrix"] @ min_row["Error Matrix"].T)
    criterionDF["Criterion Value (2-Step)"] = criterionDF.apply(
        lambda row: (row["Error Vector"].T @ w_hat_2 @ row["Error Vector"])[0][0], axis=1)

    # NEWEY-WEST ESTIMATOR FOR THE WEIGHTING MATRIX
    # Re-estimate the weighting matrix at the chosen model parameter values using its Newey-West consistent estimator
    # (note that this uses the most recent nSimulations value). As noted above, this weighting matrix is to be
    # re-estimated using these initially estimated parameter values, since the method is thought as exploring the
    # surroundings of the specific parameter set chosen. Thus, it makes sense to stick to the specific set chosen in the
    # first step (with an identity matrix)
    bandWidth = 3
    w_hat_nw = get_newey_west_weights(nMoments, nSimulations, bandWidth, min_row["Error Matrix"])
    criterionDF["Criterion Value (Newey-West)"] = criterionDF.apply(
        lambda row: (row["Error Vector"].T @ w_hat_nw @ row["Error Vector"])[0][0], axis=1)

    # ESTIMATION ERROR OF THE MODEL PARAMETERS
    # This, again uses the most recent nSimulations value
#    jac_err = get_jacobian_of_error_vector(criterionDF)
#    varCovarOfParamVector = (1 / nSimulations) * np.linalg.inv(jac_err.T @ w_hat @ jac_err)
#    print("Example of standard deviations of the parameters found (with different parameter values, though)")
#    print("BTL_CHOICE_INTENSITY = {}, Std. err. = {}".format(min_row["BTL_CHOICE_INTENSITY"],
#                                                             np.sqrt(varCovarOfParamVector[0, 0])))
#    print("SENSITIVITY_RENT_OR_PURCHASE = {}, Std. err. = {}".format(min_row["SENSITIVITY_RENT_OR_PURCHASE"],
#                                                                     np.sqrt(varCovarOfParamVector[1, 1])))

    # Print results
    criterionDF = criterionDF[[c for c in criterionDF.columns if c not in {"Error Vector", "Error Matrix",
                                                                           "Criterion Value (2-Step)",
                                                                           "Criterion Value (Newey-West)"}]]
    # with pd.option_context("display.max_columns", None, "max_colwidth", maxColWidth,
    #                        "display.expand_frame_repr", False):
    #     print(criterionDF)

    # for param in reversed(parameters):
    #     print("--- Order by {} ---".format(param))
    #     with pd.option_context("display.max_columns", None, "max_colwidth", maxColWidth,
    #                            "display.expand_frame_repr", False, "display.max_rows", None):
    #         print(criterionDF.sort_values(by=[col for col in parameters if col != param]))

    # Write to file criterionDF
    criterionDF.to_csv(".\CriterionDF11.csv", index=False, sep=";")
