# -*- coding: utf-8 -*-
"""
Class to compare values of the criterion function related with different model parameter values in order to assess
whether these parameters are relevant for the model results or not. This uses the simulation output resulting from
Launcher.py. In particular, the moments used are the means and standard deviations of all columns at Output-run*.csv
files, averaged over the different realisations *.

@author: Adrian Carro
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import platform
from pathlib import Path
import os
import re
import pandas as pd
import glob


def get_simulation_moments(_simulation_output_file, _n_skip):
    # Read results into pandas data frame, skipping first n_skip lines after the header
    _results = pd.read_csv(_simulation_output_file, delimiter=";", skipinitialspace=True,
                           skiprows=range(1, _n_skip + 1))
    _results = _results.drop(["Model time", "TotalPopulation", "HousingStock", "nNewBuild", "nUnsoldNewBuild"], axis=1)
    return np.vstack((np.array([_results.mean()]).T, np.array([_results.std()]).T))


def plot_criterion_surface(criterion_df):
    _fig = plt.figure()
    _ax = Axes3D(_fig)  # Apparently equivalent to _ax = _fig.gca(projection="3d")

    # Plot with triangular surfaces
    _ax.plot_trisurf(criterion_df["SENSITIVITY_RENT_OR_PURCHASE"],
                     criterion_df["BTL_CHOICE_INTENSITY"],
                     criterion_df["Criterion Value"],
                     cmap=matplotlib.cm.get_cmap('summer'), linewidth=1.5, edgecolor="black", alpha=0.75)

    # # Plot with rectilinear grids
    # x1 = np.linspace(criterion_df["SENSITIVITY_RENT_OR_PURCHASE"].min(),
    #                  criterion_df["SENSITIVITY_RENT_OR_PURCHASE"].max(),
    #                  len(criterion_df["SENSITIVITY_RENT_OR_PURCHASE"].unique()))
    # y1 = np.linspace(criterion_df["BTL_CHOICE_INTENSITY"].min(),
    #                  criterion_df["BTL_CHOICE_INTENSITY"].max(),
    #                  len(criterion_df["BTL_CHOICE_INTENSITY"].unique()))
    # x2, y2 = np.meshgrid(x1, y1)
    # from scipy import interpolate
    # z2 = interpolate.griddata((criterion_df["SENSITIVITY_RENT_OR_PURCHASE"], criterion_df["BTL_CHOICE_INTENSITY"]),
    #                           criterion_df["Criterion Value"], (x2, y2), method="cubic")
    # _ax.plot_surface(x2, y2, z2, rstride=1, cstride=1, cmap=matplotlib.cm.get_cmap('summer'))

    # Final plot details
    _ax.set_title("Criterion function for different parameter values")
    _ax.set_xlabel("SENSITIVITY_RENT_OR_PURCHASE")
    _ax.set_ylabel("BTL_CHOICE_INTENSITY")
    _ax.set_zlabel("Criterion Function")
    plt.show()


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
    for i in range(v, n_simulations):
        sum_over_simulations += _error_matrix[:, i] @ _error_matrix[:, i - v].T
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
    experimentName = "Discarding-Irrelevant-Params-2020-11-05"
    n_skip = 500  # Number of initial time steps to skip as burnout period
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

    # Read simulation results from the first simulation of the first parameter set as data and compute the corresponding
    # simulation moments vector
    all_data_moments = []
    for data_output_file in glob.glob("{}/Output-run1.csv".format(sortedFolders[0])):
        all_data_moments.append(get_simulation_moments(data_output_file, n_skip))
    data_moments = np.mean(all_data_moments, axis=0)
    nMoments = len(data_moments)

    # Define the weighting matrix for the different moment errors
    w_hat = np.eye(nMoments)

    # Run through results folder for each process
    criterionListDict = []
    nSimulations = None
    rex = re.compile(r"\d+$")
    i = 0
    for processFolder in sortedFolders[0:]:
        i += 1
        if i % 10 == 0:
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
        # For each simulation within this process, and thus with these parameter values...
        all_simulation_moments = []
        for simulation_output_file in glob.glob("{}/Output-run*.csv".format(processFolder)):
            # ...read simulation results and compute the corresponding simulation moments vector
            all_simulation_moments.append(get_simulation_moments(simulation_output_file, n_skip))
        nSimulations = len(all_simulation_moments)

        # Compute the error matrix, where each (r, s) element is the contribution of the sth simulated moment to the rth
        # moment error...
        error_matrix = ((np.column_stack(all_simulation_moments) - np.ones((nMoments, nSimulations)) * data_moments) /
                        (np.ones((nMoments, nSimulations)) * data_moments))
        # ...the corresponding error vector
        error_vector = (1 / 10) * error_matrix @ np.ones((10, 1))
        # ...and the corresponding criterion value (weighted sum of squared moment errors)
        criterion_value = error_vector.T @ w_hat @ error_vector
        # Finally, add this criterion value to the parameter values dictionary and append it to the list of dicts
        parameterValues["Criterion Value (Identity)"] = criterion_value[0][0]
        parameterValues["Error Vector"] = error_vector
        parameterValues["Error Matrix"] = error_matrix
        criterionListDict.append(parameterValues)

    # Create a DataFrame from the criterion list of dicts
    criterionDF = pd.DataFrame(criterionListDict, columns=criterionListDict[0].keys())
    # plot_criterion_surface(criterionDF)
    # with pd.option_context("display.max_columns", None, "max_colwidth", None, "display.expand_frame_repr", False):
    #     print(criterionDF)
    min_row = criterionDF.loc[criterionDF["Criterion Value (Identity)"].idxmin(axis=0)]
    print("Minimum Criterion Value ({:.6f}) reached for {} = {} and {} = {}".format(
        min_row["Criterion Value (Identity)"], min_row.index[0], min_row[min_row.index[0]], min_row.index[1],
        min_row[min_row.index[1]]))

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
    
    # Print results
    criterionDF = criterionDF[[c for c in criterionDF.columns if c not in {"Error Vector", "Error Matrix",
                                                                           "Criterion Value (2-Step)",
                                                                           "Criterion Value (Newey-West)"}]]
    # with pd.option_context("display.max_columns", None, "max_colwidth", maxColWidth,
    #                        "display.expand_frame_repr", False):
    #     print(criterionDF)

    for param in reversed(parameters):
        print("--- Order by {} ---".format(param))
        with pd.option_context("display.max_columns", None, "max_colwidth", maxColWidth,
                               "display.expand_frame_repr", False, "display.max_rows", None):
            print(criterionDF.sort_values(by=[col for col in parameters if col != param]))
