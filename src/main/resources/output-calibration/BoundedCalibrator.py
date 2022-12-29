# -*- coding: utf-8 -*-
"""
Class to estimate values for the model parameters using the simulation output resulting from Launcher.py and the
acceptability region as bounded by BoundaryImposer.py.

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
import time


def get_simulation_moments(_simulation_output_file, _n_skip):
    # Read results into pandas data frame, skipping first n_skip lines after the header
    _results = pd.read_csv(_simulation_output_file, delimiter=";", skipinitialspace=True,
                           skiprows=range(1, _n_skip + 1), usecols={"nRenting", "nOwnerOccupier", "Sale nSales",
                                                                    "nActiveBTL"})
    return np.vstack((np.array([_results.mean()]).T, np.array([_results.std()]).T))


def get_all_simulation_moments(_simulation_output_folder, _n_skip):
    # First, read results from core indicator files and temporarily store them in lists
    rental_yield_mean = []
    with open(_simulation_output_folder + "/coreIndicator-rentalYield.csv") as _f:
        for _line in _f:
            rental_yield_mean.append(np.mean([float(element) for element in _line.split(";")[_n_skip + 1:]]))
    spread_mean = []
    with open(_simulation_output_folder + "/coreIndicator-interestRateSpread.csv") as _f:
        for _line in _f:
            spread_mean.append(np.mean([float(element) for element in _line.split(";")[_n_skip + 1:]]))
    
    # Then, read results from general output files and create a list of vectors of moments
    _all_simulation_moments = []
    for k, _simulation_output_file in enumerate(glob.glob("{}/Output-run*.csv".format(_simulation_output_folder))):
        _results = pd.read_csv(_simulation_output_file, delimiter=";", skipinitialspace=True,
                               skiprows=range(1, _n_skip + 1), usecols={"nRenting", "nOwnerOccupier", "nActiveBTL",
                                                                        "TotalPopulation", "Sale HPI", "Rental HPI"})
        _results["fHomeOwner"] = (_results["nOwnerOccupier"] + _results["nActiveBTL"]) / _results["TotalPopulation"]
        _results["fRenting"] = _results["nRenting"] / _results["TotalPopulation"]
        _results["fActiveBTL"] = _results["nActiveBTL"] / _results["TotalPopulation"]
        _all_simulation_moments.append(np.vstack([_results["Sale HPI"].mean(), _results["Sale HPI"].std(),
                                                  get_period(_results["Sale HPI"]), _results["Rental HPI"].mean(),
                                                  _results["fHomeOwner"].mean(), _results["fRenting"].mean(),
                                                  _results["fActiveBTL"].mean(),
                                                  rental_yield_mean[k], spread_mean[k]]))
    return _all_simulation_moments


def get_period(_time_series):
    n = len(_time_series)  # Number of sample points
    fast_fourier_transform_hpi = (1.0 / n) * np.abs(fft(_time_series)[1:int(n / 2)])
    frequency_domain = np.linspace(0.0, 1.0, n)[1:int(n / 2)]  # This assumes sample spacing of 1
    return 1 / frequency_domain[fast_fourier_transform_hpi.argmax()]


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
    
    # Read list of process numbers with acceptable moments (within boundaries imposed by BoundaryImposer)
    acceptableProcessNumbers = []
    with open("AcceptableProcessNumbersRelaxed.csv", "r") as f:
        for line in f:
            acceptableProcessNumbers.append(line.strip("\n"))

    # Create sorted list of files to read
    sortedFolders = sorted([f.path for f in os.scandir(Path("{}/Results/".format(rootExperiment)))
                            if f.is_dir() and f.name.strip("Process") in acceptableProcessNumbers])

    # Create vector of data moments from inputs given
    data_moments = np.array([[HPI_MEAN, HPI_STD, HPI_PERIOD, RPI_MEAN, SHARE_HOUSEHOLDS_OWNING,
                              SHARE_HOUSEHOLDS_RENTING, SHARE_HOUSEHOLDS_ACTIVE_BTL,
                              RENTAL_YIELD_MEAN, RESIDENTIAL_MORTGAGES_SPREAD_MEAN]]).T
    nMoments = len(data_moments)

    # Define the weighting matrix for the different moment errors
    w_hat = np.eye(nMoments)
    w_hat[0][0] = 5  # HPI_MEAN
    w_hat[1][1] = 5  # HPI_STD
    w_hat[2][2] = 5  # HPI_PERIOD
    w_hat[3][3] = 1  # RPI_MEAN
    w_hat[4][4] = 1  # SHARE_HOUSEHOLDS_OWNING
    w_hat[5][5] = 2  # SHARE_HOUSEHOLDS_RENTING
    w_hat[6][6] = 2  # SHARE_ACTIVE_BTL
    w_hat[7][7] = 5  # RENTAL_YIELD_MEAN
    w_hat[8][8] = 1  # RESIDENTIAL_MORTGAGES_SPREAD_MEAN

    # Run through results folder for each process
    t0 = time.time()
    criterionListDict = []
    nSimulations = None
    rex = re.compile(r"\d+$")
    i = 0
    for processFolder in sortedFolders[0:]:
        i += 1
        if i % 10 == 0:
            print(i, time.time() - t0)
        #     t0 = time.time()
        # if i == 50:
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
        
        # Finally, add this criterion value, as well as the moments and their errors, to the parameter values
        # dictionary and append it to the list of dicts
        parameterValues["Criterion Value"] = criterion_value[0][0]
        for j, name in enumerate(["HPI_MEAN", "HPI_STD", "HPI_PERIOD", "RPI_MEAN", "SHARE_HOUSEHOLDS_OWNING",
                                  "SHARE_HOUSEHOLDS_RENTING", "SHARE_HOUSEHOLDS_ACTIVE_BTL",
                                  "RENTAL_YIELD_MEAN", "RESIDENTIAL_MORTGAGES_SPREAD_MEAN"]):
            parameterValues[name] = np.mean(all_simulation_moments, axis=0)[j][0]
            parameterValues["Error " + name] = error_vector[j][0]
        criterionListDict.append(parameterValues)

    # Create a DataFrame from the criterion list of dicts
    criterionDF = pd.DataFrame(criterionListDict, columns=criterionListDict[0].keys())

    # Select and print minimum criterion value row
    min_row = criterionDF.loc[criterionDF["Criterion Value"].idxmin(axis=0)]
    print("Minimum Criterion Value ({:.6f}) reached for ".format(min_row["Criterion Value"])
          + ", ".join(["{} = {}".format(min_row.index[i], min_row[min_row.index[i]]) for i in range(5)]))

    # Write to file criterionDF
    criterionDF.to_csv(".\CriterionDF13.csv", index=False, sep=";")
