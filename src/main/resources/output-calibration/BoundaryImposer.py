# -*- coding: utf-8 -*-
"""
Class to impose an acceptability region bounding acceptable values for the model parameters, using the simulation output
resulting from Launcher.py.

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


def get_simulation_moments(_simulation_output_folder, _n_skip):
    rental_yield_mean = []
    with open(_simulation_output_folder + "/coreIndicator-rentalYield.csv") as f:
        for _line in f:
            rental_yield_mean.append(np.mean([float(element) for element in _line.split(";")[_n_skip + 1:]]))
    spread_mean = []
    with open(_simulation_output_folder + "/coreIndicator-interestRateSpread.csv") as f:
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
                                                  _results["fActiveBTL"].mean(),
                                                  rental_yield_mean[j], spread_mean[j]]))
    return np.mean(_all_simulation_moments, axis=0)


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
    HPI_MEAN = [0.75, 1.25]  # Target mean is 1.0
    HPI_STD = [0.1, 0.5]  # Target std is 0.3424
    HPI_PERIOD = [150, 250]  # Target period is 201.0
    RPI_MEAN = [0.75, 1.25]  # Target mean is 1.0
    SHARE_HOUSEHOLDS_OWNING = [0.4875, 0.8125]  # Target mean is 0.65
    SHARE_HOUSEHOLDS_RENTING = [0.1275, 0.2125]  # Target mean is 0.17
    SHARE_HOUSEHOLDS_ACTIVE_BTL = [0.056445, 0.094075]  # Target mean is 0.07526
    RENTAL_YIELD_MEAN = [3.825, 6.375]  # Target mean is 5.10
    RESIDENTIAL_MORTGAGES_SPREAD_MEAN = [2.248875, 3.748125]  # Target mean is 2.9985

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
    data_moments = np.array([HPI_MEAN, HPI_STD, HPI_PERIOD, RPI_MEAN, SHARE_HOUSEHOLDS_OWNING,
                             SHARE_HOUSEHOLDS_RENTING, SHARE_HOUSEHOLDS_ACTIVE_BTL,
                             RENTAL_YIELD_MEAN, RESIDENTIAL_MORTGAGES_SPREAD_MEAN])

    t0 = time.time()
    # Run through results folder for each process
    acceptableProcessNumbers = []
    rex = re.compile(r"\d+$")
    i = 0
    for processFolder in sortedFolders[0:]:
        i += 1
        if i % 100 == 0:
            print(i, time.time() - t0)
        #     t0 = time.time()
        # if i == 20:
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
        simulation_moments = get_simulation_moments(processFolder, n_skip)
        
        if all(limits[0] < value[0] < limits[1] for value, limits in zip(simulation_moments, data_moments)):
            acceptableProcessNumbers.append(processNumber)

    with open("AcceptableProcessNumbersRelaxed.csv", "w") as f:
        for acceptableProcessNumber in acceptableProcessNumbers:
            f.write(acceptableProcessNumber + "\n")
