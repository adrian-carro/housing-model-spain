# -*- coding: utf-8 -*-
"""
Class to launch multiple realizations of the housing model code.

@author: Adrian Carro
"""

from __future__ import division
import numpy as np
from pathlib import Path
from shutil import rmtree
import platform
import itertools
import subprocess
import multiprocessing
import time


def create_folder(_address):
    try:
        Path(_address).mkdir(parents=False, exist_ok=False)
    except FileNotFoundError:
        print("Parents to the root \"{}\" do not exist!".format(_address))
        exit()
    except FileExistsError:
        if any(Path(_address).iterdir()):
            print("Folder \"{}\" already exists and it is not empty!".format(_address))
            if not debug:
                print("Debug mode off: If you want to continue, turn \"debug\" on.")
                exit()
            else:
                print("Debug mode on: Deleting and re-creating required folders.")
                rmtree(Path(_address))
                create_folder(_address)


def run_command(_mvn_run, _experiment_name, _general_root, _i, _width):
    print("Running process {}...".format(_i))
    if platform.system() == "Windows":
        config_file_address = "results\\{}\\ConfigFiles\\config{:0{width}d}.properties".format(_experiment_name, _i,
                                                                                               width=_width)
        output_folder_address = "results\\{}\\Results\\Process{:0{width}d}\\".format(_experiment_name, _i,
                                                                                     width=_width)
        stdout_address = "results\\{}\\Results\\Process{:0{width}d}\\stdout.txt".format(_experiment_name, _i,
                                                                                        width=_width)
    else:
        config_file_address = "results/{}/ConfigFiles/config{:0{width}d}.properties".format(_experiment_name, _i,
                                                                                            width=_width)
        output_folder_address = "results/{}/Results/Process{:0{width}d}/".format(_experiment_name, _i,
                                                                                 width=_width)
        stdout_address = "results/{}/Results/Process{:0{width}d}/stdout.txt".format(_experiment_name, _i,
                                                                                    width=_width)
    _time0 = time.time()
    subprocess.run("{} > {}".format(_mvn_run.format(config_file_address, output_folder_address), stdout_address),
                   check=True, universal_newlines=True, cwd=_general_root, shell=True)
    print("Process {} finished in {:.2f} seconds".format(_i, (time.time() - _time0)))


if __name__ == '__main__':
    # Control parameters
    debug = True  # Note this will erase everything within the "experimentName" folder within results
    # Output calibration parameters
    parameters = {"MARKET_AVERAGE_PRICE_DECAY": [0.1, 0.3, 0.5, 0.7, 0.9],
                  "BTL_PROBABILITY_MULTIPLIER": [1.6, 1.62, 1.64, 1.66, 1.68, 1.7, 1.72, 1.74, 1.76, 1.78, 1.8],
                  "BTL_CHOICE_INTENSITY": [0.1, 0.3162, 1, 3.162, 10, 31.62, 100, 316.2, 1000],
                  "PSYCHOLOGICAL_COST_OF_RENTING": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                  "SENSITIVITY_RENT_OR_PURCHASE": [0.00001, 0.00003162, 0.0001, 0.0003162, 0.001, 0.003162, 0.01,
                                                   0.03162, 0.1]}
    nProcesses = 34
    experimentName = "Output-Calibration-2020-11-17"

    # Maven commands (ADD HERE CORRECT PATHS)
    win_mvn = ""
    if platform.system() == "Windows":
        mvn_compile = win_mvn + "--batch-mode clean validate compile"
        mvn_run = win_mvn + "--batch-mode exec:java \"-Dexec.args=-configFile {} -outputFolder {}\""
    else:
        mvn_compile = "mvn --batch-mode clean validate compile"
        mvn_run = "mvn --batch-mode exec:java \"-Dexec.args=-configFile {} -outputFolder {}\""
    if debug:
        mvn_run = mvn_run[:-1] + " -dev\""
    # Addresses (ADD HERE CORRECT PATHS)
    if platform.system() == "Windows":
        generalRoot = ""
    else:
        generalRoot = ""
    defaultConfigFile = r"{}/src/main/resources/config.properties".format(generalRoot)
    rootExperiment = r"{}/results/{}".format(generalRoot, experimentName)

    # Create required folders
    create_folder(rootExperiment)
    create_folder("{}/ConfigFiles".format(rootExperiment))
    create_folder("{}/Results".format(rootExperiment))
    print("- - - - - - - - - -")
    
    # Open general control output file
    f_out = open("{}/Results/stdoutRun.txt.".format(rootExperiment), "w")

    # Read default config file
    with open(defaultConfigFile, "r") as f:
        defaultConfig = f.readlines()

    # Run through all combinations of parameter values writing the corresponding config file, based on the default one
    # nCombinations = len(list(itertools.product(*parameters.values())))
    nCombinations = len(list(zip(*parameters.values())))
    width = len(str(nCombinations))
    i = 1
    # for elements in itertools.product(*parameters.values()):
    for elements in zip(*parameters.values()):
        tempElementDict = {name: value for name, value in zip(parameters.keys(), elements)}
        with open("{}/ConfigFiles/config{:0{width}d}.properties".format(rootExperiment, i, width=width), "w") as f:
            for line in defaultConfig:
                if len(line.split()) > 0 and line.split()[0] in parameters.keys():
                    f.write(line.replace(line.split()[2], str(tempElementDict[line.split()[0]])))
                else:
                    f.write(line)
        i += 1

    # Create a folder within Results for each process to be run
    for i in range(1, nCombinations + 1):
        create_folder("{}/Results/Process{:0{width}d}".format(rootExperiment, i, width=width))

    # Compile java code (via maven)
    print("Compiling...")
    f_out.write("Compiling...\n")
    subprocess.run("{} > \"".format(mvn_compile)
                   + str(Path("{}/Results/stdoutCompilation.txt".format(rootExperiment))) + "\"",
                   check=True, universal_newlines=True, cwd=generalRoot, shell=True)

    # Launch a java simulation per config file (via maven)
    print("Launching multiprocessing pool...")
    f_out.write("Launching multiprocessing pool...\n")
    time0 = time.time()
    with multiprocessing.Pool(processes=nProcesses) as pool:
        pool.starmap(run_command, [(mvn_run, experimentName, generalRoot, i, width)
                                   for i in range(1, nCombinations + 1)])
    print("Full processing completed in {:.2f} seconds".format(time.time() - time0))
    f_out.write("Full processing completed in {:.2f} seconds".format(time.time() - time0))
    
    # Close general control output file
    f_out.close()
