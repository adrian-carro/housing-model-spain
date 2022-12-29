# coding: utf-8

# Imports
import numpy as np
from pathlib import Path
import os
import re
import glob


# Set parameters
singleParametrisation = True
fieldsToDismiss = ['SEED', 'N_SIMS']
rootOriginalRuns = r''
rootAdditionalRuns = r''

if singleParametrisation:
    # Find last run number within the original folder
    rex_folder = re.compile(r"\d+$")
    rex_file = re.compile(r"\d+(?=\.)")
    lastOriginalRunNumber = np.max([int(rex_file.search(f).group(0))
                                    for f in glob.glob("{}/*-run*.csv".format(rootOriginalRuns))])

    # Move files renaming them to avoid collisions
    for file in glob.glob("{}/*-run*.csv".format(rootAdditionalRuns)):
        oldAdditionalRunNumber = int(rex_file.search(file).group(0))
        newAdditionalRunNumber = oldAdditionalRunNumber + lastOriginalRunNumber
        newFileName = re.sub(r"\d+(?=\.)", str(newAdditionalRunNumber), Path(file).name)
        os.rename(file, rootOriginalRuns + '/' + newFileName)

else:
    # Create sorted list of folders to read from
    sortedOriginalFolders = sorted([f.path for f in os.scandir(Path("{}/Results/".format(rootOriginalRuns)))
                                    if f.is_dir()])
    sortedAdditionalFolders = sorted([f.path for f in os.scandir(Path("{}/Results/".format(rootAdditionalRuns)))
                                      if f.is_dir()])

    # Check if same number of processes/sub-folders is the same
    if len(sortedOriginalFolders) != len(sortedAdditionalFolders):
        print('The number of processes/sub-folders must be the same between both folders!')
        exit()

    # Run through results folder for each process
    nSimulations = None
    rex_folder = re.compile(r"\d+$")
    rex_file = re.compile(r"\d+(?=\.)")
    i = 0
    for originalFolder, additionalFolder in zip(sortedOriginalFolders, sortedAdditionalFolders):
        # Find process number
        if rex_folder.search(originalFolder):
            processNumber = rex_folder.search(originalFolder).group(0)
        else:
            print("Unable to find process number for folder {}".format(originalFolder))
            processNumber = None
            exit()

        # Check that config files are equal, except for the seed
        with open("{}/config{}.properties".format(originalFolder, processNumber), "r") as f1, \
                open("{}/config{}.properties".format(additionalFolder, processNumber), "r") as f2:
            for line1, line2 in zip(f1, f2):
                if line1 != line2 and line1.split()[0] not in fieldsToDismiss:
                    print('Warning: Different parameters found!')
                    print(line1[:-1])
                    print(line2[:-1])
                    exit()

        # Find last run number within the original folder
        lastOriginalRunNumber = np.max([int(rex_file.search(f).group(0))
                                        for f in glob.glob("{}/*-run*.csv".format(originalFolder))])

        # Move files renaming them to avoid collisions
        for file in glob.glob("{}/*-run*.csv".format(additionalFolder)):
            oldAdditionalRunNumber = int(rex_file.search(file).group(0))
            newAdditionalRunNumber = oldAdditionalRunNumber + lastOriginalRunNumber
            newFileName = re.sub(r"\d+(?=\.)", str(newAdditionalRunNumber), Path(file).name)
            os.rename(file, originalFolder + '/' + newFileName)
