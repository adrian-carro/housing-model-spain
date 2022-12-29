# coding: utf-8

# Imports
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

    # Create sorted list of folders to read from
    sortedAdditionalFolders = sorted([f.path for f in os.scandir(Path("{}/Results/".format(rootAdditionalRuns)))
                                      if f.is_dir()])
    originalFolder = rootOriginalRuns

    # Run through results folder for each process
    nSimulations = None
    rex_folder = re.compile(r"\d+$")
    rex_file = re.compile(r"\d+(?=\.)")

    for originalFileName in glob.glob("{}/coreIndicator-*.csv".format(originalFolder)):

        with open(originalFileName, 'a') as originalFile:

            for additionalFolder in sortedAdditionalFolders:
                additionalFileName = additionalFolder + '/' + str(Path(originalFileName).name)
                with open(additionalFileName, 'r') as additionalFile:
                    for line in additionalFile:
                        originalFile.write('\n' + line.strip('\n'))

else:
    print('Not yet implemented!')
