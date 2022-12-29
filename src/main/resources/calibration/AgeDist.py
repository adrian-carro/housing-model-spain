# -*- coding: utf-8 -*-
"""
Class to study households' age distribution based on Encuesta Financiera de las Familias 2016 data (allowing for a
comparison with Wealth and Assets Survey 2011 data). This is the code used to create files "EFF-Age9-Weighted.csv",
"EFF-Age15-Weighted.csv", "WAS-Age9-Weighted.csv" and "WAS-Age15-Weighted.csv". Note that the age variable is never
imputed, and thus this code does not require the use of multiple imputation techniques.

@author: Adrian Carro
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Set parameters
writeResults = False
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 1000)

# Read data
rootWAS = r''
dfWAS = pd.read_csv(rootWAS + r'/was_wave_3_hhold_eul_final.dta', usecols={'w3xswgt', 'HRPDVAge9W3', 'HRPDVAge15w3'})
rootEFF = r''
dfEFF = pd.read_csv(rootEFF + r'/eff_2017_imp1_csv/otras_secciones_2017_imp1.csv', sep=';',
                    usecols={'facine3', 'p1_2d_1'})
rootResults = r''

# EFF variables of interest
# facine3           Household weight
# p1_2d_1           Age of HRP

# WAS variables of interest
# HRPDVAge9W3       Age of HRP or partner [0-15, 16-24, 25-34, 35-44, 45-54, 55-64, 65-74, 75-84, 85+]
# HRPDVAge15w3      Age of HRP/Partner Banded (15) [0-16, 17-19, 20-24, 25-29, 30-34, 35-39, 40-44, 45-49,
#                                                   50-54, 55-59, 60-64, 65-69, 70-74, 75-79, 80+]

# Rename columns to be used
dfWAS.rename(columns={'w3xswgt': 'Weight'}, inplace=True)
dfWAS.rename(columns={'HRPDVAge9W3': 'Age9'}, inplace=True)
dfWAS.rename(columns={'HRPDVAge15w3': 'Age15'}, inplace=True)
dfEFF.rename(columns={'facine3': 'Weight'}, inplace=True)
dfEFF.rename(columns={'p1_2d_1': 'Age'}, inplace=True)

# Filter down to keep only columns of interest
dfWAS = dfWAS[['Age9', 'Age15', 'Weight']]
dfEFF = dfEFF[['Age', 'Weight']]

# Map age buckets to middle of bucket value by creating the corresponding dictionary
dfWAS['Age9'] = dfWAS['Age9'].map({'16-24': 20, '25-34': 30, '35-44': 40, '45-54': 50, '55-64': 60, '65-74': 70,
                                   '75-84': 80, '85+': 90})
dfWAS['Age15'] = dfWAS['Age15'].map({'17-19': 17.5, '20-24': 22.5, '25-29': 27.5, '30-34': 32.5, '35-39': 37.5,
                                     '40-44': 42.5, '45-49': 47.5, '50-54': 52.5, '55-59': 57.5, '60-64': 62.5,
                                     '65-69': 67.5, '70-74': 72.5, '75-79': 77.5, '80+': 82.5})
age9_bin_edges = [15, 25, 35, 45, 55, 65, 75, 85, 95]
age15_bin_edges = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]

# Create WAS data histograms
frequencyWAS9, xBinsWAS9 = np.histogram(dfWAS['Age9'].values, bins=age9_bin_edges, density=True,
                                        weights=dfWAS['Weight'].values)
frequencyWAS15, xBinsWAS15 = np.histogram(dfWAS['Age15'].values, bins=age15_bin_edges, density=True,
                                          weights=dfWAS['Weight'].values)

# Create EFF data histograms
frequencyEFF9, xBinsEFF9 = np.histogram(dfEFF['Age'].values, bins=age9_bin_edges, density=True,
                                        weights=dfEFF['Weight'].values)
frequencyEFF15, xBinsEFF15 = np.histogram(dfEFF['Age'].values, bins=age15_bin_edges, density=True,
                                          weights=dfEFF['Weight'].values)

if not writeResults:
    plt.figure()
    plt.bar(xBinsWAS9[:-1], frequencyWAS9, width=(xBinsWAS9[1] - xBinsWAS9[0]), label='WAS 2011')
    plt.bar(xBinsEFF9[:-1], frequencyEFF9, width=(xBinsEFF9[1]-xBinsEFF9[0]), alpha=0.5, label='EFF 2016')
    plt.legend()
    plt.title('Age9')
    plt.figure()
    plt.bar(xBinsWAS15[:-1], frequencyWAS15, width=(xBinsWAS15[1] - xBinsWAS15[0]), label='WAS 2011')
    plt.bar(xBinsEFF15[:-1], frequencyEFF15, width=(xBinsEFF15[1]-xBinsEFF15[0]), alpha=0.5, label='EFF 2016')
    plt.legend()
    plt.title('Age15')
    plt.show()
else:
    # Print WAS distributions to files
    with open(rootResults + r'/WAS-Age9.csv', 'w') as f:
        f.write('# Age (lower edge), Age (upper edge), Probability\n')
        for element, lowerEdge, upperEdge in zip(frequencyWAS9, xBinsWAS9[:-1], xBinsWAS9[1:]):
            f.write('{}, {}, {}\n'.format(lowerEdge, upperEdge, element))
    with open(rootResults + '/WAS-Age15.csv', 'w') as f:
        f.write('# Age (lower edge), Age (upper edge), Probability\n')
        for element, lowerEdge, upperEdge in zip(frequencyWAS15, xBinsWAS15[:-1], xBinsWAS15[1:]):
            f.write('{}, {}, {}\n'.format(lowerEdge, upperEdge, element))
    # Print EFF distributions to files
    with open(rootResults + '/EFF-Age9.csv', 'w') as f:
        f.write('# Age (lower edge), Age (upper edge), Probability\n')
        for element, lowerEdge, upperEdge in zip(frequencyEFF9, xBinsEFF9[:-1], xBinsEFF9[1:]):
            f.write('{}, {}, {}\n'.format(lowerEdge, upperEdge, element))
    with open(rootResults + '/EFF-Age15.csv', 'w') as f:
        f.write('# Age (lower edge), Age (upper edge), Probability\n')
        for element, lowerEdge, upperEdge in zip(frequencyEFF15, xBinsEFF15[:-1], xBinsEFF15[1:]):
            f.write('{}, {}, {}\n'.format(lowerEdge, upperEdge, element))
