# -*- coding: utf-8 -*-
"""
Class to explore the correlation between house price growth and credit growth in the model output.

@author: Adrian Carro
"""

from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt


# Set address for results
rootResults = r""

# Read model results
# House Price Growth
# This number is computed as the quarter on quarter house price index(HPI) growth, that is, as the percentage change of
# the HPI three months on three months earlier
with open(rootResults + "/coreIndicator-housePriceGrowth.csv", "r") as f:
    hpg = list(map(float, f.readline().split(";")))
# Credit Growth
# Defined as the twelve-month nominal growth rate of credit, that is, as the twelve-month cumulative net flow of
# credit divided by the stock in the initial quarter, or, in other words, as the current stock of credit minus the stock
# of credit twelve months ago divided by the stock of credit twelve months ago
with open(rootResults + "/coreIndicator-creditGrowth.csv", "r") as f:
    cg = list(map(float, f.readline().split(";")))
# Now read both House Price Growth and Credit Growth from general output file
df = pd.read_csv(rootResults + "/Output-run1.csv", skipinitialspace=True, delimiter=";")

# Plot time series of house price growth and credit growth
plt.figure()
plt.plot(range(len(hpg)), hpg, label="hpg")
# plt.plot(range(len(hpg)), 100.0*df["Sale HPI"].pct_change(periods=12).values, label="hpg2")
plt.plot(range(len(cg)), cg, label="cg")

# Plot house price growth against credit growth
plt.figure()
plt.scatter(cg[500:], hpg[500:], c="b", label="Core Indicators")
plt.scatter(cg[500:], 100.0*df["Sale HPI"].pct_change(periods=12).values[500:], c="r", label="Annual HPI growth")

plt.legend()
plt.show()
