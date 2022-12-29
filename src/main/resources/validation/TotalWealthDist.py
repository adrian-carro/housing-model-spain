# -*- coding: utf-8 -*-
"""
Class to study households' total wealth (financial + housing) distribution, for validation purposes, based on Wealth and
Assets Survey data.

@author: Adrian Carro
"""

from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def readResults(file_name, _start_time, _end_time):
    """Read micro-data from file_name, structured on a separate line per year. In particular, read from start_year until
    end_year, both inclusive"""
    # Read list of float values, one per household
    data_float = []
    with open(file_name, "r") as _f:
        for line in _f:
            if _start_time <= int(line.split(';')[0]) <= _end_time:
                for column in line.split(';')[1:]:
                    data_float.append(float(column))
    return data_float


# Set control variables and addresses. Note that available variables to print and plot are "NetPropertyWealth",
# "GrossPropertyWealth" and "GrossHousingWealth" for housing wealth and "GrossFinancialWealth", "NetFinancialWealth"
# and "LiqFinancialWealth" for financial wealth
printResults = False
plotResults = True
printPlots = False
start_time = 1000
end_time = 2000
population = "all"  # Can be "all", "non-owners", "owners"
min_log_bin_edge = 6.0
max_log_bin_edge = 16.0
financialWealthVariable = "NetFinancialWealth"
housingWealthVariable = "NetPropertyWealth"
rootData = r""
rootResults = r""

# Read Wealth and Assets Survey data for households
chunk = pd.read_csv(rootData + r"/was_wave_3_hhold_eul_final.dta", usecols={"w3xswgt", "HFINWW3_sum", "HFINWNTW3_sum",
                                                                            "DVFNSValW3_aggr", "DVCACTvW3_aggr",
                                                                            "DVCASVVW3_aggr", "DVSaValW3_aggr",
                                                                            "DVCISAVW3_aggr", "DVCaCrValW3_aggr",
                                                                            "HPROPWW3", "DVPropertyW3", "DVHValueW3",
                                                                            "DVHseValW3_sum", "DVBltValW3_sum"})

# List of household variables currently used
# HFINWW3_sum                   Gross Financial Wealth (financial assets only)
# HFINWNTW3_sum                 Household Net financial Wealth (financial assets minus financial liabilities)
# HPROPWW3                      Total (net) property wealth (net, i.e., = DVPropertyW3 - HMORTGW3)
# DVPropertyW3                  Total (gross) property wealth (sum of all property values)
# DVHValueW3                    Value of main residence
# DVHseValW3_sum                Total value of other houses
# DVBltValW3_sum                Total value of buy to let houses

# Rename the different measures of financial wealth
chunk.rename(columns={"w3xswgt": "Weight"}, inplace=True)
chunk.rename(columns={"HFINWW3_sum": "GrossFinancialWealth"}, inplace=True)
chunk.rename(columns={"HFINWNTW3_sum": "NetFinancialWealth"}, inplace=True)
chunk["LiqFinancialWealth"] = chunk["DVFNSValW3_aggr"].astype(float) + chunk["DVCACTvW3_aggr"].astype(float)\
                           + chunk["DVCASVVW3_aggr"].astype(float) + chunk["DVSaValW3_aggr"].astype(float)\
                           + chunk["DVCISAVW3_aggr"].astype(float) + chunk["DVCaCrValW3_aggr"].astype(float)
# Rename the different measures of housing wealth
chunk.rename(columns={"HPROPWW3": "NetPropertyWealth"}, inplace=True)
chunk.rename(columns={"DVPropertyW3": "GrossPropertyWealth"}, inplace=True)
chunk["GrossHousingWealth"] = chunk["DVHValueW3"].astype(float) + chunk["DVHseValW3_sum"].astype(float)\
                           + chunk["DVBltValW3_sum"].astype(float)
# Filter down to keep only these columns
chunk = chunk[["Weight", "NetFinancialWealth", "GrossFinancialWealth", "LiqFinancialWealth",
               "NetPropertyWealth", "GrossPropertyWealth", "GrossHousingWealth"]]

# Keep only selected population
if population == "all":
    pass
elif population == "non-owners":
    chunk = chunk[(chunk["Tenure"] == "Rent it") | (chunk["Tenure"] == "Rent-free")]
elif population == "owners":
    chunk = chunk[(chunk["Tenure"] == "Own it outright") | (chunk["Tenure"] == "Buying with mortgage")]
else:
    print("Population not recognised!")
    exit()

# Define bin edges and widths
number_of_bins = int(max_log_bin_edge - min_log_bin_edge) * 4 + 1
bin_edges = np.logspace(min_log_bin_edge, max_log_bin_edge, number_of_bins, base=np.e)
bin_widths = [b - a for a, b in zip(bin_edges[:-1], bin_edges[1:])]

# If printing data to files is required, histogram data and print results
financial_wealth_measures = ["NetFinancialWealth", "GrossFinancialWealth", "LiqFinancialWealth"]
housing_wealth_measures = ["NetPropertyWealth", "GrossPropertyWealth", "GrossHousingWealth"]
if printResults:
    for financial_wealth_measure in financial_wealth_measures:
        for housing_wealth_measure in housing_wealth_measures:
            # ...add total wealth column
            chunk["TotalWealth"] = chunk[financial_wealth_measure].astype(float)\
                                   + chunk[housing_wealth_measure].astype(float)
            # For the sake of using logarithmic scales, filter out any zero and negative values
            temp_chunk = chunk[(chunk["TotalWealth"] > 0.0)]
            # ...create a histogram
            frequency = np.histogram(np.log(temp_chunk["TotalWealth"].values), bins=bin_edges, density=True,
                                     weights=temp_chunk["Weight"].values)[0]
            # ...and print the distribution to a file
            with open(financial_wealth_measure + "-" + housing_wealth_measure + "-Weighted.csv", "w") as f:
                f.write("# Log Total Wealth (" + financial_wealth_measure + " + " + housing_wealth_measure
                        + ") (lower edge), Log Total Wealth (" + financial_wealth_measure + " + "
                        + housing_wealth_measure + ") (upper edge), Probability\n")
                for element, wealthLowerEdge, wealthUpperEdge in zip(frequency, bin_edges[:-1], bin_edges[1:]):
                    f.write("{}, {}, {}\n".format(wealthLowerEdge, wealthUpperEdge, element))

# If plotting data and results is required, read model results, histogram data and results and plot them
if plotResults:
    # Read model results
    resultsFinancialWealth = readResults(rootResults + r"/test/BankBalance-run1.csv", start_time, end_time)
    resultsHousingWealth = readResults(rootResults + r"/test/HousingWealth-run1.csv", start_time, end_time)
    results = [x + y for x, y in zip(resultsFinancialWealth, resultsHousingWealth)]
    # Keep only selected population
    if population == "all":
        pass
    elif population == "non-owners":
        nProperties = readResults(rootResults + r"/test/NHousesOwned-run1.csv", start_time, end_time)
        results = [x for x, y in zip(results, nProperties) if y == 0]
    elif population == "owners":
        nProperties = readResults(rootResults + r"/test/NHousesOwned-run1.csv", start_time, end_time)
        results = [x for x, y in zip(results, nProperties) if y > 0]
    # Add total wealth column with the selected measures of financial and housing wealth
    chunk["TotalWealth"] = chunk[financialWealthVariable].astype(float) + chunk[housingWealthVariable].astype(float)
    # In order to use logarithmic scales, filter out any zero and negative values from the chosen wealth column
    chunk = chunk[(chunk["TotalWealth"] > 0.0)]
    # Histogram model results
    model_hist = np.histogram([x for x in results if x > 0.0], bins=bin_edges, density=False)[0]
    model_hist = model_hist / sum(model_hist)
    # Histogram data from WAS
    WAS_hist = np.histogram(chunk[chunk["TotalWealth"] > 0.0]["TotalWealth"].values, bins=bin_edges,
                            density=False, weights=chunk[chunk["TotalWealth"] > 0.0]["Weight"].values)[0]
    WAS_hist = WAS_hist / sum(WAS_hist)
    # Plot both model results and data from WAS
    plt.figure(figsize=(8.5, 6))
    plt.bar(bin_edges[:-1], height=model_hist, width=bin_widths, align="edge", label="Model results", alpha=0.5,
            color="b")
    plt.bar(bin_edges[:-1], height=WAS_hist, width=bin_widths, align="edge", label="WAS data", alpha=0.5, color="r")
    # Final plot details
    plt.gca().set_xscale("log")
    plt.xlabel("Total Wealth (Bank Balance + Net Housing)")
    plt.ylabel("Frequency (fraction of cases)")
    plt.legend()
    plt.title("Distribution of Total Wealth (Financial + Housing)")
    if printPlots:
        plt.tight_layout()
        plt.savefig("/home/carro/Validation-{}-{}-{}.pdf".format(financialWealthVariable, housingWealthVariable,
                                                                 population), dpi=300)
    else:
        plt.show()
