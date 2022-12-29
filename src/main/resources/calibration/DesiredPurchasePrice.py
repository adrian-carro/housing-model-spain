"""
Class to study the desired purchase price bid by households when trying to acquire new properties on the sales market.

@author: Adrian Carro
"""

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy.optimize import curve_fit
import scipy.stats as stats
import pandas as pd
from matplotlib.ticker import EngFormatter
import Utilities.common_functions as cf


def add_mean_bins(scale):
    # Divide points into income bins and plot the mean price variable for each bin
    if scale == 'log':
        income_bin_edges = np.linspace(8.5, 12.25, 31, endpoint=True)
        income_bin_centers = []
        bin_means = []
        for a, b in zip(income_bin_edges[:-1], income_bin_edges[1:]):
            income_bin_centers.append((a + b) / 2.0)
            bin_means.append(np.mean(np.log(dfEDW.loc[(dfEDW[incomeVariable] >= np.exp(a))
                                                      & (dfEDW[incomeVariable] < np.exp(b)),
                                                      priceVariable])))
        plt.plot(np.exp(income_bin_centers), np.exp(bin_means), 'or', ms=4.0)
    elif scale == 'lin':
        income_bin_edges = np.exp(np.linspace(8.5, 12.25, 31, endpoint=True))
        income_bin_centers = []
        bin_means = []
        for a, b in zip(income_bin_edges[:-1], income_bin_edges[1:]):
            income_bin_centers.append((a + b) / 2.0)
            bin_means.append(np.mean(dfEDW.loc[(dfEDW[incomeVariable] >= a)
                                               & (dfEDW[incomeVariable] < b), priceVariable]))
        plt.plot(income_bin_centers, bin_means, 'D', c='magenta', ms=4.0)


def plot_residuals(_x, _y, _results):
    # Plot residuals against x variable in log scale
    plt.figure()
    plt.plot(np.log(_x), _results.resid, 'o')
    plt.axhline(y=0, ls='--', c='k')
    # Plot residuals against x variable in lin scale
    plt.figure()
    plt.plot(_x, _y - np.exp(_results.fittedvalues), 'o')
    plt.axhline(y=0, ls='--', c='k')
    # Plot distribution of residuals, including normal fit
    plt.figure()
    plt.hist(_results.resid, bins=20, density=True)
    _mu, _std = stats.norm.fit(_results.resid)  # Same as np.mean(_resid), np.std(_resid)
    x_min, x_max = plt.xlim()
    temp_x = np.linspace(x_min, x_max, 100)
    plt.plot(temp_x, stats.norm.pdf(temp_x, _mu, _std), 'r', linewidth=2)
    # Plot distribution of residuals, including normal fit
    plt.figure()
    plt.hist(_y - np.exp(_results.fittedvalues), bins=50, density=True)
    _mu, _std = stats.norm.fit(_y - np.exp(_results.fittedvalues))  # Same as np.mean(_resid), np.std(_resid)
    x_min, x_max = plt.xlim()
    temp_x = np.linspace(x_min, x_max, 100)
    plt.plot(temp_x, stats.norm.pdf(temp_x, _mu, _std), 'r', linewidth=2)


def read_and_clean_edw_data(_root_edw, _income_variable):
    # Read data
    _df_edw = cf.read_and_clean_edw_data(_root_edw, _from_year=2014, _to_year=2020)

    # If required, restrict to cases where the second income is available
    if _income_variable == 'household_income_only':
        _df_edw = _df_edw.loc[_df_edw['B_inc2'] > 0.0]
        _df_edw.rename(columns={'household_income': 'household_income_only'}, inplace=True)

    # Restrict both income and the selected price variable to positive values (thus also discarding NaNs)
    _df_edw = _df_edw.loc[_df_edw[_income_variable] > 0.0]
    _df_edw = _df_edw.loc[_df_edw[priceVariable] > 0.0]

    # Remove 1% lowest and 1% highest incomes (very low income transactions are clearly the result of other undeclared
    # sources of income, very high incomes are related to very high prices, which are not captured in the database)
    q1, q99 = np.quantile(_df_edw[_income_variable], [0.01, 0.99])
    # q1, q99 = np.quantile(_df_edw[_income_variable], [0.05, 0.95])
    _df_edw = _df_edw.loc[(_df_edw[_income_variable] > q1) & (_df_edw[_income_variable] < q99)]

    # If 'C_price' selected as price variable, then remove values below 25k
    # if priceVariable == 'C_price':
    #     _df_edw = _df_edw.loc[_df_edw[priceVariable] > 25000]

    # Manually remove the most clear vertical lines
    # if _income_variable == 'B_inc':
    #     _df_edw = _df_edw[(_df_edw[_income_variable] != 130000) & (_df_edw[_income_variable] != 43500)
    #                       & (_df_edw[_income_variable] != 21000) & (_df_edw[_income_variable] != 7500)]

    # Remove all integer income values, as they tend to be arranged into strange vertical lines
    # _df_edw = _df_edw[_df_edw['income'].apply(lambda e: e != int(e))]

    # Filter down to keep only columns of interest
    _df_edw = _df_edw[[_income_variable, priceVariable]]

    return _df_edw


def read_and_clean_eff_data(_root_eff, _multiple_imputations, _exclude_house_sale_profits):
    # EFF variables of interest
    # facine3                       Household weight
    # renthog                       Household Gross Annual income
    # p7_2                          Importe recibido por alquileres durante el ano 2016
    # p7_4a                         Plusvalias por venta de prop. inmob. durante el ano 2016
    if _multiple_imputations:
        _df_eff = pd.DataFrame()
        for i in range(1, 6):
            temp_df1 = pd.read_csv(_root_eff + r'/eff_2017_imp{0}_csv/otras_secciones_2017_imp{0}.csv'.format(i),
                                   sep=';')
            temp_df2 = pd.read_csv(_root_eff + r'/eff_2017_imp{0}_csv/seccion6_2017_imp{0}.csv'.format(i), sep=';')
            temp_df = pd.merge(temp_df1, temp_df2, on='h_2017')
            _df_eff = pd.concat([_df_eff, temp_df])
        _df_eff['facine3'] = _df_eff['facine3'] / 5.0
    else:
        temp_df1 = pd.read_csv(_root_eff + r'/eff_2017_imp1_csv/otras_secciones_2017_imp1.csv', sep=';')
        temp_df2 = pd.read_csv(_root_eff + r'/eff_2017_imp1_csv/seccion6_2017_imp1.csv', sep=';')
        _df_eff = pd.merge(temp_df1, temp_df2, on='h_2017')

    # Rename columns to be used
    _df_eff.rename(columns={'facine3': 'Weight'}, inplace=True)

    # Replace NaNs by zeros in EFF data
    _df_eff = _df_eff.fillna(0)

    # Compute annual gross non-rental income
    if not _exclude_house_sale_profits:
        _df_eff['GrossNonRentIncome'] = _df_eff['renthog'] - _df_eff['p7_2']
    else:
        _df_eff['GrossNonRentIncome'] = _df_eff['renthog'] - _df_eff['p7_2'] - _df_eff['p7_4a']
    _df_eff['GrossNonRentIncome_manual'] = _df_eff.apply(
        lambda row: cf.add_up_annual_non_renal_income_2016(row, _exclude_house_sale_profits, _individual=False), axis=1)

    _df_eff['GrossNonRentIncomeIndividual'] = _df_eff.apply(
        lambda row: cf.add_up_annual_non_renal_income_2016(row, _exclude_house_sale_profits, _individual=True), axis=1)

    # Filter down to keep only columns of interest
    # _df_eff = _df_eff[['GrossNonRentIncome', 'Weight']]
    _df_eff = _df_eff[['GrossNonRentIncome', 'GrossNonRentIncomeIndividual', 'Weight']]

    # Filter out the 1% with highest and the 1% with lowest GrossNonRentIncome
    _min_eff = cf.weighted_quantile(_df_eff['GrossNonRentIncome'], 0.01, sample_weight=_df_eff['Weight'])
    _max_eff = cf.weighted_quantile(_df_eff['GrossNonRentIncome'], 0.99, sample_weight=_df_eff['Weight'])
    _df_eff = _df_eff.loc[(_df_eff['GrossNonRentIncome'] >= _min_eff) & (_df_eff['GrossNonRentIncome'] <= _max_eff)]

    return _df_eff


def add_household_income_to_edw_data(_df_edw, _df_eff, _plot_fit):
    # Restrict to household incomes larger than the individual equivalent (note that this does only remove elements in
    # the local copy of the DataFrame)
    _df_eff = _df_eff.loc[
        (_df_eff['GrossNonRentIncomeIndividual'] > 3000) & (_df_eff['GrossNonRentIncomeIndividual'] < 150000)
        & (_df_eff['GrossNonRentIncome'] > _df_eff['GrossNonRentIncomeIndividual'])]

    # Compute linear regression
    x_lin = sm.add_constant(_df_eff['GrossNonRentIncomeIndividual'])
    y_lin = _df_eff['GrossNonRentIncome']
    model_lin = sm.OLS(y_lin, x_lin)
    lin_results_lin = model_lin.fit()

    # Compute exponential regression
    x_log = sm.add_constant(np.log(_df_eff['GrossNonRentIncomeIndividual']))
    y_log = np.log(_df_eff['GrossNonRentIncome'])
    model_log = sm.OLS(y_log, x_log)
    lin_results_log = model_log.fit()

    # If required, plot fit
    if _plot_fit:
        plt.plot(_df_eff['GrossNonRentIncomeIndividual'], _df_eff['GrossNonRentIncome'], 'o')
        x = _df_eff['GrossNonRentIncomeIndividual'].sort_values()
        plt.plot(x, lin_results_lin.predict(sm.add_constant(x)), '-', lw=2.0,
                 label='Linear Fit (y = {:.2f} + {:.2f} * x)'.format(*lin_results_lin.params))
        plt.plot(x, np.exp(lin_results_log.predict(sm.add_constant(np.log(x)))), '-', lw=2.0,
                 label='Linear Fit (y = {:.2f} * x^{:.2f})'.format(np.exp(lin_results_log.params[0]),
                                                                   lin_results_log.params[0]))
        plt.legend()
        plt.show()

    # Compute transformation of EDW individual incomes into household incomes
    _df_edw['GrossNonRentIncome_LinFit'] = lin_results_lin.predict(sm.add_constant(_df_edw['B_inc']))
    _df_edw['GrossNonRentIncome_LogFit'] = np.exp(lin_results_log.predict(sm.add_constant(np.log(_df_edw['B_inc']))))

    return _df_edw


# Set parameters
# priceVariable = 'C_price'  # Possible values are 'price' and 'Ovalue'
priceVariable = 'C_Ovaluation'  # Possible values are 'price' and 'Ovalue'
# incomeVariable = 'household_income'
incomeVariable = 'household_income_only'
# incomeVariable = 'GrossNonRentIncome_LinFit'
# incomeVariable = 'GrossNonRentIncome_LogFit'
saveFigures = False
writeResults = False
rootEWD = r''
rootEFF = r''
rootResults = r''
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 1000)

# Read European Data Warehouse data
dfEDW = read_and_clean_edw_data(rootEWD, incomeVariable)

# If income variable chosen is any of the household variants, then read and clean Encuesta Financiera de las Familias
# data and use it to compute household income from individual income and add it to EDW data
if incomeVariable == 'GrossNonRentIncome_LinFit' or incomeVariable == 'GrossNonRentIncome_LogFit':
    dfEFF = read_and_clean_eff_data(rootEFF, _multiple_imputations=True, _exclude_house_sale_profits=True)
    dfEDW = add_household_income_to_edw_data(dfEDW, dfEFF, _plot_fit=False)

# Plot raw results
plt.figure(figsize=(6.0, 4.25))
plt.plot(dfEDW[incomeVariable], dfEDW[priceVariable], 'o', ms=2.0, mew=0.0, alpha=0.5, label='Total')

# Compute and plot linear regression
x = sm.add_constant(np.log(dfEDW[incomeVariable]))
y = np.log(dfEDW[priceVariable])
model = sm.OLS(y, x)
linResults = model.fit()
mu, std = stats.norm.fit(linResults.resid)  # Same as np.mean(linResults.resid), np.std(linResults.resid)
if writeResults:
    with open(rootResults + '/DesiredPurchasePriceFits_{}.txt'.format(incomeVariable), 'w') as f:
        f.write('##############################################################################\n')
        f.write('LINEAR FIT [log(y) = a + b * log(x)]\n')
        f.write('##############################################################################\n')
        f.write(linResults.summary().as_text())
        f.write('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
        f.write('Mean of residuals = {}, Std of residuals = {}\n'.format(mu, std))
        f.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
reduced_x = np.linspace(x[incomeVariable].min(), x[incomeVariable].max(), 1000, endpoint=True)
plt.plot(np.exp(reduced_x), np.exp(linResults.predict(sm.add_constant(reduced_x))), '-',
         label='Linear Fit (y = {:.4f} * x^{:.4f})'.format(np.exp(linResults.params[0]), linResults.params[1]))
plt.fill_between(np.exp(reduced_x), np.exp(linResults.predict(sm.add_constant(reduced_x))) * np.exp(std),
                 np.exp(linResults.predict(sm.add_constant(reduced_x))) * np.exp(-std),
                 color='orange', alpha=0.2, zorder=0)

# If required, independently plot residuals
# plot_residuals(df[incomeVariable], df[priceVariable], linResults)

# Compute and plot direct nonlinear fit
popt, pcov = curve_fit(lambda income, alpha, beta: alpha * pow(income, beta), dfEDW[incomeVariable],
                       dfEDW[priceVariable], p0=(42.9036, 0.7892))
residuals = [log_price - (np.log(popt[0]) + popt[1] * log_income) for log_income, log_price
             in zip(np.log(dfEDW[incomeVariable]), np.log(dfEDW[priceVariable]))]
mu, std = stats.norm.fit(residuals)
if writeResults:
    with open(rootResults + '/DesiredPurchasePriceFits_{}.txt'.format(incomeVariable), 'a') as f:
        f.write('\n\n##############################################################################\n')
        f.write('NON LINEAR FIT [y = a * x ^ b]\n')
        f.write('##############################################################################\n')
        f.write('popt = {}\n'.format(str(popt)))
        f.write('pcov = {}\n'.format(str(pcov)))
        f.write('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
        f.write('Mean of residuals = {}, Std of residuals = {}\n'.format(mu, std))
        f.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
plt.plot(np.exp(reduced_x), [popt[0] * pow(e, popt[1]) for e in np.exp(reduced_x)], '-',
         label='Nonlinear Fit (y = {:.4f} * x ^ {:.4f})'.format(popt[0], popt[1]))

# Compute and plot weighted least squares fit
# model = sm.WLS(y, x, weights=np.sqrt(df[priceVariable]))  # Leads to same as nonlinear fit
model = sm.WLS(y, x, weights=dfEDW[priceVariable])
wLinResults = model.fit()
mu, std = stats.norm.fit(wLinResults.resid)
if writeResults:
    with open(rootResults + '/DesiredPurchasePriceFits_{}.txt'.format(incomeVariable), 'a') as f:
        f.write('\n##############################################################################\n')
        f.write('WEIGHTED FIT [log(y) = a + b * log(x)]\n')
        f.write('##############################################################################\n')
        f.write(wLinResults.summary().as_text())
        f.write('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
        f.write('Mean of residuals = {}, Std of residuals = {}\n'.format(mu, std))
        f.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
plt.plot(np.exp(reduced_x), np.exp(wLinResults.predict(sm.add_constant(reduced_x))), '-',
         label='Weighted Fit (y = {:.4f} * x^{:.4f})'.format(np.exp(wLinResults.params[0]), wLinResults.params[1]))

# For comparison, plot UK linear regression results
plt.plot(np.exp(reduced_x), np.exp([3.758955738 + 0.7992 * e for e in reduced_x]), '-',
         label='UK Linear Fit (y = {:.4f} * x^{:.4f})'.format(np.exp(3.758955738), 0.7892))

# Plot mean price per income bin (both with linear and logarithmic mean)
add_mean_bins('log')
add_mean_bins('lin')

# Plot format
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(1000, 140000)
# if priceVariable == 'price':
#     plt.ylim(10.0, 13.5)
# else:
#     plt.ylim(10.5, 13.5)
plt.gca().xaxis.set_major_formatter(EngFormatter())
plt.gca().yaxis.set_major_formatter(EngFormatter())
plt.xlabel('Income')
plt.ylabel('Ovalue')
plt.title('{} (with {} points)'.format('Total', len(dfEDW)))
plt.legend(loc='upper left')

if saveFigures:
    plt.tight_layout()
    plt.savefig(rootResults + '/DesiredPurchasePrice_{}.pdf'.format(incomeVariable), format='pdf', dpi=300,
                bbox_inches='tight')
    plt.savefig(rootResults + '/DesiredPurchasePrice_{}.png'.format(incomeVariable), format='png', dpi=300,
                bbox_inches='tight')
else:
    plt.show()
