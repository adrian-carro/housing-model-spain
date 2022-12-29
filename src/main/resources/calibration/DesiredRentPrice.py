"""
Class to study the desired rental price bid by households when trying to secure an accommodation on the rental market.

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
            bin_means.append(np.mean(np.log(dfEFF.loc[(dfEFF['GrossNonRentIncome'] >= np.exp(a))
                                                      & (dfEFF['GrossNonRentIncome'] < np.exp(b)),
                                                      'RentPrice'])))
        plt.plot(np.exp(income_bin_centers), np.exp(bin_means), 'or', ms=4.0)
    elif scale == 'lin':
        income_bin_edges = np.exp(np.linspace(8.5, 12.25, 31, endpoint=True))
        income_bin_centers = []
        bin_means = []
        for a, b in zip(income_bin_edges[:-1], income_bin_edges[1:]):
            income_bin_centers.append((a + b) / 2.0)
            bin_means.append(np.mean(dfEFF.loc[(dfEFF['GrossNonRentIncome'] >= a)
                                               & (dfEFF['GrossNonRentIncome'] < b), 'RentPrice']))
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


def read_and_clean_eff_data(_root_eff, _multiple_imputations, _exclude_house_sale_profits):
    # EFF variables of interest
    # facine3                       Household weight
    # renthog                       Household Gross Annual income
    # p7_2                          Importe recibido por alquileres durante el ano 2016
    # p7_4a                         Plusvalias por venta de prop. inmob. durante el ano 2016
    # p2_31                         Importe mensual actual del alquiler de su vivienda
    # p2_31a                        Importe mensual que deberia estar pagando por el alquiler
    # p2_43_i                       Prop i Ingresos mensuales por alquiler de esta propiedad
    # Pre-select columns of interest so as to read data more efficiently
    _vars_of_interest = ['p2_31', 'p6_6', 'p6_7', 'p7_1', 'p7_2', 'p7_4', 'p7_6', 'p7_8', 'p9_1', 'p9_20']
    _otras_secciones_cols = pd.read_csv(_root_eff + r'/eff_2017_imp1_csv/otras_secciones_2017_imp1.csv',
                                        nrows=1, sep=';').columns
    _seccion6_cols = pd.read_csv(_root_eff + r'/eff_2017_imp1_csv/seccion6_2017_imp1.csv', nrows=1, sep=';').columns
    _otras_secciones_cols_of_interest = [c for c in _otras_secciones_cols if any(e in c for e in _vars_of_interest)]
    _otras_secciones_cols_of_interest = ['facine3', 'renthog', 'h_2017'] + _otras_secciones_cols_of_interest
    _seccion6_cols_of_interest = [c for c in _seccion6_cols if any(e in c for e in _vars_of_interest)]
    _seccion6_cols_of_interest = ['h_2017'] + _seccion6_cols_of_interest
    # Read data
    if _multiple_imputations:
        _df_eff = pd.DataFrame()
        for i in range(1, 6):
            temp_df1 = pd.read_csv(_root_eff + r'/eff_2017_imp{0}_csv/otras_secciones_2017_imp{0}.csv'.format(i),
                                   sep=';', usecols=_otras_secciones_cols_of_interest)
            temp_df2 = pd.read_csv(_root_eff + r'/eff_2017_imp{0}_csv/seccion6_2017_imp{0}.csv'.format(i), sep=';',
                                   usecols=_seccion6_cols_of_interest)
            temp_df = pd.merge(temp_df1, temp_df2, on='h_2017')
            _df_eff = pd.concat([_df_eff, temp_df])
        _df_eff['facine3'] = _df_eff['facine3'] / 5.0
    else:
        temp_df1 = pd.read_csv(_root_eff + r'/eff_2017_imp1_csv/otras_secciones_2017_imp1.csv', sep=';',
                               usecols=_otras_secciones_cols_of_interest)
        temp_df2 = pd.read_csv(_root_eff + r'/eff_2017_imp1_csv/seccion6_2017_imp1.csv', sep=';',
                               usecols=_seccion6_cols_of_interest)
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

    # Define RentPrice column as rental price paid ('p2_31') or price one would have to pay ('p2_31a')
    _df_eff['RentPrice'] = _df_eff['p2_31'] + _df_eff['p2_31a']

    # Filter out NaNs, negative values and zeros
    _df_eff = _df_eff.loc[_df_eff['GrossNonRentIncome'] > 0.0]
    _df_eff = _df_eff.loc[_df_eff['RentPrice'] > 0.0]

    # Filter down to keep only columns of interest
    _df_eff = _df_eff[['Weight', 'GrossNonRentIncome', 'GrossNonRentIncomeIndividual', 'RentPrice']]

    # TODO: Think what to do about the arbitrariness of doing the income or the rent price cut first
    # Filter out the 1% with highest and the 1% with lowest GrossNonRentIncome
    _min_eff = cf.weighted_quantile(_df_eff['GrossNonRentIncome'], 0.01, sample_weight=_df_eff['Weight'])
    _max_eff = cf.weighted_quantile(_df_eff['GrossNonRentIncome'], 0.99, sample_weight=_df_eff['Weight'])
    _df_eff = _df_eff.loc[(_df_eff['GrossNonRentIncome'] >= _min_eff) & (_df_eff['GrossNonRentIncome'] <= _max_eff)]

    # Filter out the 1% with highest and the 5% with lowest RentPrice (note the asymmetry!)
    _min_eff = cf.weighted_quantile(_df_eff['RentPrice'], 0.05, sample_weight=_df_eff['Weight'])
    _max_eff = cf.weighted_quantile(_df_eff['RentPrice'], 0.99, sample_weight=_df_eff['Weight'])
    _df_eff = _df_eff.loc[(_df_eff['RentPrice'] >= _min_eff) & (_df_eff['RentPrice'] <= _max_eff)]

    return _df_eff


# Set parameters
saveFigures = False
writeResults = False
rootEFF = r''
rootResults = r''

# General printing settings
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 1000)

# Read and clean Encuesta Financiera de las Familias data
dfEFF = read_and_clean_eff_data(rootEFF, _multiple_imputations=True, _exclude_house_sale_profits=True)

# Plot raw results
plt.figure(figsize=(6.0, 4.25))
plt.plot(dfEFF['GrossNonRentIncome'], dfEFF['RentPrice'], 'o', ms=2.0, mew=0.0, alpha=0.5, label='Total')

# Compute and plot linear regression
x = sm.add_constant(np.log(dfEFF['GrossNonRentIncome']))
y = np.log(dfEFF['RentPrice'])
model = sm.OLS(y, x)
linResults = model.fit()
mu, std = stats.norm.fit(linResults.resid)  # Same as np.mean(linResults.resid), np.std(linResults.resid)
if writeResults:
    with open(rootResults + '/DesiredRentPriceFits.txt', 'w') as f:
        f.write('##############################################################################\n')
        f.write('LINEAR FIT [log(y) = a + b * log(x)]\n')
        f.write('##############################################################################\n')
        f.write(linResults.summary().as_text())
        f.write('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
        f.write('Mean of residuals = {}, Std of residuals = {}\n'.format(mu, std))
        f.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
reduced_x = np.linspace(x['GrossNonRentIncome'].min(), x['GrossNonRentIncome'].max(), 1000, endpoint=True)
plt.plot(np.exp(reduced_x), np.exp(linResults.predict(sm.add_constant(reduced_x))), '-',
         label='Linear Fit (y = {:.4f} * x^{:.4f})'.format(np.exp(linResults.params[0]), linResults.params[1]))
plt.fill_between(np.exp(reduced_x), np.exp(linResults.predict(sm.add_constant(reduced_x))) * np.exp(std),
                 np.exp(linResults.predict(sm.add_constant(reduced_x))) * np.exp(-std),
                 color='orange', alpha=0.2, zorder=0)

# If required, independently plot residuals
# plot_residuals(df[incomeVariable], df[priceVariable], linResults)

# Compute and plot direct nonlinear fit
popt, pcov = curve_fit(lambda income, alpha, beta: alpha * pow(income, beta), dfEFF['GrossNonRentIncome'],
                       dfEFF['RentPrice'], p0=(42.9036, 0.7892))
if writeResults:
    with open(rootResults + '/DesiredRentPriceFits.txt', 'a') as f:
        f.write('\n\n##############################################################################\n')
        f.write('NON LINEAR FIT [y = a * x ^ b]\n')
        f.write('##############################################################################\n')
        f.write('popt = {}\n'.format(str(popt)))
        f.write('pcov = {}\n'.format(str(pcov)))
plt.plot(np.exp(reduced_x), [popt[0] * pow(e, popt[1]) for e in np.exp(reduced_x)], '-',
         label='Nonlinear Fit (y = {:.4f} * x ^ {:.4f})'.format(popt[0], popt[1]))

# Compute and plot weighted least squares fit
# model = sm.WLS(y, x, weights=np.sqrt(df[priceVariable]))  # Leads to same as nonlinear fit
model = sm.WLS(y, x, weights=dfEFF['RentPrice'])
wLinResults = model.fit()
if writeResults:
    with open(rootResults + '/DesiredRentPriceFits.txt', 'a') as f:
        f.write('\n##############################################################################\n')
        f.write('WEIGHTED FIT [log(y) = a + b * log(x)]\n')
        f.write('##############################################################################\n')
        f.write(wLinResults.summary().as_text())
plt.plot(np.exp(reduced_x), np.exp(wLinResults.predict(sm.add_constant(reduced_x))), '-',
         label='Weighted Fit (y = {:.4f} * x^{:.4f})'.format(np.exp(wLinResults.params[0]), wLinResults.params[1]))

# For comparison, plot UK linear regression results
plt.plot(np.exp(reduced_x), np.exp([2.845872292 + 0.3463723 * e for e in reduced_x]), '-',
         label='UK Linear Fit (y = {:.4f} * x^{:.4f})'.format(np.exp(2.845872292), 0.3463723))

# Plot mean price per income bin (both with linear and logarithmic mean)
add_mean_bins('log')
add_mean_bins('lin')

# Plot format
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(8.4, 12.3)
# if priceVariable == 'price':
#     plt.ylim(10.0, 13.5)
# else:
#     plt.ylim(10.5, 13.5)
plt.gca().xaxis.set_major_formatter(EngFormatter())
plt.gca().yaxis.set_major_formatter(EngFormatter())
plt.xlabel('Income')
plt.ylabel('Rent Price')
plt.title('{} (with {} points)'.format('Total', len(dfEFF)))
plt.legend(loc='upper left')

if saveFigures:
    plt.tight_layout()
    plt.savefig(rootResults + '/DesiredRentPrice.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(rootResults + '/DesiredRentPrice.png', format='png', dpi=300, bbox_inches='tight')
else:
    plt.show()
