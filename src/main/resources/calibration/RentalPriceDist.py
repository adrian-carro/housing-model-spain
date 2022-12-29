# coding: utf-8

# Imports
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd


def weighted_quantile(values, quantiles, sample_weight=None, values_sorted=False, old_style=False):
    """
    :param values: numpy.array with data
    :param quantiles: array-like with quantiles needed, which should be in [0, 1]
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of initial array
    :param old_style: if True, will correct output to be consistent with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), 'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


def read_and_clean_eff_data(_root_eff, _multiple_imputations):
    # EFF variables of interest
    # facine3                       Household weight
    # p2_31                         Importe mensual actual del alquiler de su vivienda
    # p2_31a                        Importe mensual que deberia estar pagando por el alquiler
    # p2_43_i                       Prop i Ingresos mensuales por alquiler de esta propiedad
    if _multiple_imputations:
        _df_eff = pd.DataFrame()
        for i in range(1, 6):
            temp_df = pd.read_csv(_root_eff + r'/eff_2017_imp{0}_csv/otras_secciones_2017_imp{0}.csv'.format(i),
                                  sep=';', usecols={'facine3', 'p2_31', 'p2_31a'})
            _df_eff = pd.concat([_df_eff, temp_df])
        _df_eff['facine3'] = _df_eff['facine3'] / 5.0
    else:
        _df_eff = pd.read_csv(_root_eff + r'/eff_2017_imp1_csv/otras_secciones_2017_imp1.csv', sep=';',
                              usecols={'facine3', 'p2_31', 'p2_31a'})

    # Rename columns to be used
    _df_eff.rename(columns={'facine3': 'Weight'}, inplace=True)
    # _df_eff.rename(columns={'p2_31': 'RentPrice'}, inplace=True)

    # Replace NaNs by zeros in EFF data
    _df_eff = _df_eff.fillna(0)

    # Define RentPrice column as rental price paid ('p2_31') or price one would have to pay ('p2_31a')
    _df_eff['RentPrice'] = _df_eff['p2_31'] + _df_eff['p2_31a']

    # Filter out NaNs and zeros
    _df_eff = _df_eff.loc[_df_eff['RentPrice'] > 0.0]

    # Select columns of interest
    _df_eff = _df_eff[['Weight', 'RentPrice']]

    # Filter out the 1% with highest and the 5% with lowest RentPrice (note the asymmetry!)
    _min_eff = weighted_quantile(_df_eff['RentPrice'], 0.05, sample_weight=_df_eff['Weight'])
    _max_eff = weighted_quantile(_df_eff['RentPrice'], 0.99, sample_weight=_df_eff['Weight'])
    _df_eff = _df_eff.loc[(_df_eff['RentPrice'] >= _min_eff) & (_df_eff['RentPrice'] <= _max_eff)]

    return _df_eff


# Control variables
saveFigures = False
writeResults = False
addModelResults = True
addUKComparison = True
min_log_price_bin_edge = 4.0
max_log_price_bin_edge = 7.6
log_price_bin_width = 0.25
rootEFF = r'
rootModel = r'
rootResults = r'

# General printing settings
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 1000)

# Read and clean Encuesta Financiera de las Familias data
dfEFF = read_and_clean_eff_data(rootEFF, _multiple_imputations=True)

# Define bin edges and widths
number_of_bins = int(round(max_log_price_bin_edge - min_log_price_bin_edge, 5) / log_price_bin_width + 1)
price_bin_edges = np.logspace(min_log_price_bin_edge, max_log_price_bin_edge, number_of_bins, base=np.e)
price_bin_widths = [b - a for a, b in zip(price_bin_edges[:-1], price_bin_edges[1:])]
price_bin_centers = np.exp([e + log_price_bin_width / 2 for e in np.log(price_bin_edges[:-1])])
log_x = np.linspace(min_log_price_bin_edge, max_log_price_bin_edge, 100)

# Histogram data from EFF
EFF_hist = np.histogram(dfEFF['RentPrice'], bins=price_bin_edges, density=False, weights=dfEFF['Weight'])[0]
EFF_hist = EFF_hist / sum(EFF_hist)

# Plot data
plt.figure(figsize=(7, 5))
plt.bar(price_bin_edges[:-1], height=EFF_hist, width=price_bin_widths, align='edge', label='EFF data', alpha=0.5,
        color='tab:blue')

# Plot normal distribution with parameters from fit to UK real rental prices
if addUKComparison:
    muUK = 6.070817
    stdUK = 0.4795898
    distUK = norm(loc=muUK, scale=stdUK)
    plt.plot(np.exp(log_x), [e * 0.25 for e in distUK.pdf(log_x)], lw=2, c="tab:green",
             label=r'UK fit ($\mu = {:.2f}$, $\sigma = {:.2f}$)'.format(muUK, stdUK))

# Fit and plot normal distribution using Spanish rental prices
muSP, stdSP = norm.fit(np.log(dfEFF['RentPrice']))
plt.plot(np.exp(log_x), [e * 0.25 for e in norm.pdf(log_x, muSP, stdSP)], c='tab:blue', linewidth=2,
         label=r'SP fit ($\mu = {:.2f}$, $\sigma = {:.2f}$)'.format(muSP, stdSP))

# Plot model results if required
if addModelResults:
    # Read model results
    dfModel = pd.read_csv(rootModel + '/results/test3/RentalTransactions-run1.csv',
                          skipinitialspace=True, delimiter=';', usecols=['transactionPrice'])
    # Compute and plot histogram
    Model_hist = np.histogram(dfModel['transactionPrice'], bins=price_bin_edges, density=False)[0]
    Model_hist = Model_hist / sum(Model_hist)
    plt.bar(price_bin_edges[:-1], height=Model_hist, width=price_bin_widths, align='edge', label='Model results',
            alpha=0.5, color='tab:red')
    # Compute and plot log-normal fit
    muModel, stdModel = norm.fit(np.log(dfModel['transactionPrice']))
    plt.plot(np.exp(log_x), [e * 0.25 for e in norm.pdf(log_x, muModel, stdModel)], c='tab:red', linewidth=2,
             label=r'Model fit ($\mu = {:.2f}$, $\sigma = {:.2f}$)'.format(muModel, stdModel))

# Other figure settings
plt.xscale('log')
plt.ylabel('Relative Frequency')
plt.xlabel('Rental Price')
plt.xlim(np.exp(min_log_price_bin_edge) / 1.2, np.exp(max_log_price_bin_edge) * 1.2)
plt.legend()

if writeResults:
    with open(rootResults + '/RentalPriceDistFit.txt', 'w') as f:
        f.write('##############################################################################\n')
        f.write('NORMAL DISTRIBUTION FIT - EFF DATA\n')
        f.write('##############################################################################\n')
        f.write('Mean = {}\n'.format(muSP))
        f.write('Std = {}\n'.format(stdSP))

if saveFigures:
    plt.tight_layout()
    if addModelResults:
        figureFileName = 'RentalPriceDistWithModel'
    else:
        figureFileName = 'RentalPriceDist'
    plt.savefig(rootResults + '/{}.pdf'.format(figureFileName), format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(rootResults + '/{}.png'.format(figureFileName), format='png', dpi=300, bbox_inches='tight')
else:
    plt.show()

