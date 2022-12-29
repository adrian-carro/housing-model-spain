# -*- coding: utf-8 -*-
"""
Class to study household consumption and its limits, based on Encuesta Financiera de las Familias data.

@author: Adrian Carro
"""

import matplotlib.pyplot as plt
import numpy as np
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


def read_and_clean_eff_data(_root_eff, _multiple_imputations, _exclude_house_sale_profits):
    # EFF variables of interest
    # facine3                       Household weight
    # renthog                       Household Gross Annual income
    # p7_2                          Importe recibido por alquileres durante el ano 2016
    # p7_4a                         Plusvalias por venta de prop. inmob. durante el ano 2016
    # p9_1                          Gasto medio total en bienes de consumo en un mes (incluye comida)
    # Read data
    if _multiple_imputations:
        _df_eff = pd.DataFrame()
        for i in range(1, 6):
            temp_df = pd.read_csv(_root_eff + r'/eff_2017_imp{0}_csv/otras_secciones_2017_imp{0}.csv'.format(i),
                                  sep=';', usecols=['facine3', 'renthog', 'p7_2', 'p7_4a', 'p9_1'])
            _df_eff = pd.concat([_df_eff, temp_df])
        _df_eff['facine3'] = _df_eff['facine3'] / 5.0
    else:
        _df_eff = pd.read_csv(_root_eff + r'/eff_2017_imp1_csv/otras_secciones_2017_imp1.csv', sep=';',
                              usecols=['facine3', 'renthog', 'p7_2', 'p7_4a', 'p9_1'])

    # Rename columns to be used
    _df_eff.rename(columns={'facine3': 'Weight'}, inplace=True)
    _df_eff.rename(columns={'p9_1': 'MonthlyConsumption'}, inplace=True)

    # Replace NaNs by zeros in EFF data
    _df_eff = _df_eff.fillna(0)

    # Compute annual gross non-rental income
    if not _exclude_house_sale_profits:
        _df_eff['GrossNonRentIncome'] = _df_eff['renthog'] - _df_eff['p7_2']
    else:
        _df_eff['GrossNonRentIncome'] = _df_eff['renthog'] - _df_eff['p7_2'] - _df_eff['p7_4a']

    # Filter out NaNs, negative values and zeros
    _df_eff = _df_eff.loc[(_df_eff['GrossNonRentIncome'] > 0.0) & (_df_eff['MonthlyConsumption'] > 0.0)]

    # Filter down to keep only columns of interest
    _df_eff = _df_eff[['Weight', 'GrossNonRentIncome', 'MonthlyConsumption']]

    # Filter out the 1% with highest and the 1% with lowest GrossNonRentIncome
    # _min_eff = weighted_quantile(_df_eff['GrossNonRentIncome'], 0.01, sample_weight=_df_eff['Weight'])
    # _max_eff = weighted_quantile(_df_eff['GrossNonRentIncome'], 0.99, sample_weight=_df_eff['Weight'])
    # _df_eff = _df_eff.loc[(_df_eff['GrossNonRentIncome'] >= _min_eff) & (_df_eff['GrossNonRentIncome'] <= _max_eff)]

    # _min_eff = weighted_quantile(_df_eff['MonthlyConsumption'], 0.01, sample_weight=_df_eff['Weight'])
    # _max_eff = weighted_quantile(_df_eff['MonthlyConsumption'], 0.99, sample_weight=_df_eff['Weight'])
    # _df_eff = _df_eff.loc[(_df_eff['MonthlyConsumption'] >= _min_eff) & (_df_eff['MonthlyConsumption'] <= _max_eff)]

    return _df_eff


def add_up_annual_house_rental_income_2016(_row):
    _total_house_rental_income = 0.0
    for i in range(1, 5):
        if _row['p2_35a_{}'.format(i)] in {1, 9}:
            _total_house_rental_income += _row['p2_43_{}'.format(i)]

    return 12.0 * _total_house_rental_income


def plot_mean_consumption_per_income_percentile_bin(_df_eff):
    percentiles = np.round(np.linspace(0.0, 1.0, 101, endpoint=True), 5)
    income_edges = weighted_quantile(_df_eff['GrossNonRentIncome'], percentiles, sample_weight=_df_eff['Weight'])
    mean_consumption_fraction_per_income_bin = []
    mean_consumption_fraction_below_percentile = []
    for a, b in zip(income_edges[:-2], income_edges[1:-1]):
        mean_consumption = np.average(
            _df_eff.loc[(_df_eff['GrossNonRentIncome'] >= a)
                        & (_df_eff['GrossNonRentIncome'] < b), 'MonthlyConsumption'],
            weights=_df_eff.loc[(_df_eff['GrossNonRentIncome'] >= a) & (_df_eff['GrossNonRentIncome'] < b), 'Weight'])
        mean_consumption_fraction_per_income_bin.append(mean_consumption)
        mean_consumption_fraction_below_percentile.append(np.average(
            _df_eff.loc[(_df_eff['GrossNonRentIncome'] < b), 'MonthlyConsumption'],
            weights=_df_eff.loc[(_df_eff['GrossNonRentIncome'] < b), 'Weight']))
    mean_consumption = np.average(
        _df_eff.loc[(_df_eff['GrossNonRentIncome'] >= income_edges[-2])
                    & (_df_eff['GrossNonRentIncome'] < income_edges[-1]), 'MonthlyConsumption'],
        weights=_df_eff.loc[(_df_eff['GrossNonRentIncome'] >= income_edges[-2])
                            & (_df_eff['GrossNonRentIncome'] < income_edges[-1]), 'Weight'])
    mean_consumption_fraction_per_income_bin.append(mean_consumption)
    mean_consumption_fraction_below_percentile.append(np.average(
        _df_eff.loc[(_df_eff['GrossNonRentIncome'] < income_edges[-1]), 'MonthlyConsumption'],
        weights=_df_eff.loc[(_df_eff['GrossNonRentIncome'] < income_edges[-1]), 'Weight']))
    plt.plot(mean_consumption_fraction_per_income_bin)
    plt.show()


# Control variables
writeResults = False
rootEFF = r''
rootModel = r''
rootResults = r''

# General printing settings
np.set_printoptions(suppress=True)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 1000)

# Read Encuesta Financiera de las Familias data
dfEFF = read_and_clean_eff_data(rootEFF, _multiple_imputations=True, _exclude_house_sale_profits=True)

# Compute essential consumption as the level corresponding to the 1st percentile of lowest consumption values
essentialConsumption = weighted_quantile(dfEFF['MonthlyConsumption'], 0.01, sample_weight=dfEFF['Weight'])
lowest_1p_income = weighted_quantile(dfEFF['GrossNonRentIncome'], 0.01, sample_weight=dfEFF['Weight'])

# Plot mean consumption per income percentile bin
# plot_mean_consumption_per_income_percentile_bin(dfEFF)

# Compute monthly consumption as a fraction of annual gross non rental income
dfEFF['ConsumptionFraction'] = dfEFF['MonthlyConsumption'] / dfEFF['GrossNonRentIncome']

# Compute maximum consumption fraction as the 99th percentile
maximumConsumptionFraction = weighted_quantile(dfEFF['ConsumptionFraction'], 0.99, sample_weight=dfEFF['Weight'])

# If required, write results to file, otherwise, print to screen
if writeResults:
    with open(rootResults + '/ConsumptionLimits.txt', 'w') as f:
        f.write('##############################################################################\n')
        f.write('EFF DATA\n')
        f.write('##############################################################################\n')
        f.write('Essential Consumption = {}\n'.format(essentialConsumption))
        f.write('Essential Consumption Fraction = {}\n'.format(12.0 * essentialConsumption / lowest_1p_income))
        f.write('Maximum Consumption Fraction = {}'.format(maximumConsumptionFraction))
else:
    print('####################################### EFF #######################################')
    print('Essential Consumption = {}'.format(essentialConsumption))
    print('Essential Consumption Fraction = {}'.format(12.0 * essentialConsumption / lowest_1p_income))
    print('Maximum Consumption Fraction = {}'.format(maximumConsumptionFraction))
