# -*- coding: utf-8 -*-
"""
Class to study the probability of a household becoming a buy-to-let investor, based on Encuesta Financiera de las
Familias data.

@author: Adrian Carro
"""

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


def read_and_clean_was_data(_root_was):
    # List of household variables currently used
    # DVTotGIRw3                  Household Gross Annual (regular) income
    # DVTotNIRw3                  Household Net Annual (regular) income
    # DVGrsRentAmtAnnualw3_aggr   Household Gross Annual income from rent
    # DVNetRentAmtAnnualw3_aggr   Household Net Annual income from rent
    _df_was = pd.read_csv(_root_was + r'/was_wave_3_hhold_eul_final.dta', usecols={'w3xswgt', 'DVTotGIRw3',
                                                                                   'DVGrsRentAmtAnnualw3_aggr'})

    # Rename columns to be used and add all necessary extra columns
    _df_was.rename(columns={'w3xswgt': 'Weight'}, inplace=True)
    _df_was.rename(columns={'DVTotGIRw3': 'GrossTotalIncome'}, inplace=True)
    _df_was.rename(columns={'DVGrsRentAmtAnnualw3_aggr': 'GrossRentalIncome'}, inplace=True)
    _df_was['GrossNonRentIncome'] = _df_was['GrossTotalIncome'] - _df_was['GrossRentalIncome']

    # Filter out the 1% with highest and the 1% with lowest GrossNonRentIncome
    one_per_cent = int(round(len(_df_was.index) / 100))
    chunk_ord_by_gross = _df_was.sort_values('GrossNonRentIncome')
    max_gross_income = chunk_ord_by_gross.iloc[-one_per_cent]['GrossNonRentIncome']
    min_gross_income = chunk_ord_by_gross.iloc[one_per_cent]['GrossNonRentIncome']
    _df_was = _df_was[_df_was['GrossNonRentIncome'] <= max_gross_income]
    _df_was = _df_was[_df_was['GrossNonRentIncome'] >= min_gross_income]

    return _df_was


def read_and_clean_eff_data(_root_eff, _multiple_imputations, _exclude_house_sale_profits):
    # EFF variables of interest
    # facine3                       Household weight
    # renthog                       Household Gross Annual income
    # p7_2                          Importe recibido por alquileres durante el ano 2016
    # p7_4a                         Plusvalias por venta de prop. inmob. durante el ano 2016
    # p2_35a_i                      Prop i Tipo de propiedad (= 1 Vivienda)
    # p2_43_i                       Prop i Ingresos mensuales por alquiler de esta propiedad
    # Pre-select columns of interest so as to read data more efficiently
    _vars_of_interest = ['p2_35a', 'p2_43', 'p7_2', 'p7_4']
    _otras_secciones_cols = pd.read_csv(_root_eff + r'/eff_2017_imp1_csv/otras_secciones_2017_imp1.csv',
                                        nrows=1, sep=';').columns
    _otras_secciones_cols_of_interest = [c for c in _otras_secciones_cols if any(e in c for e in _vars_of_interest)]
    _otras_secciones_cols_of_interest = ['facine3', 'renthog'] + _otras_secciones_cols_of_interest
    # Read data
    if _multiple_imputations:
        _df_eff = pd.DataFrame()
        for i in range(1, 6):
            temp_df = pd.read_csv(_root_eff + r'/eff_2017_imp{0}_csv/otras_secciones_2017_imp{0}.csv'.format(i),
                                  sep=';', usecols=_otras_secciones_cols_of_interest)
            _df_eff = pd.concat([_df_eff, temp_df])
        _df_eff['facine3'] = _df_eff['facine3'] / 5.0
    else:
        _df_eff = pd.read_csv(_root_eff + r'/eff_2017_imp1_csv/otras_secciones_2017_imp1.csv', sep=';',
                              usecols=_otras_secciones_cols_of_interest)

    # Rename columns to be used
    _df_eff.rename(columns={'facine3': 'Weight'}, inplace=True)

    # Replace NaNs by zeros in EFF data
    _df_eff = _df_eff.fillna(0)

    # Compute annual gross non-rental income
    if not _exclude_house_sale_profits:
        _df_eff['GrossNonRentIncome'] = _df_eff['renthog'] - _df_eff['p7_2']
    else:
        _df_eff['GrossNonRentIncome'] = _df_eff['renthog'] - _df_eff['p7_2'] - _df_eff['p7_4a']

    # Compute total annual rental income, and total annual rental income from renting out dwellings
    _df_eff.rename(columns={'p7_2': 'TotalRentIncome'}, inplace=True)
    _df_eff['TotalHouseRentIncome'] = _df_eff.apply(lambda row: add_up_annual_house_rental_income_2016(row), axis=1)

    # Filter out NaNs, negative values and zeros
    _df_eff = _df_eff.loc[_df_eff['GrossNonRentIncome'] > 0.0]

    # Filter down to keep only columns of interest
    _df_eff = _df_eff[['Weight', 'GrossNonRentIncome', 'TotalRentIncome', 'TotalHouseRentIncome']]

    # Filter out the 1% with highest and the 1% with lowest GrossNonRentIncome
    _min_eff = weighted_quantile(_df_eff['GrossNonRentIncome'], 0.01, sample_weight=_df_eff['Weight'])
    _max_eff = weighted_quantile(_df_eff['GrossNonRentIncome'], 0.99, sample_weight=_df_eff['Weight'])
    _df_eff = _df_eff.loc[(_df_eff['GrossNonRentIncome'] >= _min_eff) & (_df_eff['GrossNonRentIncome'] <= _max_eff)]

    return _df_eff


def add_up_annual_house_rental_income_2016(_row):
    _total_house_rental_income = 0.0
    for i in range(1, 5):
        if _row['p2_35a_{}'.format(i)] in {1, 9}:
            _total_house_rental_income += _row['p2_43_{}'.format(i)]

    return 12.0 * _total_house_rental_income


# Control variables
writeResults = False
addModelResults = True
rootWAS = r''
rootEFF = r''
rootModel = r''
rootResults = r''

# General printing settings
np.set_printoptions(suppress=True)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 1000)

# Read Wealth and Assets Survey data for households
dfWAS = read_and_clean_was_data(rootWAS)

# Read Encuesta Financiera de las Familias data
dfEFF = read_and_clean_eff_data(rootEFF, _multiple_imputations=True, _exclude_house_sale_profits=True)

# Count number and fraction of BTL investors over total number of households
n_total_WAS = len(dfWAS)
n_BTL_WAS = len(dfWAS.loc[dfWAS['GrossRentalIncome'] > 0.0])
# n_total_WAS = sum(dfWAS['Weight'])
# n_BTL_WAS = sum(dfWAS.loc[dfWAS['GrossRentalIncome'] > 0.0, 'Weight'])
n_total_EFF = sum(dfEFF['Weight'])
n_BTL_EFF = sum(dfEFF.loc[dfEFF['TotalHouseRentIncome'] > 0.0, 'Weight'])

# If required, write results to file, otherwise, print to screen
if writeResults:
    with open(rootResults + '/BTLFractionOfHouseholds.txt', 'w') as f:
        f.write('##############################################################################\n')
        f.write('EFF DATA\n')
        f.write('##############################################################################\n')
        f.write('Total number of households = {}\n'.format(n_total_EFF))
        f.write('Number of BTL households = {}\n'.format(n_BTL_EFF))
        f.write('Fraction of BTL households over total number of households = {}\n'.format(n_BTL_EFF / n_total_EFF))
        f.write('\n##############################################################################\n')
        f.write('WAS DATA\n')
        f.write('##############################################################################\n')
        f.write('Total number of households = {}\n'.format(n_total_WAS))
        f.write('Number of BTL households = {}\n'.format(n_BTL_WAS))
        f.write('Fraction of BTL households over total number of households = {}'.format(n_BTL_WAS / n_total_WAS))
else:
    print('####################################### EFF #######################################')
    print('Total number of households = {}'.format(n_total_EFF))
    print('Number of BTL households = {}'.format(n_BTL_EFF))
    print('Fraction of BTL households over total number of households = {}'.format(n_BTL_EFF / n_total_EFF))
    print('####################################### WAS #######################################')
    print('Total number of households = {}'.format(n_total_WAS))
    print('Number of BTL households = {}'.format(n_BTL_WAS))
    print('Fraction of BTL households over total number of households = {}'.format(n_BTL_WAS / n_total_WAS))
    if addModelResults:
        # Read model results
        dfModel = pd.read_csv(rootModel + '/results/test4/Output-run1.csv', skipinitialspace=True,
                              delimiter=';', usecols=['Model time', 'nActiveBTL', 'TotalPopulation'])
        # Count number and fraction of BTL investors over total number of households
        n_total_Model = dfModel.loc[dfModel['Model time'] >= 1000, 'TotalPopulation'].mean()
        n_BTL_Model = dfModel.loc[dfModel['Model time'] >= 1000, 'nActiveBTL'].mean()
        # Print results to screen
        print('###################################### Model ######################################')
        print('Total number of households = {}'.format(n_total_Model))
        print('Number of BTL households = {}'.format(n_BTL_Model))
        print('Fraction of BTL households over total number of households = {}'.format(n_BTL_Model / n_total_Model))
