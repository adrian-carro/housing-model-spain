# -*- coding: utf-8 -*-
"""
Class to find mortgage conditions and limits (max LTV, max LTI, max DSTI, max age, maturity) using both Colegio de
Registradores and European Data Warehouse databases.

@author: Adrian Carro
"""

from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import Utilities.common_functions as cf


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


# Control variables
writeResults = False
fromYear = 2014
toYear = 2020
rootPSD = r''
rootEDW = r''
rootCdR = r''
rootEFF = r''
rootCIR = r''
rootBdE = r''
rootModelUK = r''
rootModelSP = r''
rootResults = r''

# General printing settings
np.set_printoptions(suppress=True)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 1000)

# Read and clean European Data Warehouse data
dfEDW = cf.read_and_clean_edw_data(rootEDW, _from_year=fromYear, _to_year=toYear)

# Read and clean Encuesta Financiera de las Familias data and use it to compute household income from individual income,
# add it to EDW data and recompute LTI values
dfEFF = read_and_clean_eff_data(rootEFF, _multiple_imputations=True, _exclude_house_sale_profits=True)
dfEDW = add_household_income_to_edw_data(dfEDW, dfEFF, _plot_fit=False)
dfEDW['LTI_hhldInc'] = dfEDW['LTI'] * dfEDW['B_inc'] / dfEDW['GrossNonRentIncome_LinFit']
dfEDW['DSTI_hhldInc'] = dfEDW['DSTI'] * dfEDW['B_inc'] / dfEDW['GrossNonRentIncome_LinFit']

# Read and clean Colegio de Registradores data
dfCdR = cf.read_and_clean_cdr_data(rootCdR, _with_price=False, _from_year=fromYear, _to_year=toYear)
dfCdRwP = cf.read_and_clean_cdr_data(rootCdR, _with_price=True, _from_year=fromYear, _to_year=toYear)

# Read and clean CIR data about BTL purchases
dfCIR = cf.read_and_clean_cir_data(rootCIR, _from_year=fromYear, _to_year=toYear)

# Read and clean BdE Indicadores del Mercado de la Vivienda data
dfBdE = cf.read_and_clean_bde_indicadores_data(rootBdE, _from_year=fromYear, _to_year=toYear, _columns=['PLAZO'])

# Find hard maximum LTV ratio for FTBs and HMs, defined as 99th percentile for non-BTL purchases (UK: 0.9) [UK,
# estimated with MoneyFacts data for 2011 (which contains information on the products offered)].
ltv_edw_max = np.nanpercentile(dfEDW['LTV'], [95, 99])
ltp_edw_max = np.nanpercentile(dfEDW.loc[dfEDW['LTP'] < 200.0, 'LTP'], [95, 99])
ltv_cdr_max = np.nanpercentile(dfCdR['LTV'], [95, 99])
ltp_cdr_max = np.nanpercentile(dfCdRwP['LTP'], [95, 99])
ltp_cdr_mode = dfCdRwP['LTP'].mode()[0]
ltp_cdr_frac_over_mode =\
    len(dfCdRwP.loc[(dfCdRwP['LTP'] > ltp_cdr_mode)]) / len(dfCdRwP.loc[~dfCdRwP['LTP'].isna()])
print('##############################################################################')
print('EDW - LTV: 95th percentile = {:.4f}, 99th percentile = {:.4f}'.format(*ltv_edw_max))
print('EDW - LTP: 95th percentile = {:.4f}, 99th percentile = {:.4f}'.format(*ltp_edw_max))
print('CdR - LTV: 95th percentile = {:.4f}, 99th percentile = {:.4f}'.format(*ltv_cdr_max))
print('CdR - LTP: 95th percentile = {:.4f}, 99th percentile = {:.4f}'.format(*ltp_cdr_max))
print('CdR - LTP: mode = {:.4f}, fraction over mode = {:.4f}'.format(ltp_cdr_mode, ltp_cdr_frac_over_mode))
if writeResults:
    with open(rootResults + '/MortgageLimits.txt', 'w') as f:
        f.write('##############################################################################\n')
        f.write('LTV - FTB & HM\n')
        f.write('##############################################################################\n')
        f.write('EDW - LTV: 95th percentile = {:.4f}, 99th percentile = {:.4f}\n'.format(*ltv_edw_max))
        f.write('EDW - LTP: 95th percentile = {:.4f}, 99th percentile = {:.4f}\n'.format(*ltp_edw_max))
        f.write('CdR - LTV: 95th percentile = {:.4f}, 99th percentile = {:.4f}\n'.format(*ltv_cdr_max))
        f.write('CdR - LTP: 95th percentile = {:.4f}, 99th percentile = {:.4f}\n'.format(*ltp_cdr_max))
        f.write('CdR - LTP: mode = {:.4f}, fraction over mode = {:.4f}\n'.format(ltp_cdr_mode, ltp_cdr_frac_over_mode))

# Find hard maximum LTV ratio for BTLs, defined as 99th percentile for BTL purchases (UK: 0.75) [UK, estimated with
# MoneyFacts data for 2011 (which contains information on the products offered)].
ltv_cir_max = np.nanpercentile(dfCIR['LTV_CIR'], [95, 99])
ltv_cir_cdr_max = np.nanpercentile(dfCIR['LTV_REGISTRO'], [95, 99])
ltp_cir_cdr_max = np.nanpercentile(dfCIR['LTP_REGISTRO'], [95, 99])
ltp_cir_mode = dfCIR['LTP_REGISTRO'].mode()[0]
ltp_cir_frac_over_mode =\
    len(dfCIR.loc[(dfCIR['LTP_REGISTRO'] > ltp_cir_mode)]) / len(dfCIR.loc[~dfCIR['LTP_REGISTRO'].isna()])
print('##############################################################################')
print('CIR - LTV: 95th percentile = {:.4f}, 99th percentile = {:.4f}'.format(*ltv_cir_max))
print('CIR - CdR - LTV: 95th percentile = {:.4f}, 99th percentile = {:.4f}'.format(*ltv_cir_cdr_max))
print('CIR - CdR - LTP: 95th percentile = {:.4f}, 99th percentile = {:.4f}'.format(*ltp_cir_cdr_max))
print('CIR - CdR - LTP: mode = {:.4f}, fraction over mode = {:.4f}'.format(ltp_cir_mode, ltp_cir_frac_over_mode))
if writeResults:
    with open(rootResults + '/MortgageLimits.txt', 'a') as f:
        f.write('\n##############################################################################\n')
        f.write('LTV - BTL\n')
        f.write('##############################################################################\n')
        f.write('CIR - LTV: 95th percentile = {:.4f}, 99th percentile = {:.4f}\n'.format(*ltv_cir_max))
        f.write('CIR - CdR - LTV: 95th percentile = {:.4f}, 99th percentile = {:.4f}\n'.format(*ltv_cir_cdr_max))
        f.write('CIR - CdR - LTP: 95th percentile = {:.4f}, 99th percentile = {:.4f}\n'.format(*ltp_cir_cdr_max))
        f.write('CIR - CdR - LTP: mode = {:.4f}, fraction over mode = {:.4f}\n'.format(ltp_cir_mode,
                                                                                       ltp_cir_frac_over_mode))

# Find hard maximum LTI ratio for FTBs and HMs, defined as 99th percentile for non-BTL purchases (UK: 5.4 for FTBs, 5.6
# for HMs) [UK, estimated with PSD data for 2011, 99th percentile].
lti_edw_max = np.nanpercentile(dfEDW['LTI'], [90, 95, 99])
lti2_edw_max = np.nanpercentile(dfEDW['LTI2'], [90, 95, 99])
lti3_edw_max = np.nanpercentile(dfEDW['LTI3'], [90, 95, 99])
lti_hhldInc_edw_max = np.nanpercentile(dfEDW['LTI_hhldInc'], [90, 95, 99])
print('##############################################################################')
print('EDW - LTI: 90th percentile = {:.4f}, 95th percentile = {:.4f}, 99th percentile = {:.4f}'.format(*lti_edw_max))
print('EDW - LTI with household income: 90th percentile = {:.4f}, 95th percentile = {:.4f}, '
      '99th percentile = {:.4f}'.format(*lti2_edw_max))
print('EDW - LTI with both incomes: 90th percentile = {:.4f}, 95th percentile = {:.4f}, '
      '99th percentile = {:.4f}'.format(*lti3_edw_max))
print('EDW - LTI with transformed household income: 90th percentile = {:.4f}, 95th percentile = {:.4f}, '
      '99th percentile = {:.4f}'.format(*lti_hhldInc_edw_max))
if writeResults:
    with open(rootResults + '/MortgageLimits.txt', 'a') as f:
        f.write('\n##############################################################################\n')
        f.write('LTI - FTB & HM\n')
        f.write('##############################################################################\n')
        f.write('EDW - LTI: 90th percentile = {:.4f}, 95th percentile = {:.4f}, '
                '99th percentile = {:.4f}\n'.format(*lti_edw_max))
        f.write('EDW - LTI with household income: 90th percentile = {:.4f}, 95th percentile = {:.4f}, '
                '99th percentile = {:.4f}\n'.format(*lti2_edw_max))
        f.write('EDW - LTI with both incomes: 90th percentile = {:.4f}, 95th percentile = {:.4f}, '
                '99th percentile = {:.4f}\n'.format(*lti3_edw_max))
        f.write('EDW - LTI with transformed household income: 90th percentile = {:.4f}, 95th percentile = {:.4f}, '
                '99th percentile = {:.4f}\n'.format(*lti_hhldInc_edw_max))

# Find hard maximum DSTI ratio for FTBs and HMs, defined as 99th percentile for non-BTL purchases (UK: 0.4) [UK,
# estimated with PSD data for 2011, 99th percentile].
dsti_edw_max = np.nanpercentile(dfEDW['DSTI'], [90, 95, 99])
dsti2_edw_max = np.nanpercentile(dfEDW['DSTI2'], [90, 95, 99])
dsti3_edw_max = np.nanpercentile(dfEDW['DSTI3'], [90, 95, 99])
dsti_hhldInc_edw_max = np.nanpercentile(dfEDW['DSTI_hhldInc'], [90, 95, 99])
print('##############################################################################')
print('EDW - DSTI: 90th percentile = {:.4f}, 95th percentile = {:.4f}, 99th percentile = {:.4f}'.format(*dsti_edw_max))
print('EDW - DSTI with household income: 90th percentile = {:.4f}, 95th percentile = {:.4f}, '
      '99th percentile = {:.4f}'.format(*dsti2_edw_max))
print('EDW - DSTI with both incomes: 90th percentile = {:.4f}, 95th percentile = {:.4f}, '
      '99th percentile = {:.4f}'.format(*dsti3_edw_max))
print('EDW - DSTI with transformed household income: 90th percentile = {:.4f}, 95th percentile = {:.4f}, '
      '99th percentile = {:.4f}'.format(*dsti_hhldInc_edw_max))
if writeResults:
    with open(rootResults + '/MortgageLimits.txt', 'a') as f:
        f.write('\n##############################################################################\n')
        f.write('DSTI - FTB & HM\n')
        f.write('##############################################################################\n')
        f.write('EDW - DSTI: 90th percentile = {:.4f}, 95th percentile = {:.4f}, '
                '99th percentile = {:.4f}\n'.format(*dsti_edw_max))
        f.write('EDW - DSTI with household income: 90th percentile = {:.4f}, 95th percentile = {:.4f}, '
                '99th percentile = {:.4f}\n'.format(*dsti2_edw_max))
        f.write('EDW - DSTI with both incomes: 90th percentile = {:.4f}, 95th percentile = {:.4f}, '
                '99th percentile = {:.4f}\n'.format(*dsti3_edw_max))
        f.write('EDW - DSTI with transformed household income: 90th percentile = {:.4f}, 95th percentile = {:.4f}, '
                '99th percentile = {:.4f}\n'.format(*dsti_hhldInc_edw_max))

# Find the maximum age for a non-BTL household to get a mortgage (for UK, this is defined as a maximum age to finish
# repaying its mortgages), thus akin to a retirement age. (UK: 65) [UK, defined as official retirement age in 2011]
age_edw_min = np.nanpercentile(dfEDW['age'], [1, 5, 10])
age_edw_max = np.nanpercentile(dfEDW['age'], [90, 95, 99])
print('##############################################################################')
print('EDW - Age: 1th percentile = {:.4f}, 5th percentile = {:.4f}, 10th percentile = {:.4f}'.format(*age_edw_min))
print('EDW - Age: 90th percentile = {:.4f}, 95th percentile = {:.4f}, 99th percentile = {:.4f}'.format(*age_edw_max))
if writeResults:
    with open(rootResults + '/MortgageLimits.txt', 'a') as f:
        f.write('\n##############################################################################\n')
        f.write('Age - FTB & HM\n')
        f.write('##############################################################################\n')
        f.write('EDW - Age: 1st percentile = {:.4f}, 5th percentile = {:.4f}, '
                '10th percentile = {:.4f}\n'.format(*age_edw_min))
        f.write('EDW - Age: 90th percentile = {:.4f}, 95th percentile = {:.4f}, '
                '99th percentile = {:.4f}\n'.format(*age_edw_max))

# Find the maximum age for a BTL household to get a mortgage, thus akin to a retirement age. (UK: 65) [UK, defined as
# official retirement age in 2011]
age_cir_max = np.nanpercentile(dfCIR['edad'], [90, 95, 99])
print('##############################################################################')
print('CIR - Age: 90th percentile = {:.4f}, 95th percentile = {:.4f}, 99th percentile = {:.4f}'.format(*age_cir_max))
if writeResults:
    with open(rootResults + '/MortgageLimits.txt', 'a') as f:
        f.write('\n##############################################################################\n')
        f.write('Age - BTL\n')
        f.write('##############################################################################\n')
        f.write('CIR - Age: 90th percentile = {:.4f}, 95th percentile = {:.4f}, '
                '99th percentile = {:.4f}\n'.format(*age_cir_max))

# Find mortgage maturity in years, defined as the median mortgage maturity at origination (UK: 25)
maturity_edw_mean = dfEDW['matyear'].mean()
maturity_edw_median = dfEDW['matyear'].median()
maturity_edw_mode = dfEDW['matyear'].mode()[0]
maturity_cdr_mean = dfCdR['plazohip'].mean() / 12.0
maturity_cdr_median = dfCdR['plazohip'].median() / 12.0
maturity_cdr_mode = dfCdR['plazohip'].mode()[0] / 12.0
maturity_bde_mean = dfBdE['PLAZO'].mean() / 12.0
print('##############################################################################')
print('EDW - Maturity: mean = {:.4f}, median = {:.4f}, mode = {:.4f}'.format(maturity_edw_mean, maturity_edw_median,
                                                                             maturity_edw_mode))
print('CdR - Maturity: mean = {:.4f}, median = {:.4f}, mode = {:.4f}'.format(maturity_cdr_mean, maturity_cdr_median,
                                                                             maturity_cdr_mode))
print('BdE - Maturity: mean = {:.4f}'.format(maturity_bde_mean))
if writeResults:
    with open(rootResults + '/MortgageLimits.txt', 'a') as f:
        f.write('\n##############################################################################\n')
        f.write('Maturity - FTB & HM\n')
        f.write('##############################################################################\n')
        f.write('EDW - Maturity: mean = {:.4f}, median = {:.4f}, mode = {:.4f}\n'.format(
            maturity_edw_mean, maturity_edw_median, maturity_edw_mode))
        f.write('CdR - Maturity: mean = {:.4f}, median = {:.4f}, mode = {:.4f}\n'.format(
            maturity_cdr_mean, maturity_cdr_median, maturity_cdr_mode))
        f.write('BdE - Maturity: mean = {:.4f}\n'.format(maturity_bde_mean))

# Find mortgage maturity for BTL purchases in years, defined as the median mortgage maturity at origination (UK: 25)
maturity_cir_mean = dfCIR['matyear_CIR'].mean()
maturity_cir_median = dfCIR['matyear_CIR'].median()
maturity_cir_mode = dfCIR['matyear_CIR'].mode()[0]
print('##############################################################################')
print('CIR - Maturity: mean = {:.4f}, median = {:.4f}, mode = {:.4f}'.format(maturity_cir_mean, maturity_cir_median,
                                                                             maturity_cir_mode))
if writeResults:
    with open(rootResults + '/MortgageLimits.txt', 'a') as f:
        f.write('\n##############################################################################\n')
        f.write('Maturity - BTL\n')
        f.write('##############################################################################\n')
        f.write('CIR - Maturity: mean = {:.4f}, median = {:.4f}, mode = {:.4f}\n'.format(
            maturity_cir_mean, maturity_cir_median, maturity_cir_mode))
