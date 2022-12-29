# -*- coding: utf-8 -*-
"""
Class to find credit supply parameters, such as the average interest rate and the rate of change of the interest rate
in response to a change in the demand/supply for credit.

@author: Adrian Carro
"""

from __future__ import division
import numpy as np
import pandas as pd
import Utilities.common_functions as cf


def read_and_clean_ine_mortgage_data(_root_ine):
    # Read data and reorganise it
    _df_ine = pd.read_excel(_root_ine + r'/Hipotecas constituidas sobre el total de fincas por naturaleza de la finca '
                                        r'- Mensual.xlsx', header=[6, 7, 8], skipfooter=25)
    _df_ine.drop((' ', ' ', ' '), axis='columns', inplace=True)
    _df_ine = _df_ine.stack()
    _df_ine = _df_ine.droplevel(0)
    # Select mortgage credit for dwellings
    _df_ine = _df_ine['Viviendas'][['Importe de hipotecas']]
    # Rename columns of interest
    _df_ine.rename(columns={'Importe de hipotecas': 'CREDITO'}, inplace=True)
    # Convert to euros
    _df_ine['CREDITO'] *= 1000
    # Extract year and month to separate numeric columns
    _df_ine['Y'] = pd.to_numeric(_df_ine.index.str[:4])
    _df_ine['M'] = pd.to_numeric(_df_ine.index.str[-2:])
    # Reset index and order columns
    _df_ine.reset_index(drop=True, inplace=True)
    _df_ine = _df_ine[['Y', 'M', 'CREDITO']]

    return _df_ine


def read_and_clean_ine_household_data(_root_ine):
    # Read data and reorganise it
    _df_ine = pd.read_excel(_root_ine + r'/EPA (Encuesta de Población Activa) - Hogares por número de activos y '
                                        r'número de personas.xlsx', header=[6, 7], skipfooter=18)
    _df_ine.drop((' ', ' '), axis='columns', inplace=True)
    _df_ine = _df_ine.stack()
    _df_ine = _df_ine.droplevel(0)
    # Select total number of households
    _df_ine = _df_ine[['Total']]
    # Rename columns of interest
    _df_ine.rename(columns={'Total': 'N_HOUSEHOLDS'}, inplace=True)
    # Convert to euros
    _df_ine['N_HOUSEHOLDS'] *= 1000
    # Extract year and month to separate numeric columns
    _df_ine['Y'] = pd.to_numeric(_df_ine.index.str[:4])
    _df_ine['T'] = pd.to_numeric(_df_ine.index.str[-1:])
    # Switch to monthly data by repeating the current quarterly data
    _df_ine = _df_ine.loc[_df_ine.index.repeat(3)]
    # Reset index, add month counter column and reorder columns
    _df_ine.reset_index(drop=True, inplace=True)
    _df_ine['M'] = _df_ine.index % 12 + 1
    _df_ine = _df_ine[['Y', 'M', 'N_HOUSEHOLDS']]
    # Manually add final data point for January 2022, for coherence with other datasets
    _df_ine = _df_ine.append({'Y': 2022, 'M': 1, 'N_HOUSEHOLDS': 18990000}, ignore_index=True)

    return _df_ine


def get_interest_and_credit_data(_root_bde, _root_ine):
    # Read required databases
    _df_bde = cf.read_and_clean_bde_indicadores_data(_root_bde, None, None, _columns=['INTERES'])
    _df_bde = _df_bde.loc[~_df_bde['INTERES'].isna()]
    _df_credit_supply = read_and_clean_ine_mortgage_data(_root_ine)
    _df_households = read_and_clean_ine_household_data(_root_ine)
    # Convert interest rates to points
    _df_bde['INTERES'] /= 100.0
    # Merge databases
    _df_credit_vs_rates = pd.merge(
        _df_credit_supply, _df_bde.loc[_df_bde['Y'] >= 2003, ['INTERES']].reset_index(drop=True),
        left_index=True, right_index=True)
    _df_credit_vs_rates = pd.merge(
        _df_credit_vs_rates, _df_households.loc[_df_households['Y'] >= 2003, ['N_HOUSEHOLDS']].reset_index(drop=True),
        left_index=True, right_index=True)
    # Compute credit per household column
    _df_credit_vs_rates['CREDIT_PER_HOUSEHOLD'] = _df_credit_vs_rates['CREDITO'] / _df_credit_vs_rates['N_HOUSEHOLDS']

    return _df_credit_vs_rates


# Control variables
writeResults = False
rootBdE = r''
rootINE = r''
rootEDW = r''
rootCdR = r''
rootCIR = r''
rootResults = r''

# General printing settings
np.set_printoptions(suppress=True)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 1000)

# Read and clean BdE Indicadores del Mercado de la Vivienda data
dfBdE = cf.read_and_clean_bde_indicadores_data(rootBdE, 2014, 2020, ['INTERES'])

# Use BdE Indicadores del Mercado de la Vivienda data as well as various INE databases to construct a DataFrame of
# interest rates, mortgage credit supply for dwellings and number of households in time
dfCreditVsRates = get_interest_and_credit_data(rootBdE, rootINE)

# Read and clean European Data Warehouse data
dfEDW = cf.read_and_clean_edw_data(rootEDW, 2014, 2020)
dfEDW.loc[(~dfEDW['Ointeres'].isna()) & (dfEDW['Ointeres'] < 0.1), 'Ointeres'] *= 100.0

# Read and clean Colegio de Registradores data
dfCdR = cf.read_and_clean_cdr_data(rootCdR, _with_price=False, _from_year=2014, _to_year=2020)

# Read and clean CIR data about BTL purchases
dfCIR = cf.read_and_clean_cir_data(rootCIR, 2014, 2020)

# Find the mean interest rate for each database and print it to screen and file as required
BdE_mean_interest_rate = dfBdE['INTERES'].mean()
EDW_mean_interest_rate = dfEDW.loc[(dfEDW['Ointeres'] > 0.0), 'Ointeres'].mean()
CdR_mean_interest_rate = dfCdR.loc[(dfCdR['interes'] > 0.0), 'interes'].mean()
CIR_mean_interest_rate = dfCIR.loc[dfCIR['tipo_CIR'] > 0.0, 'tipo_CIR'].mean()
print('##############################################################################')
print('BdE: Mean Interest Rate = {:.4f}'.format(BdE_mean_interest_rate))
print('EDW: Mean Interest Rate = {:.4f}'.format(EDW_mean_interest_rate))
print('CdR: Mean Interest Rate = {:.4f}'.format(CdR_mean_interest_rate))
print('CIR: Mean Interest Rate = {:.4f}'.format(CIR_mean_interest_rate))
print('##############################################################################')
print('Elasticity of interest rate to credit = {}'.format(
    dfCreditVsRates['INTERES'].diff().abs().mean() / dfCreditVsRates['CREDIT_PER_HOUSEHOLD'].diff().abs().mean()))
if writeResults:
    with open(rootResults + '/CreditSupply.txt', 'w') as f:
        f.write('##############################################################################\n')
        f.write('MORTGAGE INTEREST RATE\n')
        f.write('##############################################################################\n')
        f.write('BdE: Mean Interest Rate = {:.4f}\n'.format(BdE_mean_interest_rate))
        f.write('EDW: Mean Interest Rate = {:.4f}\n'.format(EDW_mean_interest_rate))
        f.write('CdR: Mean Interest Rate = {:.4f}\n'.format(CdR_mean_interest_rate))
        f.write('CIR: Mean Interest Rate = {:.4f}\n'.format(CIR_mean_interest_rate))
        f.write('\n##############################################################################\n')
        f.write('ELASTICITY OF INTEREST RATE TO CREDIT\n')
        f.write('##############################################################################\n')
        f.write('Elasticity of interest rate to credit = {}\n'.format(
            dfCreditVsRates['INTERES'].diff().abs().mean()
            / dfCreditVsRates['CREDIT_PER_HOUSEHOLD'].diff().abs().mean()))
