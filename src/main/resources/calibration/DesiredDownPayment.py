# -*- coding: utf-8 -*-
"""
Class to estimate the desired down-payments parameters using both Colegio de Registradores and European Data Warehouse
databases.

@author: Adrian Carro
"""

from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


def read_and_clean_edw_data(_root_edw, _price_variable):
    # _df_edw = pd.read_csv(_root_edw + '/EDWdata_JAN22.csv')
    _df_edw = pd.read_csv(_root_edw + '/EDWdata_updateMAR22_v2.csv', dtype={'M': str, 'Q': str})
    # _df_edw['age'] = _df_edw['Y'].subtract(_df_edw['B_date_birth'])
    # _df_edw['household_income'] = _df_edw[['B_inc', 'B_inc2']].sum(axis=1)

    # Remove duplicates
    _df_edw = _df_edw[~_df_edw.duplicated(keep=False)]

    # Remove unreasonably large LTPs (probably due to too small prices as compared to values)
    # _df_edw = _df_edw.loc[_df_edw['LTP'] < 200.0]

    # # Restrict to owner-occupied property
    # _df_edw = _df_edw.loc[_df_edw['property_type'] == '1']
    #
    # # Restrict both income and the selected price variable to positive values (thus also discarding NaNs)
    # _df_edw = _df_edw.loc[_df_edw['income'] > 0.0]
    # _df_edw = _df_edw.loc[_df_edw[_price_variable] > 0.0]
    #
    # # Remove 1% lowest and 1% highest incomes (very low income transactions are clearly the result of other undeclared
    # # sources of income, very high incomes are related to very high prices, which are not captured in the database)
    # q1, q99 = np.quantile(_df_edw['income'], [0.01, 0.99])
    # _df_edw = _df_edw.loc[(_df_edw['income'] > q1) & (_df_edw['income'] < q99)]
    # # Alternative way of restricting income values
    # # _df_edw = _df_edw.loc[(_df_edw['income'] > 10000.0) & (_df_edw['income'] < 100000.0)]
    #
    # # If 'price' selected as price variable, then remove values below 25k
    # if _price_variable == 'price':
    #     _df_edw = _df_edw.loc[_df_edw[_price_variable] > 25000]

    # Restrict to origination dates from a certain year on
    _df_edw = _df_edw[(_df_edw['Y'] >= 2014)]

    # # Manually remove the most clear vertical lines
    # _df_edw = _df_edw[(_df_edw['income'] != 130000) & (_df_edw['income'] != 43500) & (_df_edw['income'] != 21000)
    #                   & (_df_edw['income'] != 7500)]
    #
    # # Remove all integer income values, as they tend to be arranged into strange vertical lines
    # # _df_edw = _df_edw[_df_edw['income'].apply(lambda e: e != int(e))]
    #
    # # Filter down to keep only columns of interest
    # _df_edw = _df_edw[['income', _price_variable]]

    return _df_edw


def read_and_clean_cdr_data(_root_cdr, _with_price):
    if not _with_price:
        # Read data
        _df_cdr = pd.read_csv(rootCdR + '/Colegio_data.csv',
                              dtype={'capital': float, 'valor': float, 'interes': float, 'plazohip': int, 'Y': int,
                                     'M': str, 'Q': str, 'LTV': float, 'tipofijo': object})
        # Remove duplicates
        _df_cdr = _df_cdr[~_df_cdr.duplicated(keep=False)]
        # Restrict to origination dates from a certain year on
        _df_cdr = _df_cdr.loc[(_df_cdr['Y'] >= 2014)]
        # Return result
        return _df_cdr
    else:
        _df_cdr_wp = pd.read_csv(rootCdR + '/Colegio_data_con_LTP.csv',
                                 dtype={'capital': float, 'valor': float, 'interes': float, 'plazohip': int, 'Y': int,
                                        'M': str, 'Q': str, 'precio': float, 'LTP': float, 'tipofijo': object,
                                        'LTV': float})
        # Remove duplicates
        _df_cdr_wp = _df_cdr_wp[~_df_cdr_wp.duplicated(keep=False)]
        # Restrict to origination dates from a certain year on
        _df_cdr_wp = _df_cdr_wp.loc[(_df_cdr_wp['Y'] >= 2014)]
        return _df_cdr_wp


def read_and_clean_cir_data(_root_cir):
    # Read data
    _df_cir = pd.read_csv(_root_cir + '/CIR_BTL_data.csv', parse_dates=['M_CIR'])
    # Extract year from date column
    _df_cir['YEAR'] = _df_cir['M_CIR'].str[:4].astype(int)
    # Restrict to origination dates from a certain year on
    _df_cir = _df_cir.loc[(_df_cir['YEAR'] >= 2014)]
    # Return result
    return _df_cir


# Control variables
writeResults = False
saveFigures = False
priceVariableEDW = 'Ovalue'  # Can be 'Ovalue' or 'price'
rootEDW = r''
rootCdR = r''
rootCIR = r''
rootResults = r''

# General printing settings
np.set_printoptions(suppress=True)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 1000)

# Read and clean data
dfEDW = read_and_clean_edw_data(rootEDW, priceVariableEDW)
dfCdR = read_and_clean_cdr_data(rootCdR, _with_price=False)
dfCdRwP = read_and_clean_cdr_data(rootCdR, _with_price=True)
dfCIR = read_and_clean_cir_data(rootCIR)

# Compute down-payments (for EDW data, use only values, as prices barely available)
dfEDW['DownPayment_V'] = dfEDW['C_Ovaluation'] - dfEDW['L_amount']
dfCdR['DownPayment_V'] = dfCdR['valor'] - dfCdR['capital']
dfCdRwP['DownPayment_P'] = dfCdRwP['precio'] - dfCdRwP['capital']
dfCIR['DownPayment_V'] = dfCIR['GARfiable'] - dfCIR['principal']
dfCIR['DownPayment_P'] = dfCIR['precio'] - dfCIR['principal']
dfCIR['DownPayment_V_Fraction'] = (dfCIR['GARfiable'] - dfCIR['principal']) / dfCIR['GARfiable']
dfCIR['DownPayment_P_Fraction'] = (dfCIR['precio'] - dfCIR['principal']) / dfCIR['precio']

# Restrict to positive (log) down-payments
dfEDW = dfEDW.loc[dfEDW['DownPayment_V'] > 1.0]
dfCdR = dfCdR.loc[dfCdR['DownPayment_V'] > 1.0]
dfCdRwP = dfCdRwP.loc[dfCdRwP['DownPayment_P'] > 1.0]
dfCIR = dfCIR.loc[dfCIR['DownPayment_V'] > 1.0]
dfCIR = dfCIR.loc[dfCIR['DownPayment_P'] > 1.0]
dfCIR = dfCIR.loc[(dfCIR['DownPayment_V_Fraction'] > 0.0) & (dfCIR['DownPayment_V_Fraction'] < 100.0)]
dfCIR = dfCIR.loc[(dfCIR['DownPayment_P_Fraction'] > 0.0) & (dfCIR['DownPayment_V_Fraction'] < 100.0)]

# Actually, remove also below percentile 1
dfEDW = dfEDW.loc[dfEDW['DownPayment_V'] > np.nanpercentile(dfEDW['DownPayment_V'], 1)]
dfCdR = dfCdR.loc[dfCdR['DownPayment_V'] > np.nanpercentile(dfCdR['DownPayment_V'], 1)]
dfCdRwP = dfCdRwP.loc[dfCdRwP['DownPayment_P'] > np.nanpercentile(dfCdRwP['DownPayment_P'], 1)]
dfCIR = dfCIR.loc[dfCIR['DownPayment_V'] > np.nanpercentile(dfCIR['DownPayment_V'], 1)]
dfCIR = dfCIR.loc[dfCIR['DownPayment_P'] > np.nanpercentile(dfCIR['DownPayment_P'], 1)]

# Compute logarithmic down-payments
dfEDW['Log_DownPayment_V'] = np.log(dfEDW['DownPayment_V'])
dfCdR['Log_DownPayment_V'] = np.log(dfCdR['DownPayment_V'])
dfCdRwP['Log_DownPayment_P'] = np.log(dfCdRwP['DownPayment_P'])
dfCIR['Log_DownPayment_V'] = np.log(dfCIR['DownPayment_V'])
dfCIR['Log_DownPayment_P'] = np.log(dfCIR['DownPayment_P'])

# Normal fit of logarithmic down-payments
muEDW, stdEDW = norm.fit(dfEDW['Log_DownPayment_V'])
muCdR, stdCdR = norm.fit(dfCdR['Log_DownPayment_V'])
muCdRwP, stdCdRwP = norm.fit(dfCdRwP['Log_DownPayment_P'])
muCIR, stdCIR = norm.fit(dfCIR['Log_DownPayment_V'])
muCIRwP, stdCIRwP = norm.fit(dfCIR['Log_DownPayment_P'])

# Normal fit of down-payments fractions for BTL CIR data
muCIR_fraction, stdCIR_fraction = norm.fit(dfCIR['DownPayment_V_Fraction'])
muCIRwP_fraction, stdCIRwP_fraction = norm.fit(dfCIR['DownPayment_P_Fraction'])

# Define bins
min_x = min(dfEDW['Log_DownPayment_V'].min(), dfCdR['Log_DownPayment_V'].min(), dfCdRwP['Log_DownPayment_P'].min())
max_x = max(dfEDW['Log_DownPayment_V'].max(), dfCdR['Log_DownPayment_V'].max(), dfCdRwP['Log_DownPayment_P'].max())
min_x = np.floor(min_x * 10) / 10
max_x = np.ceil(max_x * 10) / 10
bin_edges = np.arange(min_x, max_x + 0.2, 0.2)

# Plot data
f1 = plt.figure(figsize=(6, 4))
ax1 = plt.gca()
f2 = plt.figure(figsize=(6, 4))
ax2 = plt.gca()
f3 = plt.figure(figsize=(6, 4))
ax3 = plt.gca()
ax1.hist(dfEDW['Log_DownPayment_V'], bins=bin_edges, label='EDW', density=True, alpha=0.5, color='tab:blue')
ax1.hist(dfCdR['Log_DownPayment_V'], bins=bin_edges, label='CdR', density=True, alpha=0.5, color='tab:green')
ax1.hist(dfCdRwP['Log_DownPayment_P'], bins=bin_edges, label='CdRwP', density=True, alpha=0.5, color='tab:red')
ax2.hist(dfCIR['Log_DownPayment_V'], bins=bin_edges, label='CIR', density=True, alpha=0.5, color='tab:orange')
ax2.hist(dfCIR['Log_DownPayment_P'], bins=bin_edges, label='CIRwP', density=True, alpha=0.5, color='tab:purple')
ax3.hist(dfCIR['DownPayment_V_Fraction'], bins=np.arange(0.0, 1.05, 0.05), label='CIR', density=True, alpha=0.5,
         color='tab:orange')
ax3.hist(dfCIR['DownPayment_P_Fraction'], bins=np.arange(0.0, 1.05, 0.05), label='CIRwP', density=True, alpha=0.5,
         color='tab:purple')

# Plot fits
temp_x = np.linspace(min_x, max_x, 100, endpoint=True)
ax1.plot(temp_x, norm.pdf(temp_x, muEDW, stdEDW), c='tab:blue', linewidth=2,
         label=r'EDW fit ($\mu = {:.2f}$, $\sigma = {:.2f}$)'.format(muEDW, stdEDW))
ax1.plot(temp_x, norm.pdf(temp_x, muCdR, stdCdR), c='tab:green', linewidth=2,
         label=r'CdR fit ($\mu = {:.2f}$, $\sigma = {:.2f}$)'.format(muCdR, stdCdR))
ax1.plot(temp_x, norm.pdf(temp_x, muCdRwP, stdCdRwP), c='tab:red', linewidth=2,
         label=r'CdRwP fit ($\mu = {:.2f}$, $\sigma = {:.2f}$)'.format(muCdRwP, stdCdRwP))
ax2.plot(temp_x, norm.pdf(temp_x, muCIR, stdCIR), c='tab:orange', linewidth=2,
         label=r'CIR fit ($\mu = {:.2f}$, $\sigma = {:.2f}$)'.format(muCIR, stdCIR))
ax2.plot(temp_x, norm.pdf(temp_x, muCIRwP, stdCIRwP), c='tab:purple', linewidth=2,
         label=r'CIRwP fit ($\mu = {:.2f}$, $\sigma = {:.2f}$)'.format(muCIRwP, stdCIRwP))
temp_x = np.linspace(0.0, 1.0, 100, endpoint=True)
ax3.plot(temp_x, norm.pdf(temp_x, muCIR_fraction, stdCIR_fraction), c='tab:orange', linewidth=2,
         label=r'CIR fit ($\mu = {:.2f}$, $\sigma = {:.2f}$)'.format(muCIR_fraction, stdCIR_fraction))
ax3.plot(temp_x, norm.pdf(temp_x, muCIRwP_fraction, stdCIRwP_fraction), c='tab:purple', linewidth=2,
         label=r'CIRwP fit ($\mu = {:.2f}$, $\sigma = {:.2f}$)'.format(muCIRwP_fraction, stdCIRwP_fraction))
ax1.set_xlabel('Log Down-Payment')
ax2.set_xlabel('Log Down-Payment')
ax3.set_xlabel('Down-Payment Fraction')
ax1.set_ylabel('Density')
ax2.set_ylabel('Density')
ax3.set_ylabel('Density')
ax1.set_xlim(5.0, 14.8)
ax2.set_xlim(5.0, 14.8)
ax1.set_ylim(0.0, 0.6)
ax2.set_ylim(0.0, 0.62)
ax1.legend(frameon=False, loc='upper left')
ax2.legend(frameon=False, loc='upper left')
ax3.legend(frameon=False, loc='upper right')

if writeResults:
    with open(rootResults + '/DesiredDownPaymentFits.txt', 'w') as f:
        f.write('##############################################################################\n')
        f.write('NORMAL DISTRIBUTION FIT - EDW DATA\n')
        f.write('##############################################################################\n')
        f.write('Mean = {}\n'.format(muEDW))
        f.write('Std = {}\n'.format(stdEDW))
        f.write('\n##############################################################################\n')
        f.write('NORMAL DISTRIBUTION FIT - CdR DATA\n')
        f.write('##############################################################################\n')
        f.write('Mean = {}\n'.format(muCdR))
        f.write('Std = {}\n'.format(stdCdR))
        f.write('\n##############################################################################\n')
        f.write('NORMAL DISTRIBUTION FIT - CdRwP DATA\n')
        f.write('##############################################################################\n')
        f.write('Mean = {}\n'.format(muCdRwP))
        f.write('Std = {}\n'.format(stdCdRwP))
        f.write('\n##############################################################################\n')
        f.write('NORMAL DISTRIBUTION FIT - CIR DATA\n')
        f.write('##############################################################################\n')
        f.write('Mean = {}\n'.format(muCIR))
        f.write('Std = {}\n'.format(stdCIR))
        f.write('\n##############################################################################\n')
        f.write('NORMAL DISTRIBUTION FIT - CIRwP DATA\n')
        f.write('##############################################################################\n')
        f.write('Mean = {}\n'.format(muCIRwP))
        f.write('Std = {}\n'.format(stdCIRwP))
        f.write('\n##############################################################################\n')
        f.write('NORMAL DISTRIBUTION FIT - CIR FRACTION DATA\n')
        f.write('##############################################################################\n')
        f.write('Mean = {}\n'.format(muCIR_fraction))
        f.write('Std = {}\n'.format(stdCIR_fraction))
        f.write('\n##############################################################################\n')
        f.write('NORMAL DISTRIBUTION FIT - CIRwP FRACTION DATA\n')
        f.write('##############################################################################\n')
        f.write('Mean = {}\n'.format(muCIRwP_fraction))
        f.write('Std = {}\n'.format(stdCIRwP_fraction))

if saveFigures:
    f1.tight_layout()
    f1.savefig(rootResults + '/DesiredDownPaymentDist1.pdf', format='pdf', dpi=300, bbox_inches='tight')
    f1.savefig(rootResults + '/DesiredDownPaymentDist1.png', format='png', dpi=300, bbox_inches='tight')
    f2.tight_layout()
    f2.savefig(rootResults + '/DesiredDownPaymentDist2.pdf', format='pdf', dpi=300, bbox_inches='tight')
    f2.savefig(rootResults + '/DesiredDownPaymentDist2.png', format='png', dpi=300, bbox_inches='tight')
    f3.tight_layout()
    f3.savefig(rootResults + '/DesiredDownPaymentDist3.pdf', format='pdf', dpi=300, bbox_inches='tight')
    f3.savefig(rootResults + '/DesiredDownPaymentDist3.png', format='png', dpi=300, bbox_inches='tight')
else:
    plt.show()
