"""
Class to study the distribution of purchase prices on the sales market.

@author: Adrian Carro
"""

# Imports
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd


def plot_mean_and_std_per_year():
    mu_edws = []
    std_edws = []
    mu_cdrs = []
    std_cdrs = []
    for y in range(2004, 2022):
        if len(dfEDW.loc[dfEDW['Y'] == y]) > 0:
            mu_edw, std_edw = norm.fit(np.log(dfEDW.loc[dfEDW['Y'] == y, priceVariable]))
            mu_edws.append(mu_edw)
            std_edws.append(std_edw)
        else:
            mu_edws.append(np.nan)
            std_edws.append(np.nan)
        mu_cdr, std_cdr = norm.fit(np.log(dfCdR.loc[dfCdR['Y'] == y, 'valor']))
        mu_cdrs.append(mu_cdr)
        std_cdrs.append(std_cdr)
    plt.figure()
    plt.errorbar(list(range(2004, 2022)), mu_cdrs, yerr=std_cdrs, capsize=2.0, label='CdR')
    plt.errorbar(list(range(2004, 2022)), mu_edws, yerr=std_edws, capsize=2.0, label='EDW')


# Set parameters
# priceVariable = 'price'  # Possible values are 'price' and 'Ovalue'
priceVariable = 'C_Ovaluation'  # Possible values are 'price' and 'Ovalue'
saveFigures = False
writeResults = False
addModelResults = True
fromYear = 2014
rootEDW = r''
rootCdR = r''
rootModel = r''
rootResults = r''
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 1000)

# Read data
# dfEDW = pd.read_csv(rootEDW + '/EDWdata_JAN22.csv')
dfEDW = pd.read_csv(rootEDW + '/EDWdata_updateMAR22_v2.csv', dtype={'M': str, 'Q': str})
dfCdR = pd.read_csv(rootCdR + '/Colegio_data.csv',
                    dtype={'capital': float, 'valor': float, 'interes': float, 'plazohip': int, 'Y': int, 'M': str,
                           'Q': str, 'LTV': float, 'tipofijo': object})
dfCdRwP = pd.read_csv(rootCdR + '/Colegio_data_con_LTP.csv',
                      dtype={'capital': float, 'valor': float, 'interes': float, 'plazohip': int, 'Y': int, 'M': str,
                             'Q': str, 'precio': float, 'LTP': float, 'tipofijo': object, 'LTV': float})

# Remove duplicates
dfEDW = dfEDW[~dfEDW.duplicated(keep=False)]

# Restrict to origination dates from a certain year on (tends to shift the distribution downwards and cut left tail)
dfEDW = dfEDW.loc[(dfEDW['Y'] >= fromYear)]
dfCdR = dfCdR.loc[(dfCdR['Y'] >= fromYear)]
dfCdRwP = dfCdRwP.loc[(dfCdRwP['Y'] >= fromYear)]

# Restrict positive values of the selected price variable (thus also discarding NaNs)
dfEDW = dfEDW.loc[dfEDW[priceVariable] > 0.0]
dfCdR = dfCdR.loc[dfCdR['valor'] > 0.0]
dfCdRwP = dfCdRwP.loc[dfCdRwP['precio'] > 0.0]

# Further restrict Colegio value data and Colegio price data, removing unreasonably low values
dfCdR = dfCdR.loc[dfCdR['valor'] > 5000.0]
dfCdRwP = dfCdRwP.loc[dfCdRwP['precio'] > 5000.0]

# If comparing EDW with CdR, then CdR should be cut accordingly
# dfCdR = dfCdR.loc[(dfCdR['valor'] >= dfEDW[priceVariable].min()) & (dfCdR['valor'] <= dfEDW[priceVariable].max())]

# If needed, print mean and std per year
# plot_mean_and_std_per_year()

# Plot results
plt.figure(figsize=(7.5, 5.5))
plt.hist(np.log(dfEDW[priceVariable]), bins=40, density=True, color='tab:blue', alpha=0.5, label='EDW')
plt.hist(np.log(dfCdR['valor']), bins=40, density=True, color='tab:orange', alpha=0.5, label='CdR')
plt.hist(np.log(dfCdRwP['precio']), bins=40, density=True, color='tab:green', alpha=0.5, label='CdRwP')
x_min, x_max = plt.xlim()
temp_x = np.linspace(x_min, x_max, 100)
muEDW, stdEDW = norm.fit(np.log(dfEDW[priceVariable]))
plt.plot(temp_x, norm.pdf(temp_x, muEDW, stdEDW), c='tab:blue', linewidth=2,
         label=r'EDW fit ($\mu = {:.2f}$, $\sigma = {:.2f}$)'.format(muEDW, stdEDW))
muCdR, stdCdR = norm.fit(np.log(dfCdR['valor']))
plt.plot(temp_x, norm.pdf(temp_x, muCdR, stdCdR), c='tab:orange', linewidth=2,
         label=r'CdR fit ($\mu = {:.2f}$, $\sigma = {:.2f}$)'.format(muCdR, stdCdR))
muCdRwP, stdCdRwP = norm.fit(np.log(dfCdRwP['precio']))
plt.plot(temp_x, norm.pdf(temp_x, muCdRwP, stdCdRwP), c='tab:green', linewidth=2,
         label=r'CdRwP fit ($\mu = {:.2f}$, $\sigma = {:.2f}$)'.format(muCdRwP, stdCdRwP))
if addModelResults:
    df_model = pd.read_csv(rootModel + '/results/test3/SaleTransactions-run1.csv',
                           skipinitialspace=True, delimiter=';', usecols=['transactionPrice'])
    plt.hist(np.log(df_model['transactionPrice']), bins=40, density=True, color='tab:red', alpha=0.5, label='Model')
    muModel, stdModel = norm.fit(np.log(df_model['transactionPrice']))
    plt.plot(temp_x, norm.pdf(temp_x, muModel, stdModel), c='tab:red', linewidth=2,
             label=r'Model fit ($\mu = {:.2f}$, $\sigma = {:.2f}$)'.format(muModel, stdModel))
plt.xlabel('ln(Value or Price)')
plt.ylabel('Prob. Density')
plt.xlim(9.0, 15.0)
plt.legend()

if writeResults:
    with open(rootResults + '/PriceDistFits.txt', 'w') as f:
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

if saveFigures:
    plt.tight_layout()
    if addModelResults:
        figureFileName = 'PriceDistWithModel'
    else:
        figureFileName = 'PriceDist'
    plt.savefig(rootResults + '/{}.pdf'.format(figureFileName), format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(rootResults + '/{}.png'.format(figureFileName), format='png', dpi=300, bbox_inches='tight')
else:
    plt.show()
