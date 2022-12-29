# -*- coding: utf-8 -*-
"""
Class to compare the LTV-LTI-DSTI distributions resulting from the model against those observed on the Colegio de
Registradores and European Data Warehouse databases (PSD database for UK data).

@author: Adrian Carro
"""

from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import Utilities.common_functions as cf
import seaborn as sns


def read_and_clean_model_results(_root_model, _segregate_household_types, _n_runs):
    # Read model results
    _df_model = pd.DataFrame()
    for _i in range(1, _n_runs + 1):
        _df_temp = pd.read_csv(_root_model + '/SaleTransactions-run{}.csv'.format(_i), skipinitialspace=True, sep=';',
                               usecols=['mortgagePrincipal', 'transactionPrice', 'buyerMonthlyGrossEmploymentIncome',
                                        'mortgageMonthlyPayment', 'firstTimeBuyerMortgage', 'buyToLetMortgage',
                                        'buyerAge'])
        _df_model = pd.concat([_df_model, _df_temp])

    # Add extra columns for LTV, LTI and DSTI (correcting for rounding errors)
    _df_model['LTV'] = 100.0 * _df_model['mortgagePrincipal'] / _df_model['transactionPrice'] - 1e-10
    _df_model['LTI'] = (_df_model['mortgagePrincipal'] / (12.0 * _df_model['buyerMonthlyGrossEmploymentIncome'])
                        - 1e-10)
    _df_model['DSTI'] =\
        100.0 * _df_model['mortgageMonthlyPayment'] / _df_model['buyerMonthlyGrossEmploymentIncome'] - 1e-3

    # Keep separate copies for each type of household (FTB, HM), keeping only those with a principal larger than zero
    _results = dict()
    _results['FTB'] = _df_model[_df_model['firstTimeBuyerMortgage'] & _df_model['mortgagePrincipal'] > 0.0].copy()
    _results['HM'] = _df_model[~_df_model['firstTimeBuyerMortgage'] & ~_df_model['buyToLetMortgage']
                               & _df_model['mortgagePrincipal'] > 0.0].copy()

    # Keep only columns of interest
    _results['FTB'] = _results['FTB'][['LTV', 'LTI', 'DSTI', 'buyerAge']]
    _results['HM'] = _results['HM'][['LTV', 'LTI', 'DSTI', 'buyerAge']]

    if _segregate_household_types:
        return _results
    else:
        return pd.concat([_results['FTB'], _results['HM']])


def read_and_clean_psd_data(file_name):
    _bin_edges = []
    _bin_heights = []
    with open(file_name, 'r') as f:
        # Read bins and edges
        last_bin = 0.0
        for line in f:
            if line[0] != '#':
                _bin_edges.append(float(line.split(',')[0]))
                _bin_heights.append(float(line.split(',')[2]))
                last_bin = float(line.split(',')[1])
        # Add last bin edge, with last bin being artificially assigned a width equal to the previous bin if NaN is found
        # _bin_edges.append(2.0 * _bin_edges[-1] - _bin_edges[-2])
        if np.isnan(last_bin):
            _bin_edges.append(2.0 * _bin_edges[-1] - _bin_edges[-2])
        else:
            _bin_edges.append(last_bin)
    # # Normalise heights from frequencies to fractions (not densities!)
    # _bin_heights = [element / sum(_bin_heights) for element in _bin_heights]
    # Compute bin widths
    _bin_widths = [(b - a) for a, b in zip(_bin_edges[:-1], _bin_edges[1:])]
    return _bin_edges, _bin_heights, _bin_widths


def read_and_clean_eff_data(_root_eff, _multiple_imputations, _exclude_house_sale_profits):
    # EFF variables of interest
    # facine3                       Household weight
    # renthog                       Household Gross Annual income
    # p7_2                          Importe recibido por alquileres durante el ano 2016
    # p7_4a                         Plusvalias por venta de prop. inmob. durante el ano 2016
    if _multiple_imputations:
        _df_eff = pd.DataFrame()
        for k in range(1, 6):
            temp_df1 = pd.read_csv(_root_eff + r'/eff_2017_imp{0}_csv/otras_secciones_2017_imp{0}.csv'.format(k),
                                   sep=';')
            temp_df2 = pd.read_csv(_root_eff + r'/eff_2017_imp{0}_csv/seccion6_2017_imp{0}.csv'.format(k), sep=';')
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
convertIncomes = False
saveFigures = False
dataSourceComparison = False
plotLimits = False
useKernelDensity = True
withModel = True
# incomeToUseEDW = 'individual'  # Can be 'individual', 'household', 'both'
# incomeToUseEDW = 'household'  # Can be 'individual', 'household', 'both'
incomeToUseEDW = 'both'  # Can be 'individual', 'household', 'both'
nRuns = 100
fromYear = 2016
# fromYear = 2014
toYear = 2016
# toYear = 2018
rootPSD = r''
rootEDW = r''
rootCdR = r''
rootEFF = r''
rootBdE = r''
rootModelUK = r''
rootModelSP = r''
rootResults = r''

# General printing settings
np.set_printoptions(suppress=True)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 1000)

# General settings
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'sans',
    'font.sans-serif': ['DejaVu Sans']
})

# Choice variables check-up
if incomeToUseEDW not in ['individual', 'household', 'both']:
    print('incomeToUseEDW variable not recognised!')

# Read Spanish data
# Read and clean European Data Warehouse data
dfEDW = cf.read_and_clean_edw_data(rootEDW, _from_year=fromYear, _to_year=toYear)

# Read Banco de España interest rate data and use it to transform the DSTIs of the European Data Warehouse data.
# This is necessary since the interest rates reported in the EDW are very significantly below those actually
# prevalent according to BdE data
dfBdE = cf.read_and_clean_bde_indicadores_data(rootBdE, fromYear, toYear, _columns=['INTERES'])
interestRateFactor = dfBdE['INTERES'].mean() / dfEDW['Ointeres'].mean()
dfEDW['DSTI3'] = dfEDW['DSTI3'] * interestRateFactor

# Read and clean Encuesta Financiera de las Familias data and use it to compute household income from individual
# income, add it to EDW data and recompute LTI values
if convertIncomes:
    dfEFF = read_and_clean_eff_data(rootEFF, _multiple_imputations=True, _exclude_house_sale_profits=True)
    dfEDW = add_household_income_to_edw_data(dfEDW, dfEFF, _plot_fit=False)
    dfEDW['LTI_hhldInc'] = dfEDW['LTI'] * dfEDW['B_inc'] / dfEDW['GrossNonRentIncome_LinFit']
    dfEDW['DSTI_hhldInc'] = dfEDW['DSTI'] * dfEDW['B_inc'] / dfEDW['GrossNonRentIncome_LinFit']

# Read and clean Colegio de Registradores data
dfCdRwP = cf.read_and_clean_cdr_data(rootCdR, _with_price=True, _from_year=fromYear, _to_year=toYear)
dfCdR = None
if dataSourceComparison:
    dfCdR = cf.read_and_clean_cdr_data(rootCdR, _with_price=False, _from_year=fromYear, _to_year=toYear)

# Read and clean both UK and SP model results
if withModel:
    resultsSP = read_and_clean_model_results(rootModelSP, False, nRuns)
    resultsUK = read_and_clean_model_results(rootModelUK, False, 1)
else:
    resultsSP = None
    resultsUK = None


# Read LTV-LTI-DSTI limits from config file used
limits = None
if plotLimits:
    limits = dict((line.split('=')[0].strip(), line.split('=')[1].strip()) for line in
                  open(rootModelSP + '/config.properties', 'r') if (line[0] != '#' and len(line) > 3))

# Define bin edges and widths for Spanish data and results (not slight shift of LTV edges to better capture limit cases)
if useKernelDensity:
    bin_edges_SP = {
        'LTV': list(np.round(np.linspace(0.0, 130.0001, 27), 5)),
        'LTI': list(np.round(np.linspace(0.0, 16.0001, 33), 5)),
        'DSTI': list(np.round(np.linspace(0.0, 100.001, 21), 5))
    }
else:
    bin_edges_SP = {
        'LTV': list(np.round(np.linspace(0.0, 125.0001, 26), 5)) + [500.0],
        'LTI': list(np.round(np.linspace(0.0, 14.5001, 30), 5)) + [500.0],
        'DSTI': list(np.round(np.linspace(0.0, 60.0001, 13), 5)) + [500.0]
    }

# Plot
fig = []
axes = []
for i, variable in enumerate(['LTV', 'LTI', 'DSTI']):
    f = plt.figure(figsize=(2.75, 2.75))
    fig.append(f)
    axes.append(f.gca())

# Spanish case
# LTV data
bin_heights = np.histogram(dfCdRwP['LTP'], bin_edges_SP['LTV'], density=useKernelDensity)[0]
if not useKernelDensity:
    bin_heights = bin_heights / sum(bin_heights)
bin_width = np.diff(bin_edges_SP['LTV'][:2])[0]
axes[0].bar(bin_edges_SP['LTV'][:-1], bin_heights, width=bin_width, align='edge', alpha=0.5,
            label='Data Spain\n(LTP)', color='tab:blue')
# if useKernelDensity:
#     sns.kdeplot(data=dfCdRwP['LTP'], ax=axes[1, 0], common_norm=False, lw=2.0, color='tab:blue')
# LTV model results
if withModel:
    hist = np.histogram(resultsSP['LTV'], bins=bin_edges_SP['LTV'], density=useKernelDensity)[0]
    if not useKernelDensity:
        hist = hist / sum(hist)
    axes[0].bar(bin_edges_SP['LTV'][:-1], hist, width=bin_width, align='edge', alpha=0.5, label='Model Spain',
                color='tab:orange')
# if useKernelDensity:
#     sns.kdeplot(data=resultsSP['LTV'], ax=axes[1, 0], common_norm=False, lw=2.0, color='tab:orange')
# ...potentially adding other data sources for comparison
if dataSourceComparison:
    for dfSource, ls, label, c in zip([dfEDW['LTV'], dfEDW['LTP'], dfCdR['LTV']], ['o-', '*-', '>-'],
                                      ['EDW data (LTV)', 'EDW data (LTP)', 'CdR data (LTV)'],
                                      ['tab:red', 'tab:green', 'tab:purple']):
        bin_heights = np.histogram(dfSource, bin_edges_SP['LTV'], density=useKernelDensity)[0]
        if not useKernelDensity:
            bin_heights = bin_heights / sum(bin_heights)
        axes[0].plot([e + 2.5 for e in bin_edges_SP['LTV'][:-1]], bin_heights, ls, label=label, color=c)
# LTV individual plot settings
axes[0].set_xlim(bin_edges_SP['LTV'][0], bin_edges_SP['LTV'][-2] + bin_width)
axes[0].set_xlabel('LTV')
axes[0].set_ylabel('Density')
if plotLimits:
    axes[0].axvline(x=100.0 * float(limits['BANK_LTV_HARD_MAX_HM']), c='k', ls='--', lw=2.0)
    axes[0].axvline(x=100.0 * float(limits['BANK_LTV_HARD_MAX_BTL']), c='k', ls=':', lw=2.0)

# LTI data
if incomeToUseEDW == 'individual':
    variableLTI = 'LTI'
elif incomeToUseEDW == 'household':
    variableLTI = 'LTI2'
else:
    variableLTI = 'LTI3'
bin_heights = np.histogram(dfEDW[variableLTI], bin_edges_SP['LTI'], density=useKernelDensity)[0]
if not useKernelDensity:
    bin_heights = bin_heights / sum(bin_heights)
bin_width = np.diff(bin_edges_SP['LTI'][:2])[0]
axes[1].bar(bin_edges_SP['LTI'][:-1], bin_heights, width=bin_width, align='edge', alpha=0.5, label='Data Spain',
            color='tab:blue')
# if useKernelDensity:
#     sns.kdeplot(data=dfEDW[variableLTI], ax=axes[1, 1], common_norm=False, lw=2.0, color='tab:blue')
# LTI model results
if withModel:
    hist = np.histogram(resultsSP['LTI'], bins=bin_edges_SP['LTI'], density=useKernelDensity)[0]
    if not useKernelDensity:
        hist = hist / sum(hist)
    axes[1].bar(bin_edges_SP['LTI'][:-1], hist, width=bin_width, align='edge', alpha=0.5, label='Model Spain',
                color='tab:orange')
# if useKernelDensity:
#     sns.kdeplot(data=resultsSP['LTI'], ax=axes[1, 1], common_norm=False, lw=2.0, color='tab:orange')
# LTI individual plot settings
axes[1].set_xlim(bin_edges_SP['LTI'][0], bin_edges_SP['LTI'][-2] + bin_width)
axes[1].set_xlabel('LTI')
axes[1].set_ylabel('Density')
if plotLimits:
    axes[1].axvline(x=float(limits['BANK_LTI_HARD_MAX_HM']), c='k', ls='--', lw=2.0)

# DSTI data
if incomeToUseEDW == 'individual':
    variableDSTI = 'DSTI'
elif incomeToUseEDW == 'household':
    variableDSTI = 'DSTI2'
else:
    variableDSTI = 'DSTI3'
bin_heights = np.histogram(dfEDW[variableDSTI], bin_edges_SP['DSTI'], density=useKernelDensity)[0]
if not useKernelDensity:
    bin_heights = bin_heights / sum(bin_heights)
bin_width = np.diff(bin_edges_SP['DSTI'][:2])[0]
axes[2].bar(bin_edges_SP['DSTI'][:-1], bin_heights, width=bin_width, align='edge', alpha=0.5, label='Data Spain',
            color='tab:blue')
# if useKernelDensity:
#     sns.kdeplot(data=dfEDW[variableDSTI], ax=axes[1, 2], common_norm=False, lw=2.0, color='tab:blue')
# DSTI model results
if withModel:
    hist = np.histogram(resultsSP['DSTI'], bins=bin_edges_SP['DSTI'], density=useKernelDensity)[0]
    if not useKernelDensity:
        hist = hist / sum(hist)
    axes[2].bar(bin_edges_SP['DSTI'][:-1], hist, width=bin_width, align='edge', alpha=0.5, label='Model Spain',
                color='tab:orange')
# if useKernelDensity:
#     sns.kdeplot(data=resultsSP['DSTI'], ax=axes[1, 2], common_norm=False, lw=2.0, color='tab:orange')
# DSTI individual plot settings
axes[2].set_xlim(bin_edges_SP['DSTI'][0], bin_edges_SP['DSTI'][-2] + bin_width)
axes[2].set_xlabel('DSTI')
axes[2].set_ylabel('Density')
if plotLimits:
    axes[2].axvline(x=100.0 * float(limits['BANK_AFFORDABILITY_HARD_MAX']), c='k', ls='--', lw=2.0)

# UK case
if not withModel:
    # Plot UK data and results, iterating through variables...
    j = 0
    for variable in ['LTV', 'LTI', 'DSTI']:
        # ...reading the corresponding data and plotting it
        _, bin_heights_FTB, _ = read_and_clean_psd_data(rootPSD + r'/' + variable + '_FTB.csv')
        bin_edges, bin_heights_HM, bin_widths = read_and_clean_psd_data(rootPSD + r'/' + variable + '_HM.csv')
        if variable == 'LTV':
            bin_edges = [e - 0.5 if e > 0.0 else 0.0 for e in bin_edges]
        bin_heights = [a + b for a, b in zip(bin_heights_FTB, bin_heights_HM)]
        if useKernelDensity:
            bin_heights = [h / (w * sum(bin_heights)) for h, w in zip(bin_heights, bin_widths)]
        else:
            bin_heights = [e / sum(bin_heights) for e in bin_heights]
        axes[j].bar(bin_edges[:-1], bin_heights, width=bin_widths, align='edge',
                    alpha=0.5, label='Data UK', color='tab:orange')
        # ...creating a histogram for each and plotting it
        if withModel:
            hist = np.histogram(resultsUK[variable], bins=bin_edges, density=useKernelDensity)[0]
            if not useKernelDensity:
                hist = hist / sum(hist)
            axes[j].bar(bin_edges[:-1], hist, width=bin_widths, align='edge', alpha=0.5, label='Model UK',
                        color='tab:blue')
        # ...and finally setting other plot details
        axes[j].set_xlim(min(bin_edges), max(bin_edges))
        axes[j].set_xlabel(variable)
        axes[j].set_ylabel('Density')
        j += 1

# Further formatting
if withModel:
    axes[0].legend(loc='upper left', frameon=False, handlelength=1.7, handletextpad=0.45, borderaxespad=0.2, fontsize=9)
    axes[1].legend(loc='upper right', frameon=False, handlelength=1.7, handletextpad=0.45, borderaxespad=0.2)
    axes[2].legend(loc='upper right', frameon=False, handlelength=1.7, handletextpad=0.45, borderaxespad=0.2)
else:
    axes[0].legend(loc='upper left', frameon=False, handlelength=1.7, handletextpad=0.45, borderaxespad=0.2)
    axes[1].legend(loc='upper right', frameon=False)
    axes[2].legend(loc='upper right', frameon=False)
axes[1].set_xlim(0.0, 16.0)
axes[0].set_ylim(0, 0.044)
if not withModel:
    axes[1].set_ylim(0, 0.38)
    axes[2].set_ylim(0, 0.062)
else:
    axes[2].set_yticks([0.0, 0.01, 0.02, 0.03])
axes[0].set_xticks([0, 25, 50, 75, 100, 125])
axes[0].set_yticks([0.0, 0.01, 0.02, 0.03, 0.04])

# Final figure details
if saveFigures:
    for i, f in enumerate(fig):
        f.tight_layout()
        fileName = '/LTV-LTI-DSTI-Distributions{}'.format(i)
        if useKernelDensity:
            fileName += '-KernelDensity'
        if not withModel:
            fileName += '-SpainUK'
        else:
            fileName += '-SpainModel'
        f.savefig(rootResults + fileName + '.png', format='png', dpi=300, bbox_inches='tight')
        f.savefig(rootResults + fileName + '.pdf', format='pdf', dpi=300, bbox_inches='tight')
        # f.savefig(rootResults + fileName + '-UKOnly.eps', format='eps', dpi=300, bbox_inches='tight')
else:
    plt.show()
