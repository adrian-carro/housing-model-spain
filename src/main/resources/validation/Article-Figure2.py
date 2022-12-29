"""
This creates the different plots included in Figure 2 of 'Heterogeneous Effects and Spillovers of Macroprudential Policy
in an Agent-Based Model of the UK Housing Market', both for the UK and Spanish calibrations.

@author: Adrian Carro
"""

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Utilities.common_functions as cf


def read_and_clean_model_results(_root_model, _n_runs):
    # Read model results
    _df_model = pd.DataFrame()
    for i in range(1, _n_runs + 1):
        _df_temp = pd.read_csv(_root_model + '/SaleTransactions-run{}.csv'.format(i), skipinitialspace=True, sep=';',
                               usecols=['mortgagePrincipal', 'transactionPrice', 'buyerMonthlyGrossEmploymentIncome',
                                        'mortgageMonthlyPayment', 'firstTimeBuyerMortgage', 'buyToLetMortgage',
                                        'buyerAge', 'buyerMonthlyGrossTotalIncome'])
        _df_model = pd.concat([_df_model, _df_temp])

    # Keep only FTB and HM purchases
    _df_model = _df_model.loc[~_df_model['buyToLetMortgage']]

    # Add extra columns as required
    _df_model['HPI'] = _df_model['transactionPrice'] / (12.0 * _df_model['buyerMonthlyGrossEmploymentIncome'])
    _df_model['LTV'] = 100.0 * _df_model['mortgagePrincipal'] / _df_model['transactionPrice'] - 1e-10
    _df_model['LTI'] = (_df_model['mortgagePrincipal'] / (12.0 * _df_model['buyerMonthlyGrossEmploymentIncome'])
                        - 1e-10)
    # _df_model['DSTI'] = \
    #     100.0 * _df_model['mortgageMonthlyPayment'] / _df_model['buyerMonthlyGrossEmploymentIncome'] - 1e-3

    return _df_model


def get_house_price_quintile(_value, _borders):
    if np.isnan(_value):
        return np.nan
    else:
        _group_number = 1
        for _border in _borders:
            if _value >= _border:
                _group_number += 1
            else:
                break
        return _group_number


def read_and_clean_eff_data(_root_eff, _multiple_imputations, _exclude_house_sale_profits):
    # EFF variables of interest
    # facine3                       Household weight
    # renthog                       Household Gross Annual income
    # p7_2                          Importe recibido por alquileres durante el ano 2016
    # p7_4a                         Plusvalias por venta de prop. inmob. durante el ano 2016
    # p2_33                         Number of other properties, apart from home
    # p2_35a_i                      Prop i Tipo de propiedad (= 1 Vivienda)
    # p2_43_i                       Prop i Ingresos mensuales por alquiler de esta propiedad
    # Pre-select columns of interest so as to read data more efficiently
    _vars_of_interest = ['p2_33', 'p2_35a', 'p2_43', 'p7_2', 'p7_4', 'p2_1']
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
    _df_eff.rename(columns={'p2_33': 'NumberOtherProperties'}, inplace=True)
    _df_eff.rename(columns={'p2_1': 'Tenure'}, inplace=True)

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
    _df_eff = _df_eff[['Weight', 'GrossNonRentIncome', 'TotalRentIncome', 'TotalHouseRentIncome',
                       'NumberOtherProperties', 'Tenure', 'renthog']]

    # Filter out the 1% with highest and the 1% with lowest GrossNonRentIncome
    _min_eff = cf.weighted_quantile(_df_eff['GrossNonRentIncome'], 0.01, sample_weight=_df_eff['Weight'])
    _max_eff = cf.weighted_quantile(_df_eff['GrossNonRentIncome'], 0.99, sample_weight=_df_eff['Weight'])
    _df_eff = _df_eff.loc[(_df_eff['GrossNonRentIncome'] >= _min_eff) & (_df_eff['GrossNonRentIncome'] <= _max_eff)]

    return _df_eff


def add_up_annual_house_rental_income_2016(_row):
    _total_house_rental_income = 0.0
    for i in range(1, 5):
        if _row['p2_35a_{}'.format(i)] in {1, 9}:
            _total_house_rental_income += _row['p2_43_{}'.format(i)]

    return 12.0 * _total_house_rental_income


# Parameters
saveFigures = False
includeResultsUK = False
fromYear = 2014
# fromYear = 2015
toYear = 2020
# toYear = 2017
nRuns = 20
my_color = ['tab:blue', 'tab:orange']
rootUK = r''
rootPSD = r''
rootEDW = r''
rootCdR = r''
rootEFF = r''
rootModelSP = r''
rootResults = r''

# General settings
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'sans',
    'font.sans-serif': ['DejaVu Sans']
})

# Read and clean European Data Warehouse data
dfEDW = cf.read_and_clean_edw_data(rootEDW, _from_year=fromYear, _to_year=toYear)

# Read and clean Colegio de Registradores data
dfCdRwP = cf.read_and_clean_cdr_data(rootCdR, _with_price=True, _from_year=fromYear, _to_year=toYear)

# Read Encuesta Financiera de las Familias data
dfEFF = read_and_clean_eff_data(rootEFF, _multiple_imputations=True, _exclude_house_sale_profits=True)
incomeQuintileEdges = cf.weighted_quantile(dfEFF['GrossNonRentIncome'], [0.2, 0.4, 0.6, 0.8],
                                           sample_weight=dfEFF['Weight'])
incomeQuintileEdges = [dfEFF['GrossNonRentIncome'].min()] + list(incomeQuintileEdges) \
                      + [dfEFF['GrossNonRentIncome'].max() + 0.001]

# Read and clean both UK and SP model results
resultsSP = read_and_clean_model_results(rootModelSP, nRuns)
monthlyGrossRentalIncomeSP = []
numberOfPropertiesSP = []
dfModelSP = pd.DataFrame()
for i in range(1, nRuns + 1):
    monthlyGrossRentalIncomeSP.extend(cf.read_micro_results(
        rootModelSP + '/MonthlyGrossRentalIncome-run{}.csv'.format(i), 1000, 3500))
    numberOfPropertiesSP.extend(cf.read_micro_results(rootModelSP + '/NHousesOwned-run{}.csv'.format(i), 1000, 3500))
    dfModelSP = pd.concat([dfModelSP, pd.read_csv(rootModelSP + '/Output-run{}.csv'.format(i), skipinitialspace=True,
                                                  delimiter=';', skiprows=range(1, 1000))])
dfModelSP.reset_index(drop=True, inplace=True)
dfModelSP['fHomeless'] = dfModelSP['nHomeless'] / dfModelSP['TotalPopulation']
dfModelSP['fRenting'] = dfModelSP['nRenting'] / dfModelSP['TotalPopulation']
dfModelSP['fOwnerOccupier'] = (dfModelSP['nOwnerOccupier'] + dfModelSP['nActiveBTL']) / dfModelSP['TotalPopulation']
dfModelSP = dfModelSP.mean()

if includeResultsUK:
    # Loan-to-value ratio: bar chart
    LTV = pd.read_excel(rootUK + '/Python charts.XLSX', sheet_name='LTV')
    LTV.columns = ['index', 'PSD', 'Simulation']
    LTV.plot.bar(x='index', y=['PSD', 'Simulation'], color=my_color, figsize=(3.8, 2.6))
    plt.xlabel('Loan-to-value ratio')
    plt.ylabel('Share of total mortgages (\%)')
    plt.legend(['PSD data', 'Model'], frameon=False, loc='upper left')
    labels = ['0\,-\,10', '10\,-\,20', '20\,-\,30', '30\,-\,40', '40\,-\,50', '50\,-\,60', '60\,-\,70', '70\,-\,80',
              '80\,-\,90']
    plt.gca().set_xticklabels(labels)
    plt.xticks(rotation=30)
    if saveFigures:
        plt.tight_layout()
        plt.savefig(rootResults + '/LTV.png', format='png', dpi=300, bbox_inches='tight')

if includeResultsUK:
    # Loan-to-income ratio: bar chart
    LTI = pd.read_excel(rootUK + '/Python charts.XLSX', sheet_name='LTI')
    LTI.columns = ['index', 'PSD', 'Simulation']
    LTI.plot.bar(x='index', y=['PSD', 'Simulation'], color=my_color, figsize=(3.8, 2.6))
    plt.xlabel('Loan-to-income ratio')
    plt.ylabel('Share of total mortgages (\%)')
    plt.legend(['PSD data', 'Model'], frameon=False, loc='upper right')
    labels = ['0\,-\,1', '1\,-\,2', '2\,-\,3', '3\,-\,4', '4\,-\,5', '5\,+']
    plt.gca().set_xticklabels(labels)
    plt.xticks(rotation=0)
    if saveFigures:
        plt.tight_layout()
        plt.savefig(rootResults + '/LTI.png', format='png', dpi=300, bbox_inches='tight')

if includeResultsUK:
    # House-price-to-income ratio: bar chart
    HPI = pd.read_excel(rootUK + '/Python charts.XLSX', sheet_name='HPI')
    HPI.columns = ['index', 'PSD', 'Simulation']
    HPI.plot.bar(x='index', y=['PSD', 'Simulation'], color=my_color, figsize=(3.8, 2.6), width=0.6)
    plt.xlabel('House-price-to-income ratio', labelpad=0)
    plt.ylabel('Share of total mortgages (\%)')
    plt.legend(['PSD data', 'Model'], frameon=False, loc='upper right')
    labels = ['0\,-\,1', '1\,-\,2', '2\,-\,3', '3\,-\,4', '4\,-\,5', '5\,-\,6', '6\,-\,7', '7\,-\,8', '8\,-\,9',
              '9\,-\,10', '10\,-\,20']
    plt.gca().set_xticklabels(labels)
    plt.xticks(rotation=45)
    plt.xlim(-1, 17)
    plt.ylim(0, 27)
    if saveFigures:
        plt.tight_layout()
        plt.savefig(rootResults + '/HPI.png', format='png', dpi=300, bbox_inches='tight')

# SPAIN - House-price-to-income ratio: bar chart
plt.figure(figsize=(3.9, 2.6))
bin_edges = list(range(0, 16)) + [20]
bin_width = 0.3
# Data
dfEDW['HPI'] = dfEDW['C_Ovaluation'] / dfEDW['household_income']
hist = np.histogram(dfEDW['HPI'], bins=bin_edges, density=False)[0]
hist = 100.0 * hist / sum(hist)
plt.bar([e - bin_width for e in bin_edges[:-1]], hist, width=bin_width, align='edge', label='Data Spain',
        color=my_color[0])
# Model
hist = np.histogram(resultsSP['HPI'], bins=bin_edges, density=False)[0]
hist = 100.0 * hist / sum(hist)
plt.bar(bin_edges[:-1], hist, width=bin_width, align='edge', label='Model Spain', color=my_color[1])
# Format
plt.xlabel('House-price-to-income ratio', labelpad=0)
plt.ylabel('Share of total mortgages (\%)')
plt.legend(frameon=False, loc='upper right')
labels = ['0\,-\,1', '1\,-\,2', '2\,-\,3', '3\,-\,4', '4\,-\,5', '5\,-\,6', '6\,-\,7', '7\,-\,8', '8\,-\,9',
          '9\,-\,10', '10\,-\,11', '11\,-\,12', '12\,-\,13', '13\,-\,14', '14\,-\,15', '15\,-\,20']
plt.gca().set_xticks(bin_edges[:-1])
plt.gca().set_xticklabels(labels)
plt.xticks(rotation=45)
# plt.ylim(0, 27)
if saveFigures:
    plt.tight_layout()
    plt.savefig(rootResults + '/HPI-SP.png', format='png', dpi=300, bbox_inches='tight')
    plt.savefig(rootResults + '/HPI-SP.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(rootResults + '/HPI-SP.eps', format='eps', dpi=300, bbox_inches='tight')

if includeResultsUK:
    # Mortgagor age: bar chart
    Age_borrower = pd.read_excel(rootUK + '/Python charts.XLSX', sheet_name='Age')
    Age_borrower.columns = ['index', 'PSD', 'Simulation']
    Age_borrower.plot.bar(x='index', y=['PSD', 'Simulation'], color=my_color, figsize=(3.8, 2.6))
    plt.xlabel('Owner-occupier Mortgagor Age')
    plt.ylabel('Share of total mortgages (\%)')
    plt.legend(['PSD data', 'Model'], frameon=False, loc='upper right')
    labels = ['16\,-\,24', '25\,-\,34', '35\,-\,44', '45\,-\,54', '55\,-\,64', '65\,+']
    plt.gca().set_xticklabels(labels)
    plt.xticks(rotation=0)
    if saveFigures:
        plt.tight_layout()
        plt.savefig(rootResults + '/Ageborrower.png', format='png', dpi=300, bbox_inches='tight')

# SPAIN - Mortgagor age: bar chart
plt.figure(figsize=(3.8, 2.6))
bin_edges = [16, 25, 35, 45, 55, 65, 100]
bin_width = 2.8
# Data
hist = np.histogram(dfEDW['age'], bins=bin_edges, density=False)[0]
hist = 100.0 * hist / sum(hist)
plt.bar([e - bin_width for e in bin_edges[:-1]], hist, width=bin_width, align='edge', label='Data Spain',
        color=my_color[0])
# Model
hist = np.histogram(resultsSP.loc[resultsSP['mortgagePrincipal'] > 0.0, 'buyerAge'], bins=bin_edges, density=False)[0]
hist = 100.0 * hist / sum(hist)
plt.bar(bin_edges[:-1], hist, width=bin_width, align='edge', label='Model Spain', color=my_color[1])
# Format
plt.xlabel('Owner-occupier Mortgagor Age')
plt.ylabel('Share of total mortgages (\%)')
plt.legend(frameon=False, loc='upper right')
labels = ['16\,-\,24', '25\,-\,34', '35\,-\,44', '45\,-\,54', '55\,-\,64', '65\,+']
plt.gca().set_xticks(bin_edges[:-1])
plt.gca().set_xticklabels(labels)
plt.xticks(rotation=0)
if saveFigures:
    plt.tight_layout()
    plt.savefig(rootResults + '/Ageborrower-SP.png', format='png', dpi=300, bbox_inches='tight')
    plt.savefig(rootResults + '/Ageborrower-SP.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(rootResults + '/Ageborrower-SP.eps', format='eps', dpi=300, bbox_inches='tight')

if includeResultsUK:
    # House Price Quintiles High LTV: bar chart
    HPQ = pd.read_excel(rootUK + '/Python charts.XLSX', sheet_name='HPQ',
                        usecols=['Unnamed: 0', 'High LTV - PSD', 'High LTV - Benchmark'])
    HPQ.columns = ['index', 'High LTV - PSD', 'High LTV - Benchmark']
    HPQ.plot.bar(x='index', y=['High LTV - PSD', 'High LTV - Benchmark'], color=my_color, figsize=(3.8, 2.6))
    plt.xlabel('House Price Quintile')
    plt.ylabel('Share of high-LTV mortgages (\%)', horizontalalignment='right', y=1.0)
    plt.legend(['PSD data', 'Model'], frameon=False, loc='upper right')
    plt.xticks(rotation=0)
    if saveFigures:
        plt.tight_layout()
        plt.savefig(rootResults + '/HPQ_HIGH_LTV_validation.png', format='png', dpi=300,
                    bbox_inches='tight')

# SPAIN - BTL properties: bar chart
plt.figure(figsize=(3.8, 2.6))
bin_edges = [0.99, 1.01, 4.01, 9.01, 24.1, 500]
bin_positions = [1, 2, 3, 4, 5]
bin_width = 0.3
# Data
hist = np.histogram(dfEFF.loc[dfEFF['NumberOtherProperties'] > 0.0, 'NumberOtherProperties'], bins=bin_edges,
                    density=False, weights=dfEFF.loc[dfEFF['NumberOtherProperties'] > 0.0, 'Weight'])[0]
hist = 100.0 * hist / sum(hist)
plt.bar([e - bin_width for e in bin_positions], hist, width=bin_width, align='edge', label='Data Spain',
        color=my_color[0])
# Model
hist = np.histogram([e - 1.0 for e in numberOfPropertiesSP if e > 1.0], bins=bin_edges, density=False)[0]
hist = 100.0 * hist / sum(hist)
plt.bar(bin_positions, hist, width=bin_width, align='edge', label='Model Spain', color=my_color[1])
# Format
plt.xlabel('Number of BTL properties per investor')
plt.ylabel('Share of total BTL investors (\%)', horizontalalignment='right', y=1.0)
plt.legend(frameon=False, loc='upper right')
labels = ['1 only', '2\,-\,4', '5\,-\,9', '10\,-\,24', '25\,+']
plt.gca().set_xticks(bin_positions)
plt.gca().set_xticklabels(labels)
plt.xticks(rotation=0)
if saveFigures:
    plt.tight_layout()
    plt.savefig(rootResults + '/BTLproperty_val-SP.png', format='png', dpi=300, bbox_inches='tight')
    plt.savefig(rootResults + '/BTLproperty_val-SP.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(rootResults + '/BTLproperty_val-SP.eps', format='eps', dpi=300, bbox_inches='tight')

if includeResultsUK:
    # Tenure shares: bar chart
    bin_centers = [1, 2, 3]
    bin_width = 0.3
    plt.figure(figsize=(3.8, 2.6))
    # TODO: Check source of this data
    plt.bar([e - bin_width for e in bin_centers], [0.65, 0.17, 1.0 - 0.65 - 0.17], width=bin_width, align='edge',
            label='WAS', color=my_color[0])
    plt.bar(bin_centers, [0.62, 0.21, 1.0 - 0.62 - 0.21], width=bin_width, align='edge', label='Model',
            color=my_color[1])
    plt.xlabel('Tenure')
    plt.ylabel('Share of households (\%)')
    plt.legend(frameon=False, loc='upper right')
    labels = ['Owner-occupying', 'Renting', 'Social housing']
    plt.gca().set_xticks(bin_centers)
    plt.gca().set_xticklabels(labels)
    plt.xticks(rotation=0)
    plt.ylim(0, 0.8)
    if saveFigures:
        plt.tight_layout()
        plt.savefig(rootResults + '/TenureShares-UK.png', format='png', dpi=300, bbox_inches='tight')

# SPAIN - Tenure shares: bar chart
bin_centers = [1, 2, 3]
bin_width = 0.3
plt.figure(figsize=(3.8, 2.6))
# Data
n_EFF_total = sum(dfEFF['Weight'])
n_EFF_renting = sum(dfEFF.loc[dfEFF['Tenure'] == 1.0, 'Weight'])
f_EFF_renting = n_EFF_renting / n_EFF_total
n_EFF_owning = sum(dfEFF.loc[dfEFF['Tenure'] == 2.0, 'Weight'])
f_EFF_owning = n_EFF_owning / n_EFF_total
f_EFF_social_housing = 1.0 - f_EFF_renting - f_EFF_owning
plt.bar([e - bin_width for e in bin_centers], [f_EFF_owning, f_EFF_renting, f_EFF_social_housing],
        width=bin_width, align='edge', label='Data Spain', color=my_color[0])
# Model
plt.bar(bin_centers, [dfModelSP['fOwnerOccupier'], dfModelSP['fRenting'], dfModelSP['fHomeless']], width=bin_width,
        align='edge', label='Model Spain', color=my_color[1])
# Format
plt.xlabel('Tenure')
plt.ylabel('Share of households (\%)')
plt.legend(frameon=False, loc='upper right')
labels = ['Owner-occupying', 'Renting', 'Social housing']
plt.gca().set_xticks(bin_centers)
plt.gca().set_xticklabels(labels)
plt.xticks(rotation=0)
plt.ylim(0, 0.8)
if saveFigures:
    plt.tight_layout()
    plt.savefig(rootResults + '/TenureShares-SP.png', format='png', dpi=300, bbox_inches='tight')
    plt.savefig(rootResults + '/TenureShares-SP.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(rootResults + '/TenureShares-SP.eps', format='eps', dpi=300, bbox_inches='tight')

# If required, show all figures
if not saveFigures:
    plt.show()
