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
                       'NumberOtherProperties', 'Tenure']]

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
fromYear = 2014
toYear = 2020
my_color = ['#0A4A69', '#802667']
rootUK = r''
rootEDW = r''
rootEFF = r''
rootResults = r''

# General settings
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'sans',
    'font.sans-serif': ['DejaVu Sans']
})

# Read and clean European Data Warehouse data
dfEDW = cf.read_and_clean_edw_data(rootEDW, _from_year=fromYear, _to_year=toYear)

# Read Encuesta Financiera de las Familias data
dfEFF = read_and_clean_eff_data(rootEFF, _multiple_imputations=True, _exclude_house_sale_profits=True)

# House-price-to-income ratio
plt.figure(figsize=(5.8, 3.8))
bin_edges = list(range(0, 16)) + [20]
bin_width = 0.3
# UK
HPI = pd.read_excel(rootUK + '/Python charts.XLSX', sheet_name='HPI')
HPI.columns = ['index', 'PSD', 'Simulation']
plt.bar([e - bin_width for e in bin_edges[:-6]], HPI['PSD'], width=bin_width, align='edge', label='PSD data (UK)',
        color=my_color[0])
# Spain
dfEDW['HPI'] = dfEDW['C_Ovaluation'] / dfEDW['household_income']
hist = np.histogram(dfEDW['HPI'], bins=bin_edges, density=False)[0]
hist = 100.0 * hist / sum(hist)
plt.bar(bin_edges[:-1], hist, width=bin_width, align='edge', label='EDW data (Spain)', color=my_color[1])
# Format
plt.xlabel('House-price-to-income ratio', labelpad=0)
plt.ylabel('Share of total mortgages (\%)')
plt.legend(frameon=False, loc='upper right')
labels = ['0\,-\,1', '1\,-\,2', '2\,-\,3', '3\,-\,4', '4\,-\,5', '5\,-\,6', '6\,-\,7', '7\,-\,8', '8\,-\,9',
          '9\,-\,10', '10\,-\,11', '11\,-\,12', '12\,-\,13', '13\,-\,14', '14\,-\,15', '15\,-\,20']
plt.gca().set_xticks(bin_edges[:-1])
plt.gca().set_xticklabels(labels)
plt.xticks(rotation=30)
if saveFigures:
    plt.tight_layout()
    plt.savefig(rootResults + '/CombinedData-HPI.png', format='png', dpi=300, bbox_inches='tight')

# Mortgagor age
plt.figure(figsize=(5.8, 3.8))
bin_edges = [16, 25, 35, 45, 55, 65, 100]
bin_width = 2.8
# UK
Age_borrower = pd.read_excel(rootUK + '/Python charts.XLSX', sheet_name='Age')
Age_borrower.columns = ['index', 'PSD', 'Simulation']
plt.bar([e - bin_width for e in bin_edges[:-1]], Age_borrower['PSD'], width=bin_width, align='edge',
        label='PSD data (UK)', color=my_color[0])
# Spain
hist = np.histogram(dfEDW['age'], bins=bin_edges, density=False)[0]
hist = 100.0 * hist / sum(hist)
plt.bar(bin_edges[:-1], hist, width=bin_width, align='edge', label='EDW data (Spain)', color=my_color[1])
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
    plt.savefig(rootResults + '/CombinedData-BorrowerAge.png', format='png', dpi=300, bbox_inches='tight')

# Tenure shares
plt.figure(figsize=(5.8, 3.8))
bin_centers = [1, 2, 3]
bin_width = 0.3
# UK TODO: Check source of this data (apparently, not WAS)
plt.bar([e - bin_width for e in bin_centers], [65, 17, 100 - 65 - 17], width=bin_width, align='edge',
        label='WAS data (UK)', color=my_color[0])
# Spain
n_EFF_total = sum(dfEFF['Weight'])
n_EFF_renting = sum(dfEFF.loc[dfEFF['Tenure'] == 1.0, 'Weight'])
f_EFF_renting = 100.0 * n_EFF_renting / n_EFF_total
n_EFF_owning = sum(dfEFF.loc[dfEFF['Tenure'] == 2.0, 'Weight'])
f_EFF_owning = 100.0 * n_EFF_owning / n_EFF_total
f_EFF_social_housing = 100.0 - f_EFF_renting - f_EFF_owning
plt.bar(bin_centers, [f_EFF_owning, f_EFF_renting, f_EFF_social_housing], width=bin_width, align='edge',
        label='EFF data (Spain)', color=my_color[1])
# Format
plt.xlabel('Tenure')
plt.ylabel('Share of households (\%)')
plt.legend(frameon=False, loc='upper right')
labels = ['Owner-occupying', 'Renting', 'Social housing']
plt.gca().set_xticks(bin_centers)
plt.gca().set_xticklabels(labels)
plt.xticks(rotation=0)
if saveFigures:
    plt.tight_layout()
    plt.savefig(rootResults + '/CombinedData-TenureShares.png', format='png', dpi=300, bbox_inches='tight')

# If required, show all figures
if not saveFigures:
    plt.show()
