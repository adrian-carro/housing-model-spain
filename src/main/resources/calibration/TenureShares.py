# -*- coding: utf-8 -*-
"""
Class to study the share of households by tenure based on Encuesta Financiera de las Familias and Encuesta Continua de
Hogares data.

@author: Adrian Carro
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Utilities.common_functions as cf


def read_and_clean_eff_data(_root_eff, _multiple_imputations):
    # EFF variables of interest
    # facine3       Household weight
    # p2_1          Regimen de tenencia de su vivienda principal (1 en alquiler, 2 en propiedad, 3 cesion gratuita)
    # Read data
    if _multiple_imputations:
        _df_eff = pd.DataFrame()
        for i in range(1, 6):
            temp_df = pd.read_csv(_root_eff + r'/eff_2017_imp{0}_csv/otras_secciones_2017_imp{0}.csv'.format(i),
                                  sep=';', usecols=['facine3', 'p2_1', 'p1_2d_1'])
            _df_eff = pd.concat([_df_eff, temp_df])
        _df_eff['facine3'] = _df_eff['facine3'] / 5.0
    else:
        _df_eff = pd.read_csv(_root_eff + r'/eff_2017_imp1_csv/otras_secciones_2017_imp1.csv', sep=';',
                              usecols=['facine3', 'p2_1', 'p1_2d_1'])

    # Rename columns to be used
    _df_eff.rename(columns={'facine3': 'Weight', 'p2_1': 'Tenure', 'p1_2d_1': 'Age'}, inplace=True)

    # Filter down to keep only columns of interest
    _df_eff = _df_eff[['Weight', 'Tenure', 'Age']]

    return _df_eff


def read_and_clean_ech_data(_root_ine):
    # Field names for renaming
    names = ['Year', 'Total', 'Owner-occupying (without mortgage)', 'Owner-occupying (with mortgage)', 'Renting',
             'Social housing']
    # Read data
    _df_ech = pd.read_excel(_root_ine + r'/ECH (Encuesta Contínua de Hogares) - Número de hogares según el tipo de '
                                        r'hogar y el régimen de tenencia de la vivienda.xlsx', header=None,
                            skiprows=list(range(0, 8)) + list(range(16, 200)), usecols=range(6), names=names,
                            index_col='Year')
    # Convert to natural units
    _df_ech *= 1000.0

    return _df_ech


def read_and_clean_ecv_data(_root_ine):
    # Read data and reorganise it
    _df_ecv = pd.read_excel(_root_ine + r'/ECV (Encuesta de Condiciones de Vida) - Hogares por régimen de tenencia '
                                        r'de la vivienda y tipo de hogar.xlsx', header=[6, 7], skipfooter=13)
    _df_ecv.drop((' ', ' '), axis='columns', inplace=True)
    _df_ecv = _df_ecv.stack()
    _df_ecv = _df_ecv.droplevel(0)
    _df_ecv.index = _df_ecv.index.map(int)
    # Rename columns
    _df_ecv.rename(columns={'Alquiler a precio de mercado': 'Private renting',
                            'Alquiler inferior al precio de mercado': 'Social renting',
                            'Cesión': 'Social housing',
                            'Propiedad': 'Owner-occupying'}, inplace=True)
    # Convert to fractions instead of percentages
    _df_ecv /= 100.0

    return _df_ecv


def read_and_clean_model_results(_root_model, _n_runs):
    # Read model results
    _df_model = pd.DataFrame()
    for _i in range(1, _n_runs + 1):
        _df_temp = pd.read_csv(_root_model + '/Output-run{}.csv'.format(_i), skipinitialspace=True, delimiter=';',
                               usecols=['Model time', 'nHomeless', 'nRenting', 'nOwnerOccupier', 'nActiveBTL',
                                        'TotalPopulation'])
        _df_model = pd.concat([_df_model, _df_temp])
    # Remove burnout period
    _df_model = _df_model.loc[_df_model['Model time'] > 1000]
    # Compute fraction of households per tenure
    _df_model['fHomeless'] = _df_model['nHomeless'] / _df_model['TotalPopulation']
    _df_model['fRenting'] = _df_model['nRenting'] / _df_model['TotalPopulation']
    _df_model['fOwnerOccupier'] = (_df_model['nOwnerOccupier'] + _df_model['nActiveBTL']) / _df_model['TotalPopulation']
    # Restrict to columns of interest
    _df_model = _df_model[['fHomeless', 'fRenting', 'fOwnerOccupier']]
    _df_model = _df_model.mean()

    return _df_model


def plot_ownership_rate_by_age(_root_model, _n_runs, _df_eff):
    number_of_properties = []
    age = []
    _n_runs = 10
    for i in range(1, _n_runs + 1):
        number_of_properties.extend(cf.read_micro_results(_root_model + '/NHousesOwned-run{}.csv'.format(i),
                                                          1000, 3500))
        age.extend(cf.read_micro_results(_root_model + '/Age-run{}.csv'.format(i), 1000, 3500))
    df_age_n_properties = pd.DataFrame({'Age': age, 'NProperties': number_of_properties})

    plt.figure()
    bar_centers = [1, 2, 3, 4, 5, 6]
    bar_width = 0.3
    bin_edges = [16, 25, 35, 45, 55, 65, 100]
    households_by_age = np.histogram(_df_eff['Age'], bins=bin_edges, weights=_df_eff['Weight'])[0]
    owners_by_age = np.histogram(_df_eff.loc[_df_eff['Tenure'] == 2, 'Age'], bins=bin_edges,
                                 weights=_df_eff.loc[_df_eff['Tenure'] == 2, 'Weight'])[0]
    plt.bar([e - bar_width for e in bar_centers], owners_by_age / households_by_age,
            width=bar_width, align='edge', label='EFF')

    households_by_age = np.histogram(df_age_n_properties['Age'], bins=bin_edges)[0]
    owners_by_age = np.histogram(df_age_n_properties.loc[df_age_n_properties['NProperties'] > 0, 'Age'],
                                 bins=bin_edges)[0]
    plt.bar([e for e in bar_centers], owners_by_age / households_by_age, width=bar_width, align='edge', label='Model')

    labels = ['16\,-\,24', '25\,-\,34', '35\,-\,44', '45\,-\,54', '55\,-\,64', '65\,+']
    plt.gca().set_xticks(bar_centers)
    plt.gca().set_xticklabels(labels)
    plt.legend()
    plt.show()


# Control variables
writeResults = False
savePlots = False
nRuns = 10
rootEFF = r''
rootINE = r''
rootModel = r''
rootResults = r''

# General printing settings
np.set_printoptions(suppress=True)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 1000)

# Control parameters
year = 2016

# Read Encuesta Financiera de las Familias data
dfEFF = read_and_clean_eff_data(rootEFF, _multiple_imputations=True)

# Read Encuesta Continua de Hogares
dfECH = read_and_clean_ech_data(rootINE)

# Read Encuesta de Condiciones de Vida
dfECV = read_and_clean_ecv_data(rootINE)

# Read model results
dfModel = read_and_clean_model_results(rootModel, nRuns)

# Count share of households by tenure with EFF data
n_EFF_total = sum(dfEFF['Weight'])
n_EFF_renting = sum(dfEFF.loc[dfEFF['Tenure'] == 1.0, 'Weight'])
f_EFF_renting = n_EFF_renting / n_EFF_total
n_EFF_owning = sum(dfEFF.loc[dfEFF['Tenure'] == 2.0, 'Weight'])
f_EFF_owning = n_EFF_owning / n_EFF_total
f_EFF_social_housing = 1.0 - f_EFF_renting - f_EFF_owning
# Count share of households by tenure with ECH data
n_ECH_total = dfECH.loc[year]['Total']
n_ECH_renting = dfECH.loc[year]['Renting']
f_ECH_renting = n_ECH_renting / n_ECH_total
n_ECH_owning = dfECH.loc[year]['Owner-occupying (without mortgage)']\
               + dfECH.loc[year]['Owner-occupying (with mortgage)']
f_ECH_owning = n_ECH_owning / n_ECH_total
f_ECH_social_housing = 1.0 - f_ECH_renting - f_ECH_owning
# Count share of households by tenure with ECV data
f_ECV_renting = dfECV.loc[year]['Private renting']
f_ECV_owning = dfECV.loc[year]['Owner-occupying']
f_ECV_social_housing = 1.0 - f_ECV_renting - f_ECV_owning


# If required, write results to file, otherwise, print to screen
if writeResults:
    with open(rootResults + '/TenureShares.txt', 'w') as f:
        f.write('##############################################################################\n')
        f.write('EFF DATA\n')
        f.write('##############################################################################\n')
        f.write('Share of households privately renting = {}\n'.format(f_EFF_renting))
        f.write('Share of households owner-occupying = {}\n'.format(f_EFF_owning))
        f.write('Share of households in social housing = {}\n'.format(f_EFF_social_housing))
        f.write('\n##############################################################################\n')
        f.write('ECH DATA\n')
        f.write('##############################################################################\n')
        f.write('Share of households privately renting = {}\n'.format(f_ECH_renting))
        f.write('Share of households owner-occupying = {}\n'.format(f_ECH_owning))
        f.write('Share of households in social housing = {}\n'.format(f_ECH_social_housing))
        f.write('\n##############################################################################\n')
        f.write('ECV DATA\n')
        f.write('##############################################################################\n')
        f.write('Share of households privately renting = {}\n'.format(f_ECV_renting))
        f.write('Share of households owner-occupying = {}\n'.format(f_ECV_owning))
        f.write('Share of households in social housing = {}'.format(f_ECV_social_housing))
else:
    print('####################################### EFF #######################################')
    print('Share of households privately renting = {}'.format(f_EFF_renting))
    print('Share of households owner-occupying = {}'.format(f_EFF_owning))
    print('Share of households in social housing = {}'.format(f_EFF_social_housing))
    print('####################################### ECH #######################################')
    print('Share of households privately renting = {}'.format(f_ECH_renting))
    print('Share of households owner-occupying = {}'.format(f_ECH_owning))
    print('Share of households in social housing = {}'.format(f_ECH_social_housing))
    print('####################################### ECV #######################################')
    print('Share of households privately renting = {}'.format(f_ECV_renting))
    print('Share of households owner-occupying = {}'.format(f_ECV_owning))
    print('Share of households in social housing = {}'.format(f_ECV_social_housing))
    print('####################################### Model #######################################')
    print('Share of households privately renting = {}'.format(dfModel['fRenting']))
    print('Share of households owner-occupying = {}'.format(dfModel['fOwnerOccupier']))
    print('Share of households in social housing = {}'.format(dfModel['fHomeless']))
    # If required, plot ownership rate by age
    # plot_ownership_rate_by_age(rootModel, nRuns, dfEFF)
    # Plot, including model results for comparison
    plt.figure()
    bin_centers = [0, 1, 2]
    bin_width = 0.2
    plt.bar([e - 2.0 * bin_width for e in bin_centers], [f_EFF_owning, f_EFF_renting, f_EFF_social_housing],
            width=bin_width, align='edge', label='EFF')
    plt.bar([e - bin_width for e in bin_centers], [f_ECH_owning, f_ECH_renting, f_ECH_social_housing],
            width=bin_width, align='edge', label='ECH')
    plt.bar([e for e in bin_centers], [f_ECV_owning, f_ECV_renting, f_ECV_social_housing],
            width=bin_width, align='edge', label='ECV')
    plt.bar([e + bin_width for e in bin_centers],
            [dfModel['fOwnerOccupier'], dfModel['fRenting'], dfModel['fHomeless']],
            width=bin_width, align='edge', label='Model')
    plt.gca().set_xticks(bin_centers)
    plt.gca().set_xticklabels(['Owner-occupying', 'Renting', 'Social housing'])
    plt.legend()
    if savePlots:
        plt.tight_layout()
        plt.savefig(rootResults + '/TenureShares.png', format='png', dpi=300, bbox_inches='tight')
    else:
        plt.show()
