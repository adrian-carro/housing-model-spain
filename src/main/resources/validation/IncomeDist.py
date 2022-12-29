# -*- coding: utf-8 -*-
"""
Class to study households' income distribution for validation purposes.

@author: Adrian Carro
"""

from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import DescrStatsW
import statsmodels.api as sm


def read_results(file_name, _start_time, _end_time):
    """Read micro-data from file_name, structured on a separate line per year. In particular, read from start_year until
    end_year, both inclusive"""
    # Read list of float values, one per household
    data_float = []
    with open(file_name, 'r') as _f:
        for line in _f:
            if _start_time <= int(line.split(';')[0]) <= _end_time:
                for column in line.split(';')[1:]:
                    data_float.append(float(column))
    return data_float


def read_and_clean_was_data(_root_was, _population):
    _df_was = pd.read_csv(rootWAS + r'/was_wave_3_hhold_eul_final.dta',
                          usecols={'w3xswgt', 'DVTotGIRw3', 'DVGrsRentAmtAnnualw3_aggr', 'Ten1W3'})
    # List of household variables currently used
    # DVTotGIRw3                  Household Gross Annual (regular) income
    # DVGrsRentAmtAnnualw3_aggr   Household Gross Annual income from rent

    # List of other household variables of possible interest
    # DVTotNIRw3                  Household Net Annual (regular) income
    # DVNetRentAmtAnnualw3_aggr   Household Net Annual income from rent

    # Rename columns to be used and add all necessary extra columns
    _df_was.rename(columns={'w3xswgt': 'Weight'}, inplace=True)
    _df_was.rename(columns={'Ten1W3': 'Tenure'}, inplace=True)
    _df_was.rename(columns={'DVTotGIRw3': 'GrossTotalIncome'}, inplace=True)
    _df_was.rename(columns={'DVGrsRentAmtAnnualw3_aggr': 'GrossRentalIncome'}, inplace=True)
    _df_was['GrossNonRentIncome'] = _df_was['GrossTotalIncome'] - _df_was['GrossRentalIncome']

    # Filter down to keep only columns of interest
    _df_was = _df_was[['GrossTotalIncome', 'GrossRentalIncome', 'GrossNonRentIncome', 'Weight', 'Tenure']]

    # Restrict to positive values of income
    _df_was = _df_was.loc[_df_was['GrossNonRentIncome'] > 0.0]

    # Filter out the 1% with highest and 1% with lowest GrossNonRentIncome
    _one_per_cent = int(round(len(_df_was.index) / 100))
    _chunk_ord_by_gross = _df_was.sort_values('GrossNonRentIncome')
    _max_gross_income = _chunk_ord_by_gross.iloc[-_one_per_cent]['GrossNonRentIncome']
    _min_gross_income = _chunk_ord_by_gross.iloc[_one_per_cent]['GrossNonRentIncome']
    _df_was = _df_was[_df_was['GrossNonRentIncome'] <= _max_gross_income]
    _df_was = _df_was[_df_was['GrossNonRentIncome'] >= _min_gross_income]

    # Keep only selected population
    if _population == 'all':
        pass
    elif _population == 'non-owners':
        _df_was = _df_was[(_df_was['Tenure'] == 'Rent it') | (_df_was['Tenure'] == 'Rent-free')]
    elif _population == 'owners':
        _df_was = _df_was[(_df_was['Tenure'] == 'Own it outright') | (_df_was['Tenure'] == 'Buying with mortgage')]
    else:
        print('Population not recognised!')
        exit()

    return _df_was


def read_and_clean_edw_data(_root_edw, _income_variable):
    # _df_edw = pd.read_csv(_root_edw + '/EDWdata_JAN22.csv', usecols=['income', 'property_type'],
    #                       dtype={'income': float, 'property_type': str})
    # _df_edw = pd.read_csv(_root_edw + '/EDWdata_JAN22.csv')
    _df_edw = pd.read_csv(_root_edw + '/EDWdata_updateMAR22_v2.csv', dtype={'M': str, 'Q': str})
    _df_edw['age'] = _df_edw['Y'].subtract(_df_edw['B_date_birth'])
    _df_edw['household_income'] = _df_edw[['B_inc', 'B_inc2']].sum(axis=1, skipna=False)

    # Remove duplicates
    _df_edw = _df_edw[~_df_edw.duplicated(keep=False)]

    # # Restrict to owner-occupied property
    # _df_edw = _df_edw.loc[_df_edw['property_type'] == '1']

    # Restrict both income and the selected price variable to positive values (thus also discarding NaNs)
    _df_edw = _df_edw.loc[_df_edw[_income_variable] > 0.0]
    # _df_edw = _df_edw.loc[_df_edw[priceVariable] > 0.0]  # Used at DesiredPurchasePrice.py

    # Remove 1% lowest and 1% highest incomes (very low income transactions are clearly the result of other undeclared
    # sources of income, very high incomes are related to very high prices, which are not captured in the database)
    q1, q99 = np.quantile(_df_edw[_income_variable], [0.01, 0.99])
    _df_edw = _df_edw.loc[(_df_edw[_income_variable] > q1) & (_df_edw[_income_variable] < q99)]
    # Alternative way of restricting income values
    # _df_edw = _df_edw.loc[(_df_edw['income'] > 10000.0) & (_df_edw['income'] < 100000.0)]

    # If 'price' selected as price variable, then remove values below 25k
    # if priceVariable == 'price':
    #     _df_edw = _df_edw.loc[_df_edw[priceVariable] > 25000]  # Used at DesiredPurchasePrice.py

    # Restrict to origination dates from a certain year on
    _df_edw = _df_edw[(_df_edw['Y'] >= 2014)]

    # Manually remove the most clear vertical lines
    if _income_variable == 'B_inc':
        _df_edw = _df_edw[(_df_edw[_income_variable] != 130000) & (_df_edw[_income_variable] != 43500)
                          & (_df_edw[_income_variable] != 21000) & (_df_edw[_income_variable] != 7500)]

    # Remove all integer income values, as they tend to be arranged into strange vertical lines
    # _df_edw = _df_edw[_df_edw['income'].apply(lambda e: e != int(e))]

    # Filter down to keep only columns of interest
    _df_edw = _df_edw[['B_inc', 'household_income']]

    return _df_edw


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
        lambda row: add_up_annual_non_renal_income_2016(row, _exclude_house_sale_profits, _individual=False), axis=1)

    _df_eff['GrossNonRentIncomeIndividual'] = _df_eff.apply(
        lambda row: add_up_annual_non_renal_income_2016(row, _exclude_house_sale_profits, _individual=True), axis=1)

    # Filter down to keep only columns of interest
    # _df_eff = _df_eff[['GrossNonRentIncome', 'Weight']]
    _df_eff = _df_eff[['GrossNonRentIncome', 'GrossNonRentIncomeIndividual', 'Weight']]

    # Filter out the 1% with highest and the 1% with lowest GrossNonRentIncome
    _min_eff = weighted_quantile(_df_eff['GrossNonRentIncome'], 0.01, sample_weight=_df_eff['Weight'])
    _max_eff = weighted_quantile(_df_eff['GrossNonRentIncome'], 0.99, sample_weight=_df_eff['Weight'])
    _df_eff = _df_eff.loc[(_df_eff['GrossNonRentIncome'] >= _min_eff) & (_df_eff['GrossNonRentIncome'] <= _max_eff)]

    return _df_eff


def read_and_clean_model_results(_root_model, _population, _start_time, _end_time):
    _model_results = read_results(_root_model + r'/MonthlyGrossEmploymentIncome-run1.csv', _start_time, _end_time)

    # Keep only selected population
    if _population == 'all':
        pass
    elif _population == 'non-owners':
        n_properties = read_results(_root_model + r'/test/NHousesOwned-run1.csv', _start_time, _end_time)
        _model_results = [x for x, y in zip(_model_results, n_properties) if y == 0]
    elif _population == 'owners':
        n_properties = read_results(_root_model + r'/test/NHousesOwned-run1.csv', _start_time, _end_time)
        _model_results = [x for x, y in zip(_model_results, n_properties) if y > 0]

    return _model_results


def add_up_annual_non_renal_income_2016(_row, _exclude_house_sale_profits, _individual):
    _main_person_annual_income_fields = [
        # INGRESOS POR CUENTA AJENA (BRUTO ANUAL)
        'p6_64_1',      # 'p.6.64. Miem. 1. Importe recibido como empleado por cuenta ajena durante 2016'
        # INGRESOS POR CUENTA AJENA EN ESPECIE (BRUTO ANUAL)
        'p6_66_1',      # 'p.6.66. Miem. 1. Importe recibido en especie durante 2016'
        # INGRESOS COMO DESEMPLEADO (BRUTO ANUAL)
        'p6_68_1',      # 'p.6.68. Miem. 1. Importe recibido por prestaciones por desempleo durante 2016'
        # INGRESOS POR INDEMNIZACION POR DESPIDO (BRUTO ANUAL)
        'p6_70_1',      # 'p.6.70. Miem. 1. Importe recibido por indemnizaciones por despido durante 2016'
        # INGRESOS POR CUENTA PROPIA (BRUTO ANUAL)
        'p6_72_1',      # 'p.6.72. Miem. 1. Importe recibido por su trabajo por cuenta propia en 2016'
        # INGRESOS POR PENSION VIUDEDAD/ORFANDAD (BRUTO ANUAL)
        'p6_74b_1',     # 'p.6.74b. Miem. 1. Importe recibido por pensiones de viudedad/orfandad en 2016'
        # INGRESOS POR PENSION JUBILACION/INCAPACIDAD(BRUTO ANUAL)
        'p6_74_1',      # 'p.6.74. Miem. 1. Importe recibido por pensiones jubilacion/incapacidad en 2016'
        # INGRESOS POR PENSION PRIVADA (BRUTO ANUAL)
        'p6_74c_1',     # 'p.6.74c. Miem. 1. Importe recibido por planes de pensiones privados en 2016'
    ]

    _other_members_annual_income_fields = [
        # INGRESOS POR CUENTA AJENA (BRUTO ANUAL)
        'p6_64_2',      # 'p.6.64. Miem. 2. Importe recibido como empleado por cuenta ajena durante 2016'
        'p6_64_3',      # 'p.6.64. Miem. 3. Importe recibido como empleado por cuenta ajena durante 2016'
        'p6_64_4',      # 'p.6.64. Miem. 4. Importe recibido como empleado por cuenta ajena durante 2016'
        'p6_64_5',      # 'p.6.64. Miem. 5. Importe recibido como empleado por cuenta ajena durante 2016'
        'p6_64_6',      # 'p.6.64. Miem. 6. Importe recibido como empleado por cuenta ajena durante 2016'
        'p6_64_7',      # 'p.6.64. Miem. 7. Importe recibido como empleado por cuenta ajena durante 2016'
        'p6_64_8',      # 'p.6.64. Miem. 8. Importe recibido como empleado por cuenta ajena durante 2016'
        'p6_64_9',      # 'p.6.64. Miem. 9. Importe recibido como empleado por cuenta ajena durante 2016'
        # INGRESOS POR CUENTA AJENA EN ESPECIE (BRUTO ANUAL)
        'p6_66_2',      # 'p.6.66. Miem. 2. Importe recibido en especie durante 2016'
        'p6_66_3',      # 'p.6.66. Miem. 3. Importe recibido en especie durante 2016'
        'p6_66_4',      # 'p.6.66. Miem. 4. Importe recibido en especie durante 2016'
        'p6_66_5',      # 'p.6.66. Miem. 5. Importe recibido en especie durante 2016'
        'p6_66_6',      # 'p.6.66. Miem. 6. Importe recibido en especie durante 2016'
        'p6_66_7',      # 'p.6.66. Miem. 7. Importe recibido en especie durante 2016'
        'p6_66_8',      # 'p.6.66. Miem. 8. Importe recibido en especie durante 2016'
        'p6_66_9',      # 'p.6.66. Miem. 9. Importe recibido en especie durante 2016'
        # INGRESOS COMO DESEMPLEADO (BRUTO ANUAL)
        'p6_68_2',      # 'p.6.68. Miem. 2. Importe recibido por prestaciones por desempleo durante 2016'
        'p6_68_3',      # 'p.6.68. Miem. 3. Importe recibido por prestaciones por desempleo durante 2016'
        'p6_68_4',      # 'p.6.68. Miem. 4. Importe recibido por prestaciones por desempleo durante 2016'
        'p6_68_5',      # 'p.6.68. Miem. 5. Importe recibido por prestaciones por desempleo durante 2016'
        'p6_68_6',      # 'p.6.68. Miem. 6. Importe recibido por prestaciones por desempleo durante 2016'
        'p6_68_7',      # 'p.6.68. Miem. 7. Importe recibido por prestaciones por desempleo durante 2016'
        'p6_68_8',      # 'p.6.68. Miem. 8. Importe recibido por prestaciones por desempleo durante 2016'
        'p6_68_9',      # 'p.6.68. Miem. 9. Importe recibido por prestaciones por desempleo durante 2016'
        # INGRESOS POR INDEMNIZACION POR DESPIDO (BRUTO ANUAL)
        'p6_70_2',      # 'p.6.70. Miem. 2. Importe recibido por indemnizaciones por despido durante 2016'
        'p6_70_3',      # 'p.6.70. Miem. 3. Importe recibido por indemnizaciones por despido durante 2016'
        'p6_70_4',      # 'p.6.70. Miem. 4. Importe recibido por indemnizaciones por despido durante 2016'
        'p6_70_5',      # 'p.6.70. Miem. 5. Importe recibido por indemnizaciones por despido durante 2016'
        'p6_70_6',      # 'p.6.70. Miem. 6. Importe recibido por indemnizaciones por despido durante 2016'
        'p6_70_7',      # 'p.6.70. Miem. 7. Importe recibido por indemnizaciones por despido durante 2016'
        'p6_70_8',      # 'p.6.70. Miem. 8. Importe recibido por indemnizaciones por despido durante 2016'
        'p6_70_9',      # 'p.6.70. Miem. 9. Importe recibido por indemnizaciones por despido durante 2016'
        # INGRESOS POR CUENTA PROPIA (BRUTO ANUAL)
        'p6_72_2',      # 'p.6.72. Miem. 2. Importe recibido por su trabajo por cuenta propia en 2016'
        'p6_72_3',      # 'p.6.72. Miem. 3. Importe recibido por su trabajo por cuenta propia en 2016'
        'p6_72_4',      # 'p.6.72. Miem. 4. Importe recibido por su trabajo por cuenta propia en 2016'
        'p6_72_5',      # 'p.6.72. Miem. 5. Importe recibido por su trabajo por cuenta propia en 2016'
        'p6_72_6',      # 'p.6.72. Miem. 6. Importe recibido por su trabajo por cuenta propia en 2016'
        'p6_72_7',      # 'p.6.72. Miem. 7. Importe recibido por su trabajo por cuenta propia en 2016'
        'p6_72_8',      # 'p.6.72. Miem. 8. Importe recibido por su trabajo por cuenta propia en 2016'
        'p6_72_9',      # 'p.6.72. Miem. 9. Importe recibido por su trabajo por cuenta propia en 2016'
        # INGRESOS POR PENSION VIUDEDAD/ORFANDAD (BRUTO ANUAL)
        'p6_74b_2',     # 'p.6.74b. Miem. 2. Importe recibido por pensiones de viudedad/orfandad en 2016'
        'p6_74b_3',     # 'p.6.74b. Miem. 3. Importe recibido por pensiones de viudedad/orfandad en 2016'
        'p6_74b_4',     # 'p.6.74b. Miem. 4. Importe recibido por pensiones de viudedad/orfandad en 2016'
        'p6_74b_5',     # 'p.6.74b. Miem. 5. Importe recibido por pensiones de viudedad/orfandad en 2016'
        'p6_74b_6',     # 'p.6.74b. Miem. 6. Importe recibido por pensiones de viudedad/orfandad en 2016'
        'p6_74b_7',     # 'p.6.74b. Miem. 7. Importe recibido por pensiones de viudedad/orfandad en 2016'
        'p6_74b_8',     # 'p.6.74b. Miem. 8. Importe recibido por pensiones de viudedad/orfandad en 2016'
        'p6_74b_9',     # 'p.6.74b. Miem. 9. Importe recibido por pensiones de viudedad/orfandad en 2016'
        # INGRESOS POR PENSION JUBILACION/INCAPACIDAD(BRUTO ANUAL)
        'p6_74_2',      # 'p.6.74. Miem. 2. Importe recibido por pensiones jubilacion/incapacidad en 2016'
        'p6_74_3',      # 'p.6.74. Miem. 3. Importe recibido por pensiones jubilacion/incapacidad en 2016'
        'p6_74_4',      # 'p.6.74. Miem. 4. Importe recibido por pensiones jubilacion/incapacidad en 2016'
        'p6_74_5',      # 'p.6.74. Miem. 5. Importe recibido por pensiones jubilacion/incapacidad en 2016'
        'p6_74_6',      # 'p.6.74. Miem. 6. Importe recibido por pensiones jubilacion/incapacidad en 2016'
        'p6_74_7',      # 'p.6.74. Miem. 7. Importe recibido por pensiones jubilacion/incapacidad en 2016'
        'p6_74_8',      # 'p.6.74. Miem. 8. Importe recibido por pensiones jubilacion/incapacidad en 2016'
        'p6_74_9',      # 'p.6.74. Miem. 9. Importe recibido por pensiones jubilacion/incapacidad en 2016'
        # INGRESOS POR PENSION PRIVADA (BRUTO ANUAL)
        'p6_74c_2',     # 'p.6.74c. Miem. 2. Importe recibido por planes de pensiones privados en 2016'
        'p6_74c_3',     # 'p.6.74c. Miem. 3. Importe recibido por planes de pensiones privados en 2016'
        'p6_74c_4',     # 'p.6.74c. Miem. 4. Importe recibido por planes de pensiones privados en 2016'
        'p6_74c_5',     # 'p.6.74c. Miem. 5. Importe recibido por planes de pensiones privados en 2016'
        'p6_74c_6',     # 'p.6.74c. Miem. 6. Importe recibido por planes de pensiones privados en 2016'
        'p6_74c_7',     # 'p.6.74c. Miem. 7. Importe recibido por planes de pensiones privados en 2016'
        'p6_74c_8',     # 'p.6.74c. Miem. 8. Importe recibido por planes de pensiones privados en 2016'
        'p6_74c_9',     # 'p.6.74c. Miem. 9. Importe recibido por planes de pensiones privados en 2016'
    ]

    _household_annual_income_fields = [
        # HOGAR - OTROS INGRESOS NO DECLARADOS POR PERSONA (BRUTO ANUAL)
        'p6_75b',  # 'p.6.75b. Importe recibido por pensiones de viudedad/orfandad durante 2016'
        'p6_75d1',  # 'p.6.75d1. Importe recibido de ex pareja con la que no conviven durante 2016'
        'p6_75d2',  # 'p.6.75d2. Importe recibido de familiares fuera del hogar durante 2016'
        'p6_75d3',  # 'p.6.75d3. Importe recibido por ayudas economicas publicas durante 2016'
        'p6_75d4',  # 'p.6.75d4. Importe recibido por becas durante 2016'
        'p6_76b',  # 'p.6.76.b. Valoracion total de lo recibido por pertenecer a C. de Admin. en 2016'
        'p6_75f',  # 'p.6.75.f. Importe recibido por seguro/premio/herencia/indeminizacion en 2016'
        # HOGAR - RENTAS DE ACTIVOS REALES (BRUTO ANUAL)
        # 'p7_2',         # 'p.7.2. Importe recibido por alquileres durante el ano 2016'
        # 'p7_4a',        # 'p.7.4.a. Plusvalias por venta de prop. inmob. durante el ano 2016'
        'p7_6a',  # 'p.7.6.a. Plusvalias por venta de joyas, etc. durante el ano 2016?'
        # HOGAR - RENTAS DE ACTIVOS FINANCIEROS (BRUTO ANUAL)
        'p7_8a',  # 'p.7.8.a. Plusvalias recibidas por venta de act. financieros durante el ano 2016'
        'p7_10',  # 'p.7.10. Intereses en cuentas bancarias recibidos durante el ano 2016'
        'p7_12',  # 'p.7.12. Ingresos recibidos por dividendos, opciones, prestamos durante 2016'
        'p7_12a',  # 'p.7.12.a. Rentas recibidas del negocio por miembro que no lo gestiona en 2016'
        # HOGAR - OTROS INGRESOS (BRUTO ANUAL)
        'p7_14'  # 'p.7.14. Ingresos recibidos por otros conceptos durante 2016'
    ]

    _annual_income_fields = _main_person_annual_income_fields + _household_annual_income_fields

    if not _individual:
        _annual_income_fields.extend(_other_members_annual_income_fields)

    if not _exclude_house_sale_profits:
        _annual_income_fields.append('p7_4a')

    # annual_losses_fields = [
    #     # HOGAR - PERDIDAS POR ACTIVOS REALES (BRUTO ANUAL)
    #     'p7_4b',        # 'p.7.4.b. Perdidas por venta de prop. inmobiliarias durante el ano 2016'
    #     'p7_6b',        # 'p.7.6.b. Perdidas por venta de joyas, etc. durante el ano 2016'
    #     # HOGAR - PERDIDAS POR ACTIVOS FINANCIEROS (BRUTO ANUAL)
    #     'p7_8b',        # 'p.7.8.b. Perdidas por venta de activos financieros durante el ano 2016'
    # ]
    #
    # extraordinary_income_fields = [
    #     # HOGAR - INGRESOS EXTRAORDINARIOS (BRUTO)
    #     'p9_20',        # 'p.9.20. Valor aproximado de la herencia, regalo o donacion cuando lo recibieron'
    #     'p9_14',        # 'p.9.14. Cuantia de la renta extraordinaria recibida'
    # ]
    #
    # extraordinary_income_year_fields = [
    #     # HOGAR - INGRESOS EXTRAORDINARIOS (BRUTO)
    #     'p9_19',        # 'p.9.19. Ano en que recibieron la herencia, regalo o donacion mas importante'
    #     'p9_13',        # 'p.9.13. Ano en que recibieron la renta extraordinaria'
    # ]

    _total_income = 0.0
    for _field in _annual_income_fields:
        if _row[_field] > 0.0:
            _total_income += _row[_field]
    # for _field in annual_losses_fields:
    #     if _row[_field] > 0.0:
    #         _total_income -= _row[_field]
    # for _year, _field in zip(extraordinary_income_year_fields, extraordinary_income_fields):
    #     if _row[_year] == 2016 and _row[_field] > 0.0:
    #         _total_income += _row[_field]
    return _total_income


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


# Set control variables and addresses. Note that available variables to print and plot are 'GrossTotalIncome',
# 'NetTotalIncome', 'GrossRentalIncome', 'NetRentalIncome', 'GrossNonRentIncome' and 'NetNonRentIncome'
incomeVariable = 'household_income'  # Available values: 'B_inc', 'household_income'
printPlots = False
multipleImputations = True
useManualIncome = False  # TODO: Decide whether to use manual or ready-made calculation of total income
excludeHouseSaleProfits = True  # TODO: Decide whether to remove also house sale profits, as endogenous too
startTime = 1000
endTime = 2000
min_log_income_bin_edge = 8.0
max_log_income_bin_edge = 12.2
log_income_bin_width = 0.2
variableToPlot = 'GrossNonRentIncome'
rootWAS = r''
rootEDW = r''
rootEFF = r''
rootModel = r''
rootResults = r''

# General printing settings
np.set_printoptions(suppress=True)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 1000)

# Read and clean Wealth and Assets Survey data for households
dfWAS = read_and_clean_was_data(rootWAS, _population='all')

# Read and clean European Data Warehouse data
dfEDW = read_and_clean_edw_data(rootEDW, incomeVariable)

# Read and clean Encuesta Financiera de las Familias data
dfEFF = read_and_clean_eff_data(rootEFF, multipleImputations, excludeHouseSaleProfits)

# Use EFF data to compute household income from individual income for EDW data
dfEDW = add_household_income_to_edw_data(dfEDW, dfEFF, _plot_fit=False)

# Read and clean model results
modelResults = read_and_clean_model_results(rootModel, _population='all', _start_time=startTime, _end_time=endTime)

# Define bin edges and widths
number_of_bins = int(round(max_log_income_bin_edge - min_log_income_bin_edge, 5) / log_income_bin_width + 1)
income_bin_edges = np.logspace(min_log_income_bin_edge, max_log_income_bin_edge, number_of_bins, base=np.e)
income_bin_widths = [b - a for a, b in zip(income_bin_edges[:-1], income_bin_edges[1:])]
income_bin_centers = np.exp([e + log_income_bin_width / 2 for e in np.log(income_bin_edges[:-1])])

# Print means and standard deviations to screen
print('Model mean = {}, std = {}'.format(np.average([12.0 * x for x in modelResults if x > 0.0]),
                                         np.std([12.0 * x for x in modelResults if x > 0.0])))
weightedStatsWAS = DescrStatsW(dfWAS[variableToPlot], weights=dfWAS['Weight'], ddof=0)
print('WAS mean = {}, std = {}'.format(weightedStatsWAS.mean, weightedStatsWAS.std))
print('Note EDW data has been selected for income variable {}'.format(incomeVariable))
print('EDW mean = {}, std = {}'.format(np.average(dfEDW['B_inc']), np.std(dfEDW['B_inc'])))
print('EDW hhld mean = {}, std = {}'.format(np.nanmean(dfEDW['household_income']),
                                            np.nanstd(dfEDW['household_income'])))
weightedStatsEFF = DescrStatsW(dfEFF[variableToPlot], weights=dfEFF['Weight'], ddof=0)
print('EFF mean = {}, std = {}'.format(weightedStatsEFF.mean, weightedStatsEFF.std))
weightedStatsEFF_ind = DescrStatsW(dfEFF['GrossNonRentIncomeIndividual'], weights=dfEFF['Weight'], ddof=0)
print('EFF_ind mean = {}, std = {}'.format(weightedStatsEFF_ind.mean, weightedStatsEFF_ind.std))

# Read model results, histogram data and results and plot them
# Histogram model results
model_hist = np.histogram([12.0 * x for x in modelResults if x > 0.0], bins=income_bin_edges, density=False)[0]
model_hist = model_hist / sum(model_hist)
# Histogram data from WAS
WAS_hist = np.histogram(dfWAS[variableToPlot], bins=income_bin_edges, density=False, weights=dfWAS['Weight'])[0]
WAS_hist = WAS_hist / sum(WAS_hist)
# Histogram data from EDW
EDW_hist = np.histogram(dfEDW[incomeVariable], bins=income_bin_edges, density=False)[0]
EDW_hist = EDW_hist / sum(EDW_hist)
# Histogram data from EFF
EFF_hist = np.histogram(dfEFF[variableToPlot], bins=income_bin_edges, density=False, weights=dfEFF['Weight'])[0]
EFF_hist = EFF_hist / sum(EFF_hist)
# Histogram data from EFF (individual incomes)
EFF_ind_hist = np.histogram(dfEFF['GrossNonRentIncomeIndividual'], bins=income_bin_edges, density=False,
                            weights=dfEFF['Weight'])[0]
EFF_ind_hist = EFF_ind_hist / sum(EFF_ind_hist)
# Plot both model results and data
plt.figure(figsize=(7.0, 5.0))
plt.bar(income_bin_edges[:-1], height=model_hist, width=income_bin_widths, align='edge',
        label='Model results', alpha=0.5, color='tab:blue')
plt.bar(income_bin_edges[:-1], height=WAS_hist, width=income_bin_widths, align='edge',
        label='WAS data', alpha=0.5, color='tab:orange')
plt.bar(income_bin_edges[:-1], height=EDW_hist, width=income_bin_widths, align='edge',
        label='EDW data', alpha=0.5, color='tab:green')
plt.bar(income_bin_edges[:-1], height=EFF_hist, width=income_bin_widths, align='edge',
        label='EFF data', alpha=0.5, color='tab:red')
plt.bar(income_bin_edges[:-1], height=EFF_ind_hist, width=income_bin_widths, align='edge',
        label='EFF ind data', alpha=0.5, color='tab:purple')
# Add lines and points for ease of comparison
plt.plot(income_bin_centers, model_hist, '-o', label='Model results', color='tab:blue')
plt.plot(income_bin_centers, WAS_hist, '-o', label='WAS data', color='tab:orange')
plt.plot(income_bin_centers, EDW_hist, '-o', label='EDW data', color='tab:green')
plt.plot(income_bin_centers, EFF_hist, '-o', label='EFF data', color='tab:red')
plt.plot(income_bin_centers, EFF_ind_hist, '-o', label='EFF ind data', color='tab:purple')
# Final plot details
plt.gca().set_xscale('log')
plt.xlabel('Income')
plt.ylabel('Frequency (fraction of cases)')
plt.legend()
plt.title('Distribution of {}'.format(variableToPlot))

# Print to screen and plot mean and std of log-distribution
# logChunk = np.log(dfWAS[variableToPlot])
# print(np.mean(logChunk))
# print(np.std(logChunk))
# plt.axvline(np.exp(np.mean(logChunk) - 2 * np.std(logChunk)), ls='--')
# plt.axvline(np.exp(np.mean(logChunk)), ls='-')
# plt.axvline(np.exp(np.mean(logChunk) + 2 * np.std(logChunk)), ls='--')

if printPlots:
    plt.tight_layout()
    plt.savefig(rootResults + '/IncomeDist-EDW-{}.pdf'.format(incomeVariable), format='pdf', dpi=300,
                bbox_inches='tight')
    plt.savefig(rootResults + '/IncomeDist-EDW-{}.png'.format(incomeVariable), format='png', dpi=300,
                bbox_inches='tight')
else:
    plt.show()
