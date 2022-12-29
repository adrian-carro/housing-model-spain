# -*- coding: utf-8 -*-
"""
Class to study the probability of a household becoming a buy-to-let investor depending on its income percentile, based
on Encuesta Financiera de las Familias data.

@author: Adrian Carro
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


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

    # Compute income percentiles (using gross non-rent income) of all households
    _df_was['GrossNonRentIncomePercentile'] = [stats.percentileofscore(_df_was['GrossNonRentIncome'].values, x, 'weak')
                                               for x in _df_was['GrossNonRentIncome']]

    return _df_was


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


def read_and_clean_eff_data(_root_eff, _multiple_imputations, _exclude_house_sale_profits):
    # EFF variables of interest
    # facine3                       Household weight
    # renthog                       Household Gross Annual income
    # p7_2                          Importe recibido por alquileres durante el ano 2016
    # p7_4a                         Plusvalias por venta de prop. inmob. durante el ano 2016
    # p2_35a_i                      Prop i Tipo de propiedad (= 1 Vivienda)
    # p2_43_i                       Prop i Ingresos mensuales por alquiler de esta propiedad
    # Pre-select columns of interest so as to read data more efficiently
    _vars_of_interest = ['p2_35a', 'p2_43', 'p6_6', 'p6_7', 'p7_1', 'p7_2', 'p7_4', 'p7_6', 'p7_8', 'p9_1', 'p9_20']
    _otras_secciones_cols = pd.read_csv(_root_eff + r'/eff_2017_imp1_csv/otras_secciones_2017_imp1.csv',
                                        nrows=1, sep=';').columns
    _seccion6_cols = pd.read_csv(_root_eff + r'/eff_2017_imp1_csv/seccion6_2017_imp1.csv', nrows=1, sep=';').columns
    _otras_secciones_cols_of_interest = [c for c in _otras_secciones_cols if any(e in c for e in _vars_of_interest)]
    _otras_secciones_cols_of_interest = ['facine3', 'renthog', 'h_2017'] + _otras_secciones_cols_of_interest
    _seccion6_cols_of_interest = [c for c in _seccion6_cols if any(e in c for e in _vars_of_interest)]
    _seccion6_cols_of_interest = ['h_2017'] + _seccion6_cols_of_interest
    # Read data
    if _multiple_imputations:
        _df_eff = pd.DataFrame()
        for i in range(1, 6):
            temp_df1 = pd.read_csv(_root_eff + r'/eff_2017_imp{0}_csv/otras_secciones_2017_imp{0}.csv'.format(i),
                                   sep=';', usecols=_otras_secciones_cols_of_interest)
            temp_df2 = pd.read_csv(_root_eff + r'/eff_2017_imp{0}_csv/seccion6_2017_imp{0}.csv'.format(i), sep=';',
                                   usecols=_seccion6_cols_of_interest)
            temp_df = pd.merge(temp_df1, temp_df2, on='h_2017')
            _df_eff = pd.concat([_df_eff, temp_df])
        _df_eff['facine3'] = _df_eff['facine3'] / 5.0
    else:
        temp_df1 = pd.read_csv(_root_eff + r'/eff_2017_imp1_csv/otras_secciones_2017_imp1.csv', sep=';',
                               usecols=_otras_secciones_cols_of_interest)
        temp_df2 = pd.read_csv(_root_eff + r'/eff_2017_imp1_csv/seccion6_2017_imp1.csv', sep=';',
                               usecols=_seccion6_cols_of_interest)
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
    _df_eff['GrossNonRentIncomeManual'] = _df_eff.apply(
        lambda row: add_up_annual_non_rental_income_2016(row, _exclude_house_sale_profits, _individual=False), axis=1)
    _df_eff['GrossNonRentIncomeIndividual'] = _df_eff.apply(
        lambda row: add_up_annual_non_rental_income_2016(row, _exclude_house_sale_profits, _individual=True), axis=1)

    # Compute total annual rental income, and total annual rental income from renting out dwellings
    _df_eff.rename(columns={'p7_2': 'TotalRentIncome'}, inplace=True)
    _df_eff['TotalHouseRentIncome'] = _df_eff.apply(lambda row: add_up_annual_house_rental_income_2016(row), axis=1)

    # Filter out NaNs, negative values and zeros
    _df_eff = _df_eff.loc[_df_eff['GrossNonRentIncome'] > 0.0]

    # Filter down to keep only columns of interest
    _df_eff = _df_eff[['Weight', 'GrossNonRentIncome', 'GrossNonRentIncomeManual', 'GrossNonRentIncomeIndividual',
                       'TotalRentIncome', 'TotalHouseRentIncome']]

    # Filter out the 1% with highest and the 1% with lowest GrossNonRentIncome
    # _min_eff = weighted_quantile(_df_eff['GrossNonRentIncome'], 0.01, sample_weight=_df_eff['Weight'])
    # _max_eff = weighted_quantile(_df_eff['GrossNonRentIncome'], 0.99, sample_weight=_df_eff['Weight'])
    # _df_eff = _df_eff.loc[(_df_eff['GrossNonRentIncome'] >= _min_eff) & (_df_eff['GrossNonRentIncome'] <= _max_eff)]

    return _df_eff


def add_up_annual_non_rental_income_2016(_row, _exclude_house_sale_profits, _individual):
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


def add_up_annual_house_rental_income_2016(_row):
    _total_house_rental_income = 0.0
    for i in range(1, 5):
        if _row['p2_35a_{}'.format(i)] in {1, 9}:
            _total_house_rental_income += _row['p2_43_{}'.format(i)]

    return 12.0 * _total_house_rental_income


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


# Control variables
saveFigures = False
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

# Compute BTL probability per income bin for WAS data
probability_BTL_WAS = []
for a in range(0, 100, 5):
    n_total = len(dfWAS[(a < dfWAS['GrossNonRentIncomePercentile'])
                        & (dfWAS['GrossNonRentIncomePercentile'] <= a + 1.0)])
    n_BTL = len(dfWAS[(a < dfWAS['GrossNonRentIncomePercentile'])
                      & (dfWAS['GrossNonRentIncomePercentile'] <= a + 1.0)
                      & (dfWAS['GrossRentalIncome'] > 0.0)])
    probability_BTL_WAS.append(n_BTL / n_total)

# Compute BTL probability per income bin for EFF data
percentiles = np.round(np.linspace(0.0, 1.0, 21), 5)
income_edges = weighted_quantile(dfEFF['GrossNonRentIncome'], percentiles, sample_weight=dfEFF['Weight'])
probability_BTL_EFF = []
for a, b in zip(income_edges[:-2], income_edges[1:-1]):
    total_weight = sum(dfEFF.loc[(dfEFF['GrossNonRentIncome'] >= a) & (dfEFF['GrossNonRentIncome'] < b), 'Weight'])
    BTL_weight = sum(dfEFF.loc[(dfEFF['GrossNonRentIncome'] >= a) & (dfEFF['GrossNonRentIncome'] < b)
                               & (dfEFF['TotalHouseRentIncome'] > 0.0), 'Weight'])
    probability_BTL_EFF.append(BTL_weight / total_weight)
total_weight = sum(dfEFF.loc[(dfEFF['GrossNonRentIncome'] >= income_edges[-2])
                             & (dfEFF['GrossNonRentIncome'] < income_edges[-1]), 'Weight'])
BTL_weight = sum(dfEFF.loc[(dfEFF['GrossNonRentIncome'] >= income_edges[-2])
                           & (dfEFF['GrossNonRentIncome'] < income_edges[-1])
                           & (dfEFF['TotalHouseRentIncome'] > 0.0), 'Weight'])
probability_BTL_EFF.append(BTL_weight / total_weight)

# Plot model results if required
probability_BTL_model = []  # To avoid undefined warning
if addModelResults:
    # Read model results
    model_incomes = read_results(rootModel + '/results/test4/MonthlyGrossEmploymentIncome-run1.csv', 1000, 2000)
    model_n_properties = read_results(rootModel + '/results/test4/NHousesOwned-run1.csv', 1000, 2000)
    # Compute model income edges from model incomes and the required percentile values
    model_income_edges = np.percentile(model_incomes, [100.0 * e for e in percentiles])
    # Compute the corresponding income bin for each model income (correcting for last edge to be included in last bin)
    total_households_bin_numbers = np.digitize(model_incomes, model_income_edges, right=False)
    total_households_bin_numbers[total_households_bin_numbers == len(percentiles)] = len(percentiles) - 1
    # Compute value counts for all households for each bin (present in the model results)
    total_households_unique, total_households_counts = np.unique(total_households_bin_numbers, return_counts=True)
    # Select only incomes of active BTL households
    model_BTL_incomes = [income for income, n_properties in zip(model_incomes, model_n_properties) if n_properties > 1]
    # Compute the corresponding income bin for each BTL income (correcting for last edge to be included in last bin)
    BTL_households_bin_numbers = np.digitize(model_BTL_incomes, model_income_edges, right=False)
    BTL_households_bin_numbers[BTL_households_bin_numbers == len(percentiles)] = len(percentiles) - 1
    # Compute value counts for BTL households for each bin (present in the model results)
    BTL_households_unique, BTL_households_counts = np.unique(BTL_households_bin_numbers, return_counts=True)
    BTL_households_unique = list(BTL_households_unique)
    BTL_households_counts = list(BTL_households_counts)
    # Correct BTL_households_counts in case any bin is absent (adding a zero count)
    for i in range(1, len(percentiles)):
        if i not in BTL_households_unique:
            BTL_households_unique = BTL_households_unique[:i - 1] + [i] + BTL_households_unique[i - 1:]
            BTL_households_counts = BTL_households_counts[:i - 1] + [0] + BTL_households_counts[i - 1:]
    # Finally, use value counts for total households and BTL households to compute BTL probability per income bin
    probability_BTL_model = [BTL / total for BTL, total in zip(BTL_households_counts, total_households_counts)]

# Plot results
plt.figure(figsize=(6, 4.5))
plt.bar(percentiles[:-1], probability_BTL_WAS, width=0.05, align='edge', label='WAS', alpha=0.5, color='tab:blue')
plt.bar(percentiles[:-1], probability_BTL_EFF, width=0.05, align='edge', label='EFF', alpha=0.5, color='tab:green')
if addModelResults:
    plt.plot([e + 0.025 for e in percentiles[:-1]], probability_BTL_model, 'o-', label='Model', color='tab:red')
plt.legend()
plt.xlabel('Income Percentile')
plt.ylabel('BTL Probability')

# Write to file probability of being a BTL investor for each percentile bin
if writeResults:
    with open(rootResults + '/BTLProbabilityPerIncomePercentileBin.csv', 'w') as f:
        f.write('# Gross non-rental income percentile (lower edge), gross non-rental income percentile (upper edge), '
                'BTL probability\n')
        for i, prob in enumerate(probability_BTL_EFF):
            f.write('{}, {}, {}\n'.format(percentiles[i], percentiles[i + 1], prob))

if saveFigures:
    plt.tight_layout()
    plt.savefig(rootResults + '/BTLProbabilityPerIncomePercentileBin.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(rootResults + '/BTLProbabilityPerIncomePercentileBin.png', format='png', dpi=300, bbox_inches='tight')
else:
    plt.show()

