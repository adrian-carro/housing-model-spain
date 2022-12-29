# -*- coding: utf-8 -*-
"""
Class to collect common functions used in a number of other classes within this project.

@author: Adrian Carro
"""

from __future__ import division
import numpy as np
import pandas as pd


def read_and_clean_edw_data(_root_edw, _from_year, _to_year):
    # Read data
    _df_edw = pd.read_csv(_root_edw + '/EDWdata_updateMAR22_v2.csv', dtype={'M': str, 'Q': str})

    # Remove duplicates
    _df_edw = _df_edw[~_df_edw.duplicated(keep=False)]

    # Restrict to origination dates within a certain range
    if _from_year is not None:
        _df_edw = _df_edw[(_df_edw['Y'] >= _from_year)]
    if _to_year is not None:
        _df_edw = _df_edw[(_df_edw['Y'] <= _to_year)]

    # Compute age and household income from available columns
    _df_edw['age'] = _df_edw['Y'].subtract(_df_edw['B_date_birth'])
    _df_edw['household_income'] = _df_edw[['B_inc', 'B_inc2']].sum(axis=1)

    # Add a third LTI measure, LTI3, equal to LTI2 when available and to LTI when LTI2 is unavailable
    _df_edw['LTI3'] = _df_edw['LTI2']
    _df_edw.loc[_df_edw['LTI3'].isna(), 'LTI3'] = _df_edw.loc[_df_edw['LTI3'].isna(), 'LTI']

    # Add a third DSTI measure, DSTI3, equal to DSTI2 when available and to DSTI when DSTI2 is unavailable
    _df_edw['DSTI3'] = _df_edw['DSTI2']
    _df_edw.loc[_df_edw['DSTI3'].isna(), 'DSTI3'] = _df_edw.loc[_df_edw['DSTI3'].isna(), 'DSTI']

    return _df_edw


def read_and_clean_cdr_data(_root_cdr, _with_price, _from_year, _to_year):
    if not _with_price:
        # Read data
        _df_cdr = pd.read_csv(_root_cdr + '/Colegio_data.csv',
                              dtype={'capital': float, 'valor': float, 'interes': float, 'plazohip': int, 'Y': int,
                                     'M': str, 'Q': str, 'LTV': float, 'tipofijo': object})

        # Remove duplicates
        _df_cdr = _df_cdr[~_df_cdr.duplicated(keep=False)]

        # Restrict to origination dates within a certain range
        if _from_year is not None:
            _df_cdr = _df_cdr.loc[(_df_cdr['Y'] >= 2014)]
        if _to_year is not None:
            _df_cdr = _df_cdr.loc[(_df_cdr['Y'] <= 2020)]

        return _df_cdr
    else:
        # Read data
        _df_cdr_wp = pd.read_csv(_root_cdr + '/Colegio_data_con_LTP.csv',
                                 dtype={'capital': float, 'valor': float, 'interes': float, 'plazohip': int, 'Y': int,
                                        'M': str, 'Q': str, 'precio': float, 'LTP': float, 'tipofijo': object,
                                        'LTV': float})

        # Remove duplicates
        _df_cdr_wp = _df_cdr_wp[~_df_cdr_wp.duplicated(keep=False)]

        # Restrict to origination dates within a certain range
        if _from_year is not None:
            _df_cdr_wp = _df_cdr_wp.loc[(_df_cdr_wp['Y'] >= 2014)]
        if _to_year is not None:
            _df_cdr_wp = _df_cdr_wp.loc[(_df_cdr_wp['Y'] <= 2020)]

        return _df_cdr_wp


def read_and_clean_cir_data(_root_cir, _from_year, _to_year):
    # Read data
    _df_cir = pd.read_csv(_root_cir + '/CIR_BTL_data.csv', parse_dates=['M_CIR'])

    # Extract year from date column
    _df_cir['YEAR'] = _df_cir['M_CIR'].str[:4].astype(int)

    # Restrict to origination dates within a certain range
    if _from_year is not None:
        _df_cir = _df_cir.loc[(_df_cir['YEAR'] >= 2014)]
    if _to_year is not None:
        _df_cir = _df_cir.loc[(_df_cir['YEAR'] <= 2020)]

    return _df_cir


def read_and_clean_bde_indicadores_data(_root_bde, _from_year, _to_year, _columns):
    # Read data
    _df_bde = pd.read_excel(_root_bde + '/si_1_5.xlsx', skiprows=[1, 2, 3, 4, 5], skipfooter=2, na_values=['-'])

    # Rename dates column (NOMBRE DE LA SERIE) and create separate year column (Y)
    _df_bde.rename(columns={'NOMBRE DE LA SERIE': 'FECHA'}, inplace=True)
    _df_bde['Y'] = pd.to_numeric(_df_bde['FECHA'].str[4:8])

    # Restrict to origination dates within a certain range
    if _from_year is not None:
        _df_bde = _df_bde.loc[(_df_bde['Y'] >= _from_year)]
    if _to_year is not None:
        _df_bde = _df_bde.loc[(_df_bde['Y'] <= _to_year)]

    # Rename columns of interest
    if 'PLAZO' in _columns:
        _df_bde.rename(columns={'DHIRINOAPLDMDUVT.T': 'PLAZO'}, inplace=True)
        _df_bde['PLAZO'].astype(float)
    if 'INTERES' in _columns:
        _df_bde.rename(columns={'D_IMVTIPLV': 'INTERES'}, inplace=True)
        _df_bde['INTERES'].astype(float)

    # Restrict to columns of interest
    _df_bde = _df_bde[['FECHA', 'Y'] + _columns]

    return _df_bde


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


def read_micro_results(file_name, _start_time, _end_time):
    """Read micro-data from file_name, structured on a separate line per time step. In particular, read from start_time
    until end_time, both inclusive"""
    # Read list of float values, one per household
    data_float = []
    with open(file_name, "r") as _f:
        for line in _f:
            if _start_time <= int(line.split(';')[0]) <= _end_time:
                for column in line.split(';')[1:]:
                    data_float.append(float(column))
    return data_float
