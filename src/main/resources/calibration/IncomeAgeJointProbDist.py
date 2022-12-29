# -*- coding: utf-8 -*-
"""
Class to study households' income distribution depending on their age based on Encuesta Financiera de las Familias 2016
data (allowing for a comparison with Wealth and Assets Survey 2011 data). This is the code used to create files
"EFF-AgeGrossIncomeJointDist.csv" and "WAS-AgeGrossIncomeJointDist.csv".

@author: Adrian Carro
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def add_up_annual_non_renal_income_2016(row, exclude_house_sale_profits):
    annual_income_fields = [
        # INGRESOS POR CUENTA AJENA (BRUTO ANUAL)
        'p6_64_1',      # 'p.6.64. Miem. 1. Importe recibido como empleado por cuenta ajena durante 2016'
        'p6_64_2',      # 'p.6.64. Miem. 2. Importe recibido como empleado por cuenta ajena durante 2016'
        'p6_64_3',      # 'p.6.64. Miem. 3. Importe recibido como empleado por cuenta ajena durante 2016'
        'p6_64_4',      # 'p.6.64. Miem. 4. Importe recibido como empleado por cuenta ajena durante 2016'
        'p6_64_5',      # 'p.6.64. Miem. 5. Importe recibido como empleado por cuenta ajena durante 2016'
        'p6_64_6',      # 'p.6.64. Miem. 6. Importe recibido como empleado por cuenta ajena durante 2016'
        'p6_64_7',      # 'p.6.64. Miem. 7. Importe recibido como empleado por cuenta ajena durante 2016'
        'p6_64_8',      # 'p.6.64. Miem. 8. Importe recibido como empleado por cuenta ajena durante 2016'
        'p6_64_9',      # 'p.6.64. Miem. 9. Importe recibido como empleado por cuenta ajena durante 2016'
        # INGRESOS POR CUENTA AJENA EN ESPECIE (BRUTO ANUAL)
        'p6_66_1',      # 'p.6.66. Miem. 1. Importe recibido en especie durante 2016'
        'p6_66_2',      # 'p.6.66. Miem. 2. Importe recibido en especie durante 2016'
        'p6_66_3',      # 'p.6.66. Miem. 3. Importe recibido en especie durante 2016'
        'p6_66_4',      # 'p.6.66. Miem. 4. Importe recibido en especie durante 2016'
        'p6_66_5',      # 'p.6.66. Miem. 5. Importe recibido en especie durante 2016'
        'p6_66_6',      # 'p.6.66. Miem. 6. Importe recibido en especie durante 2016'
        'p6_66_7',      # 'p.6.66. Miem. 7. Importe recibido en especie durante 2016'
        'p6_66_8',      # 'p.6.66. Miem. 8. Importe recibido en especie durante 2016'
        'p6_66_9',      # 'p.6.66. Miem. 9. Importe recibido en especie durante 2016'
        # INGRESOS COMO DESEMPLEADO (BRUTO ANUAL)
        'p6_68_1',      # 'p.6.68. Miem. 1. Importe recibido por prestaciones por desempleo durante 2016'
        'p6_68_2',      # 'p.6.68. Miem. 2. Importe recibido por prestaciones por desempleo durante 2016'
        'p6_68_3',      # 'p.6.68. Miem. 3. Importe recibido por prestaciones por desempleo durante 2016'
        'p6_68_4',      # 'p.6.68. Miem. 4. Importe recibido por prestaciones por desempleo durante 2016'
        'p6_68_5',      # 'p.6.68. Miem. 5. Importe recibido por prestaciones por desempleo durante 2016'
        'p6_68_6',      # 'p.6.68. Miem. 6. Importe recibido por prestaciones por desempleo durante 2016'
        'p6_68_7',      # 'p.6.68. Miem. 7. Importe recibido por prestaciones por desempleo durante 2016'
        'p6_68_8',      # 'p.6.68. Miem. 8. Importe recibido por prestaciones por desempleo durante 2016'
        'p6_68_9',      # 'p.6.68. Miem. 9. Importe recibido por prestaciones por desempleo durante 2016'
        # INGRESOS POR INDEMNIZACION POR DESPIDO (BRUTO ANUAL)
        'p6_70_1',      # 'p.6.70. Miem. 1. Importe recibido por indemnizaciones por despido durante 2016'
        'p6_70_2',      # 'p.6.70. Miem. 2. Importe recibido por indemnizaciones por despido durante 2016'
        'p6_70_3',      # 'p.6.70. Miem. 3. Importe recibido por indemnizaciones por despido durante 2016'
        'p6_70_4',      # 'p.6.70. Miem. 4. Importe recibido por indemnizaciones por despido durante 2016'
        'p6_70_5',      # 'p.6.70. Miem. 5. Importe recibido por indemnizaciones por despido durante 2016'
        'p6_70_6',      # 'p.6.70. Miem. 6. Importe recibido por indemnizaciones por despido durante 2016'
        'p6_70_7',      # 'p.6.70. Miem. 7. Importe recibido por indemnizaciones por despido durante 2016'
        'p6_70_8',      # 'p.6.70. Miem. 8. Importe recibido por indemnizaciones por despido durante 2016'
        'p6_70_9',      # 'p.6.70. Miem. 9. Importe recibido por indemnizaciones por despido durante 2016'
        # INGRESOS POR CUENTA PROPIA (BRUTO ANUAL)
        'p6_72_1',      # 'p.6.72. Miem. 1. Importe recibido por su trabajo por cuenta propia en 2016'
        'p6_72_2',      # 'p.6.72. Miem. 2. Importe recibido por su trabajo por cuenta propia en 2016'
        'p6_72_3',      # 'p.6.72. Miem. 3. Importe recibido por su trabajo por cuenta propia en 2016'
        'p6_72_4',      # 'p.6.72. Miem. 4. Importe recibido por su trabajo por cuenta propia en 2016'
        'p6_72_5',      # 'p.6.72. Miem. 5. Importe recibido por su trabajo por cuenta propia en 2016'
        'p6_72_6',      # 'p.6.72. Miem. 6. Importe recibido por su trabajo por cuenta propia en 2016'
        'p6_72_7',      # 'p.6.72. Miem. 7. Importe recibido por su trabajo por cuenta propia en 2016'
        'p6_72_8',      # 'p.6.72. Miem. 8. Importe recibido por su trabajo por cuenta propia en 2016'
        'p6_72_9',      # 'p.6.72. Miem. 9. Importe recibido por su trabajo por cuenta propia en 2016'
        # INGRESOS POR PENSION VIUDEDAD/ORFANDAD (BRUTO ANUAL)
        'p6_74b_1',     # 'p.6.74b. Miem. 1. Importe recibido por pensiones de viudedad/orfandad en 2016'
        'p6_74b_2',     # 'p.6.74b. Miem. 2. Importe recibido por pensiones de viudedad/orfandad en 2016'
        'p6_74b_3',     # 'p.6.74b. Miem. 3. Importe recibido por pensiones de viudedad/orfandad en 2016'
        'p6_74b_4',     # 'p.6.74b. Miem. 4. Importe recibido por pensiones de viudedad/orfandad en 2016'
        'p6_74b_5',     # 'p.6.74b. Miem. 5. Importe recibido por pensiones de viudedad/orfandad en 2016'
        'p6_74b_6',     # 'p.6.74b. Miem. 6. Importe recibido por pensiones de viudedad/orfandad en 2016'
        'p6_74b_7',     # 'p.6.74b. Miem. 7. Importe recibido por pensiones de viudedad/orfandad en 2016'
        'p6_74b_8',     # 'p.6.74b. Miem. 8. Importe recibido por pensiones de viudedad/orfandad en 2016'
        'p6_74b_9',     # 'p.6.74b. Miem. 9. Importe recibido por pensiones de viudedad/orfandad en 2016'
        # INGRESOS POR PENSION JUBILACION/INCAPACIDAD(BRUTO ANUAL)
        'p6_74_1',      # 'p.6.74. Miem. 1. Importe recibido por pensiones jubilacion/incapacidad en 2016'
        'p6_74_2',      # 'p.6.74. Miem. 2. Importe recibido por pensiones jubilacion/incapacidad en 2016'
        'p6_74_3',      # 'p.6.74. Miem. 3. Importe recibido por pensiones jubilacion/incapacidad en 2016'
        'p6_74_4',      # 'p.6.74. Miem. 4. Importe recibido por pensiones jubilacion/incapacidad en 2016'
        'p6_74_5',      # 'p.6.74. Miem. 5. Importe recibido por pensiones jubilacion/incapacidad en 2016'
        'p6_74_6',      # 'p.6.74. Miem. 6. Importe recibido por pensiones jubilacion/incapacidad en 2016'
        'p6_74_7',      # 'p.6.74. Miem. 7. Importe recibido por pensiones jubilacion/incapacidad en 2016'
        'p6_74_8',      # 'p.6.74. Miem. 8. Importe recibido por pensiones jubilacion/incapacidad en 2016'
        'p6_74_9',      # 'p.6.74. Miem. 9. Importe recibido por pensiones jubilacion/incapacidad en 2016'
        # INGRESOS POR PENSION PRIVADA (BRUTO ANUAL)
        'p6_74c_1',     # 'p.6.74c. Miem. 1. Importe recibido por planes de pensiones privados en 2016'
        'p6_74c_2',     # 'p.6.74c. Miem. 2. Importe recibido por planes de pensiones privados en 2016'
        'p6_74c_3',     # 'p.6.74c. Miem. 3. Importe recibido por planes de pensiones privados en 2016'
        'p6_74c_4',     # 'p.6.74c. Miem. 4. Importe recibido por planes de pensiones privados en 2016'
        'p6_74c_5',     # 'p.6.74c. Miem. 5. Importe recibido por planes de pensiones privados en 2016'
        'p6_74c_6',     # 'p.6.74c. Miem. 6. Importe recibido por planes de pensiones privados en 2016'
        'p6_74c_7',     # 'p.6.74c. Miem. 7. Importe recibido por planes de pensiones privados en 2016'
        'p6_74c_8',     # 'p.6.74c. Miem. 8. Importe recibido por planes de pensiones privados en 2016'
        'p6_74c_9',     # 'p.6.74c. Miem. 9. Importe recibido por planes de pensiones privados en 2016'
        # HOGAR - OTROS INGRESOS NO DECLARADOS POR PERSONA (BRUTO ANUAL)
        'p6_75b',       # 'p.6.75b. Importe recibido por pensiones de viudedad/orfandad durante 2016'
        'p6_75d1',      # 'p.6.75d1. Importe recibido de ex pareja con la que no conviven durante 2016'
        'p6_75d2',      # 'p.6.75d2. Importe recibido de familiares fuera del hogar durante 2016'
        'p6_75d3',      # 'p.6.75d3. Importe recibido por ayudas economicas publicas durante 2016'
        'p6_75d4',      # 'p.6.75d4. Importe recibido por becas durante 2016'
        'p6_76b',       # 'p.6.76.b. Valoracion total de lo recibido por pertenecer a C. de Admin. en 2016'
        'p6_75f',       # 'p.6.75.f. Importe recibido por seguro/premio/herencia/indeminizacion en 2016'
        # HOGAR - RENTAS DE ACTIVOS REALES (BRUTO ANUAL)
        # 'p7_2',         # 'p.7.2. Importe recibido por alquileres durante el ano 2016'
        # 'p7_4a',        # 'p.7.4.a. Plusvalias por venta de prop. inmob. durante el ano 2016'
        'p7_6a',        # 'p.7.6.a. Plusvalias por venta de joyas, etc. durante el ano 2016?'
        # HOGAR - RENTAS DE ACTIVOS FINANCIEROS (BRUTO ANUAL)
        'p7_8a',        # 'p.7.8.a. Plusvalias recibidas por venta de act. financieros durante el ano 2016'
        'p7_10',        # 'p.7.10. Intereses en cuentas bancarias recibidos durante el ano 2016'
        'p7_12',        # 'p.7.12. Ingresos recibidos por dividendos, opciones, prestamos durante 2016'
        'p7_12a',       # 'p.7.12.a. Rentas recibidas del negocio por miembro que no lo gestiona en 2016'
        # HOGAR - OTROS INGRESOS (BRUTO ANUAL)
        'p7_14'         # 'p.7.14. Ingresos recibidos por otros conceptos durante 2016'
    ]

    if not exclude_house_sale_profits:
        annual_income_fields.append('p7_4a')

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
    for _field in annual_income_fields:
        if row[_field] > 0.0:
            _total_income += row[_field]
    # for _field in annual_losses_fields:
    #     if row[_field] > 0.0:
    #         _total_income -= row[_field]
    # for _year, _field in zip(extraordinary_income_year_fields, extraordinary_income_fields):
    #     if row[_year] == 2016 and row[_field] > 0.0:
    #         _total_income += row[_field]
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


# Set parameters
incomeBinWidth = 0.2
multipleImputations = True
useManualIncome = False  # TODO: Decide whether to use manual or ready-made calculation of total income
excludeHouseSaleProfits = True  # TODO: Decide whether to remove also house sale profits, as endogenous too
writeResults = False
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 1000)

# Read data
rootWAS = r''
dfWAS = pd.read_csv(rootWAS + r'/was_wave_3_hhold_eul_final.dta',
                    usecols={"w3xswgt", "DVTotGIRw3", "DVGrsRentAmtAnnualw3_aggr", "HRPDVAge9W3"})
rootEFF = r''
if multipleImputations:
    dfEFF = pd.DataFrame()
    for i in range(1, 6):
        if not useManualIncome:
            temp_df = pd.read_csv(rootEFF + r'/eff_2017_imp{0}_csv/otras_secciones_2017_imp{0}.csv'.format(i), sep=';',
                                  usecols={'facine3', 'p1_2d_1', 'p7_2', 'p7_4a', 'renthog'})
        else:
            temp_df1 = pd.read_csv(rootEFF + r'/eff_2017_imp{0}_csv/otras_secciones_2017_imp{0}.csv'.format(i), sep=";")
            temp_df2 = pd.read_csv(rootEFF + r'/eff_2017_imp{0}_csv/seccion6_2017_imp{0}.csv'.format(i), sep=";")
            temp_df = pd.merge(temp_df1, temp_df2, on="h_2017")
        dfEFF = pd.concat([dfEFF, temp_df])
    dfEFF['facine3'] = dfEFF['facine3'] / 5.0
else:
    if not useManualIncome:
        dfEFF = pd.read_csv(rootEFF + r'/eff_2017_imp1_csv/otras_secciones_2017_imp1.csv', sep=';',
                            usecols={'facine3', 'p1_2d_1', 'p7_2', 'p7_4a', 'renthog'})
    else:
        temp_df1 = pd.read_csv(rootEFF + r'/eff_2017_imp1_csv/otras_secciones_2017_imp1.csv', sep=";")
        temp_df2 = pd.read_csv(rootEFF + r'/eff_2017_imp1_csv/seccion6_2017_imp1.csv', sep=";")
        dfEFF = pd.merge(temp_df1, temp_df2, on="h_2017")
rootResults = r''

# EFF variables of interest
# facine3                       Household weight
# p1_2d_1                       Age of HRP
# renthog                       Household Gross Annual income
# p7_2                          Importe recibido por alquileres durante el ano 2016
# p7_4a                         Plusvalias por venta de prop. inmob. durante el ano 2016

# WAS variables of interest
# DVTotGIRw3                    Household Gross Annual (regular) income
# DVGrsRentAmtAnnualw3_aggr     Household Gross Annual income from rent
# HRPDVAge9W3                   Age of HRP or partner [0-15, 16-24, 25-34, 35-44, 45-54, 55-64, 65-74, 75-84, 85+]

# Rename columns to be used
dfWAS.rename(columns={'w3xswgt': 'Weight'}, inplace=True)
dfWAS.rename(columns={'HRPDVAge9W3': 'Age'}, inplace=True)
dfEFF.rename(columns={'facine3': 'Weight'}, inplace=True)
dfEFF.rename(columns={'p1_2d_1': 'Age'}, inplace=True)

# Replace NaNs by zeros in EFF data
dfEFF = dfEFF.fillna(0)

# Compute annual gross non-rental income
dfWAS['GrossNonRentIncome'] = dfWAS['DVTotGIRw3'] - dfWAS['DVGrsRentAmtAnnualw3_aggr']
if not useManualIncome:
    if not excludeHouseSaleProfits:
        dfEFF['GrossNonRentIncome'] = dfEFF['renthog'] - dfEFF['p7_2']
    else:
        dfEFF['GrossNonRentIncome'] = dfEFF['renthog'] - dfEFF['p7_2'] - dfEFF['p7_4a']
else:
    dfEFF['GrossNonRentIncome'] = dfEFF.apply(
        lambda row: add_up_annual_non_renal_income_2016(row, excludeHouseSaleProfits), axis=1)

# Filter down to keep only columns of interest
dfWAS = dfWAS[["Age", "GrossNonRentIncome", "Weight"]]
dfEFF = dfEFF[["Age", "GrossNonRentIncome", "Weight"]]

# Filter out NaNs, negative values and zeros
dfEFF = dfEFF.loc[(dfEFF['GrossNonRentIncome'] > 0.0)]

# Filter out the 1% with highest and the 1% with lowest GrossNonRentIncome
minWAS = weighted_quantile(dfWAS['GrossNonRentIncome'], 0.01, sample_weight=dfWAS['Weight'])
maxWAS = weighted_quantile(dfWAS['GrossNonRentIncome'], 0.99, sample_weight=dfWAS['Weight'])
minEFF = weighted_quantile(dfEFF['GrossNonRentIncome'], 0.01, sample_weight=dfEFF['Weight'])
maxEFF = weighted_quantile(dfEFF['GrossNonRentIncome'], 0.99, sample_weight=dfEFF['Weight'])
dfWAS = dfWAS.loc[(dfWAS['GrossNonRentIncome'] >= minWAS) & (dfWAS['GrossNonRentIncome'] <= maxWAS)]
dfEFF = dfEFF.loc[(dfEFF['GrossNonRentIncome'] >= minEFF) & (dfEFF['GrossNonRentIncome'] <= maxEFF)]

# Map age buckets to middle of bucket value by creating the corresponding dictionary
dfWAS["Age"] = dfWAS["Age"].map({"16-24": 20, "25-34": 30, "35-44": 40, "45-54": 50, "55-64": 60, "65-74": 70,
                                 "75-84": 80, "85+": 90})

# Create a 2D histogram of the data with logarithmic income bins (no normalisation here as we want column normalisation,
# to be introduced when plotting or printing) (EFF could have 16, 21, 26 or 27 bin edges)
incMinBinEdgeWAS = np.round(np.floor(np.log(minWAS) / incomeBinWidth) * incomeBinWidth, 5)
incMaxBinEdgeWAS = np.round(np.ceil(np.log(maxWAS) / incomeBinWidth) * incomeBinWidth, 5)
incMinBinEdgeEFF = np.round(np.floor(np.log(minEFF) / incomeBinWidth) * incomeBinWidth, 5)
incMaxBinEdgeEFF = np.round(np.ceil(np.log(maxEFF) / incomeBinWidth) * incomeBinWidth, 5)
binEdgesWAS = np.round(np.linspace(incMinBinEdgeWAS, incMaxBinEdgeWAS,
                                   int((incMaxBinEdgeWAS - incMinBinEdgeWAS) / incomeBinWidth) + 1, endpoint=True), 5)
binEdgesEFF = np.round(np.linspace(incMinBinEdgeEFF, incMaxBinEdgeEFF,
                                   int((incMaxBinEdgeEFF - incMinBinEdgeEFF) / incomeBinWidth) + 1, endpoint=True), 5)
ageBinEdges = [15, 25, 35, 45, 55, 65, 75, 85, 95]
frequencyWAS = np.histogram2d(dfWAS["Age"], np.log(dfWAS["GrossNonRentIncome"]), bins=[ageBinEdges, binEdgesWAS],
                              normed=True, weights=dfWAS["Weight"])[0]
frequencyEFF = np.histogram2d(dfEFF["Age"], np.log(dfEFF["GrossNonRentIncome"]), bins=[ageBinEdges, binEdgesEFF],
                              normed=True, weights=dfEFF["Weight"])[0]

if not writeResults:
    plt.hist(np.log(dfWAS['GrossNonRentIncome']), bins=binEdgesWAS, density=True, weights=dfWAS['Weight'],
             alpha=0.5, label='WAS')
    plt.hist(np.log(dfEFF['GrossNonRentIncome']), bins=binEdgesEFF, density=True, weights=dfEFF['Weight'],
             alpha=0.5, label='EFF')
    plt.xlim(7.8, 12.4)
    plt.ylim(0, 0.6)
    plt.legend()
    plt.show()
else:
    # Write joint distributions to files
    with open(rootResults + '/WAS-AgeGrossIncomeJointDist.csv', 'w') as f:
        f.write('# Age (lower edge), Age (upper edge), Log Gross Income (lower edge), Log Gross Income (upper edge), '
                'Probability\n')
        for line, ageLowerEdge, ageUpperEdge in zip(frequencyWAS, ageBinEdges[:-1], ageBinEdges[1:]):
            for element, incomeLowerEdge, incomeUpperEdge in zip(line, binEdgesWAS[:-1], binEdgesWAS[1:]):
                f.write('{}, {}, {}, {}, {}\n'.format(ageLowerEdge, ageUpperEdge, incomeLowerEdge, incomeUpperEdge,
                                                      element / sum(line)))
    fileName = '/EFF-AgeGrossIncomeJointDist'
    if useManualIncome:
        fileName += '-Manual'
    if excludeHouseSaleProfits:
        fileName += '-NoHouseSaleProfits'
    if multipleImputations:
        fileName += '-MI'
    fileName += '.csv'
    with open(rootResults + fileName, 'w') as f:
        f.write('# Age (lower edge), Age (upper edge), Log Gross Income (lower edge), Log Gross Income (upper edge), '
                'Probability\n')
        for line, ageLowerEdge, ageUpperEdge in zip(frequencyEFF, ageBinEdges[:-1], ageBinEdges[1:]):
            for element, incomeLowerEdge, incomeUpperEdge in zip(line, binEdgesEFF[:-1], binEdgesEFF[1:]):
                f.write('{}, {}, {}, {}, {}\n'.format(ageLowerEdge, ageUpperEdge, incomeLowerEdge, incomeUpperEdge,
                                                      element / sum(line)))
