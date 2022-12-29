# -*- coding: utf-8 -*-
"""
Class to study households' wealth distribution depending on their income based on Encuesta Financiera de las Familias
2016 data (allowing for a comparison with Wealth and Assets Survey 2011 data). This is the code used to create files
"EFF-GrossIncome[type]WealthJointDist.csv" and "WAS-GrossIncome[type]WealthJointDist.csv", with possible types Gross,
Net and Liq.

@author: Adrian Carro
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_liquid_financial_wealth_from_was(row):
    """
    Compute liquid financial wealth from WAS data as a sum of:
    * 'DVFNSValW3_aggr', Hhold value of National Savings Product
    * 'DVCACTvW3_aggr', Hhold value of children's trust funds
    * 'DVCASVVW3_aggr', Hhold value of others children's savings
    * 'DVSaValW3_aggr', Hhold value of Savings Accounts
    * 'DVCISAVW3_aggr', Hhold value of Cash ISA
    * 'DVCaCrValW3_aggr', Hhold value of current accounts in credit
    """
    return (row["DVFNSValW3_aggr"] + row["DVCACTvW3_aggr"] + row["DVCASVVW3_aggr"] + row["DVSaValW3_aggr"]
            + row["DVCISAVW3_aggr"] + row["DVCaCrValW3_aggr"])


def get_real_assets(row):
    """
    Compute real assets as sum of: value of main residence + value of other real estate properties + value of jewellery,
    works of art and antiques + value of business related to self-employment
    """

    # Value of main residence
    # 'p.2.1b. Posesion de la totalidad de la vivienda principal o de una parte'
    # 'p.2.1c. Porcentaje del valor de su vivienda principal que les pertenece'
    # 'p.2.5. Valor actual de la vivienda principal'
    np2_5 = 0.0
    if row['p2_1b'] == 1 and row['p2_5'] > 0.0:
        np2_5 += row['p2_5']
    elif row['p2_1b'] == 2 and row['p2_5'] > 0.0 and row['p2_1c'] > 0.0:
        np2_5 += row['p2_5'] * (row['p2_1c'] / 100)

    # Value of other real estate properties
    # 'p.2.37. Prop i. Porcentaje de la propiedad que pertenece al hogar'
    # 'p.2.39. Prop i. Valor actual de la propiedad'
    # 'p.2.33. Numero de otras propiedades inmobiliarias que poseen'
    otraspr = 0.0
    for j in range(1, 4):
        if row['p2_33'] >= j and row['p2_39_{}'.format(j)] > 0 and row['p2_37_{}'.format(j)] > 0:
            otraspr += row['p2_39_{}'.format(j)] * (row['p2_37_{}'.format(j)] / 100)
    if row['p2_33'] >= 4 and row['p2_39_4'] > 0:
        otraspr += row['p2_39_4']

    # Value of business related to self-employment
    # 'p.4.101. El hogar posee algun negocio gestionado por algun miembro del hogar'
    # 'p.4.111. Neg 1. Valor actual del negocio ,descontadas sus deudas pendientes'
    valhog = 0.0
    if row['p4_101'] == 1:
        for j in range(1, 7):
            if row['p4_111_{}'.format(j)] > 0.0:
                valhog += row['p4_111_{}'.format(j)]

    # Return sum of total assets' components: value of main residence + value of other real estate properties + value of
    # jewellery, works of art and antiques + value of business related to self-employment
    if row['p2_84'] > 0.0:
        return np2_5 + otraspr + row['p2_84'] + valhog
    else:
        return np2_5 + otraspr + valhog


def get_financial_assets(row):
    """
    Compute financial assets as sum of: value of accounts and deposits usable for payments, value of listed shares,
    value of unlisted shares and other equity, value of fixed-income securities, value of mutual funds, value of
    portfolios under management, value of home-purchase savings accounts and accounts not usable for payments, value of
    pension schemes, value of life insurances, value of other financial assets, such as business debts towards the
    household
    """

    actfinanc = 0.0

    # Value of accounts and deposits usable for payments
    # 'p.4.7. Cuentas utilizables para pagos, Saldo total actual'
    if row['p4_7_3'] > 0.0:
        actfinanc += row['p4_7_3']

    # Value of listed shares
    # 'p.4.15. Valor de la cartera de acciones cotizadas'
    if row['p4_15'] > 0.0:
        actfinanc += row['p4_15']

    # Value of unlisted shares and other equity
    # 'p.4.24. Valor de la cartera de acciones no cotizadas y participaciones'
    if row['p4_24'] > 0.0:
        actfinanc += row['p4_24']

    # Value of fixed-income securities
    # 'p.4.35. Valor de su cartera en valores de renta fija'
    if row['p4_35'] > 0.0:
        actfinanc += row['p4_35']

    # Value of mutual funds
    # 'p.4.28a. Valor total de la cartera en fondos de inversion'
    # 'p.4.31.i Fondo i. Valor total de su cartera en este fondo'
    # TODO: Decide between specification as maximum declared (True) or as in published definitions (False)
    use_maximum_declared = False
    if use_maximum_declared:
        allf = 0.0
        for j in range(1, 11):
            if row['p4_31_{}'.format(j)] > 0.0:
                allf += row['p4_31_{}'.format(j)]
        allf = max(allf, row['p4_28a'])
        if allf > 0.0:
            actfinanc += allf
    else:
        allf = 0.0
        if row['p4_28'] <= 10:
            for j in range(1, 11):
                if row['p4_31_{}'.format(j)] > 0.0:
                    allf += row['p4_31_{}'.format(j)]
        else:
            allf += row['p4_28a']
        if allf > 0.0:
            actfinanc += allf

    # Value of portfolios under management
    # 'p.4.43. Valor de estos activos adicionales en carteras gestionadas'
    if row['p4_43'] > 0.0:
        actfinanc += row['p4_43']

    # Value of home-purchase savings accounts and accounts not usable for payments
    # 'p.4.3. Poseen cuentas de ahorro vivienda'
    # 'p.4.4. Poseen cuentas o depositos de ahorro que no puedan ser usadas para pagos'
    # 'p.4.7.1 Cuenta ahorro vivienda, Saldo total actual'
    # 'p.4.7.2 Cuentas no utilizables para pagos, Saldo total actual'
    salcuentas = 0.0
    if row['p4_3'] == 1:
        salcuentas = salcuentas + row['p4_7_1']
    if row['p4_4'] == 1:
        salcuentas = salcuentas + row['p4_7_2']
    if salcuentas > 0.0:
        actfinanc += salcuentas

    # Value of pension schemes
    # 'p.5.1. Estan adscritos a algun tipo de plan de pensiones'
    # 'p.5.7.i Plan i. Valor actualizado de su patrimonio en este plan de pensiones'
    valor = 0.0
    if row['p5_1'] == 1:
        for j in range(1, 11):
            if row['p5_7_{}'.format(j)] > 0.0:
                valor += row['p5_7_{}'.format(j)]
    if valor > 0.0:
        actfinanc += valor

    # Value of life insurances
    # 'p.5.13.i Seg i. Modalidad de la poliza'
    # 'p.5.14.i Seg i. Valoracion del seguro'
    valseg = 0
    for j in range(1, 7):
        if (row['p5_13_{}'.format(j)] in [2, 3]) and (row['p5_14_{}'.format(j)] > 0.0):
            valseg += row['p5_14_{}'.format(j)]
    if valseg > 0.0:
        actfinanc += valseg

    # Value of business debts towards the household
    # 'p.4.116.i Neg i. Importe por el que el negocio debe dinero al hogar'
    valdeuhog = 0.0
    for j in range(1, 6):
        if row['p4_116_{}'.format(j)] > 0.0:
            valdeuhog += row['p4_116_{}'.format(j)]

    # Value of other financial assets
    # 'p.4.38. Importe que se le debe al hogar en conjunto'
    odeuhog = 0.0
    if valdeuhog > 0.0:
        odeuhog += valdeuhog
    if row['p4_38'] > 0.0:
        odeuhog += row['p4_38']
    if odeuhog > 0.0:
        actfinanc += odeuhog

    return actfinanc


def get_liquid_financial_assets(row):
    """
    Compute liquid financial assets, that is, those usable for house purchases. This is computed as the sum of: value of
    accounts and deposits usable for payments, value of listed shares, value of fixed-income securities, value of mutual
    funds, value of portfolios under management, value of home-purchase savings accounts and accounts not usable for
    payments. In other words, this includes all financial assets except: unlisted shares, pension schemes, life
    insurances, business debts towards the household and other financial assets.
    """

    actfinanc = 0.0

    # Value of accounts and deposits usable for payments
    # 'p.4.7. Cuentas utilizables para pagos, Saldo total actual'
    if row['p4_7_3'] > 0.0:
        actfinanc += row['p4_7_3']

    # Value of listed shares
    # 'p.4.15. Valor de la cartera de acciones cotizadas'
    if row['p4_15'] > 0.0:
        actfinanc += row['p4_15']

    # Value of unlisted shares and other equity - EXCLUDED

    # Value of fixed-income securities
    # 'p.4.35. Valor de su cartera en valores de renta fija'
    if row['p4_35'] > 0.0:
        actfinanc += row['p4_35']

    # Value of mutual funds
    # 'p.4.28a. Valor total de la cartera en fondos de inversion'
    # 'p.4.31.i Fondo i. Valor total de su cartera en este fondo'
    # TODO: Decide between specification as maximum declared (True) or as in published definitions (False)
    use_maximum_declared = False
    if use_maximum_declared:
        allf = 0.0
        for j in range(1, 11):
            if row['p4_31_{}'.format(j)] > 0.0:
                allf += row['p4_31_{}'.format(j)]
        allf = max(allf, row['p4_28a'])
        if allf > 0.0:
            actfinanc += allf
    else:
        allf = 0.0
        if row['p4_28'] <= 10:
            for j in range(1, 11):
                if row['p4_31_{}'.format(j)] > 0.0:
                    allf += row['p4_31_{}'.format(j)]
        else:
            allf += row['p4_28a']
        if allf > 0.0:
            actfinanc += allf

    # Value of portfolios under management
    # 'p.4.43. Valor de estos activos adicionales en carteras gestionadas'
    if row['p4_43'] > 0.0:
        actfinanc += row['p4_43']

    # Value of home-purchase savings accounts and accounts not usable for payments
    # 'p.4.3. Poseen cuentas de ahorro vivienda'
    # 'p.4.4. Poseen cuentas o depositos de ahorro que no puedan ser usadas para pagos'
    # 'p.4.7.1 Cuenta ahorro vivienda, Saldo total actual'
    # 'p.4.7.2 Cuentas no utilizables para pagos, Saldo total actual'
    salcuentas = 0.0
    if row['p4_3'] == 1:
        salcuentas = salcuentas + row['p4_7_1']
    if row['p4_4'] == 1:
        salcuentas = salcuentas + row['p4_7_2']
    if salcuentas > 0.0:
        actfinanc += salcuentas

    # Value of pension schemes - EXCLUDED

    # Value of life insurances - EXCLUDED

    # Value of business debts towards the household - EXCLUDED

    # Value of other financial assets - EXCLUDED

    return actfinanc


def get_outstanding_secured_debt(row):
    """
    Compute total outstanding secured debt, that is, including all loans with a real asset as guarantee or collateral.
    This is computed as the sum of: outstanding debt from purchase of main residence, outstanding debt from
    purchase of other real estate properties, outstanding debts from mortgages and other loans with real guarantee not
    related to real estate
    """
    # Outstanding debt from purchase of main residence
    # 'p.2.8a. No. de prestamos pendientes por la compra de la vivienda principal'
    # 'p.2.12.i Prest i. Importe pendiente de amortizar del prestamo'
    dvivpral = 0.0
    for j in range(1, min(int(row['p2_8a']) + 1, 5)):
        if row['p2_12_{}'.format(j)]:
            dvivpral += row['p2_12_{}'.format(j)]

    # Outstanding debt from purchase of other real estate properties
    # 'p.2.51.i Prop i. No. de prestamos pendientes para compra de esta propiedad'
    # 'p.2.55.i.j Prop i. Prest j. Importe pendiente de amortizar del pretamo'
    dprop1 = 0.0
    for j in range(1, min(int(row['p2_51_1']) + 1, 4)):
        if row['p2_55_1_{}'.format(j)] > 0.0:
            dprop1 += row['p2_55_1_{}'.format(j)]
    dprop2 = 0.0
    for j in range(1, min(int(row['p2_51_2']) + 1, 4)):
        if row['p2_55_2_{}'.format(j)] > 0.0:
            dprop2 += row['p2_55_2_{}'.format(j)]
    dprop3 = 0.0
    for j in range(1, min(int(row['p2_51_3']) + 1, 4)):
        if row['p2_55_3_{}'.format(j)] > 0.0:
            dprop3 += row['p2_55_3_{}'.format(j)]
    dprop4 = 0.0
    if row['p2_55_4'] > 0.0:
        dprop4 += row['p2_55_4']
    deuoprop = dprop1 + dprop2 + dprop3 + dprop4

    # Outstanding debts from mortgages and other loans with real guarantee not related to real estate
    # 'p.3.2.i Prest i. Tipo de prestamo'
    # 'p.3.6.i Prest i. Importe pendiente de amortizar de este prestamo'
    phipo = 0.0
    for j in range(1, 9):
        if (row['p3_2_{}'.format(j)] in [1, 2, 10]) and (row['p3_6_{}'.format(j)] > 0.0):
            phipo += row['p3_6_{}'.format(j)]

    return dvivpral + deuoprop + phipo


def get_outstanding_unsecured_debt(row):
    """
    Compute total outstanding unsecured debt, that is, including all loans with no real asset as guarantee or
    collateral. This is computed as the sum of: outstanding debts from personal loans, outstanding credit card balances
    and other outstanding debts
    """
    # Outstanding debts from personal loans
    # 'p.3.2.i Prest i. Tipo de prestamo'
    # 'p.3.6.i Prest i. Importe pendiente de amortizar de este prestamo'
    pperso = 0.0
    for j in range(1, 9):
        if (row['p3_2_{}'.format(j)] == 3) and (row['p3_6_{}'.format(j)] > 0.0):
            pperso += row['p3_6_{}'.format(j)]

    # Outstanding credit card balances
    # 'p.8.5a. Importe que deben actualmente de deudas contraidas con tarj. de credito'
    ptmos_tarj = 0.0
    if row['p8_5a'] > 0.0:
        ptmos_tarj += row['p8_5a']

    # Other outstanding debts
    # 'p.3.2.i Prest i. Tipo de prestamo'
    # 'p.3.6.i Prest i. Importe pendiente de amortizar de este prestamo'
    potrasd = 0.0
    for j in range(1, 9):
        if (row['p3_2_{}'.format(j)] in [4, 5, 6, 7, 8, 9, 97]) and (row['p3_6_{}'.format(j)] > 0.0):
            potrasd += row['p3_6_{}'.format(j)]

    return pperso + ptmos_tarj + potrasd


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
wealthMeasure = 'Gross'  # Available wealth measures: Gross, Net and Liq
incomeBinWidth = 0.2
wealthBinWidth = 0.5
multipleImputations = True
excludeHouseSaleProfits = True  # TODO: Decide whether to remove also house sale profits, as endogenous too
writeResults = False
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 1000)

# Read data
rootWAS = r''
dfWAS = pd.read_csv(rootWAS + r'/was_wave_3_hhold_eul_final.dta',
                    usecols={"w3xswgt", "DVTotGIRw3", "DVGrsRentAmtAnnualw3_aggr", "HFINWW3_sum", "HFINWNTW3_sum",
                             "DVFNSValW3_aggr", "DVCACTvW3_aggr", "DVCASVVW3_aggr", "DVSaValW3_aggr", "DVCISAVW3_aggr",
                             "DVCaCrValW3_aggr"})
rootEFF = r''
if multipleImputations:
    dfEFF = pd.DataFrame()
    for i in range(1, 6):
        temp_df = pd.read_csv(rootEFF + r'/eff_2017_imp{0}_csv/otras_secciones_2017_imp{0}.csv'.format(i), sep=';')
        dfEFF = pd.concat([dfEFF, temp_df])
    dfEFF['facine3'] = dfEFF['facine3'] / 5.0
else:
    dfEFF = pd.read_csv(rootEFF + r'/eff_2017_imp1_csv/otras_secciones_2017_imp1.csv', sep=';',
                        usecols={'facine3', 'p1_2d_1', 'p7_2', 'p7_4a', 'renthog'})
rootResults = r''

# EFF variables of interest
# facine3                       Household weight
# renthog                       Household Gross Annual income
# p7_2                          Importe recibido por alquileres durante el ano 2016
# p7_4a                         Plusvalias por venta de prop. inmob. durante el ano 2016

# WAS variables of interest
# HFINWNTW3_sum               Household Net financial Wealth (financial assets minus financial liabilities)
# HFINWW3_sum                 Gross Financial Wealth (financial assets only )
# DVTotGIRw3                  Household Gross Annual (regular) income
# DVGrsRentAmtAnnualw3_aggr   Household Gross annual income from rent

# Rename columns to be used
dfWAS.rename(columns={'w3xswgt': 'Weight'}, inplace=True)
dfEFF.rename(columns={'facine3': 'Weight'}, inplace=True)

# Replace NaNs by zeros in EFF data
dfEFF = dfEFF.fillna(0)

# Compute annual gross non-rental income
dfWAS['GrossNonRentIncome'] = dfWAS['DVTotGIRw3'] - dfWAS['DVGrsRentAmtAnnualw3_aggr']
if not excludeHouseSaleProfits:
    dfEFF['GrossNonRentIncome'] = dfEFF['renthog'] - dfEFF['p7_2']
else:
    dfEFF['GrossNonRentIncome'] = dfEFF['renthog'] - dfEFF['p7_2'] - dfEFF['p7_4a']

# Compute selected wealth measure and filter frame down to variables of interest
if wealthMeasure == 'Gross':
    dfWAS.rename(columns={'HFINWW3_sum': 'GrossFinancialWealth'}, inplace=True)
    dfEFF['GrossFinancialWealth'] = dfEFF.apply(lambda row: get_financial_assets(row), axis=1)
    dfWAS = dfWAS[['GrossNonRentIncome', 'GrossFinancialWealth', 'Weight']]
    dfEFF = dfEFF[['GrossNonRentIncome', 'GrossFinancialWealth', 'Weight']]
elif wealthMeasure == 'Net':
    dfWAS.rename(columns={'HFINWNTW3_sum': 'NetFinancialWealth'}, inplace=True)
    dfEFF['NetFinancialWealth'] = dfEFF.apply(
        lambda row: get_financial_assets(row) - get_outstanding_unsecured_debt(row), axis=1)
    dfWAS = dfWAS[['GrossNonRentIncome', 'NetFinancialWealth', 'Weight']]
    dfEFF = dfEFF[['GrossNonRentIncome', 'NetFinancialWealth', 'Weight']]
elif wealthMeasure == 'Liq':
    dfWAS['LiqFinancialWealth'] = dfWAS.apply(lambda row: get_liquid_financial_wealth_from_was(row), axis=1)
    dfEFF['LiqFinancialWealth'] = dfEFF.apply(lambda row: get_liquid_financial_assets(row), axis=1)
    dfWAS = dfWAS[['GrossNonRentIncome', 'LiqFinancialWealth', 'Weight']]
    dfEFF = dfEFF[['GrossNonRentIncome', 'LiqFinancialWealth', 'Weight']]
# dfEFF['GrossRealWealth'] = dfEFF.apply(lambda row: get_real_assets(row), axis=1)
# dfEFF['NetRealWealth'] = dfEFF.apply(
#     lambda row: row['GrossRealWealth'] - get_outstanding_secured_debt(row), axis=1)
# dfEFF['GrossTotalWealth'] = dfEFF.apply(lambda row: row['GrossRealWealth'] + row['GrossFinancialWealth'], axis=1)
# dfEFF['NetTotalWealth'] = dfEFF.apply(lambda row: row['NetRealWealth'] + row['NetFinancialWealth'], axis=1)

# Filter out NaNs, negative values and zeros
dfEFF = dfEFF.loc[(dfEFF['GrossNonRentIncome'] > 0.0)]

# Filter out the 1% with highest and the 1% with lowest GrossNonRentIncome
minIncomeWAS = weighted_quantile(dfWAS['GrossNonRentIncome'], 0.01, sample_weight=dfWAS['Weight'])
maxIncomeWAS = weighted_quantile(dfWAS['GrossNonRentIncome'], 0.99, sample_weight=dfWAS['Weight'])
minIncomeEFF = weighted_quantile(dfEFF['GrossNonRentIncome'], 0.01, sample_weight=dfEFF['Weight'])
maxIncomeEFF = weighted_quantile(dfEFF['GrossNonRentIncome'], 0.99, sample_weight=dfEFF['Weight'])
dfWAS = dfWAS.loc[(dfWAS['GrossNonRentIncome'] >= minIncomeWAS) & (dfWAS['GrossNonRentIncome'] <= maxIncomeWAS)]
dfEFF = dfEFF.loc[(dfEFF['GrossNonRentIncome'] >= minIncomeEFF) & (dfEFF['GrossNonRentIncome'] <= maxIncomeEFF)]

# Filter out negative values of the selected wealth measure, so as to be able to use logarithmic methods
dfWAS = dfWAS.loc[(dfWAS[wealthMeasure + 'FinancialWealth'] > 0.0)]
dfEFF = dfEFF.loc[(dfEFF[wealthMeasure + 'FinancialWealth'] > 0.0)]

# Filter out the 0.01% with highest selected wealth measure, so as to avoid outliers driving determination of bin edges
maxWealthWAS = weighted_quantile(dfWAS[wealthMeasure + 'FinancialWealth'], 0.9999, sample_weight=dfWAS['Weight'])
maxWealthEFF = weighted_quantile(dfEFF[wealthMeasure + 'FinancialWealth'], 0.9999, sample_weight=dfEFF['Weight'])
dfWAS = dfWAS.loc[(dfWAS[wealthMeasure + 'FinancialWealth'] <= maxWealthWAS)]
dfEFF = dfEFF.loc[(dfEFF[wealthMeasure + 'FinancialWealth'] <= maxWealthEFF)]

# Define income bin edges
incMinBinEdgeWAS = np.round(np.floor(np.log(minIncomeWAS) / incomeBinWidth) * incomeBinWidth, 5)
incMaxBinEdgeWAS = np.round(np.ceil(np.log(maxIncomeWAS) / incomeBinWidth) * incomeBinWidth, 5)
incMinBinEdgeEFF = np.round(np.floor(np.log(minIncomeEFF) / incomeBinWidth) * incomeBinWidth, 5)
incMaxBinEdgeEFF = np.round(np.ceil(np.log(maxIncomeEFF) / incomeBinWidth) * incomeBinWidth, 5)
incBinEdgesWAS = np.round(np.linspace(incMinBinEdgeWAS, incMaxBinEdgeWAS,
                                      int((incMaxBinEdgeWAS - incMinBinEdgeWAS) / incomeBinWidth) + 1,
                                      endpoint=True), 5)
incBinEdgesEFF = np.round(np.linspace(incMinBinEdgeEFF, incMaxBinEdgeEFF,
                                      int((incMaxBinEdgeEFF - incMinBinEdgeEFF) / incomeBinWidth) + 1,
                                      endpoint=True), 5)

# Define wealth bin edges
wMinBinEdgeWAS = 0.0
wMaxBinEdgeWAS = np.round(np.ceil(np.log(maxWealthWAS) / wealthBinWidth) * wealthBinWidth, 5)
wMinBinEdgeEFF = 0.0
wMaxBinEdgeEFF = np.round(np.ceil(np.log(maxWealthEFF) / wealthBinWidth) * wealthBinWidth, 5)
wBinEdgesWAS = np.round(np.linspace(wMinBinEdgeWAS, wMaxBinEdgeWAS,
                                    int((wMaxBinEdgeWAS - wMinBinEdgeWAS) / wealthBinWidth) + 1, endpoint=True), 5)
wBinEdgesEFF = np.round(np.linspace(wMinBinEdgeEFF, wMaxBinEdgeEFF,
                                    int((wMaxBinEdgeEFF - wMinBinEdgeEFF) / wealthBinWidth) + 1, endpoint=True), 5)

if not writeResults:
    # If not writing to file, then plot wealth distribution
    plt.hist(np.log(dfWAS[wealthMeasure + 'FinancialWealth']), bins=wBinEdgesWAS, density=True,
             weights=dfWAS['Weight'], alpha=0.5, label='WAS')
    plt.hist(np.log(dfEFF[wealthMeasure + 'FinancialWealth']), bins=wBinEdgesEFF, density=True,
             weights=dfEFF['Weight'], alpha=0.5, label='EFF')
    plt.legend()
    plt.show()
else:
    # If writing to file, create a 2D histogram of the data with logarithmic income and wealth bins (no normalisation
    # here as we want column normalisation, to be introduced when plotting or printing)
    frequencyWAS = np.histogram2d(np.log(dfWAS['GrossNonRentIncome']), np.log(dfWAS[wealthMeasure + "FinancialWealth"]),
                                  bins=[incBinEdgesWAS, wBinEdgesWAS], normed=True, weights=dfWAS['Weight'])[0]
    frequencyEFF = np.histogram2d(np.log(dfEFF['GrossNonRentIncome']), np.log(dfEFF[wealthMeasure + "FinancialWealth"]),
                                  bins=[incBinEdgesEFF, wBinEdgesEFF], normed=True, weights=dfEFF['Weight'])[0]

    # Print joint distributions to files
    with open(rootResults + '/WAS-GrossIncome{}WealthJointDist.csv'.format(wealthMeasure), 'w') as f:
        f.write('# Log Gross Income (lower edge), Log Gross Income (upper edge), Log {0} Wealth (lower edge), '
                'Log {0} Wealth (upper edge), Probability\n'.format(wealthMeasure))
        for line, incomeLowerEdge, incomeUpperEdge in zip(frequencyWAS, incBinEdgesWAS[:-1], incBinEdgesWAS[1:]):
            for element, wealthLowerEdge, wealthUpperEdge in zip(line, wBinEdgesWAS[:-1], wBinEdgesWAS[1:]):
                f.write("{}, {}, {}, {}, {}\n".format(incomeLowerEdge, incomeUpperEdge, wealthLowerEdge,
                                                      wealthUpperEdge, element / sum(line)))
    with open(rootResults + '/EFF-GrossIncome{}WealthJointDist.csv'.format(wealthMeasure), 'w') as f:
        f.write('# Log Gross Income (lower edge), Log Gross Income (upper edge), Log {0} Wealth (lower edge), '
                'Log {0} Wealth (upper edge), Probability\n'.format(wealthMeasure))
        for line, incomeLowerEdge, incomeUpperEdge in zip(frequencyEFF, incBinEdgesEFF[:-1], incBinEdgesEFF[1:]):
            for element, wealthLowerEdge, wealthUpperEdge in zip(line, wBinEdgesEFF[:-1], wBinEdgesEFF[1:]):
                f.write("{}, {}, {}, {}, {}\n".format(incomeLowerEdge, incomeUpperEdge, wealthLowerEdge,
                                                      wealthUpperEdge, element / sum(line)))
