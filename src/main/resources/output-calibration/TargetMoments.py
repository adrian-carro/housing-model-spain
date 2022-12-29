# coding: utf-8

# Imports
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import pandas as pd
from scipy.fftpack import fft


def get_period(_df, _time_field, _values_field):
    # Prepare time series and sample spacing
    _time_series = _df[_values_field].values
    _sample_spacing = _df.iloc[1][_time_field] - _df.iloc[0][_time_field]
    # Find period
    n = len(_time_series)  # Number of sample points
    fast_fourier_transform_hpi = (1.0 / n) * np.abs(fft(_time_series)[1:int(n / 2)])
    frequency_domain = np.linspace(0.0, 1.0 / _sample_spacing, n)[1:int(n / 2)]
    return 1 / frequency_domain[fast_fourier_transform_hpi.argmax()]


def plot_cycles(_df_data, _std_data, _period_data, _df_model, _std_model, _period_model):
    shift = 1000

    # noinspection PyTypeChecker
    plt.figure(figsize=(5, 4))

    # Plot axis 2
    plt.axhline(100, c='k', ls='--')
    plt.plot(_df_data['Time'], 100.0 + _df_data['HPI Long Trend'], '-o', c='green', lw=1.5, ms=4.0,
             label=r'Data (std = {:.1f}, $\tau$ = {:d})'.format(_std_data, int(_period_data)))
    plt.plot([b - shift for b in _df_model[1::3]['Time']],
             [np.mean([a, b, c]) for a, b, c in zip(_df_model[0::3]['HPI'], _df_model[1::3]['HPI'],
                                                    _df_model[2::3]['HPI'])],
             '-o', c='red', lw=1.5, ms=4.0,
             label=r'Model (std = {:.1f}, $\tau$ = {:.1f})'.format(stdHPIModel, periodHPIModel))

    # Figure and axes settings
    plt.xlim(0, 603)
    plt.xlabel('Time (months)')
    plt.ylabel('HPI')
    plt.legend(loc='upper left')


def plot_spectral_density(series, sample_spacing):
    n = len(series)  # Number of sample points
    t = sample_spacing  # Sample spacing
    fast_fourier_transform = fft(series)
    frequency_domain = np.linspace(0.0, 1.0 / t, n)
    _x = frequency_domain[1:int(n / 2)]
    _y = 1.0 / n * np.abs(fast_fourier_transform[1:int(n / 2)])
    plt.figure()
    plt.plot(_x, _y, 'o-')


# Set parameters
writeResults = False
timeStepsToDiscard = 1000
nRuns = 100
rootOECD = r''
rootModelSP = r''
rootResults = r''

# Read OECD data
dfOECD = pd.read_csv(rootOECD + '/HOUSE_PRICES_ES_12122021141313619.csv',
                     skipinitialspace=True, usecols=['IND', 'Value'])
dfOECD = dfOECD.loc[dfOECD['IND'] == 'RHP', ['Value']]
dfOECD['HPI'] = dfOECD['Value'] / dfOECD.iloc[0]['Value']
dfOECD['Time'] = range(0, 3 * len(dfOECD), 3)
dfOECD = dfOECD[['Time', 'HPI']]

# De-trend data
dfOECD['HPI Short Trend'] = sm.tsa.filters.hpfilter(np.array(dfOECD['HPI']), lamb=1600)[0]
dfOECD['HPI Long Trend'] = sm.tsa.filters.hpfilter(np.array(dfOECD['HPI']), lamb=129600)[0]
stdHPIOECD = np.std(dfOECD['HPI Long Trend'])
periodHPIOECD = get_period(dfOECD, _time_field='Time', _values_field='HPI Long Trend')

# Read model results
meanHPIModel = []
stdHPIModel = []
periodHPIModel = []
meanRPIModel = []
shareOwnersModel = []
shareRentingModel = []
shareBTLModel = []
rentalYieldModel = []
spreadModel = []
for i in range(1, nRuns + 1):
    dfModel = pd.read_csv(rootModelSP + '/Output-run{}.csv'.format(i), skipinitialspace=True, delimiter=';',
                          usecols=['Model time', 'Sale HPI', 'Rental HPI', 'nRenting', 'nOwnerOccupier', 'nActiveBTL',
                                   'TotalPopulation', 'Rental ExpAvFlowYield', 'interestRate'])
    dfModel.rename(columns={'Model time': 'Time', 'Sale HPI': 'HPI', 'Rental HPI': 'RPI'}, inplace=True)
    dfModel = dfModel.loc[dfModel['Time'] >= timeStepsToDiscard]
    # Compute moments
    meanHPIModel.append(np.mean(dfModel['HPI']))
    stdHPIModel.append(np.std(dfModel['HPI']))
    periodHPIModel.append(get_period(dfModel, _time_field='Time', _values_field='HPI'))
    meanRPIModel.append(np.mean(dfModel['RPI']))
    shareOwnersModel.append(np.mean((dfModel['nOwnerOccupier'] + dfModel['nActiveBTL']) / dfModel['TotalPopulation']))
    shareRentingModel.append(np.mean(dfModel['nRenting'] / dfModel['TotalPopulation']))
    shareBTLModel.append(np.mean(dfModel['nActiveBTL'] / dfModel['TotalPopulation']))
    rentalYieldModel.append(np.mean(dfModel['Rental ExpAvFlowYield']))
    spreadModel.append(np.mean(dfModel['interestRate']) - 0.000102459)

if writeResults:
    with open(rootResults + '/TargetMoments.txt', 'w') as f:
        f.write('----------------------------------------------------------------------------------\n')
        f.write('| Moment \t\t\t\t | Data value \t\t\t\t | Model value \t\t\t\t |\n')
        f.write('----------------------------------------------------------------------------------\n')
        f.write('| HPI mean \t\t\t\t | {:.6f} \t\t\t\t | {:.6f} (\u00B1{:.6f}) \t |\n'.format(
            1.0, np.mean(meanHPIModel), np.std(meanHPIModel)))
        f.write('| Std. HPI \t\t\t\t | {:.6f} \t\t\t\t | {:.6f} (\u00B1{:.6f}) \t |\n'.format(
            stdHPIOECD, np.mean(stdHPIModel), np.std(stdHPIModel)))
        f.write('| Period HPI \t\t\t | {:.6f} \t\t\t\t | {:.6f} (\u00B1{:.6f}) \t |\n'.format(
            periodHPIOECD, np.mean(periodHPIModel), np.std(periodHPIModel)))
        f.write('| RPI mean \t\t\t\t | {:.6f} \t\t\t\t | {:.6f} (\u00B1{:.6f}) \t |\n'.format(
            1.0, np.mean(meanRPIModel), np.std(meanRPIModel)))
        f.write('| Share owning \t\t\t | {:.6f} \t\t\t\t | {:.6f} (\u00B1{:.6f}) \t |\n'.format(
            0.77531905, np.mean(shareOwnersModel), np.std(shareOwnersModel)))
        f.write('| Share renting \t\t | {:.6f} \t\t\t\t | {:.6f} (\u00B1{:.6f}) \t |\n'.format(
            0.17334471, np.mean(shareRentingModel), np.std(shareRentingModel)))
        f.write('| Share BTL investing \t | {:.6f} \t\t\t\t | {:.6f} (\u00B1{:.6f}) \t |\n'.format(
            0.07809251, np.mean(shareBTLModel), np.std(shareBTLModel)))
        f.write('| Rental yield mean \t | {:.6f} \t\t\t\t | {:.6f} (\u00B1{:.6f}) \t |\n'.format(
            0.044175, np.mean(rentalYieldModel), np.std(rentalYieldModel)))
        f.write('| Interest spread mean \t | {:.6f} \t\t\t\t | {:.6f} (\u00B1{:.6f}) \t |\n'.format(
            0.023536, np.mean(spreadModel), np.std(spreadModel)))
        f.write('----------------------------------------------------------------------------------\n')
else:
    print('----------------------------------------------------------------------------------')
    print('| Moment \t\t\t\t | Data value \t\t\t\t | Model value \t\t\t\t |')
    print('----------------------------------------------------------------------------------')
    print('| HPI mean \t\t\t\t | {:.6f} \t\t\t\t | {:.6f} (\u00B1{:.6f}) \t |'.format(
        1.0, np.mean(meanHPIModel), np.std(meanHPIModel)))
    print('| Std. HPI \t\t\t\t | {:.6f} \t\t\t\t | {:.6f} (\u00B1{:.6f}) \t |'.format(
        stdHPIOECD, np.mean(stdHPIModel), np.std(stdHPIModel)))
    print('| Period HPI \t\t\t | {:.6f} \t\t\t\t | {:.6f} (\u00B1{:.6f}) \t |'.format(
        periodHPIOECD, np.mean(periodHPIModel), np.std(periodHPIModel)))
    print('| RPI mean \t\t\t\t | {:.6f} \t\t\t\t | {:.6f} (\u00B1{:.6f}) \t |'.format(
        1.0, np.mean(meanRPIModel), np.std(meanRPIModel)))
    print('| Share owning \t\t\t | {:.6f} \t\t\t\t | {:.6f} (\u00B1{:.6f}) \t |'.format(
        0.77531905, np.mean(shareOwnersModel), np.std(shareOwnersModel)))
    print('| Share renting \t\t | {:.6f} \t\t\t\t | {:.6f} (\u00B1{:.6f}) \t |'.format(
        0.17334471, np.mean(shareRentingModel), np.std(shareRentingModel)))
    print('| Share BTL investing \t | {:.6f} \t\t\t\t | {:.6f} (\u00B1{:.6f}) \t |'.format(
        0.07809251, np.mean(shareBTLModel), np.std(shareBTLModel)))
    print('| Rental yield mean \t | {:.6f} \t\t\t\t | {:.6f} (\u00B1{:.6f}) \t |'.format(
        0.044175, np.mean(rentalYieldModel), np.std(rentalYieldModel)))
    print('| Interest spread mean \t | {:.6f} \t\t\t\t | {:.6f} (\u00B1{:.6f}) \t |'.format(
        0.023536, np.mean(spreadModel), np.std(spreadModel)))
    print('----------------------------------------------------------------------------------')
