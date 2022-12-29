# coding: utf-8

# Imports
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import pandas as pd
from scipy.fftpack import fft


def plot_cycles_and_de_trending(save_figure, _data_df, _root_results):
    # Open figure
    # noinspection PyTypeChecker
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 10))

    # Plot axis 1
    # ax1.plot(data_time_BoE, data_HPI_BoE, '-o', lw=1.5, label='Data BoE')
    ax1.plot(_data_df['Time'], _data_df['HPI'], '-o', c='blue', lw=1.5, ms=4.0, label='Data OECD')
    # ax1.plot(data_time_OECD, data_HPI_OECD_de_trend_quad, '-o', lw=1.5, ms=4.0,
    #          label='Data OECD (quadtratic de-trending)')
    ax1.plot(_data_df['Time'], _data_df['HPI Trend (short)'], '-o', c='orange', lw=1.5, ms=4.0,
             label=r'Data OECD (HP de-trending, short trend, $\lambda = 1600$)')
    ax1.plot(_data_df['Time'], _data_df['HPI Trend'], '-o', c='green', lw=1.5, ms=4.0,
             label=r'Data OECD (HP de-trending, long trend, $\lambda = 129600$)')

    # Plot axis 2
    ax2.axhline(100, c='k', ls='--')
    ax2.plot(_data_df['Time'], _data_df['De-trended HPI'], '-o', c='green', lw=1.5, ms=4.0,
             label='Data OECD (HP de-trending, long cycle, $\lambda = 129600$)')
    ax2.plot(_data_df['Time'],
             [100.0 + a - b for a, b in zip(_data_df['HPI Trend (short)'], _data_df['HPI Trend'])],
             '-o', c='red', lw=1.5, ms=4.0, label='Data OECD (HP de-trending, short - long trend)')
    ax2.plot([b - modelShift for a, b, c in zip(model_time[0::3], model_time[1::3], model_time[2::3])],
             [np.mean([a, b, c]) for a, b, c in zip(model_HPI[0::3], model_HPI[1::3], model_HPI[2::3])],
             '-o', c='purple', lw=1.5, ms=4.0, label='Model')

    # Figure and axes settings
    plt.xlim(0, 603)
    ax1.tick_params(axis='x', direction='in', pad=-22)
    ax2.set_xlabel('Time (months)')
    ax1.set_ylabel('HPI')
    ax2.set_ylabel('HPI')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    plt.tight_layout(h_pad=0)

    # If required, save figure
    if save_figure:
        plt.savefig(_root_results + '/HousePriceIndex_DeTrending.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(_root_results + '/HousePriceIndex_DeTrending.png', format='png', dpi=300, bbox_inches='tight')


def get_period(_time_series):
    n = len(_time_series)  # Number of sample points
    fast_fourier_transform_hpi = (1.0 / n) * np.abs(fft(_time_series)[1:int(n / 2)])
    frequency_domain = np.linspace(0.0, 1.0, n)[1:int(n / 2)]  # This assumes sample spacing of 1
    return 1 / frequency_domain[fast_fourier_transform_hpi.argmax()]


def plot_cycles(save_figure, _data_df, _root_results, _mean_std_model, _mean_period_model, _with_model):
    # noinspection PyTypeChecker
    plt.figure(figsize=(5.0, 3.5))
    if _with_model:
        # plt.plot(_data_df['Time'], _data_df['De-trended HPI'], '-o', c='green', lw=1.5, ms=4.0,
        #          label=r'Data (std = {:.2f}, $\tau$ = {:.2f})'.format(std_data, period_data))
        plt.plot(range(0, 3 * len(_data_df), 3), _data_df['RescaledDeTrendedValue'], '-', c='green', lw=1.5, ms=4.0,
                 label=r'Data Spain')
        # plt.plot([b - modelShift for a, b, c in zip(model_time[0::3], model_time[1::3], model_time[2::3])],
        #          [np.mean([a, b, c]) for a, b, c in zip(model_HPI[0::3], model_HPI[1::3], model_HPI[2::3])],
        #          '-o', c='red', lw=1.5, ms=4.0,
        #          label=r'Model (std = {:.2f}, $\tau$ = {:.2f})'.format(_mean_std_model, _mean_period_model))
        plt.plot([b - modelShift for a, b, c in zip(model_time[0::3], model_time[1::3], model_time[2::3])],
                 [np.mean([a, b, c]) - 100.0 for a, b, c in zip(model_HPI[0::3], model_HPI[1::3], model_HPI[2::3])],
                 '-', c='red', lw=1.5, ms=4.0, label=r'Model Spain')
    else:
        # _data_df.plot(ax=plt.gca(), x='Date', y='De-trended HPI', marker='o', ls='-', c='tab:green', lw=1.5, ms=4.0,
        #               label=r'Data (std = {:.2f}, $\tau$ = {:.2f})'.format(std_data, period_data))
        plt.plot(_data_df['TIME'], _data_df['RescaledDeTrendedValue'], '-o', c='green', lw=1.5, ms=4.0,
                 label=r'Data')
    plt.axhline(0, c='k', ls='--', zorder=100)

    # Figure and axes settings
    if _with_model:
        plt.xlim(0, 573)
    plt.ylim(-75, 125)
    if _with_model:
        plt.xlabel('Time (months)')
    else:
        plt.xlabel('Date')
    plt.ylabel('HPI (Cyclical Component)')
    plt.legend(loc='upper left', frameon=False)

    # If required, save figure
    if save_figure:
        plt.savefig(_root_results + '/HousePriceIndex-SP{}.png'.format('-withModel' if _with_model else ''),
                    format='png', dpi=300, bbox_inches='tight')
        plt.savefig(_root_results + '/HousePriceIndex-SP{}.pdf'.format('-withModel' if _with_model else ''),
                    format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(_root_results + '/HousePriceIndex-SP{}.eps'.format('-withModel' if _with_model else ''),
                    format='eps', dpi=300, bbox_inches='tight')


def plot_spectral_density(series, sample_spacing):
    n = len(series)  # Number of sample points
    t = sample_spacing  # Sample spacing
    fast_fourier_transform = fft(series)
    frequency_domain = np.linspace(0.0, 1.0 / t, n)
    x = frequency_domain[1:int(n / 2)]
    y = 1.0 / n * np.abs(fast_fourier_transform[1:int(n / 2)])
    plt.figure()
    plt.plot(x, y, 'o-')


def plot_uk_cycles(_root_oecd_uk, _root_model, _root_results, _save_figure, _model_shift_uk, _with_model):
    # Read only time and HPI from model results files
    model_df_uk = pd.read_csv(_root_model + '/Output-run1.csv', skipinitialspace=True, delimiter=';',
                              usecols=['Model time', 'Sale HPI'])
    model_hpi_uk = list(100 * np.array(model_df_uk.loc[model_df_uk['Model time'] >= 1000, 'Sale HPI']))
    model_time_uk = range(0, len(model_hpi_uk))
    model_hpi_uk = model_hpi_uk[166 + _model_shift_uk:270 + _model_shift_uk] + model_hpi_uk[:-50 + _model_shift_uk]\
        + model_hpi_uk[80 + _model_shift_uk:166 + _model_shift_uk]\
        + model_hpi_uk[-50 + _model_shift_uk:80 + _model_shift_uk] + model_hpi_uk[270 + _model_shift_uk:]

    # Read time and HPI from OECD data for UK
    # data_df_uk = pd.read_csv(_root_oecd_uk + '/HPI - UK HPI data (Ruben).csv', header=None, skiprows=1,
    #                          skipinitialspace=True, usecols=[0, 3], parse_dates=[0], dayfirst=True)
    # data_df_uk.rename({0: 'Date', 3: 'HPI'}, inplace=True, axis=1)
    # data_df_uk['Time'] = range(0, 3 * len(data_df_uk), 3)

    # De-trend data
    # data_df_uk['De-trended HPI'] = sm.tsa.filters.hpfilter(np.array(data_df_uk['HPI']), lamb=129600)[0] + 100.0

    # Print information to screen: standard deviation
    # std_data_uk = np.std(data_df_uk['De-trended HPI'])
    # std_model_uk = np.std(model_hpi_uk)

    # /////////
    # Read full OECD data
    data_df_uk = pd.read_csv(rootOECD + '/OECD_Full_Data.csv', usecols=['LOCATION', 'TIME', 'Value'],
                             parse_dates=['TIME'])
    # Select UK
    data_df_uk = data_df_uk.loc[data_df_uk['LOCATION'] == 'GBR']
    # Select time period
    _start_date = pd.to_datetime('1973-01-01')
    _end_date = pd.to_datetime('2020-10-01')
    data_df_uk = data_df_uk.loc[data_df_uk['TIME'] >= _start_date].copy()
    data_df_uk = data_df_uk.loc[data_df_uk['TIME'] <= _end_date]
    # Compute rescaled variables
    data_df_uk['RescaledValue'] = 100.0 * data_df_uk['Value'] / data_df_uk.loc[data_df_uk['TIME'] == start_date,
                                                                               'Value'].values[0]
    _hp_filter_rescaled = sm.tsa.filters.hpfilter(np.array(data_df_uk['RescaledValue'].values), lamb=129600)
    data_df_uk['RescaledDeTrendedValue'] = _hp_filter_rescaled[0]
    data_df_uk['RescaledTrendOfValue'] = _hp_filter_rescaled[1]
    # /////////

    # Print information to screen: period of cycles
    _n_samples = len(data_df_uk['RescaledDeTrendedValue'])  # Number of sample points
    _sample_spacing = 3.0  # Sample spacing
    fast_fourier_transform_hpi_uk = fft(data_df_uk['RescaledDeTrendedValue'].tolist())
    frequency_domain = np.linspace(0.0, 1.0 / _sample_spacing, _n_samples)
    _x = frequency_domain[1:70]
    _y = 1.0 / _n_samples * np.abs(fast_fourier_transform_hpi_uk[1:70])
    period_data_uk = 1 / _x[_y.argmax()]
    period_model_uk = get_period(model_hpi_uk)

    # Plot
    plt.figure(figsize=(5.5, 4))
    if _with_model:
        plt.plot(range(0, 3 * len(data_df_uk), 3), data_df_uk['RescaledDeTrendedValue'], '-', c='green', lw=1.5,
                 ms=4.0, label=r'Data UK')
        plt.plot([b - _model_shift_uk for a, b, c in zip(model_time_uk[0::3], model_time_uk[1::3],
                                                         model_time_uk[2::3])],
                 [np.mean([a, b, c]) - 100.0 for a, b, c in zip(model_hpi_uk[0::3], model_hpi_uk[1::3],
                                                                model_hpi_uk[2::3])],
                 '-', c='red', lw=1.5, ms=4.0,
                 # label=r'Model (std = {:.2f}, $\tau$ = {:.2f})'.format(std_model_uk, period_model_uk))
                 label=r'Model')
    else:
        data_df_uk.plot(ax=plt.gca(), x='TIME', y='RescaledDeTrendedValue', ls='-', c='tab:green', lw=1.5,
                        ms=4.0, label=r'Data UK')
    plt.axhline(0, c='k', ls='--', zorder=100)

    # Figure and axes settings
    if _with_model:
        plt.xlim(0, 573)
    plt.ylim(-75, 125)
    if _with_model:
        plt.xlabel('Time (months)')
    else:
        plt.xlabel('Date')
    plt.ylabel('HPI (Cyclical Component)')
    plt.legend(loc='upper left')

    # If required, save figure
    if _save_figure:
        plt.savefig(_root_results + '/HousePriceIndex-UK{}.pdf'.format('-withModel' if _with_model else ''),
                    format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(_root_results + '/HousePriceIndex-UK{}.png'.format('-withModel' if _with_model else ''),
                    format='png', dpi=300, bbox_inches='tight')


def plot_data_cycle_for_both_countries(_data_df, _root_oecd_uk, _root_results, _save_figures):
    # Read time and HPI from OECD data for UK
    _data_df_uk = pd.read_csv(_root_oecd_uk + '/HPI - UK HPI data (Ruben).csv', header=None, skiprows=1,
                              skipinitialspace=True, usecols=[0, 3], parse_dates=[0], dayfirst=True)
    _data_df_uk.rename({0: 'Date', 3: 'HPI'}, inplace=True, axis=1)

    # De-trend data
    _data_df_uk['De-trended HPI'] = sm.tsa.filters.hpfilter(np.array(_data_df_uk['HPI']), lamb=129600)[0] + 100.0

    # Plot
    plt.figure(figsize=(5.5, 4))
    _data_df_uk.plot(ax=plt.gca(), x='Date', y='De-trended HPI', marker='o', ls='-', c='tab:blue', lw=1.5, ms=4.0,
                     label='UK')
    _data_df.plot(ax=plt.gca(), x='Date', y='De-trended HPI', marker='o', ls='-', c='tab:orange', lw=1.5, ms=4.0,
                  label='Spain')
    # _data_df_uk.plot(ax=plt.gca(), x='Date', y='HPI', marker='o', ls='-', c='tab:blue', lw=1.5, ms=4.0,
    #                 label='UK')
    # data_df.plot(ax=plt.gca(), x='Date', y='HPI', marker='o', ls='-', c='tab:orange', lw=1.5, ms=4.0,
    #              label='SP')

    plt.axhline(100, c='k', ls='--', zorder=100)
    plt.ylabel('HPI')
    plt.legend(loc='upper left')
    if _save_figures:
        plt.savefig(_root_results + '/HousePriceIndex-Combined.png', format='png', dpi=300, bbox_inches='tight')
    else:
        plt.show()
        exit()


# Set parameters
modelShift = 1570  # For benchmark2
modelShiftUK = 350
saveFigures = False
withModel = True
nRuns = 10
timeStepsToDiscard = 1000
rootModel = ''
rootOECD = ''
rootResults = ''
rootModelUK = ''
rootOECDUK = ''

# General settings
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'sans',
    'font.sans-serif': ['DejaVu Sans']
})

# Adapt SP shift taking into account initial time steps to discard
modelShift = modelShift - timeStepsToDiscard

# Read time and HPI from OECD data
data_df = pd.read_csv(rootOECD + '/HOUSE_PRICES_ES_12122021141313619.csv',
                      skipinitialspace=True, usecols=['TIME', 'IND', 'Value'], parse_dates=['TIME'])
data_df = data_df.loc[data_df['IND'] == 'RHP']
data_df['Value'] = 100.0 * data_df['Value'] / data_df['Value'].iloc[0]
data_df.rename({'Value': 'HPI', 'TIME': 'Date'}, inplace=True, axis=1)
data_df['Time'] = range(0, 3 * len(data_df), 3)

# /////////
# Read full OECD data
df = pd.read_csv(rootOECD + '/OECD_Full_Data.csv', usecols=['LOCATION', 'TIME', 'Value'],
                 parse_dates=['TIME'])
# Select Spain
df = df.loc[df['LOCATION'] == 'ESP']
# Select time period
start_date = pd.to_datetime('1973-01-01')
end_date = pd.to_datetime('2020-10-01')
df = df.loc[df['TIME'] >= start_date].copy()
df = df.loc[df['TIME'] <= end_date]
# Compute rescaled variables
df['RescaledValue'] = 100.0 * df['Value'] / df.loc[df['TIME'] == start_date, 'Value'].values[0]
hp_filter_rescaled = sm.tsa.filters.hpfilter(np.array(df['RescaledValue'].values), lamb=129600)
df['RescaledDeTrendedValue'] = hp_filter_rescaled[0]
df['RescaledTrendOfValue'] = hp_filter_rescaled[1]
# /////////

# De-trend data
data_df['De-trended HPI (short)'] = sm.tsa.filters.hpfilter(data_df['HPI'], lamb=1600)[0] + 100.0
data_df['De-trended HPI'] = sm.tsa.filters.hpfilter(np.array(data_df['HPI']), lamb=129600)[0] + 100.0
data_df['HPI Trend (short)'] = sm.tsa.filters.hpfilter(data_df['HPI'], lamb=1600)[1]
data_df['HPI Trend'] = sm.tsa.filters.hpfilter(np.array(data_df['HPI']), lamb=129600)[1]

# If required, plot cycle for both countries
# plot_data_cycle_for_both_countries(data_df, rootOECDUK, rootResults, saveFigures)

# Std data
std_data = np.std(data_df['De-trended HPI'])

# Period data
N = len(data_df['De-trended HPI'])  # Number of sample points
T = 3.0  # Sample spacing
fastFourierTransformHPI = fft(data_df['De-trended HPI'].tolist())
frequencyDomain = np.linspace(0.0, 1.0/T, N)
x = frequencyDomain[1:70]
y = 1.0/N * np.abs(fastFourierTransformHPI[1:70])
period_data = 1 / x[y.argmax()]

# Print std and period of data cycles to screen
# print('Std of OECD cyclical component (HP de-trending, long trend, $\lambda = 129600$) = {}'.format(std_data))
# print('Period of OECD data cycles (HP de-trending, long trend, $\lambda = 129600$) = {}'.format(period_data))

stds = []
periods = []
for i in range(1, nRuns + 1):
    # Read only time and HPI from model results files
    model_df = pd.read_csv(rootModel + '/Output-run{}.csv'.format(i),
                           skipinitialspace=True, delimiter=';', usecols=['Model time', 'Sale HPI'])
    model_HPI = 100 * np.array(model_df.loc[model_df['Model time'] >= timeStepsToDiscard, 'Sale HPI'])
    # model_HPI = model_HPI - np.mean(model_HPI) + 100.0
    model_time = range(0, len(model_HPI))

    # Compute, store and, if required, print to screen std and period of model cycles
    std_model = np.std(model_HPI)
    # print('Std of model = {}'.format(std_model))
    stds.append(std_model)
    period_model = get_period(model_HPI)
    # print('Period of model cycles = {}'.format(period_model))
    periods.append(period_model)

# Print individual run and mean results to screen
print('Data, std of HPI', std_data)
print('Data, period of HPI cycles', period_data)
print('Stds of HPI', stds)
print('Mean value of std of HPI', np.mean(stds))
print('Periods of HPI cycles', periods)
print('Mean period of HPI cycles', np.mean(periods))

# plot_spectral_density(data_HPI_OECD_de_trend_hp_long[0], 3.0)

# plot_spectral_density(model_HPI, 1.0)

# plot_cycles_and_de_trending(saveFigures, data_df, rootResults)

plot_cycles(saveFigures, df, rootResults, np.mean(stds), np.mean(periods), _with_model=withModel)

# plot_uk_cycles(rootOECDUK, rootModelUK, rootResults, saveFigures, modelShiftUK, _with_model=withModel)

if not saveFigures:
    plt.show()
