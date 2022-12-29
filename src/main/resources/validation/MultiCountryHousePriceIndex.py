# coding: utf-8

# Imports
from __future__ import division
import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
import numpy as np
import statsmodels.api as sm
import pandas as pd


def explore_countries_and_time():
    # for date in pd.date_range('1971-01-01', '1972-01-01', freq='QS'):
    # for i, date in enumerate(pd.date_range('1971-01-01', '2000-01-01', freq='QS')):
    # for i, date in enumerate(pd.date_range('1971-01-01', '1974-01-01', freq='QS')):
    for i, date in enumerate(pd.date_range('1973-01-01', '1973-01-01', freq='QS')):
        # for j, final_date in enumerate(pd.date_range('2017-01-01', '2020-10-01', freq='QS')):
        for j, final_date in enumerate(pd.date_range('2020-10-01', '2020-10-01', freq='QS')):

            # f1, ax1 = plt.subplots()
            f2, ax2 = plt.subplots()

            df = df_original.loc[df_original['TIME'] >= date].copy()
            df = df.loc[df['TIME'] <= final_date]
            # df = df_original.loc[df_original['TIME'] <= final_date].copy()

            # Add  de-trended HPI
            # for country in df['LOCATION'].unique():
            for c in ['BEL', 'FRA', 'DEU', 'GRC', 'IRL', 'ITA', 'NLD', 'PRT', 'ESP', 'GBR']:

                if df.loc[df['LOCATION'] == c, 'TIME'].min() <= date:

                    df.loc[df['LOCATION'] == c, 'RescaledValue'] = \
                        100.0 * df.loc[df['LOCATION'] == c, 'Value'] / df.loc[(df['LOCATION'] == c) &
                                                                              (df['TIME'] == date), 'Value'].values[0]
                    hp_filter = sm.tsa.filters.hpfilter(np.array(df.loc[df['LOCATION'] == c, 'Value'].values),
                                                        lamb=myLambda)
                    df.loc[df['LOCATION'] == c, 'DeTrendedValue'] = hp_filter[0]
                    df.loc[df['LOCATION'] == c, 'TrendOfValue'] = hp_filter[1]
                    hp_filter_rescaled = sm.tsa.filters.hpfilter(np.array(df.loc[df['LOCATION'] == c,
                                                                                 'RescaledValue'].values),
                                                                 lamb=myLambda)
                    df.loc[df['LOCATION'] == c, 'RescaledDeTrendedValue'] = hp_filter_rescaled[0]
                    df.loc[df['LOCATION'] == c, 'RescaledTrendOfValue'] = hp_filter_rescaled[1]

                    if c == 'ESP' or c == 'GBR':
                        ax2.plot(df.loc[df['LOCATION'] == c, 'TIME'], df.loc[df['LOCATION'] == c,
                                                                             'RescaledDeTrendedValue'],
                                 label='{} {:.2f}'.format(c, np.std(df.loc[df['LOCATION'] == c,
                                                                           'RescaledDeTrendedValue'])))
                        ax2.set_title(date)

                    # print(country, np.std(df.loc[df['LOCATION'] == country, 'DeTrendedValue']),
                    #       np.std(df.loc[df['LOCATION'] == country, 'RescaledDeTrendedValue']))

            std_esp = np.std(df.loc[df['LOCATION'] == 'ESP', 'RescaledDeTrendedValue'])
            std_gbr = np.std(df.loc[df['LOCATION'] == 'GBR', 'RescaledDeTrendedValue'])

            # if 30.0 <= std_gbr <= 37.0 and 37.0 <= std_esp <= 43.0:
            #     print(date, i, j, final_date, std_esp, std_gbr)
            print(date, i, j, final_date, std_esp, std_gbr)

            # ax1.legend()
            ax2.legend()

            # plt.tight_layout()
            # plt.savefig(rootResults + '/HPI/TestCut{}.png'.format(i), format='png', dpi=300, bbox_inches='tight')


def plot_hpi(_save_figures, _root_results, _countries_list, _to_highlight, _string_to_add):

    # De-trended data
    plt.figure(figsize=(5.0, 3.5))
    # Add  de-trended HPI
    for c in _countries_list:
        if df.loc[df['LOCATION'] == c, 'TIME'].min() <= start_date:
            df.loc[df['LOCATION'] == c, 'RescaledValue'] = \
                100.0 * df.loc[df['LOCATION'] == c, 'Value'] / df.loc[(df['LOCATION'] == c) &
                                                                      (df['TIME'] == start_date), 'Value'].values[0]
            hp_filter = sm.tsa.filters.hpfilter(np.array(df.loc[df['LOCATION'] == c, 'Value'].values),
                                                lamb=myLambda)
            df.loc[df['LOCATION'] == c, 'DeTrendedValue'] = hp_filter[0]
            df.loc[df['LOCATION'] == c, 'TrendOfValue'] = hp_filter[1]
            hp_filter_rescaled = sm.tsa.filters.hpfilter(np.array(df.loc[df['LOCATION'] == c,
                                                                         'RescaledValue'].values), lamb=myLambda)
            df.loc[df['LOCATION'] == c, 'RescaledDeTrendedValue'] = hp_filter_rescaled[0]
            df.loc[df['LOCATION'] == c, 'RescaledTrendOfValue'] = hp_filter_rescaled[1]
            if c in _to_highlight:
                plt.plot(df.loc[df['LOCATION'] == c, 'TIME'], df.loc[df['LOCATION'] == c, 'RescaledDeTrendedValue'],
                         '-', lw=1.5, color=colors[c], label='{}'.format(countryNames[c]),
                         zorder=100 if c == 'ESP' else 50)
            else:
                # plt.plot(df.loc[df['LOCATION'] == c, 'TIME'], df.loc[df['LOCATION'] == c, 'RescaledDeTrendedValue'],
                #          '-', lw=1.0, color='grey')
                plt.plot(df.loc[df['LOCATION'] == c, 'TIME'], df.loc[df['LOCATION'] == c, 'RescaledDeTrendedValue'],
                         ':', lw=1.5, color=colors[c], label='{}'.format(countryNames[c]))
    plt.axhline(0, c='k', ls='--', zorder=100)
    plt.xlabel('Date')
    plt.ylabel('HPI (Cyclical Component)')
    plt.xlim(start_date, end_date)
    # plt.gca().xaxis.set_minor_locator(mdates.YearLocator(2))
    plt.legend(loc='upper left', ncol=2 if len(_countries_list) > 2 else 1, fontsize=8.5, frameon=False,
               labelspacing=0.45, handlelength=1.6, borderaxespad=0.25)
    if len(_countries_list) == 2:
        string = 'UK-Spain'
    else:
        string = 'Multi-Country'
    if _save_figures:
        plt.tight_layout()
        plt.savefig(_root_results + '/HousePriceIndex-' + string + _string_to_add + '.png', format='png', dpi=300,
                    bbox_inches='tight')
        plt.savefig(_root_results + '/HousePriceIndex-' + string + _string_to_add + '.pdf', format='pdf', dpi=300,
                    bbox_inches='tight')
        plt.savefig(_root_results + '/HousePriceIndex-' + string + _string_to_add + '.eps', format='eps', dpi=300,
                    bbox_inches='tight')
        # Store also version with zoom on the most recent cycle
        plt.xlim(pd.to_datetime('1992-01-01'), end_date)
        if len(_countries_list) > 2:
            plt.ylim(-142, 152)
            plt.legend(loc='lower left', ncol=2, prop={'size': 8})
        plt.tight_layout()
        plt.savefig(_root_results + '/HousePriceIndex-' + string + _string_to_add + '-Zoom.png', format='png', dpi=300,
                    bbox_inches='tight')

    # Unfiltered data
    plt.figure(figsize=(6.0, 4))
    for c in _countries_list:
        if df.loc[df['LOCATION'] == c, 'TIME'].min() <= start_date:
            if c in _to_highlight:
                plt.plot(df.loc[df['LOCATION'] == c, 'TIME'], df.loc[df['LOCATION'] == c, 'RescaledValue'],
                         '-', lw=1.5, color=colors[c], label='{}'.format(countryNames[c]))
            else:
                plt.plot(df.loc[df['LOCATION'] == c, 'TIME'], df.loc[df['LOCATION'] == c, 'RescaledValue'],
                         '-', lw=1.0, color='grey')
    plt.xlabel('Date')
    plt.ylabel('HPI')
    plt.xlim(start_date, end_date)
    # plt.gca().xaxis.set_minor_locator(mdates.YearLocator(2))
    plt.legend(loc='upper left', ncol=2 if len(_countries_list) > 2 else 1, prop={'size': 8})
    if _save_figures:
        plt.tight_layout()
        plt.savefig(_root_results + '/HousePriceIndex-' + string + '-Unfiltered' + _string_to_add + '.png',
                    format='png', dpi=300, bbox_inches='tight')


# Set parameters
saveFigures = False
myLambda = 129600
rootOECD = ''
rootResults = ''
colors = {'ESP': 'tab:red', 'GBR': 'tab:blue', 'BEL': 'tab:green', 'FRA': 'tab:pink', 'DEU': 'tab:orange',
          'GRC': 'tab:purple', 'IRL': 'tab:green', 'ITA': 'tab:gray', 'NLD': 'tab:olive', 'PRT': 'tab:purple',
          'DNK': 'tab:brown', 'FIN': 'tab:cyan', 'SWE': 'black'}
countryNames = {'ESP': 'Spain', 'GBR': 'UK', 'BEL': 'Belgium', 'FRA': 'France', 'DEU': 'Germany',
                'GRC': 'Greece', 'IRL': 'Ireland', 'ITA': 'Italy', 'NLD': 'Netherlands', 'PRT': 'Portugal',
                'DNK': 'Denmark', 'FIN': 'Finland', 'SWE': 'Sweden'}

# General printing settings
np.set_printoptions(suppress=True)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 1000)

# General settings
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'sans',
    'font.sans-serif': ['DejaVu Sans']
})

# Read multi-country OECD data
df_original = pd.read_csv(rootOECD + '/OECD_Full_Data.csv', usecols=['LOCATION', 'TIME', 'Value'],
                          parse_dates=['TIME'])

start_date = pd.to_datetime('1973-01-01')
end_date = pd.to_datetime('2020-10-01')
df = df_original.loc[df_original['TIME'] >= start_date].copy()
df = df.loc[df['TIME'] <= end_date]

# plot_hpi(saveFigures, rootResults, ['ESP', 'GBR'], ['ESP', 'GBR'], '')
# plot_hpi(saveFigures, rootResults,
#          ['FRA', 'DEU', 'GRC', 'IRL', 'ITA', 'NLD', 'PRT', 'DNK', 'FIN', 'SWE', 'ESP', 'GBR'],
#          ['FRA', 'DEU', 'GRC', 'IRL', 'ITA', 'NLD', 'PRT', 'DNK', 'FIN', 'SWE', 'ESP', 'GBR'], '-highlightAll')
# plot_hpi(saveFigures, rootResults,
#          ['FRA', 'DEU', 'GRC', 'IRL', 'ITA', 'NLD', 'PRT', 'DNK', 'FIN', 'SWE', 'ESP', 'GBR'],
#          ['DEU', 'IRL', 'ESP', 'GBR'], '-highlight4')
# plot_hpi(saveFigures, rootResults,
#          ['FRA', 'DEU', 'GRC', 'IRL', 'ITA', 'NLD', 'PRT', 'DNK', 'FIN', 'SWE', 'ESP', 'GBR'],
#          ['DEU', 'IRL', 'ESP'], '-highlight3')
plot_hpi(saveFigures, rootResults,
         ['ESP', 'GBR', 'FRA', 'DEU', 'GRC', 'IRL', 'ITA', 'NLD', 'PRT', 'DNK', 'FIN', 'SWE'],
         # ['FRA', 'DEU', 'GRC', 'IRL', 'ITA', 'NLD', 'PRT', 'DNK', 'FIN', 'SWE', 'ESP', 'GBR'],
         ['ESP', 'GBR'], '-highlight2New')
# plot_hpi(saveFigures, rootResults,
#          ['FRA', 'DEU', 'GRC', 'IRL', 'ITA', 'NLD', 'PRT', 'DNK', 'FIN', 'SWE', 'ESP', 'GBR'],
#          ['ESP'], '-highlight1')

if not saveFigures:
    plt.show()
