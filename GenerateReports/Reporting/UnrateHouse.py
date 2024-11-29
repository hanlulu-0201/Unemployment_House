import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from datetime import datetime, date
from statsmodels.tsa.stattools import adfuller
from matplotlib.ticker import FormatStrFormatter
from pandas.plotting import scatter_matrix
from sklearn import datasets

def process_raw_data():
    df_unrate = pd.read_csv(r'D:\JupyterNotebook\HG_Vora\UNRATE.csv')
    df_unrate = df_unrate.set_index(pd.to_datetime(df_unrate['DATE']))
    df_unrate = df_unrate.drop(['DATE'], axis=1)

    df_house = pd.read_csv(r'D:\JupyterNotebook\HG_Vora\MSPNHSUS.csv')
    df_house = df_house.set_index(pd.to_datetime(df_house['DATE']))
    df_house = df_house.drop(['DATE'], axis=1)

    master = pd.concat([df_unrate, df_house], axis=1)
    master['house_diff'] = master['MSPNHSUS'].diff(1)
    master['house_return'] = master['MSPNHSUS'].pct_change(1)
    master = master.dropna()

    return master

def stationary_test(df):
    colList = ['UNRATE','MSPNHSUS','house_diff','house_return']
    descriptionList = ['Unemployment Rate','Median Housing Price Over Time',
                       'First diff of housing Price','Return of housing Price']

    resultdf = pd.DataFrame(columns=['Column', 'P-Value', 'Result', 'Description'])
    for col in colList:
        dftest = adfuller(df[col], autolag='AIC')
        if dftest[0] < dftest[4]["5%"]:
            result = 'stationary'
        else:
            result = 'non-stationary'
        # Create a new row as a dictionary
        new_row = [col, dftest[1], result, descriptionList[col]]
        # Append the new row
        resultdf.loc[len(resultdf)] = new_row

    return resultdf

def timeseries_recession_graph(master, dir):
    # Created a graph to visualize the raw data
    fig, ax1 = plt.subplots(figsize=(18, 8))
    line1 = ax1.plot(master.index, 'UNRATE', data=master, color='tab:blue', label='Unemployment')
    ax1.set_xlabel('Year', fontdict={'fontsize': 15, 'fontweight': 'medium'})
    ax1.set_ylabel('Unemploymet Rate %', fontdict={'fontsize': 15, 'fontweight': 'medium'})
    ax1.set_title('Unemploymet Rate and Housing Price Over Time', fontdict={'fontsize': 20, 'fontweight': 'medium'})
    ax1.grid(True)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    line2 = ax2.plot(master.index, 'MSPNHSUS', data=master, color='tab:gray', alpha=0.5, label='House Price')
    ax2.tick_params(axis='y')
    ax2.set_ylabel('Median House Sales Price $', size=15)

    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left')

    # Year period for recessions defined by Wikipedia list of recessions
    ax3 = ax1.twinx()
    ax3.axvspan(datetime(1980, 1, 1), datetime(1980, 7, 1), alpha=0.1, color='gray')
    ax3.axis('off')
    ax4 = ax1.twinx()
    ax4.axvspan(datetime(1981, 7, 1), datetime(1982, 11, 1), alpha=0.1, color='gray')
    ax4.axis('off')
    ax5 = ax1.twinx()
    ax5.axvspan(datetime(1990, 7, 1), datetime(1991, 3, 1), alpha=0.1, color='gray')
    ax5.axis('off')
    ax6 = ax1.twinx()
    ax6.axvspan(datetime(2001, 3, 1), datetime(2001, 11, 1), alpha=0.1, color='gray')
    ax6.axis('off')
    ax7 = ax1.twinx()
    ax7.axvspan(datetime(2007, 12, 1), datetime(2009, 6, 1), alpha=0.1, color='gray')
    ax7.axis('off')
    ax8 = ax1.twinx()
    ax8.axvspan(datetime(2020, 2, 1), datetime(2020, 4, 1), alpha=0.1, color='gray')
    ax8.axis('off')
    ax9 = ax1.twinx()
    ax9.axvspan(datetime(1973, 11, 1), datetime(1975, 3, 1), alpha=0.1, color='gray')
    ax9.axis('off')
    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    fig.autofmt_xdate()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(dir)

    return 0

def correlation_plot(master, list,dir):
    # diagonal is graphed by kernel density estimation (KDE)
    ax = scatter_matrix(master[list],
                        color="#0392cf", alpha=0.5, figsize=(10, 10), diagonal='kde', marker='.')

    for i in range(np.shape(ax)[0]):
        for j in range(np.shape(ax)[1]):
            if i < j:
                ax[i, j].set_visible(False)

    for ax in ax.ravel():
        ax.set_xlabel(ax.get_xlabel().replace(' ', '\n'), fontsize=7, rotation=90, weight='medium')
        ax.set_ylabel(ax.get_ylabel().replace(' ', '\n'), fontsize=7, rotation=90, weight='medium')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks([])

    plt.suptitle('Correlation plot of Initial Data', size=15, weight='bold', va='bottom', x=0.5, y=0.93)

    handles = [plt.plot([], [], color=plt.cm.brg(i / 2.), ls="", marker="o",markersize=np.sqrt(10))[0] for i in range(3)]
    handles = [handles[0], handles[2], handles[1]]
    # plt.legend(handles, loc=(1.02,0))
    plt.savefig(dir)

def correlation_matrix(master, list,dir):
    corr_matrix = master[list]
    np.bool = np.bool_
    corr = round(corr_matrix.corr(), 2)

    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(corr, annot_kws={'size': 10}, cmap='vlag_r', xticklabels=corr.columns.values,
                yticklabels=corr.columns.values, annot=True, mask=mask)
    heat_map = plt.gcf()
    heat_map.set_size_inches(10, 6)
    plt.suptitle('Correlation Matrix of Initial Data', size=15, weight='bold', va='bottom', x=0.5, y=0.93)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig(dir)



def generate_pdf_report(date: str, tempdir: str):
    master = process_raw_data()
    stationary_result = stationary_test(master)
    timeseries_recession_graph(master, r'D:\JupyterNotebook\HG_Vora\timeseries1.png')
    first_corr = ['UNRATE', 'MSPNHSUS', 'house_diff', 'house_return']
    first_matrix = ['UNRATE', 'house_diff', 'house_return']
    correlation_plot(master, first_corr, r'D:\JupyterNotebook\HG_Vora\correlationplot1.png')
    correlation_matrix(master,first_matrix,r'D:\JupyterNotebook\HG_Vora\corrmatrix1.png')

    return


def generate_excel_report(date: str, tempdir: str):
    return