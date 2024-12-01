import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from datetime import datetime, date
from statsmodels.tsa.stattools import adfuller
from matplotlib.ticker import FormatStrFormatter
from pandas.plotting import scatter_matrix
from plotly.offline import plot
import subprocess
import plotly.graph_objs as go
from plotly.subplots import make_subplots

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
    for col in range(len(colList)):
        dftest = adfuller(df[colList[col]], autolag='AIC')
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
    plt.clf()
    return 0

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
    plt.clf()
    return 0

def ols_regression_lag(master,nlag, col):
    R2_result = []
    prevR2 = 0

    for i in range(1, nlag + 1):
        master_loop = master.copy()
        column_names = [col]
        current_lag = i
        while i > 0:
            text = 'lag' + '_' + str(i)
            master_loop[text] = master_loop[col].shift(i)
            column_names.append(text)
            i = i - 1
        master_loop = master_loop.dropna()

        x = master_loop[column_names]
        reg_model = sm.OLS(master_loop['UNRATE'], x)
        result = reg_model.fit()
        changeR2 = result.rsquared - prevR2
        prevR2 = result.rsquared
        R2_result.append({'Total Lag': current_lag, 'R2': result.rsquared, 'Change in R2': changeR2})

    R2_df = pd.DataFrame(R2_result)
    return R2_df

def regression_plot(R2_df, dir):
    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax1.plot('Total Lag', 'R2', data=R2_df, color='tab:red', label='R square')
    ax1.set_xlabel('Lags', fontdict={'fontsize': 15, 'fontweight': 'medium'})
    ax1.set_ylabel('R square', fontdict={'fontsize': 15, 'fontweight': 'medium'})
    ax1.set_title('R square over different lags', fontdict={'fontsize': 20, 'fontweight': 'medium'})

    ax1.set_xticks(np.linspace(min(R2_df['Total Lag']), max(R2_df['Total Lag']), 5, dtype=int))
    ax1.grid(True)
    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    fig.autofmt_xdate()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(dir)
    plt.clf()
    return 0


def generate_latex_report(df, image_path, report_path):
    # Create a LaTeX document with embedded image and dataframe
    latex_code = r'''
\documentclass{article}
\usepackage{graphicx}
\usepackage{longtable}
\usepackage{geometry}
\geometry{a4paper, margin=1in}
\graphicspath{{C''' + image_path + r'''}}

\begin{document}

\title{Sample LaTeX Report with Image and Dataframe}
\author{Your Name}
\date{\today}
\maketitle

\section{Introduction}
This is a simple LaTeX report generated using Python. Below is a plot generated from Python:

\begin{figure}[h!]
\centering
\includegraphics[width=0.8\textwidth]{C:/Users/siaha/PycharmProjects/Unemployment_House/Analytics_Output/timeseries1.png}
\caption{Time Serie Test}
\end{figure}


\section{Dataframe}
The following table represents some sample data:

\begin{longtable}{|c|c|c|}
\hline
\textbf{Index} & \textbf{Value1} & \textbf{Value2} \\
\hline
\endfirsthead
\hline
\textbf{Index} & \textbf{Value1} & \textbf{Value2} \\
\hline
\endhead
''' + generate_dataframe_latex(df) + r'''

\end{longtable}

\end{document}
    '''
    # Write the LaTeX code to a .tex file
    with open(report_path, 'w') as file:
        file.write(latex_code)


# Function to convert a pandas DataFrame to LaTeX table format
def generate_dataframe_latex(df):
    latex_str = ""
    for index, row in df.iterrows():
        latex_str += f"{row['Description']} & {row['P-Value']} & {row['Result']}  \\\\\n"
    return latex_str


# Function to compile LaTeX document into PDF
def compile_latex_to_pdf(latex_file):
    subprocess.run([r"C:\Users\siaha\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex", latex_file], check=True)

def generate_pdf_report(date: str, tempdir: str):
    #Generate All Table and Images we need
    master = process_raw_data()
    stationary_result = stationary_test(master)
    timeseries_recession_graph(master, r'/Analytics_Output\\timeseries1.png')
    first_corr = ['UNRATE', 'MSPNHSUS', 'house_diff', 'house_return']
    first_matrix = ['UNRATE', 'house_diff', 'house_return']
    correlation_plot(master, first_corr, r'/Analytics_Output\\correlationplot1.png')
    correlation_matrix(master, first_matrix, r'/Analytics_Output\\corrmatrix1.png')
    regression_result = ols_regression_lag(master,24, 'house_diff')
    regression_plot(regression_result, r'/Analytics_Output\\regression_result.png')

    # Generate LaTeX report
    report_path = r"/Analytics_Output\\latexTemplate.tex"
    image_path= r'/Analytics_Output'
    generate_latex_report(stationary_result, image_path, report_path)

    # Compile LaTeX file to PDF
    compile_latex_to_pdf(report_path)

    return 0


def generate_html_report(date: str, tempdir: str):
    master = process_raw_data()
    stationary_result = stationary_test(master)

    # Create a figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=master.index, y=master['UNRATE'], mode='lines', name='Unemployment Rate'),
                  secondary_y=False)
    # Add the second trace (new line)
    fig.add_trace(go.Scatter(x=master.index, y=master['MSPNHSUS'], mode='lines', name='Median House Price',
                             line=dict(dash='dash')), secondary_y=True)

    # Update layout to include slider
    fig.update_layout(
        title='Interactive Unemployment Rate Time Series Graph',
        xaxis=dict(
            title='Date',
            rangeselector=dict(
                buttons=[
                    dict(count=60, label="5y", step="month", stepmode="backward"),
                    dict(count=120, label="10y", step="month", stepmode="backward"),
                    dict(count=180, label="15y", step="month", stepmode="backward"),
                    dict(step="all")
                ],
                bgcolor='rgba(50, 50, 50, 0.8)',  # Dark background for the range selector
                activecolor='rgba(80, 80, 80, 1)',  # Darker shade for active button
                bordercolor='rgba(200, 200, 200, 0.6)'  # Subtle border color for contrast
            ),
            rangeslider=dict(visible=True),  # Add the range slider
            type="date"
        ),
        yaxis=dict(title='Unemployment Rate', side='left'),  # Primary y-axis on the left
        yaxis2=dict(title='Median House Price', side='right'),  # Secondary y-axis on the right
        template='plotly_dark'
    )

    # Save the interactive graph to HTML
    graph_html = plot(fig, output_type='div', include_plotlyjs='cdn')

    # Create an HTML table from the DataFrame
    table_html = stationary_result[['Description', 'P-Value', 'Result']].to_html(
        index=False,
        classes='table table-dark table-striped',
        justify='center',
        escape=False,
        table_id = 'data-table'
    )

    # Create an HTML template with custom filtering functionality
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>US Unemployment Rate Vs. Median House Sale Price in the US</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{
                background-color: #2c2c2c;
                color: white;
            }}
            h1 {{
                text-align: center;
                margin-top: 20px;
            }}
            .table-container {{
                margin: 20px auto;
                width: 90%;
                text-align: center;
            }}
            .table {{
                font-size: 16px;
                width: 100%;
            }}
            .table th, .table td {{
                padding: 15px;
            }}
            .filter-container {{
                margin-bottom: 10px;  /* Reduced margin */
                text-align: center;
                width: 50%;           /* Adjust width to make it smaller */
                margin-left: auto;    /* Center horizontally */
                margin-right: auto;   /* Center horizontally */
                display: flex;        /* Use flexbox for alignment */
                flex-direction: column; /* Align label and select vertically */
                align-items: center;  /* Center content horizontally */
            }}
            .filter-container label {{
                margin-bottom: 5px;  /* Add some space between the label and the select */
                font-size: 16px;     /* Adjust font size if needed */
            }}
            .filter-container select {{
                padding: 5px;         /* Reduce padding for smaller dropdown */
                background-color: #444;
                color: white;
                border: 1px solid #555;
                width: 100%;          /* Ensure it takes up the full width of the container */
                max-width: 200px;     /* Optionally, set a max width for the dropdown */
            }}
        </style>
        <script>
            function filterTable() {{
                console.log('test1')
                var filterValue = document.getElementById("filter-dropdown").value;
                var table = document.getElementById("data-table");
                var rows = table.getElementsByTagName("tr");
                console.log(table)  
                for (var i = 1; i < rows.length; i++) {{
                    var cells = rows[i].getElementsByTagName("td");
                    var resultCell = cells[2]; // 3rd column (Result)
                    console.log('test3')    
                    if (resultCell) {{
                        var resultText = resultCell.textContent || resultCell.innerText;
                        if (filterValue === "" || resultText === filterValue) {{
                            rows[i].style.display = "";
                        }} else {{
                            rows[i].style.display = "none";
                        }}
                    }}
                }}
            }}
        </script>
    </head>
    <body>
    <h1>US Unemployment Rate Vs. Median House Sale Price in the US</h1>
    <p>Unemployment is an important indicators used to explain US economy performance, and it is proved to be highly correlated to recession. On the other hand, Housing market is always involved either directly or indirectly in US recession, especially in 2008. </p>
    <p>This analysis wants to explore whether housing price can be used as a predictor to US recession (using umeployment rate to represent the recession cycle). They are very likely to have a negative correlation as housing market usually goes down when unemployment goes up. It's also very likely there would be lagging effects between the two, as housing market usually starts to go down before umemployment starts to go up. If there are measurable lags, how many months would that be?</p>
    {graph_html}
    <!-- Dropdown filter for Result column -->
    <div class="filter-container">
        <label for="filter-dropdown" style="color: white;">Filter by Result:</label>
        <select id="filter-dropdown" class="form-select" onchange="filterTable()">
            <option value="">All</option>
            <option value="stationary">Stationary</option>
            <option value="non-stationary">Non-Stationary</option>
        </select>
    </div>

    <div class="table-container">
        <h2>Time Series Data Table</h2>
            {table_html}
    </div>
    </body>
    </html>
    """

    # Save the HTML report
    output_path = r"C:\Users\siaha\PycharmProjects\Unemployment_House\Analytics_Output\unemployment_house_report.html"
    with open(output_path, "w") as f:
        f.write(html_template)

    print(f"Report saved to {output_path}")

    return 0


def generate_excel_report(date: str, tempdir: str):
    master = process_raw_data()
    stationary_result = stationary_test(master)
    regression_result = ols_regression_lag(master, 24, 'house_diff')
    # Write the DataFrames to an Excel file
    report_path = r"C:\Users\siaha\PycharmProjects\Unemployment_House\Analytics_Output\unemployment_house_report.xlsx"
    with pd.ExcelWriter(report_path) as writer:
        stationary_result[['Description','P-Value', 'Result']].to_excel(writer, sheet_name='Stationary_Analytics', index=False)
        regression_result.to_excel(writer, sheet_name='Regression_Result', index=False,header=True)
        master.to_excel(writer, sheet_name='Raw', index=True)

    return

