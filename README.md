# Relationship between the US Unemployment Rate and Median House Sale Price in the US.

Unemployment is an important indicators used to explain US economy performance, and it is proved to be highly correlated to recession. On the other hand, Housing market is always involved either directly or indirectly in US recession, especially in 2008.

This analysis wants to explore whether housing price can be used an a predictor to US recession (using umeployment rate to represent the recession cycle). They are very likely to have a negative correlation as housing market usually goes down when unemployment goes up. It's also very likely there would be lagging effects between the two, as housing market usually starts to go down before umemployment starts to go up. If there are measurable lags, how many months would that be?

Data source from FRED:

Median Sales Price for New Houses Sold in the United States (MSPNHSUS)
https://fred.stlouisfed.org/series/MSPNHSUS

Unemployment Rate (UNRATE)
https://fred.stlouisfed.org/series/UNRATE

All analysis output is stored in the "Analytics_Output" folder.

Detailed Analysis can be found in the jupyterNotebook file .
### Key Take-aways :
1. Unemployment goes up during recession, and housing price tends to go down
2. Housing price starts to go down before recession
3. There is a negative correlation between unemployment rate and Housing price
4. Housing price could be used as a predictor for unemployment rate with 11 month lags

### Reporting options:
This project provides different options for reporting, includes the following:
1. pdf report based on LaTex template
2. Interactive html report 
3. Excel reports (which will be used as input to Tableau)
(No Unit Test for reporting functions now)

### Tableau Dashboard Example
read from the data in Excel reports
https://public.tableau.com/app/profile/hanlu.xia/viz/unemployment_house_report/Dashboard1?publish=yes

### Instruction on Running
1. Cloning the project from GitHub
2. Change the args in the example command to your local directory
3. Example: python GenerateReports/main.py -r unratehouse_html -d 2024/12/01 -o C:\Users\siaha\PycharmProjects\Unemployment_House\Analytics_Output\\
4. Run in Command Prompt
5. You can also change the -r to run different report: unratehouse_html, unratehouse_pdf, unratehouse_excel
