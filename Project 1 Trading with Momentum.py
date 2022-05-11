#Project 1 Trading with Momentum

import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.offline as offline_py
import matplotlib.pyplot as plt

# Load Data
df = pd.read_csv('eod-quotemedia.csv', parse_dates = ['date'], index_col = False)
close = df.reset_index().pivot(index = 'date', columns = 'ticker', values = 'adj_close')

# Using Apple as an example
apple_ticker = 'AAPL'


# Resample Adjusted Prices
def resample_prices(close_prices, freq = 'M'):
    '''
    Resample close prices for each ticker at specified frequency
    '''
    return close_prices.resample(freq).last()

monthly_close = resample_prices(close)

# Compute Log Returns
# Log Returns (R(t)) are coputed from Prices (P(t)) as follows: R(t) = log(P(t)) - log(P(t-1))
def compute_log_returns(prices):
    '''
    Compute log returns for each ticker

    '''
    return np.log(prices) - np.log(prices).shift(1) 
    #return np.log(prices / prices.shift(1)) #can also calculate it this way

monthly_close_returns = compute_log_returns(monthly_close)

# Shift returns function
def shift_returns(returns, shift_n):
    '''
    Generate shifted returns
    '''
    return returns.shift(shift_n)

prev_returns = shift_returns(monthly_close_returns, 1)
lookahead_returns = shift_returns(monthly_close_returns, -1)

plt.plot(prev_returns.loc[:, apple_ticker])

# Generate Trading Signal
'''
Our Trading Strategy: For each month-end observation period, rank the stocks by 
previous returns, from the highest to the lowest. Select the top performing stocks 
for the long portfolio, and the bottom performing stocks for the short portfolio.
'''
def get_top_n(prev_returns, top_n):
    '''
    Select top performing stocks
    '''
    top_stocks = pd.DataFrame(index = prev_returns.index, columns = prev_returns.columns)
    for index, row in prev_returns.iterrows(): #iterating across rows - each row yields a pandas Data Series
        top_stocks.loc[index] = row.isin(row.nlargest(top_n)).astype(np.int) #finding n_largest values, creating a True/False boolean, and converting it to np.int
        
    return top_stocks

top_bottom_n = 50
df_long = get_top_n(prev_returns, top_bottom_n)
df_short = get_top_n(prev_returns*-1, top_bottom_n)

# Projected Returns
'''
We'll start by computing the net returns this portfolio would return. For simplicity, 
we'll assume every stock gets an equal dollar amount of investment. This makes it easier 
to compute a portfolio's returns as the simple arithmetic average of the individual stock returns.

Implement the portfolio_returns function to compute the expected portfolio returns. 
Using df_long to indicate which stocks to long and df_short to indicate which stocks to 
short, calculate the returns using lookahead_returns. To help with calculation, we've 
provided you with n_stocks as the number of stocks we're investing in a single period.
'''
def portfolio_returns(df_long, df_short, lookahead_returns, n_stocks):
    '''
    Compute expected returns for the portfolio, assuming equal investment in each long/short stock.
    '''
    return (df_long - df_short) * lookahead_returns / n_stocks #dividing by n_stocks because we assume equal weighted portfolio
           # this is an array of 1s and 0s, so '-' sign in front of df_short would convert those 1s into -1s

expected_portfolio_returns = portfolio_returns(df_long, df_short, lookahead_returns, 2*top_bottom_n)

plt.plot(expected_portfolio_returns.T.sum())
plt.title('Portfolio Returbs')
    
# Statostocal Tests
# Computing Annualized Rate of Return
expected_portfolio_returns_by_date=  expected_portfolio_returns.T.sum().dropna()
portfolio_ret_mean = expected_portfolio_returns_by_date.mean()
portfolio_ret_ste = expected_portfolio_returns_by_date.sem()
portfolio_ret_annual_rate = (np.exp(portfolio_ret_mean * 12) - 1) * 100

print("""
Mean:                       {:.6f}
Standard Error:             {:.6f}
Annualized Rate of Return:  {:.2f}%
""".format(portfolio_ret_mean, portfolio_ret_ste, portfolio_ret_annual_rate))

# Running a T-Test
'''
Our null hypothesis (H0) is that the actual mean return from the signal is zero. 
We'll perform a one-sample, one-sided t-test on the observed mean return, to see if we can reject (H0).

We'll need to first compute the t-statistic, and then find its corresponding p-value. The p-value will 
indicate the probability of observing a t-statistic equally or more extreme than the one we observed 
if the null hypothesis were true. A small p-value means that the chance of observing the t-statistic 
we observed under the null hypothesis is small, and thus casts doubt on the null hypothesis. It's good 
practice to set a desired level of significance or alpha before computing the p-value, and then reject the null hypothesis if p < alpha.
'''
from scipy import stats
def analyze_alpha(expected_portfolio_returns_by_date):
    """
    Perform t-test with the null hyoptheiss being that the epxected mean return is zero
    """
    t_statistic, p_value = stats.ttest_1samp(expected_portfolio_returns_by_date, 0)
    return t_statistic, p_value / 2

t_value, p_value = analyze_alpha(expected_portfolio_returns_by_date)
print("""
Alpha analysis:
 t-value:        {:.3f}
 p-value:        {:.6f}
""".format(t_value, p_value))

# Out p-value = 0.073359. Because it is greater than 0.05, we cannot reject the null hypothesis that the population mean is zero.

    