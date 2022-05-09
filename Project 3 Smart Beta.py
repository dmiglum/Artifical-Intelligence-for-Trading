#Project 3 Smart Beta


import pandas as pd
import numpy as np
import helper
#import project_helper
#import project_tests
import plotly.graph_objs as go
import plotly.offline as offline_py
import matplotlib.pyplot as plt

def large_dollar_volume_stocks(df, price_column, volume_column, top_percent):
    """
    Get the stocks with the largest dollar volume stocks.
    """
    dollar_traded = df.groupby('ticker').apply(lambda row: sum(row[volume_column] * row[price_column]))

    return dollar_traded.sort_values().tail(int(len(dollar_traded) * top_percent)).index.values.tolist()


df = pd.read_csv('eod-quotemedia.csv', parse_dates = ['date'], index_col = False) #from a different link, because other one didn't have csv data

percent_top_dollar = 0.2

high_volume_symbols = large_dollar_volume_stocks(df, 'adj_close', 'adj_volume', percent_top_dollar)
df = df[df['ticker'].isin(high_volume_symbols)]

close = df.reset_index().pivot(index='date', columns='ticker', values='adj_close')
volume = df.reset_index().pivot(index='date', columns='ticker', values='adj_volume')
dividends = df.reset_index().pivot(index='date', columns='ticker', values='dividends')

def generate_dollar_volume_weights(close, volume):
    """
    Generate dollar volume weights.
    """
    assert close.index.equals(volume.index)
    assert close.columns.equals(volume.columns)
    
    market_cap = close * volume
    dollar_volume_weights = market_cap.div(market_cap.sum(axis = 1), axis = 0).fillna(0)
    
    return dollar_volume_weights

index_weights = generate_dollar_volume_weights(close, volume)


def calculate_dividend_weights(dividends):
    """
    Calculate dividend weights.
    """
    cum_dividends = dividends.cumsum(axis = 0)
    dividend_weights = cum_dividends.div(cum_dividends.sum(axis = 1), axis = 0)
    
    return dividend_weights

etf_weights = calculate_dividend_weights(dividends)


def generate_returns(prices):
    """
    Generate returns for ticker and date.
    """
    returns = prices / prices.shift(1) - 1
    return returns

returns = generate_returns(close)


def generate_weighted_returns(returns, weights):
    """
    Generate weighted returns.
    """
    assert returns.index.equals(weights.index)
    assert returns.columns.equals(weights.columns)  
    weighted_returns = returns * weights
    
    return weighted_returns
    
index_weighted_returns = generate_weighted_returns(returns, index_weights)
etf_weighted_returns = generate_weighted_returns(returns, etf_weights)


def calculate_cumulative_returns(returns):
    """
    Calculate cumulative returns.
    """
    cumulative_returns = (1 + returns.sum(axis = 1)).cumprod(axis = 0)
    
    return cumulative_returns

index_weighted_cumulative_returns = calculate_cumulative_returns(index_weighted_returns)
etf_weighted_cumulative_returns = calculate_cumulative_returns(etf_weighted_returns)


def tracking_error(benchmark_returns_by_date, etf_returns_by_date):
    """
    Calculate the tracking error.
    """
    tracking_error = np.sqrt(252) * np.std(etf_returns_by_date - benchmark_returns_by_date, ddof=1)
    return tracking_error

smart_beta_tracking_error = tracking_error(np.sum(index_weighted_returns, 1), np.sum(etf_weighted_returns, 1))
print('Smart Beta Tracking Error: {}'.format(smart_beta_tracking_error))



def get_covariance_returns(returns):
    """
    Calculate covariance matrices.
    """
    returns_covariance = np.cov(returns.fillna(0).transpose())
    
    return returns_covariance

covariance_returns = get_covariance_returns(returns)
covariance_returns = pd.DataFrame(covariance_returns, returns.columns, returns.columns)

covariance_returns_correlation = np.linalg.inv(np.diag(np.sqrt(np.diag(covariance_returns))))
covariance_returns_correlation = pd.DataFrame(
    covariance_returns_correlation.dot(covariance_returns).dot(covariance_returns_correlation),
    covariance_returns.index,
    covariance_returns.columns)


import cvxpy as cvx

def get_optimal_weights(covariance_returns, index_weights, scale=2.0):
    """
    Find the optimal weights.
    """
    m = covariance_returns.shape[0]
    x = cvx.Variable(m)
    
    objective_function = cvx.Minimize(cvx.quad_form(x, covariance_returns) + scale * cvx.norm(x - index_weights))
    constraints = [x >=0, sum(x) == 1]
    
    cvx.Problem(objective_function, constraints).solve()
    
    return x.value

raw_optimal_single_rebalance_etf_weights = get_optimal_weights(covariance_returns.values, index_weights.iloc[-1])

optimal_single_rebalance_etf_weights = pd.DataFrame(
    np.tile(raw_optimal_single_rebalance_etf_weights, (len(returns.index), 1)),
    returns.index,
    returns.columns)

optim_etf_returns = generate_weighted_returns(returns, optimal_single_rebalance_etf_weights)
optim_etf_cumulative_returns = calculate_cumulative_returns(optim_etf_returns)

optim_etf_tracking_error = tracking_error(np.sum(index_weighted_returns, 1), np.sum(optim_etf_returns, 1))
print('Optimized ETF Tracking Error: {}'.format(optim_etf_tracking_error))


def rebalance_portfolio(returns, index_weights, shift_size, chunk_size):
    """
    Get weights for each rebalancing of the portfolio.
    """   
    n_days = returns.shape[0]
    all_rebalance_weights = []
    
    for index in range(shift_size, n_days, shift_size):
        start_index = index - chunk_size
        if start_index < 0:
            continue
            
        period_returns = returns[start_index:index]
        period_index_weights = index_weights.iloc[index - 1]
        
        cov_returns = get_covariance_returns(period_returns)
        optimal_weights = get_optimal_weights(cov_returns, period_index_weights)
        all_rebalance_weights.append(optimal_weights)
        
    return all_rebalance_weights

chunk_size = 250
shift_size = 5
all_rebalance_weights = rebalance_portfolio(returns, index_weights, shift_size, chunk_size)  #takes a little bit of time


def get_portfolio_turnover(all_rebalance_weights, shift_size, rebalance_count, n_trading_days_in_year=252):
    """
    Calculage portfolio turnover.
    """
    all_rebalance_weights = pd.DataFrame(all_rebalance_weights)
    total_turnover = np.abs(all_rebalance_weights.iloc[1:, :] - all_rebalance_weights.iloc[:-1, :].values).values.sum()
    
    num_reb_per_year = n_trading_days_in_year / shift_size
    annualized_turnover = total_turnover / rebalance_count * num_reb_per_year
    
    return annualized_turnover

print(get_portfolio_turnover(all_rebalance_weights, shift_size, len(all_rebalance_weights) - 1))