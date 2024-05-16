# Pricing Models Interface Monte Carlo - Dylan

# Comp simulation that relies on rptd rdm sampling for results

#to run: streamlit run /Users/dylanhans/PycharmProjects/pricing_models/MonteCarlo.py
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf

yf.pdr_override()

def get_data(stocks, start, end):
    try:
        stockData = pdr.get_data_yahoo(stocks, start, end)
        if stockData.empty:
            raise ValueError("No data returned from Yahoo Finance")
        stockData = stockData['Close']
        returns = stockData.pct_change()
        covMatrix = returns.cov()  # Assuming you want to compute covariance of returns
        meanReturns = returns.mean()
        return meanReturns, covMatrix
    except Exception as e:
        st.error(f"Make a selection")
        return None, None

# Streamlit app title
st.title('Monte Carlo Simulation')

# Sidebar for user input
st.sidebar.title('Simulation Parameters')
stocks = st.sidebar.multiselect('Select stocks', ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'])
startDate = st.sidebar.date_input("Start Date", dt.date.today() - dt.timedelta(days=300))
endDate = st.sidebar.date_input("End Date", dt.date.today())

# Fetch data
meanReturns, covMatrix = get_data(stocks, startDate, endDate)

# If data is successfully retrieved, proceed with simulation and visualization
if meanReturns is not None and covMatrix is not None:
    # Randomly generate weights
    weights = np.random.random(len(meanReturns))
    weights /= np.sum(weights)

    # Number of simulations
    MonteCarlo_sims = 100
    # Number of days
    T = 100

    # Initialize arrays
    MatrixMean = np.full(shape=(T, len(weights)), fill_value=meanReturns)
    MatrixMean = MatrixMean.T
    Portfolio_sims = np.full(shape=(T, MonteCarlo_sims), fill_value=0.0)
    InitialPort = 10000

    # Run simulations
    for m in range(0, MonteCarlo_sims):
        Z = np.random.normal(size=(T, len(weights)))
        L = np.linalg.cholesky(covMatrix)
        dailyReturns = MatrixMean + np.inner(L, Z)
        Portfolio_sims[:,m] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*InitialPort

    # Plot simulations
    fig, ax = plt.subplots()
    for m in range(0, MonteCarlo_sims):
        ax.plot(Portfolio_sims[:, m])

    ax.set_xlabel('Days')
    ax.set_ylabel('Portfolio Value ($)')
    ax.set_title('Stock Portfolio MC')
    st.pyplot(fig)
