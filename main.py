import pandas as pd
import numpy as np
import yfinance as yf

from scripts.src.strategy import Strategy
from scripts.utils import *
from scripts.strategies.PairsTrading import PairsTrading
from scripts.strategies.SMA import SMAStrategy

if __name__ == '__main__':

    data = yf.download('AAPL', start='2010-01-01', end='2020-01-01', progress=False)

    # Define the parameters for the strategies
    # params_pairs = {'entry_threshold': 1.5, 
    #                 'exit_threshold': 0.5, 
    #                 'order_size': 1.0, 
    #                 'spread_type': 'zscore'}

    # pairs_strategy = PairsTrading('Pairs Trading', data, params_pairs, 1.0)

    # pairs_strategy.run()
    # pairs_strategy.plot()

    params_sma = {'short_window': 20,
                    'long_window': 100,
                    'order_size': 1.0}
    
    sma_strategy = SMAStrategy('SMA Strategy', data, params_sma, 1.0)
    sma_strategy.run()
    sma_strategy.plot('/Users/Samuel/Documents/Projects/Algorithmic-Trading/results/SMA')
    
