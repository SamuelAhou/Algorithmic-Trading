from src.strategy import Strategy
import pandas as pd
import numpy as np
import statsmodels.api as sm

"""
def pairs_trading_strategy(closeA: pd.Series, closeB: pd.Series,
                           entry_threshold: float, exit_threshold: float,
                           spread_type: str) -> tuple:
    
    # Filter NA 
    filter_na = closeA.notna() & closeB.notna()
    closeA = closeA[filter_na]
    closeB = closeB[filter_na]

    # Compute the returns
    retA = closeA.pct_change()
    retB = closeB.pct_change()

    # Compute the spread
    X = sm.tools.add_constant(closeB)
    model = sm.regression.linear_model.OLS(closeA, X)
    model = model.fit()
    hedge_ratio = model.params.iloc[1]

    # Compute the spread
    if spread_type == 'zscore':
        spread = closeA - hedge_ratio*closeB
        spread = (spread - spread.rolling(20).mean())/spread.rolling(20).std()
    elif spread_type == 'ratio':
        spread = closeA/closeB
    elif spread_type == 'log-difference':
        spread = np.log(closeA) - np.log(closeB)
    else:
        raise ValueError('Invalid spread type')

    # Compute the longs, shorts and exits
    longs = []
    shorts = []
    exits = []

    for i in range(1, len(spread)):
        if spread.iloc[i] > entry_threshold and spread.iloc[i-1] < entry_threshold:
            shorts.append(i)
        elif spread.iloc[i] < -entry_threshold and spread.iloc[i-1] > -entry_threshold:
            longs.append(i)
        elif abs(spread.iloc[i]) < exit_threshold and abs(spread.iloc[i-1]) > exit_threshold:
            exits.append(i)

    return longs, shorts, exits, spread

"""

"""
Strategy class is the parent class for all strategies. It contains the basic
methods that all strategies should have.

Attributes:

    - name: str
        The name of the strategy.
    - data: pd.DataFrame 
        The data that the strategy will use to make decisions.
        Can contain multiple assets.
        Should contain the following columns for each asset:
            - 'open': The opening price of the asset.
            - 'high': The highest price of the asset.
            - 'low': The lowest price of the asset.
            - 'close': The closing price of the asset.
            - 'volume': The volume
    - params: dict
        The parameters of the strategy.
    - init_cash: float
        The initial cash that the strategy has.

    - signals: pd.DataFrame
        The signals that the strategy generates. 
        Contains an arbitrary but fixed number of columns for each asset.
        This dataframe is created by the generate_signals method.
    - positions: pd.DataFrame
        The positions that the strategy generates for each asset in the data.
        Contains 2 columns for each asset in the data:
            - 'position': The current position of the strategy for the asset.
            - 'order_size': The size of the order that the strategy will place.
        This dataframe is created by the generate_positions method.
    - pnl: pd.DataFrame
        The profit and loss that the strategy generates.
        Contains a single column for each asset in the data:
            - 'pnl': The profit and loss of the strategy for the asset.
        This dataframe is created by the generate_pnl method.
    
Methods:

    - __init__(self, name, data)
        Initializes the strategy with the given name and data.
    - generate_signals(self)
        Generates the signals for the strategy.
    - generate_positions(self)
        Generates the positions for the strategy.
    - run(self)
        Runs the strategy and generates the pnl.
    - evaluate(self)
        Computes some metrics to evaluate the strategy.
    - plot(self)
        Plots the signals, positions and pnl.
    - save(self, path)
        Saves the signals, positions and pnl to the given path.
    - load(self, path)
        Loads the signals, positions and pnl from the given path.
"""

class PairsTrading(Strategy):

    def __init__(self, name: str, data: pd.DataFrame, params: dict):
        super().__init__(name, data, params)
        self.entry_threshold = params['entry_threshold']
        self.exit_threshold = params['exit_threshold']
        self.spread_type = params['spread_type']
        self.assets = data.columns

    def generate_signals(self):
        closeA = self.data[self.assets[0]].Close
        closeB = self.data[self.assets[1]].Close

        # Filter NA
        filter_na = closeA.notna() & closeB.notna()
        closeA = closeA[filter_na]
        closeB = closeB[filter_na]

        # Compute the spread
        if self.spread_type == 'zscore':
            X = sm.tools.add_constant(closeB)
            model = sm.regression.linear_model.OLS(closeA, X)
            model = model.fit()
            hedge_ratio = model.params.iloc[1]

            spread = closeA - hedge_ratio*closeB
            spread = (spread - spread.rolling(20).mean())/spread.rolling(20).std()
        elif self.spread_type == 'ratio':
            spread = closeA/closeB
        elif self.spread_type == 'log-difference':
            spread = np.log(closeA) - np.log(closeB)
        else:
            raise ValueError('Invalid spread type')
        
        self.signals['spread'] = spread


    def generate_positions(self):

        self.positions = pd.DataFrame(index=self.data.index, columns=index)

        for asset in self.assets:
            pass


        
    def generate_pnl(self):
        self.pnl = pd.DataFrame(index=self.data.index)

        for asset in self.data.columns:
            self.pnl[asset] = self.positions[asset + '_position'].shift(1)*self.data[asset].pct_change()

    def evaluate(self):
        self.metrics = {}

        for asset in self.data.columns:
            self.metrics[asset] = {
                'total_return': self.pnl[asset].sum(),
                'annualized_return': self.pnl[asset].mean()*252,
                'annualized_volatility': self.pnl[asset].std()*np.sqrt(252),
                'sharpe_ratio': self.pnl[asset].mean()/self.pnl[asset].std()*np.sqrt(252)
            }

    def plot(self):
        pass

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass