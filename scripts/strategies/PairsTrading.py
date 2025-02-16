from ..src.strategy import Strategy
import pandas as pd
import numpy as np
import statsmodels.api as sm


class PairsTrading(Strategy):

    def __init__(self, name: str, data: pd.DataFrame, params: dict, init_cash: float= 100_000.0):
        super().__init__(name, data, params, init_cash)
        
        assert type(params['entry_threshold']) == float and type(params['exit_threshold']) == float
        self.entry_threshold = params['entry_threshold']
        self.exit_threshold = params['exit_threshold']

        assert type(params['order_size']) == float
        self.order_size = params['order_size']

        assert params['spread_type'] in ['zscore', 'ratio', 'log-difference']
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

        for i in range(1, len(self.signals)):
            # Asset 1 overperforms Asset 2 -> Short Asset 1, Long Asset 2
            if self.signals['spread'].iloc[i] > self.entry_threshold and self.signals['spread'].iloc[i-1] < self.entry_threshold:
                self.positions.loc[self.data.index[i], (self.assets[0], 'position')] -= self.order_size
                self.positions.loc[self.data.index[i], (self.assets[0], 'order_size')] = -self.order_size 
                self.positions.loc[self.data.index[i], (self.assets[1], 'position')] += self.order_size
                self.positions.loc[self.data.index[i], (self.assets[1], 'order_size')] = self.order_size
            # Asset 1 underperforms Asset 2 -> Long Asset 1, Short Asset 2
            elif self.signals['spread'].iloc[i] < -self.entry_threshold and self.signals['spread'].iloc[i-1] > -self.entry_threshold:
                self.positions.loc[self.data.index[i], (self.assets[0], 'position')] += self.order_size
                self.positions.loc[self.data.index[i], (self.assets[0], 'order_size')] = self.order_size
                self.positions.loc[self.data.index[i], (self.assets[1], 'position')] -= self.order_size
                self.positions.loc[self.data.index[i], (self.assets[1], 'order_size')] = -self.order_size
            # Spread reverts to the mean -> Close positions
            elif abs(self.signals['spread'].iloc[i]) < self.exit_threshold and abs(self.signals['spread'].iloc[i-1]) > self.exit_threshold:
                self.positions.loc[self.data.index[i], (self.assets[0], 'position')] = 0    
                self.positions.loc[self.data.index[i], (self.assets[0], 'order_size')] = 0
                self.positions.loc[self.data.index[i], (self.assets[1], 'position')] = 0
                self.positions.loc[self.data.index[i], (self.assets[1], 'order_size')] = 0

