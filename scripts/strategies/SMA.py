import pandas as pd
import numpy as np
from ..src.strategy import Strategy
from rich.progress import track


class SMAStrategy(Strategy):

    def __init__(self, name, data, params, init_cash):
        super().__init__(name, data, params, init_cash)
        self.short_window = params['short_window']
        self.long_window = params['long_window']
        self.order_size = params['order_size']
        

    def generate_signals(self):
        
        self.signals['short_mavg'] = self.data['Close'].rolling(window=self.short_window, min_periods=1, center=False).mean()
        self.signals['long_mavg'] = self.data['Close'].rolling(window=self.long_window, min_periods=1, center=False).mean()

        self.signals['long'] = 0
        self.signals['short'] = 0
        self.signals['exits'] = 0

        self.signals.loc[self.data.index[self.short_window:], 'long'] = np.where(self.signals.loc[self.data.index[self.short_window:], 'short_mavg'] > 
                                                                                 self.signals.loc[self.data.index[self.short_window:], 'long_mavg'], 1, 0)
        self.signals['long'] = np.where(self.signals['long'].diff() == 1, 1, 0)
        self.signals.loc[self.data.index[self.short_window:], 'short'] = np.where(self.signals.loc[self.data.index[self.short_window:], 'short_mavg'] < 
                                                                                  self.signals.loc[self.data.index[self.short_window:], 'long_mavg'], -1, 0)
        self.signals['short'] = np.where(self.signals['short'].diff() == -1, -1, 0)

        self.signals['positions'] = self.signals['long'] + self.signals['short']


        entries = self.signals.index[self.signals['positions'] != 0].tolist()
        for entry in track(entries, description='Generating Signals ...'):
            # Remember entry prices for long and short positions
            if self.signals['positions'].loc[entry] == 1:
                long_entry_price = self.data['Close'].loc[entry]

                # Compute the indices where the price has increased by 1% or decreased by 0.5% after entry
                long_increased = self.data['Close'].loc[entry:] > long_entry_price * 1.01
                long_decreased = self.data['Close'].loc[entry:] < long_entry_price * 0.995

                # Exit Signal for long position
                long_exit_idx = (long_increased | long_decreased).ne(0).idxmax().iloc[0]
                # Find the next short signal after the exit signal
                next_short_signal = self.signals['short'].loc[long_exit_idx:].ne(0).idxmax()

                # If the next short signal is before the exit signal, then exit the position
                if next_short_signal < long_exit_idx:
                    long_exit_idx = next_short_signal

                self.signals.loc[long_exit_idx, 'exits'] = 1

            elif self.signals['positions'].loc[entry] == -1:
                short_entry_price = self.data['Close'].loc[entry]

                short_decreased = self.data['Close'].loc[entry:] < short_entry_price * 0.99
                short_increased = self.data['Close'].loc[entry:] > short_entry_price * 1.005

                short_exit_idx = (short_increased | short_decreased).ne(0).idxmax().iloc[0]
                next_long_signal = self.signals['long'].loc[short_exit_idx:].ne(0).idxmax()

                if next_long_signal < short_exit_idx:
                    short_exit_idx = next_long_signal
                
                self.signals.loc[short_exit_idx, 'exits'] = 1

        self.signals.drop(['long', 'short'], axis=1, inplace=True)

    def generate_positions(self):

        print('Generating Positions ...')

        buys = self.signals['positions'] == 1
        sells = self.signals['positions'] == -1
        exits = self.signals['exits'] == 1

        self.positions.loc[buys, (self.assets[0], 'order_size')] = self.order_size
        self.positions.loc[sells, (self.assets[0], 'order_size')] = -self.order_size
        self.positions.loc[exits, (self.assets[0], 'position')] = 0
        self.positions.loc[exits, (self.assets[0], 'order_size')] = -self.positions.loc[exits, (self.assets[0], 'position')]

        self.positions[(self.assets[0], 'position')] = self.positions[(self.assets[0], 'order_size')].cumsum()