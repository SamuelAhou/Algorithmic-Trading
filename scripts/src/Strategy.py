import pandas as pd
from matplotlib import pyplot as plt
import talib as ta

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

class Strategy:

    def __init__(self, name: str, data: pd.DataFrame, params: dict, init_cash: float=100_000):
        """
        Initializes the strategy with the given name and data.

        Args:
            name (str): The name of the strategy.
            data (pd.DataFrame): The data that the strategy will use to make decisions.
            params (dict): The parameters of the strategy.
            init_cash (float): The initial cash that the strategy has.
        
        Returns:
            None
        """

        self.name = name
        self.data = data
        self.params = params
        self.init_cash = init_cash

        self.signals = pd.DataFrame(index=self.data.index)
        self.positions = pd.DataFrame(index=self.data.index)
        self.pnl = pd.DataFrame(index=self.data.index)


    def generate_signals(self):
        """
        Generates the signals for the strategy.
        
        Returns:
            None
        """

        raise NotImplementedError


    def generate_positions(self):
        """
        Generates the positions for the strategy.

        Returns:
            None
        """

        raise NotImplementedError


    def run(self) -> pd.DataFrame:
        """
        Runs the strategy and generates the pnl.

        Returns:
            pd.DataFrame: The profit and loss of the strategy.
        """

        self.generate_signals()
        self.generate_positions()

        self.pnl['cash'] = self.init_cash
        self.pnl['pnl'] = 0
        self.pnl['returns'] = 0
        
        for i in range(1, len(self.data)):
            self.pnl['cash'].iloc[i] = self.pnl['cash'].iloc[i-1] + self.positions['position'].iloc[i-1] * self.data['close'].iloc[i-1]
            self.pnl['pnl'].iloc[i] = self.pnl['cash'].iloc[i] - self.init_cash
            self.pnl['returns'].iloc[i] = self.pnl['cash'].iloc[i] / self.pnl['cash'].iloc[i-1] - 1

        return self.pnl
    

    def evaluate(self) -> dict:
        """
        Computes some metrics to evaluate the strategy. The metrics are:
            - Sharpe ratio
            - Maximum drawdown
            - Annualized return
            - Annualized volatility
            - Calmar ratio
            - Sortino ratio
            - Average return
            - Average loss
            - Average win
            - Win rate
            - Loss rate
            - Number of trades
        
        Returns:
            dict: A dictionary containing the metrics.
        """

        pnl = self.pnl['pnl']
        returns = pnl.pct_change()
        returns = returns.dropna()

        # Sharpe ratio
        sharpe_ratio = returns.mean() / returns.std()

        # Maximum
        max_drawdown = 0
        max_drawdown = (pnl / pnl.cummax() - 1).min()
        max_drawdown = max_drawdown.min()

        # Annualized return
        annualized_return = returns.mean() * 252
        
        # Annualized volatility
        annualized_volatility = returns.std() * (252 ** 0.5)

        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown)

        # Sortino ratio
        average_loss = returns[returns < 0].mean()
        sortino_ratio = (annualized_return - 0) / average_loss

        # Average return
        average_return = returns.mean()
        
        # Average loss
        average_loss = returns[returns < 0].mean()

        # Average win
        average_win = returns[returns > 0].mean()

        # Win rate
        win_rate = (returns > 0).mean()

        # Loss rate
        loss_rate = (returns < 0).mean()

        # Number of trades (Count changes in positions)
        number_of_trades = (self.positions['order_size'] != 0).sum()

        return {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'average_return': average_return,
            'average_loss': average_loss,
            'average_win': average_win,
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'number_of_trades': number_of_trades
        }
    

    def plot(self):

        raise NotImplementedError
    

    def save(self, path):
        """
        Saves the signals, positions and pnl to the given path.

        Args:
            path (str): The path to save the signals, positions and pnl.

        Returns:
            None
        """
        self.signals.to_csv(path + '/signals.csv')
        self.positions.to_csv(path + '/positions.csv')
        self.pnl.to_csv(path + '/pnl.csv')

    
    def load(self, path):
        """
        Loads the signals, positions and pnl from the given path.

        Args:
            path (str): The path to load the signals, positions and pnl.

        Returns:
            None
        """
        self.signals = pd.read_csv(path + '/signals.csv', index_col=0)
        self.positions = pd.read_csv(path + '/positions.csv', index_col=0)
        self.pnl = pd.read_csv(path + '/pnl.csv', index_col=0)
    