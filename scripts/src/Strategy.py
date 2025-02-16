import pandas as pd
import matplotlib.pyplot as plt
from rich.progress import track


"""
Strategy class is the parent class for all strategies. It contains the basic
methods that all strategies should have.

Attributes:

    - name: str
        The name of the strategy.
    - data: pd.DataFrame 
        The data that the strategy will use to make decisions.
        Can contain multiple assets.
        Should contain the following columns:
            - 'Open': The opening price of the asset.
            - 'High': The highest price of the asset.
            - 'Low': The lowest price of the asset.
            - 'Close': The closing price of the asset.
            - 'Volume': The volume
        MultiIndex columns with the first level being the column names and the second level the assets.
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

        self.assets = self.data.columns.get_level_values(1).unique()

        self.signals = pd.DataFrame(index=self.data.index)

        positions_iter = [self.assets, ['position', 'order_size']]
        positions_index = pd.MultiIndex.from_product(positions_iter)
        self.positions = pd.DataFrame(index=self.data.index, columns=positions_index)

        pnl_cols = ['cash', 'pnl', 'returns'] 
        self.pnl = pd.DataFrame(index=self.data.index, columns=pnl_cols)
        self.pnl['cash'] = self.init_cash
        self.pnl['pnl'] = 0
        self.pnl['returns'] = 0

        self.metrics = {}


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


    def run(self):
        """
        Runs the strategy and generates the pnl.
        """

        self.generate_signals()
        self.generate_positions()
        
        for i in track(range(1, len(self.data)), description= 'Running Strategy...'):
            idx = self.data.index[i]
            idx_prev = self.data.index[i-1]

            self.pnl.loc[idx, 'cash'] = self.pnl.loc[idx_prev, 'cash']

            for asset in self.assets:
                order_size = self.positions.loc[idx_prev, (asset, 'order_size')]
                position = self.positions.loc[idx_prev, (asset, 'position')]

                # Update cash
                self.pnl.loc[idx, 'cash'] -= order_size * self.data['Close'][asset].loc[idx]

                # Update pnl
                self.pnl.loc[idx, 'pnl'] += position * (self.data['Close'][asset].loc[idx] - self.data['Close'][asset].loc[idx_prev])

        # Compute returns
        self.pnl['returns'] = self.pnl['cash'].pct_change(fill_method=None)
        self.pnl.loc[self.data.index[0], 'returns'] = 0
            

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
        """

        print('Evaluating Strategy...')

        pnl = self.pnl['pnl']
        returns = pnl.pct_change(fill_method=None)
        returns = returns.dropna()

        # Sharpe ratio
        self.metrics['sharpe_ratio'] = returns.mean() / returns.std()

        # Maximum
        max_drawdown = (self.pnl['returns'].cumsum() - 
                        self.pnl['returns'].cumsum().cummax())/self.pnl['returns'].cumsum().cummax()
        self.metrics['max_drawdown'] = max_drawdown.min()

        # Annualized return
        self.metrics['annualized_return'] = returns.mean() * 252
        
        # Annualized volatility
        self.metrics['annualized_volatility'] = returns.std() * (252 ** 0.5)

        # Calmar ratio
        self.metrics['calmar_ratio'] = self.metrics['annualized_return'] / abs(max_drawdown)

        # Sortino ratio
        self.metrics['average_loss'] = returns[returns < 0].mean()
        self.metrics['sortino_ratio'] = (self.metrics['annualized_return'] - 0) / self.metrics['average_loss']

        # Average return
        self.metrics['average_return'] = returns.mean()

        # Average win
        self.metrics['average_win'] = returns[returns > 0].mean()

        # Win rate
        self.metrics['win_rate'] = (returns > 0).mean()

        # Loss rate
        self.metrics['loss_rate'] = (returns < 0).mean()

        # Number of trades (Count changes in positions), sum over all assets
        self.metrics['number_of_trades'] = (self.positions.loc[:, (slice(None), 'order_size')] != 0).sum().sum()
    

    def plot(self, dir_path: str, start_date=None, end_date=None):
        """
        Plots the assets' close prices, volume, signals, positions and pnl.
        Saves the plot to dir_path/{self.name}.pdf file.
        Args:
            dir_path (str): The directory path to save the plot.
            start_date (str, optional): The start date to plot. Defaults to self.data.index[0].
            end_date (str, optional): The end date to plot. Defaults to self.data.index[-1].

        Returns:
            None
        """

        print('Plotting ...')

        plt.rcParams.update({'text.usetex': True,
                     'font.size': 16,
                     'legend.fontsize': 12,
                     'legend.title_fontsize': 16,
                     'legend.loc': 'upper right',
                     'figure.titlesize': 16,
                     'axes.labelsize': 14,
                     })

        if start_date is None:
            start_date = self.data.index[0]
        if end_date is None:
            end_date = self.data.index[-1]
        

        fig = plt.figure(figsize=(15, 15))
        gs = fig.add_gridspec(4, 1, width_ratios = [1], height_ratios=[4, 1, 4, 4])

        ax = []
        ax.append(fig.add_subplot(gs[0, 0]))
        ax.append(fig.add_subplot(gs[1, 0], sharex=ax[0]))
        ax.append(fig.add_subplot(gs[2, 0], sharex=ax[0]))
        ax.append(fig.add_subplot(gs[3, 0], sharex=ax[0]))

        title = self.name + ' - ' + 'Sharpe = ' + str(self.metrics['sharpe_ratio']) + \
                ', ' + 'Max Drawdown = ' + str(self.metrics['max_drawdown']) + \
                ', ' + 'Annualized Return = ' + str(self.metrics['annualized_return'])
        
        # Plot close prices
        ax[0].plot(self.data.loc[start_date:end_date, ('Close', slice(None))])
        ax[0].set_title(title)
        ax[0].set_ylabel('Close Price')
        ax[0].legend(self.assets)
        ax[0].grid()

        # Plot volume
        for asset in self.assets:
            ax[1].bar(self.data.loc[start_date:end_date].index, self.data.loc[start_date:end_date, ('Volume', asset)], alpha=0.7)
        ax[1].set_ylabel('Volume')
        ax[1].legend(self.assets)
        ax[1].grid()
        
        # Plot signals
        ax[2].plot(self.signals.loc[start_date:end_date])
        ax[2].set_ylabel('Signals')
        ax[2].legend(self.signals.columns)
        ax[2].grid()

        # Plot positions
        ax[3].plot(self.positions.loc[start_date:end_date, (slice(None), 'position')])
        ax[3].set_ylabel('Positions')
        ax[3].legend(self.assets)
        ax[3].grid()

        plt.setp(ax[0].get_xticklabels(), visible=False)
        plt.setp(ax[1].get_xticklabels(), visible=False)
        plt.setp(ax[2].get_xticklabels(), visible=False)

        plt.savefig(dir_path + '/' + self.name + '.pdf')
        plt.close()
    

    def save(self, dir_path):
        """
        Saves the signals, positions and pnl to the given path in CSV format.

        Args:
            path (str): The path to save the signals, positions and pnl.

        Returns:
            None
        """
        self.signals.to_csv(dir_path + '/signals.csv')
        self.positions.to_csv(dir_path + '/positions.csv')
        self.pnl.to_csv(dir_path + '/pnl.csv')

    
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
    