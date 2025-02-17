# Algorithmic Trading Framework in Python

This repository contains a Python framework for algorithmic trading. The framework is designed to be used for backtesting strategies. 

### Features 

The framework has the following features:

- **Data**: The framework uses historical data for backtesting. The data should be downloaded using **yfinance**.
- **Strategies**: The framework supports multiple strategies that can be backtested.
- **Metrics**: The framework calculates various metrics for the strategies, such as Sharpe ratio, maximum drawdown, etc.
- **Visualization**: The framework provides visualizations for the strategies.

### Structure

The framework is structured as follows:

- `data/`: Contains the data used for backtesting.
- `scripts/src`: Contains the source code for the framework. In this directory, the following files are present:
  - `strategy.py`: Contains the base class for strategies. All strategies inherit from this class.
- `scripts/strategies/`: Contains strategies that can be backtested.
- `scripts/utils/`: Contains utility functions that are used in the framework.

### Usage

To use the framework, follow these steps:

1. Download the historical data using **yfinance**.
2. Create a new strategy by inheriting from the `Strategy` class. For that you need to implement the `generate_signals` method that generates the signals used by the strategy. Then, implement the `generate_positions` method that generates the positions based on the signals.
3. Backtest the strategy using the `run` function defined in the `Strategy` class.
4. Visualize the results using the `plot` function defined in the `Strategy` class.

You can find multiple metrics in the attribute `metrics` of the strategy object after running the backtest.

### Example

Here is an example of how to use the framework:

```python

# Import the necessary libraries
import yfinance as yf
from scripts.src.Strategy import Strategy

# Download the data
data = yf.download('AAPL', start='2010-01-01', end='2021-01-01')

# Define a simple moving average strategy

class SimpleMovingAverage(Strategy):

    def __init__(self, name: str, data: pd.DataFrame, params: dict, init_cash: float = 100_000.0):
        super().__init__(name, data, params, init_cash)
        self.short_window = params['short_window']
        self.long_window = params['long_window']
        self.signals = pd.DataFrame(index=data.index)

    def generate_signals(self):
        
        self.signals['short_mavg'] = data['Close'].rolling(window=self.short_window, min_periods=1, center=False).mean()
        self.signals['long_mavg'] = data['Close'].rolling(window=self.long_window, min_periods=1, center=False).mean()

        self.signals['signal'] = 0.0

        self.signals['signal'][self.short_window:] = np.where(self.signals['short_mavg'][self.short_window:] > self.signals['long_mavg'][self.short_window:], 1.0, 0.0)


    def generate_positions(self):
        self.signals['positions'] = self.signals['signal'].diff()
    
# Backtest the strategy
strategy = SimpleMovingAverage(data)

strategy.run()
strategy.plot()

```

Here is a sample output from the `SMA.py` strategy (in `scripts/strategies`) with short/long MA of 20 and 100 days on Apple stock from 2010 to 2020:


<img src="results/SMA/SMA%20Strategy.png" alt="drawing" width="800"/>









