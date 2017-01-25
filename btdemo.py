import bt
import matplotlib
matplotlib.get_backend()

# download data
from bt.algos import SelectWhere

data = bt.get('aapl,msft', start='2016-01-01')

# calculate moving average DataFrame using pandas' rolling_mean
import pandas as pd
# a rolling mean is a moving average, right?
sma = pd.rolling_mean(data, 50)

bt.merge(data, sma).plot(figsize=(15, 5))

signal = data > sma

# first we create the Strategy
s = bt.Strategy('above50sma', [SelectWhere(data > sma),
                               bt.algos.WeighEqually(),
                               bt.algos.Rebalance()])

# now we create the Backtest
t = bt.Backtest(s, data)

# and let's run it!
res = bt.run(t)
res.plot('d')
print(type(res))
res.display()