import os
import sys
import pandas as pd
import bt
from bt.algos import SelectWhere

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

def f(x):
    if x[0]:
        x.fill(True)
    print(x)
    return x



def run(pred_file):
    #df = pd.read_pickle(os.path.join(root, "data", "cross", "pred2017-01-09.pkl"))
    df = pd.read_pickle(pred_file)
    df = df["2010-01-01":]
    df.reset_index(drop=False, inplace = True)
    df["datetime"] = pd.to_datetime(df["index"])
    df = df.set_index('datetime')
    del df["index"]

    signal = df > 0.7
    for col in range(signal.shape[1]):
        eng = 0
        for i in range(signal.shape[0]):
            if signal.iloc[i, col]:
                eng = 4
            elif eng > 0:
                signal.iloc[i, col] = True
                eng -= 1
            else:
                pass


    # first we create the Strategy
    s = bt.Strategy('above50sma', [SelectWhere(signal),
                                   bt.algos.WeighEqually(),
                                   bt.algos.Rebalance()])
    # now we create the Backtest
    t = bt.Backtest(s, df, initial_capital=9000)

    # and let's run it!
    res = bt.run(t)
    res.plot('d')
    ser = res._get_series('d').rebase()
    plot = ser.plot()
    fig = plot.get_figure()
    fig.savefig(os.path.join(root, "report", "backtest.png"))
    #res.display()

