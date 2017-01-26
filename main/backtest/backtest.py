import os
import sys
import pandas as pd
import numpy as np
import bt
from bt.algos import SelectWhere

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

from main import base

def f(x):
    if x[0]:
        x.fill(True)
    print(x)
    return x

def assert_cotinue(df):
    for sym in df.columns:
        s = df[sym]
        assert isinstance(s, pd.Series)
        BEGIN = 0; BODY=1; END=2
        phrase = BEGIN
        for date, each in s.iteritems():
            if BEGIN == phrase:
                if not np.isnan(each):
                    phrase = BODY
                    continue
            elif BODY == phrase:
                if np.isnan(each):
                    phrase = END
                    continue
            else:
                if not np.isnan(each):
                    assert(False)
def transfer():
    df_pred = pd.read_pickle(os.path.join(root, "data", "cross", "pred%s.pkl" % base.last_trade_date()))
    df_pred = df_pred[df_pred.date >= "2010-01-01"]
    index = df_pred[df_pred.sym == "MSFT"]["date"].unique(); index.sort()
    index = pd.to_datetime(index)
    columns = df_pred["sym"].unique(); columns.sort()
    df_price = pd.DataFrame(index = index, columns = columns)
    df_thred = pd.DataFrame(index = index, columns = columns)

    for i, each in df_pred.iterrows():
        if each["date"] not in df_price.index:
            continue
        df_price.set_value(each["date"], each["sym"], each["close"])
        df_thred.set_value(each["date"], each["sym"], each["pred"])
    # assert
    assert_cotinue(df_price)
    assert_cotinue(df_thred)

    return (df_price, df_thred)

        
        
     
def run():
    df_price, df_thred = transfer()
    signal = df_thred > 0.7
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
    t = bt.Backtest(s, df_price, initial_capital=9000)

    # and let's run it!
    res = bt.run(t)
    res.plot('d')
    ser = res._get_series('d').rebase()
    plot = ser.plot()
    fig = plot.get_figure()
    fig.savefig(os.path.join(root, "report", "backtest.png"))
    #res.display()

if __name__ == '__main__':
    run()