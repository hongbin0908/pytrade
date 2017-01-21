import os
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from .. import model_work as work
import matplotlib.pyplot as plt

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')

def test_cv():
    df = pd.read_pickle(os.path.join(root, '..', 'data', 'ta', 'base1', 'AAPL.pkl'))
    assert isinstance(df, pd.DataFrame)
    npDates = df["date"].unique()
    df.set_index(["date"], drop=True, inplace=True)
    assert df.shape == df.loc[npDates.tolist()].shape
    cv = TimeSeriesSplit(n_splits=5)
    for (train, test) in cv.split(npDates):
        train_size = len(df.loc[npDates[train]])
        test_size = len(df.loc[npDates[test]])
    assert len(df) == train_size + test_size

def test_date_split():
    df = pd.read_pickle(os.path.join(root, '..', 'data', 'ta', 'base1', 'AAPL.pkl'))
    for (train, test) in work.date_split(df, n_splits=5):
        train_size = len(train)
        test_size = len(test)
    assert len(df) == train_size + test_size

def test_RandClassifer():
    from ..model_work import RandClassifer
    classifer = RandClassifer(0.8)
    X = np.arange(10000)
    y = classifer.predict_proba(X)
    plt.hist(y[:,0], 30, normed=True)
    plt.savefig(os.path.join(local_path, "test_RandClassifer.png"))




