import sys
from ..model import feat_select
import pandas as pd
def test_ana_fmetas():
    df = pd.DataFrame(data=[
        ["name1", "fname1",1,2,100,1,1.10,100],
        ["name2", "fname2",2,3,101,-1,0.90,100],
        ["name3", "fname3",1,2,99,1,1.20,100],
        ["name4", "fname4",1,2,99,-1,0.80,100],

    ],
    columns=["name", "fname", "start", "end", "score", "direct",
                "chvfa", "n_samples"]
    )

    #feat_select.ana_fmetas(df, sys.stdout)


def test_apply():
    dfmetas = pd.DataFrame(data=[
        ["name1", "fname1",1,2,100,1,1.10,100],
        ["name2", "fname1",2,3,101,-1,0.90,100],
        ["name3", "fname2",1,2,99,1,1.20,100],
        ["name4", "fname2",2,3,99,-1,0.80,100],

    ],
    columns=["name", "fname", "start", "end", "score", "direct",
                "p_chvfa", "n_samples"]
    )

    p2 = pd.DataFrame(data=[
        [1.1, 1.1,1.1],
        [1.1, 1.1,0.9],
        [2.1, 2.1,1.1],
        [2.1, 2.1,0.9],

    ],
    columns=["fname1", "fname2", "label5"]
    )

    print feat_select.apply(dfmetas, p2, "label5", "_p2")

