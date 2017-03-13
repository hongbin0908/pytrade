import sys
import os
import pandas as pd
import numpy as np
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)
from main import base

class Selector:
    def __init__(self, confs):
        self.confs = confs
    def _selector_name(self):
        pass
    def get_name(self):
        name = self._selector_name()+"-"
        for conf in self.confs:
            name += conf.name_bitlize + "-"
        return name
    def _select(self,df, start, end, score):
        pass
    def work(self):
        appender = []
        for conf in self.confs: 
            df = pd.read_pickle(conf.get_bitlize_file())
            df["ds"] = df["date"] + df["sym"]
            df = df.set_index("ds")
            res = self._select(df,\
                    conf.model_split.train_start,\
                    conf.model_split.train_end,\
                    conf.scores[0].get_name())
            tabase_names = base.get_tabase_names(df)
            df_new = pd.concat([df[tabase_names], df[list(res.index)]], axis=1, join_axes=[df.index])
            assert len(df_new) == len(df)
            appender.append(df_new)
        return pd.concat(appender, axis=1)


class PnRatioSelector(Selector):
    """
    pn ratio selector is simplize of mutual index.
    n11: when a feature is True and label is 1
    n10: when a feature is True and label is 0.
    pn ratio = n11/(n11+n10)

    common ratio = num(label=1)/num(all)
    obvirsly, when pn ratio > common ratio,the feature is good
    """
    def __init__(self, confs, threshold=0.54):
        self.confs = confs
        self.threshold = threshold
    def _selector_name(self):
        return "pn_ratio"
    def _select(self,df,start,end, score):
        df = df[(df.date >= start) & (df.date < end)]
        #df = df[(df.date >= "2013-01-01") & (df.date < "2014-01-01")]
        feat_names = base.get_feat_names(df)
        label = df.loc[:,score]
        res = pd.DataFrame(data = None, index=feat_names)
        df = df[feat_names]
        n11 = df[label > 0.5].sum()
        n10 = df[label < 0.5].sum()
        res["n11"] = n11
        res["n10"] = n10
        res["pn_ratio"] = res["n11"]/(res["n10"]+res["n11"])
        #res = res[(res["pn_ratio"]<=self.threshold)&(res["pn_ratio"]>=1-self.threshold)]
        res = res.sort_values("pn_ratio", ascending=False)
        return res.tail(10)
class MiSelector(Selector):
    def _selector_name(self):
        return "mi"
    def _select(self, df, start, end, score):
        """
        http://nlp.stanford.edu/IR-book/html/htmledition/mutual-information-1.html#mifeatsel
        """
        df = df[(df.date >= start) & (df.date < end)]
        feat_names = base.get_feat_names(df)
        label = df.loc[:,score]
        res = pd.DataFrame(data = None, index=feat_names)
        df = df[feat_names]
        n11 = df[label > 0.5].sum()
        n10 = df[label < 0.5].sum()
        n01 = (1-df[label>0.5]).sum()
        n00 = (1-df[label<0.5]).sum()
        n = df.count()
        n1_ = n11+n10
        n0_ = n01+n00
        n_1 = n01+n11
        n_0 = n00+n10
    
        assert 0 == (n11 + n01 - df[label>0.5].count()).sum() 
        assert 0 == (n11 + n01 + n10 + n00 - df.count()).sum() 
        
    
        mi = n11/n*np.log2(n*n11/(n1_*n_1)) + n01/n*np.log2(n*n01/(n0_*n_1)) \
                + n10/n*np.log2(n*n10/(n1_*n_0)) + n00/n*np.log2(n*n00/(n0_*n_0))
    
        res["mi"] = mi
        res["pn_ratio"] = n11/(n11+n10)
        res = res.sort_values("mi", ascending=False)

        return res
class Correlation(Selector):
    def _selector_name(self):
        return "correlation"
    def _select(self, df, start, end, score):
        df = df[(df.date >= start)&(df.date < end)]
        feat_names = base.get_feat_names(df)
        label = df.loc[:, score]

