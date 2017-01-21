#!/usr/bin/env python
# author hongbin0908@126.com

import os, sys
import pandas as pd

class BaseScore():
    def get_name(self):
        pass


class ScoreRank():
    def __init__(self, interval, threshold):
        self.interval = interval
        self.threshold = threshold
    def get_name(self):
        return "score_rank_%d_%d" % (self.interval, int(self.threshold * 100))
    def agn_score(self, df):
        score_name = self.get_name()
        if score_name in df.columns:
            return df
        rel_score = ScoreRelative(self.interval)
        df = rel_score.agn_score(df)
        df.loc[:, "rank_tmp"] = df.groupby('date')[rel_score.get_name()].rank()
        tmp = pd.DataFrame(
            pd.Series(df.groupby('date')[rel_score.get_name()].count(),
                                     name="count_"))
        tmp.reset_index(drop=False, inplace=True)
        df = df.merge(tmp, left_on='date', right_on="date", how='left')
        df.loc[:, score_name] = \
            df.apply(lambda x: 1 if x["rank_tmp"] > x["count_"] * self.threshold \
                else 0, axis=1)
        del df["rank_tmp"]
        return df


class ScoreRelative():
    def __init__(self, interval):
        self.interval = interval

    def get_name(self):
        return "score_rel_%d" % self.interval

    def agn_score(self, df):
        if self.get_name() in df.columns:
            return df
        df.sort_values("date", ascending=True, inplace=True)
        df.loc[:, "close_shift"] = \
            df.groupby("sym")["close"].shift(-1 * self.interval)
        df.loc[:, self.get_name()] = df["close_shift"]/df["close"]
        del df["close_shift"]
        return df


class ScoreLabel():
    def __init__(self, interval, threshold):
        self.interval = interval
        self.threshold = threshold

    def get_name(self):
        return "score_label_%d_%d" % (self.interval, int(self.threshold * 100))

    def agn_score(self, df):
        if self.get_name() in df.columns:
            return df
        rel_score = ScoreRelative(self.interval)
        df = rel_score.agn_score(df)
        df.loc[:, self.get_name()] = \
            df.apply(lambda x: 1 if x[rel_score.get_name()] > self.threshold else 0,
                     axis=1)
        return df
