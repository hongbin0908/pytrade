#!/usr/bin/env python
# author hongbin0908@126.com
import os, sys

class BaseScore():
    def get_name(self):
        pass


class ScoreRelative(BaseScore):
    def __init__(self, interval, index=False):
        self.interval = interval
        self.index = index

    def get_name(self):
        return "score_rel_%d_%d" % (self.interval, self.index)

    def agn_score(self, df):
        if self.get_name() in df.columns:
            return df
        df.sort_values("date", ascending=True, inplace=True)
        if self.index:
            df.loc[:,"close2"] = df.loc[:,"close"]/df.loc[:,"iclose"]*2000
        else:
            df.loc[:,"close2"] = df.loc[:,"close"]
        df.loc[:, "close_shift"] = df["close2"].shift(-1 * self.interval)
        df.loc[:, self.get_name()] = df["close_shift"]/df["close2"]
        del df["close_shift"]
        return df


class ScoreRelativeOpen(BaseScore):
    def __init__(self, interval):
        self.interval = interval

    def get_name(self):
        return "score_rel_open_%d" % self.interval

    def agn_score(self, df):
        if self.get_name() in df.columns:
            return df
        df.sort_values("date", ascending=True, inplace=True)
        df.loc[:,"open2"] = df.loc[:,"open"].shift(-1)
        df.loc[:, "open_shift"] = df["open2"].shift(-1 * self.interval)
        df.loc[:, self.get_name()] = df["open_shift"]/df["open2"]
        del df["open_shift"]
        del df["open2"]
        return df


class ScoreLabel(BaseScore):
    def __init__(self, interval, threshold, index=False):
        self.interval = interval
        self.threshold = threshold
        self.index = index

    def get_name(self):
        return "score_label_%d_%d_%d" % (self.interval, int(self.threshold * 100), self.index)

    def agn_score(self, df):
        if self.get_name() in df.columns:
            return df
        rel_score = ScoreRelative(self.interval, self.index)
        df = rel_score.agn_score(df)
        assert len(df) > 0
        df.loc[:, self.get_name()] = \
            df.apply(lambda x: 1 if x[rel_score.get_name()] >= self.threshold else 0,
                     axis=1)
        return df
