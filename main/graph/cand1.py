#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author hongbin@youzan.com
import os,sys
import talib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.finance import candlestick
from matplotlib.dates import DateFormatter, WeekdayLocator,DayLocator, MONDAY
from matplotlib.ticker import FixedLocator,FuncFormatter
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)
import main.base as base


def main(argv):
    sym = argv[0]
    start = argv[1]
    end = argv[2]
    df = base.ta('call1s1_dow', sym)
    df = df.dropna()
    df = df[(df.date >= start) & (df.date <= end)]
    df["idx"] = np.arange(len(df))
    up = (df.open<=df.close)
    down = (df.open>df.close)
    fig = plt.figure(figsize = (10,5))
    ax = fig.add_subplot(211)

    ax.vlines(df.idx, df.low, df.high, edgecolor='black', linewidth=1.0)
    ax.vlines(df.idx[up], df.open[up], df.close[up], edgecolor ='green', linewidth=3.0)
    ax.vlines(df.idx[down], df.open[down], df.close[down], edgecolor ='red', linewidth=3.0)


    dates = [base.strDate2num(each) for each in df.date]
    print dates
    years = set([d.year for d in dates])

    mdindex = []
    for y in sorted(years):
        months = set([d.month for d in dates if d.year == y])
        for m in sorted(months):
            monthday = min([dt for dt in dates if dt.year == y and dt.month==m])
            mdindex.append(dates.index(monthday))
    xMajorLocator = FixedLocator(np.array(mdindex))
    def x_major_formatter(idx, pos = None):
        return dates[int(idx)].strftime('%Y-%m-%d')
    xMajorFormatter = FuncFormatter(x_major_formatter)
    ax.xaxis.set_major_locator(xMajorLocator)
    ax.xaxis.set_major_formatter(xMajorFormatter)


    wdindex = {}
    for d in dates:
        isoyear, weekno = d.isocalendar()[0:2]
        dmark = isoyear * 100 + weekno
        if dmark not in wdindex:
            wdindex[dmark] = dates.index(d)
    xMinorLocator = FixedLocator(np.array(sorted(wdindex.values())))
    def x_minor_formatter(idx, pos = None):
        return dates[int(idx)].strftime("%m-%d")
    xMinorFormatter = FuncFormatter(x_minor_formatter)
    ax.xaxis.set_minor_locator(xMinorLocator)
    ax.xaxis.set_minor_formatter(xMinorFormatter)

    ax2 = fig.add_subplot(212,sharex=ax)
    ax2.plot(df.idx, df.ta_STOCHRSI_slowd_2_10_6)
    for malabel in ax2.get_xticklabels(minor=False):
        malabel.set_fontsize(12)
        malabel.set_horizontalalignment('right')
        malabel.set_rotation('45')

    for milabel in ax2.get_xticklabels(minor=True):
        milabel.set_fontsize(10)
        milabel.set_color('blue')
        milabel.set_horizontalalignment('right')
        milabel.set_rotation('45')
    plt.show()

if __name__ == '__main__':
    main(sys.argv[1:])



