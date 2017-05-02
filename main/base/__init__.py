#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#@author  Bin Hong

"""
the base method use by pytrade
"""

import sys,os
import json
import numpy as np
import pandas as pd
import math
import multiprocessing
import datetime as dt
import time
import platform
import socket
import zipfile
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

def is_test_flag():
    if platform.platform().startswith("Windows"):
        return True
    elif platform.platform().startswith("Darwin"):
        return True
    elif 'PYTEST' in os.environ and os.environ['PYTEST'] == '1':
        return True
    else:
        return False


def to_pickles(df, picklename):
    if not os.path.exists(picklename):
        os.makedirs(picklename)
    LEN = 200000000/len(df.columns)
    size_ = len(df)
    cursor1 = 0
    while cursor1 < size_:
        cursor2 = cursor1 + LEN
        if cursor2 > size_:
            cursor2 = size_
        tmp = df.iloc[cursor1:cursor2]
        tmp.to_pickle(os.path.join(picklename, "%d.pkl"%cursor1))
        cursor1 = cursor2

def from_pickles(picklename):
    flist = get_file_list(picklename, ext=".pkl")
    toappend = []
    for f in flist:
        toappend.append(pd.read_pickle(f))
    return pd.concat(toappend)


def get_file_list(rootdir, ext=".csv"):
    file_list = []
    for f in os.listdir(rootdir):
        if f == None or not f.endswith(ext):
            continue
        file_list.append(os.path.join(rootdir, f))
    return file_list

def get_stock_from_path(pathname):
    return os.path.splitext(os.path.split(pathname)[-1])[0]
def get_sym_from_path(pathname):
    return os.path.splitext(os.path.split(pathname)[-1])[0]

def get_feat_names(df):
    """
    the the columns of feature names to train
    """
    return sorted([x for x in df.columns if x.startswith('ta_')])
def get_tabase_names(df):
    """
    the the columns of feature names to train
    """
    return list(set(df.columns) - set([x for x in df.columns if x.startswith('ta_')]))

def get_all(taname, lsym, start="",end=""):
    sym2df = {}
    for sym in lsym:
        df = pd.read_pickle(os.path.join(dir_ta(taname), sym + ".pkl"))
        if len(start) > 0:
            df = df[(df.date >=start)&(df.date<end)]
        df["sym"] = sym
        sym2df[sym] = df
    return sym2df

def get_range(df, start ,end):
    """
    get the date between start(include) and end(*include*)
    """
    return df[(df.date>=start) & (df.date<=end)]

def merge(sym2feats,start="", end=""):
    df_merged = None
    to_appends = []
    for sym in sym2feats.keys():
        df = sym2feats[sym]
        to_appends.append(df)
    if len(to_appends) > 0:
        df_merged =  pd.concat(to_appends)
    assert isinstance(df_merged, pd.DataFrame)
    df_merged.sort_values(["date"], ascending=True, inplace=True)
    return df_merged

def get_merged_with_na(taname, lsym,start="", end=""):
    sym2ta = get_all(taname, lsym,start,end)
    df = merge(sym2ta,start,end)
    if df is None:
        return None
    if len(df) > 0 and "ta_NATR_7" in df:
        df = df[df['ta_NATR_7']>1.0]
    return df

def get_merged(taname, lsym,start = "", end =""):
    df = get_merged_with_na(taname, lsym,start,end)
    if df is None:
        return df
    df = df.replace([np.inf,-np.inf],np.nan).dropna()
    return df


def extract_feat_label(df, scorename, drop = True):
    if drop:
        df = df.replace([np.inf,-np.inf],np.nan).dropna()
    feat_names = get_feat_names(df)
    npFeat = df.loc[:,feat_names].values.copy()
    npLabel = df.loc[:,scorename].values.copy()
    return npFeat, npLabel

def get_last_trade_date():
    """
    get the last trade date
    """
    df = pd.read_csv(os.path.join(local_path, '..','..','data' , 'yeod', 'index', '^GSPC.csv'))
    return df.date.max()


def strDate2num(str):
    df =dt.datetime.strptime(str, "%Y-%m-%d")
    return df
    #time_sec_float = time.mktime(df.timetuple())
    #return time_sec_float

def dir_eod():
    return os.path.join(local_path, '..','..', 'data', 'yeod')


def zip_folder(folder_path, output_path):
    """Zip the contents of an entire folder (with that folder included
    in the archive). Empty subfolders will be included in the archive
    as well.
    """
    parent_folder = os.path.dirname(folder_path)
    # Retrieve the paths of the folder contents.
    contents = os.walk(folder_path)
    try:
        zip_file = zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED)
        for root, folders, files in contents:
            # Include all subfolders, including empty ones.
            for folder_name in folders:
                absolute_path = os.path.join(root, folder_name)
                relative_path = absolute_path.replace(parent_folder + '\\',
                                                      '')
                print("Adding '%s' to archive." % absolute_path)
                zip_file.write(absolute_path, relative_path)
            for file_name in files:
                absolute_path = os.path.join(root, file_name)
                relative_path = absolute_path.replace(parent_folder + '\\',
                                                      '')
                print("Adding '%s' to archive." % absolute_path)
                zip_file.write(absolute_path, relative_path)
        print("'%s' created successfully." % output_path)
    except IOError as exc:
        traceback.print_exc()
        assert(0)
    except OSError as exc:
        traceback.print_exc()
        assert(0)
    except zipfile.BadZipfile as exc:
        taceback.print_exc()
        assert(0)
    finally:
        zip_file.close()
if __name__ == "__main__":
    print(is_test_flag())

