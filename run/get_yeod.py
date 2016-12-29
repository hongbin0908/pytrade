#!/usr/bin/env python3

import os, sys, time
import pandas as pd
import numpy as np
import pandas_datareader.yahoo.daily as yahoo
import multiprocessing
import subprocess
import logging

logging.basicConfig(level=logging.DEBUG)

local_path = os.path.dirname(__file__)


cmdstr = """
    cd {target};
    rm -rf sp100_current.tar.gz;
    wget http://hongindex.com/sp100_current.tar.gz;
    mkdir -p yeod
    rm -rf yeod/*
    tar xvzf sp100_current.tar.gz -C yeod
    """.format(**{"target":os.path.join(local_path, "..", "data")})
logging.debug(cmdstr)
subprocess.check_output(cmdstr, shell=True)
