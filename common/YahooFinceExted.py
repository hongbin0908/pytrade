# PyAlgoTrade
# 
# Copyright 2011 Gabriel Martin Becedillas Ruiz
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#   http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
.. moduleauthor Bin Hong <hongbin0908@126.com>
"""

import sys,os,urllib2
local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")

from pyalgotrade.tools import yahoofinance


def __download_instrument_prices_full(instrument):
    url = "http://ichart.finance.yahoo.com/table.csv?s=%s&g=d&ignore=.csv" % instrument
    print url
    f = urllib2.urlopen(url, timeout=60)
    if f.headers['Content-Type'] != 'text/csv':
        raise Exception("Failed to download data: %s" % f.getcode())
    buff = f.read()
    # Remove the BOM
    while not buff[0].isalnum():
        buff = buff[1:]
    return buff

def download_daily_bars_full(instrument, csvFile):
    """ Download bars for all years.
        A full version of download_daily_bars in pyalgotrade.tools.yahoofinance
    """
    bars = __download_instrument_prices_full(instrument)
    f = open(csvFile, "w")
    f.write(bars)
    f.close()
