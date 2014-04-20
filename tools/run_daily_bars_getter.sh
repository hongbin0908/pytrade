#!/bin/bash
#########################
#@author hongbin0908@126.com
#@date 
#@desc get stock prices for pytrade
# the data located in /home/work/stock_data/ fixed
#########################
export PATH=/bin/:/usr/bin/:$PATH
export SCRIPT_PATH=`cd $(dirname $0); pwd`
cd "$SCRIPT_PATH"

mkdir -p /home/work/workplace/stock_data
mkdir -p ${SCRIPT_PATH}/../log/
logfile=${SCRIPT_PATH}/../log/get_stock_prices.py.log.$(date '+%Y%m%d')
echo "tools/daily_bars_getter.py start ..." >> $logfile

while [ 1 ]; do
    python $SCRIPT_PATH/daily_bars_getter.py  2&>1 >>  $logfile
    sleep 3600
done

echo "tools/daily_bars_getter.py start ..." >> $logfile


