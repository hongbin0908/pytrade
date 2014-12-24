#!/bin/sh 
######################### 
#@author binred@outlook.com 
#@date 
#@desc TODO
######################### 
export PATH=/usr/bin:$PATH 
export SCRIPT_PATH=`dirname $(readlink -f $0)` # get the path of the script 
pushd . > /dev/null 
cd "$SCRIPT_PATH" 
 
cd ../tools/ ; 
#python daily_bars_getter.py
cd "$SCRIPT_PATH" 

rm -rf /home/work/workplace/stock_data/BMC.csv 

for ((i=90;i>=0;i++)); do
    datestr=$(date -d"$i days ago" +%Y-%m-%d) 
    #time python model_build_price_series2.py --utildate=${datestr} || exit 1
    time python model_tuner.py  --input=data/prices_series/Extractor4_5 --utildate=${datestr} || exit 1
done


popd  > /dev/null # return the directory orignal

