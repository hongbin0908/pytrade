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
python daily_bars_getter.py
cd "$SCRIPT_PATH" 

for ((i=90;i>=0;i++)); do
    datestr=$(date -d"$i days ago" +%Y-%m-%d) 
    python model_build_price_series2.py --utildate=${datestr}
    python model_tuner.py  --input=data/prices_series/Extractor4_5 --utildate=${datestr}
done


popd  > /dev/null # return the directory orignal

