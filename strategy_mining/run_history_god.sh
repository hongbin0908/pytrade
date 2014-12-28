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

start_date=$1
end_date=$2
i=0
while true
do
    datestr=$(date -d"$start_date + $i day" "+%Y-%m-%d") 
    time python model_build_price_series2.py --utildate=${datestr} --window=7 || exit 1
    time python model_tuner.py  --input=data/prices_series/Extractor4_7_5 --utildate=${datestr} --trainmodel=Nothing || continue
    time python model_tuner.py  --input=data/prices_series/Extractor4_7_5 --utildate=${datestr} --trainmodel=God || continue
    time python model_tuner.py  --input=data/prices_series/Extractor4_7_5 --utildate=${datestr} --trainmodel=Gdbc1 || continue
    time python post_check.py  --direct=data/prices_series/Nothing/${datestr}  || exit 1
    time python post_check.py  --direct=data/prices_series/God/${datestr}  || exit 1
    time python post_check.py  --direct=data/prices_series/Gdbc1/${datestr}  || exit 1
    let i+=1
done


popd  > /dev/null # return the directory orignal

