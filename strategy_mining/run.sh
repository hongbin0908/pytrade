wget http://ichart.yahoo.com/table.csv\?s=$1
mv table.csv?s=$1 $1
python rsi_analysis.py $1
