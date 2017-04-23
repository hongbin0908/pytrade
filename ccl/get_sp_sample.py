import sys
import os
import numpy
import pandas

def load_data(filename):
	try:
		info = pandas.read_csv(filename, delimiter=',', header=1)
	except Exception as e:
		sys.stderr.write("error: info = %s\n" %(e))
		sys.exit(1)
	return info

def build_sample(info, stock_name = "A", interval = 15):
	for i in range(interval, info.shape[0]-1):
		x_date = info.iloc[i-interval, 0]
		x_date1 = info.iloc[i, 0]
		x = info.iloc[i-interval:i, 5].tolist()
		x_str = ' '.join(map(str, x))
		label = info.iloc[i+1, 8] / info.iloc[i, 8]
		if label >1:
			label = 1
		else:
			label = 0
		print("%s %s %s %s %d" %(stock_name, x_date, x_date1, x_str, label))
if __name__ == "__main__":
	result = load_data("../data/yeod/index/^GSPC.csv")
	i = "../data/yeod/index/^GSPC.csv"
	stock_name = i.strip().split("/")[-1].replace(".csv", "")
	build_sample(result, stock_name, 15)
