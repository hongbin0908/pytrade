import sys
import os
fd = open(sys.argv[1], "r")
max_list= None
min_list = None

for s in fd:
    line_list = s.split(", ")
    if max_list == None:
        max_list = [ -1 for i in range(len(line_list))]
    if min_list == None:
        min_list = [100000000 for i in range(len(line_list))]
    for index, f in enumerate(line_list[:-1]):
        value = float(f)
        if max_list[index] < value:
            max_list[index] =value
        if min_list[index] > value:
            min_list[index] = value
fd.close()
fd = open(sys.argv[1], "r")
for s in fd:
    tmp_list = []
    line_list = s.rstrip().split(", ")
    for index, f in enumerate(line_list[:-1]):
        tmp_value = int(100*(float(f)-min_list[index])/(max_list[index]-float(f)+0.001))
        tmp_list.append(str(tmp_value))
    print ', '.join(tmp_list) + ", " + line_list[-1]
fd.close()
