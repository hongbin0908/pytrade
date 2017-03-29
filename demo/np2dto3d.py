import numpy as np

a = np.arange(30).reshape(6,5)

def d2tod3(fro, window):
    row = fro.shape[0]
    feat_num = fro.shape[1]

    d1 = row - window +1
    d2 = window
    d3 = feat_num

    to = np.zeros(d1*d2*d3).reshape(d1,d2,d3)
    for i in range(len(fro)-window + 1):
        to[i] = fro[i:i+window]
    return to

print(d2tod3(a, window=2))