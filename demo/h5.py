import pandas as pd
import numpy as np
import pandas.util.testing as tm

colors = np.random.choice(['red', 'green'], size = 10)
foods  = np.random.choice(['egg', 'ham'], size = 10)
index = pd.MultiIndex.from_arrays([colors, foods], names=['color', 'food'])
df = pd.DataFrame(np.random.randn(10,2), index=index)

df.to_hdf("./test.h5", 'test')

df = pd.read_hdf("./test.h5")
print(df)