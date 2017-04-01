import pandas as pd
import numpy as np
import
colors =
index = pd.MultiIndex.from_arrays([colors, foods], names=['color', 'food'])
df = pd.DataFrame(np.random.randn(10,2), index=index)