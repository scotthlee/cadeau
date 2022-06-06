import numpy as np
import pandas as pd
import time
import seaborn as sns

from matplotlib import pyplot as plt
from importlib import reload

import optimizers as ops
import tools.metrics as tm


# Importing the original data
records = pd.read_csv('data/test_data.csv')

# Making them combined
y = records.stroke.values
X = records.iloc[:, records.columns != 'stroke']

# Trying the different solvers on the simple problem
start = time.time()
ip = ops.IntegerProgram()
ip.fit(X, y)
end = time.time()
ip_time = end - start

start = time.time()
nola = ops.NonlinearApproximation()
nola.fit(X, y)
end = time.time()
nola_time = end - start

start = time.time()
fe = ops.FullEnumeration()
fe.fit(X, y, max_n=None)
end = time.time()
fe_time = end - start

# Trying the full enumeration and approximation on the compound problem
start = time.time()
fe.fit(X, y, compound=True, write_full=True)
end = time.time()
fe_comp_time = end - start

start = time.time()
nola.fit(X, y, compound=True)
end = time.time()
nola_comp_time = end - start

