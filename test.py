import numpy as np
import pandas as pd
import tracemalloc
import time
import seaborn as sns

from matplotlib import pyplot as plt
from importlib import reload

from kudos import optimizers as ops
from kudos import metrics

if __name__ == '__main__':
    # Importing the original data
    records = pd.read_csv('data/tb_data.csv')

    # Making them combined
    y = records.mtb.values.astype(np.uint8)
    X = records.iloc[:, records.columns != 'mtb'].fillna(0)
    X = X.astype(np.uint8)

    # Setting up a pruner to whittle down the variables
    pruner = ops.FeaturePruner()
    pruner.fit(X, y, factor=5)

    # Trying the different solvers on the simple problem
    start = time.time()
    ip = ops.IntegerProgram()
    ip.fit(pruner.X, y)
    end = time.time()
    ip_time = end - start

    start = time.time()
    nola = ops.NonlinearApproximation()
    nola.fit(pruner.X, y)
    end = time.time()
    nola_time = end - start

    tracemalloc.start()
    start = time.time()
    fe = ops.FullEnumeration()
    fe.fit(pruner.X, y, chunksize=int(1e4))
    end = time.time()
    fe_size, fe_peak = tracemalloc.get_traced_memory()
    fe_time = end - start

    tracemalloc.clear_traces()
    tracemalloc.start()
    start = time.time()
    fe_share = ops.FullEnumeration(share_memory=True)
    fe_share.fit(pruner.X, y, chunksize=int(1e4))
    end = time.time()
    fe_share_size, fe_share_peak = tracemalloc.get_traced_memory()
    fe_share_time = end - start

    # Trying the full enumeration and approximation on the compound problem
    start = time.time()
    fe.fit(pruner.X, y, compound=True, write_full=False)
    end = time.time()
    fe_comp_time = end - start

    start = time.time()
    nola.fit(pruner.X, y, compound=True)
    end = time.time()
    nola_comp_time = end - start

