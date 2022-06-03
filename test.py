import numpy as np
import scipy as sp
import pandas as pd
import ortools
import time
import os

from importlib import reload
from ortools.linear_solver import pywraplp
from scipy.special import expit, erf
from scipy import optimize

import optimizers as ops
import tools.metrics as tm


# Importing the original data
records = pd.read_csv('data/test_data.csv')

# Making them combined
y = records.stroke.values
X = records.iloc[:, records.columns != 'stroke']
