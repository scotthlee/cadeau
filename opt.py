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

import tools.optimization as to


# Globals
UNIX = True
USE_TODAY = False
COMBINED = True

# Importing the original data
records = pd.read_csv('data/test_data.csv')

# Making them combined
y = records.pcr.values
X = records[var_list].values

# And now trying it as a linear program; first setting up the constraints
H = 100
N = X.shape[0]
npos = len(pos)
nneg = len(neg)

T_con = np.identity(N) * H * -1
m_con_pos = np.ones((npos, 1))
m_con_neg = np.ones((nneg, 1)) * -1
m_con = np.concatenate([m_con_pos, m_con_neg])
Z_con = np.concatenate([xp * -1, xn], axis=0)
mn_con = np.concatenate([np.ones(Ns) * -1, 
                         np.ones(1), 
                         np.zeros(N)]).reshape(1, -1)
con_mat = np.concatenate([Z_con, m_con, T_con], axis=1)
con_mat = np.concatenate([con_mat, mn_con], axis=0).astype(np.int16)

# Setting up the bounds on the variables
bnds = [(0, 1)] * Ns
bnds += [(1, Ns)]
bnds += [(0, 1)] * N

# Setting up the bounds on the constraints
con_bounds = np.concatenate([np.zeros(npos), 
                             np.ones(nneg) * -1,
                             np.zeros(1)])

# And setting up the objective
divs = [1 / xp.shape[0]] * xp.shape[0]
divs += [1 / xn.shape[0]] * xn.shape[0]
obj = np.concatenate([np.zeros(Ns),
                      np.zeros(1),
                      np.array(divs)])

opt = sp.optimize.linprog(
    c=obj,
    A_ub=con_mat,
    b_ub=con_bounds,
    method='highs',
    bounds=bnds
)

# Trying with or tools
divs = [1 / (xp.shape[0] / 10000)] * xp.shape[0]
divs += [1 / (xn.shape[0] / 10000)] * xn.shape[0]
obj = np.concatenate([np.zeros(Ns),
                      np.zeros(1),
                      np.array(divs)])

def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data['constraint_coeffs'] = [[int(i) for i in j]for j in con_mat]
    data['bounds'] = [i for i in con_bounds]
    data['obj_coeffs'] = [int(i) for i in obj]
    data['num_vars'] = con_mat.shape[1]
    data['num_constraints'] = con_mat.shape[0]
    return data

data = create_data_model()

# Create the mip solver with the SCIP backend.
solver = pywraplp.Solver.CreateSolver('SCIP')

x = {}
for j in range(data['num_vars']):
    x[j] = solver.IntVar(bnds[j][0], bnds[j][1], '')

neg_inf = -solver.infinity()
for i in range(data['num_constraints']):
    constraint = solver.RowConstraint(neg_inf, data['bounds'][i], '')
    for j in range(data['num_vars']):
        constraint.SetCoefficient(x[j], data['constraint_coeffs'][i][j])

objective = solver.Objective()
for j in range(data['num_vars']):
    objective.SetCoefficient(x[j], data['obj_coeffs'][j])

objective.SetMinimization()

status = solver.Solve()

if status == pywraplp.Solver.OPTIMAL:
    print('Objective value =', solver.Objective().Value())
    for j in range(data['num_vars']):
        print(x[j].name(), ' = ', x[j].solution_value())
    print()
    print('Problem solved in %f milliseconds' % solver.wall_time())
    print('Problem solved in %d iterations' % solver.iterations())
    print('Problem solved in %d branch-and-bound nodes' % solver.nodes())
else:
    print('The problem does not have an optimal solution.')

good = np.round([x[i].solution_value() for i in range(16)]).astype(np.int8)
