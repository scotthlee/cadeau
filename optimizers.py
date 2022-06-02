"""Wrapper classes for different optimization functions."""
import pandas as pd
import numpy as np
import scipy as sp
import math
import tqdm
import ortools

from multiprocessing import Pool
from ortools.linear_solver import pywraplp
from itertools import combinations

import tools.inference as ti
import tools.metrics as tm
from tools.generic import smash_log, unique_combo


def mz_to_str(m, z):
    

class TotalEnumerator:
    """A brute-force combinatorial optimizer. Hulk smash!"""
    def __init__(self,
                 n_jobs=None):
        self.n_jobs = n_jobs
        return
    
    def fit(self, X, y,
            max_n=5,
            metric='j',
            metric_mode='max',
            compound=False,
            use_reverse=False,
            top_n=None,
            batch_keep_n=15,
            return_results=True):
        """Fits the optimizer.
        
        Parameters
        ----------
          max_n : int, default=5
            The maximum allowable combination size.
          metric : str, default='j'
            The classification metric to be optimized. Must be a function in \
            tools.metrics.
          metric_mode : str, default='max'
            Whether to minimize ('min') or maximimze ('max') the metric.
          complex : bool, default=False
            Whether to search compound combinations. Performance will \
            probably be higher
          use_reverse: bool, default=False
            Whether to include reversed symptoms (e.g., 'not X'); this will \
            double the size of the feature space.
          n_jobs : int, default=-1
            Number of jobs for multiprocessing. -1 runs them all.
          top_n : int, default=100
            Number of top-performing combinations to save.
          batch_keep_n: int, default=15
            Number of top combos to keep from each batch. Only the top \
            combinations will be kept as candidates for the final cut. Usually \
            only applies for compound combinations.
        """
        n_symp = X.shape[1]
        symptom_list = X.columns.values
        X = X.values.astype(np.uint8)
        
        if not compound:
            # Optional reversal
            if use_reverse:
                X_rev = -X + 1
                X = np.concatenate((X, X_rev), axis=1)
            
            # Setting up the combinations
            n_symp = X.shape[1]
            n_combos = [list(combinations(range(1, n_symp + 1), i)) 
                        for i in range(1, max_n + 1)]
            
            # Dropping impossible symptom pairings
            if use_reverse:
                clashes = [[i, i + n_symp] for i in range(n_symp)]
                keepers = [[np.sum([c[0] in l and c[1] in l
                                    for c in clashes]) == 0
                            for l in combos]
                         for combos in n_combos]
                n_combos = [[c for j, c in enumerate(combos) if keepers[i][j]]
                            for i, combos in enumerate(n_combos)]
                symptom_list += ['no_' + s for s in symptom_list]
            
            # Running the search loop
            symp_out = []
            score_fn = getattr(tm, metric)
            for i, combos in enumerate(n_combos):
                c_out = []
                X_combos = [X[:, np.array(c) - 1] for c in combos]
                for m in range(len(combos[0])):
                    inputs = [(y, np.array(np.array(np.sum(x, axis=1) > m,
                                                      dtype=np.uint8) > 0, 
                                             dtype=np.uint8))
                               for x in X_combos]
                    with Pool(processes=self.n_jobs) as p:
                        res = p.starmap(score_fn, inputs)
                    
                    res = pd.DataFrame(pd.Series(res), columns=[metric])
                    res['m'] = m + 1
                    res['n'] = i + 1
                    c_out.append(res)
                symp_out.append(c_out)
            
            # Getting the combo names
            combo_names = [[' '.join([symptom_list[i] 
                                      for i in np.array(c) - 1])
                            for c in combos]
                           for combos in n_combos]
            
            # Filling in the combo names
            for i, dfs in enumerate(symp_out):
                for j in range(len(dfs)):
                    dfs[j]['rule'] = [str(j) + ' of ' + s
                                      for s in combo_names[i]]    
            
            results = pd.concat([pd.concat(dfs, axis=0)
                                 for dfs in symp_out], axis=0)
            results.sort_values(metric,
                                ascending=(metric_mode != 'max'),
                                inplace=True)
            if top_n:
                results = results.iloc[:top_n, :]
            
            self.results = results
            return
        
        def predict(self, X):
            pass


class NonlinearApproximator:
    """A nonlinear approximation to the integer program. Keeping it fast and 
    loose.
    """
    def __int__(self):
        return
    
    def fit(self, X, y,
            compound=False,
            num_bags=2,
            max_m=None,
            max_n=None,
            min_sens=None,
            min_spec=None):
        """Fits the optimizer.
        
        Parameters
        ----------
          max_n : int, default=5
            The maximum allowable combination size.
          metric : str, default='j'
            The classification metric to be optimized.
          metric_mode : str, default='max'
            Whether to minimize ('min') or maximimze ('max') the metric.
          complex : bool, default=False
            Whether to search compound combinations. Performance will \
            probably be higher
          use_reverse: bool, default=False
            Whether to include reversed symptoms (e.g., 'not X'); this will \
            double the size of the feature space.
          n_jobs : int, default=-1
            Number of jobs for multiprocessing. -1 runs them all.
          top_n : int, default=100
            Number of top-performing combinations to save.
          batch_keep_n: int, default=15
            Number of top combos to keep from each batch. Only the top \
            combinations will be kept as candidates for the final cut. Usually \
            only applies for compound combinations.
        """
        # Setting up the problem
        self.var_names = X.columns.values
        X = X.values
        
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        
        X = np.concatenate([X[pos], X[neg]], axis=0)
        y = np.concatenate([y[pos], y[neg]], axis=0)
        
        xp = X[:len(pos), :]
        xn = X[len(pos):, :]
        
        N = X.shape[0]
        Ns = X.shape[1]
        bnds = ((0, 1),) * Ns
        bnds += ((1, Ns),)
        init = np.zeros(Ns + 1)
        
        if not compound:
            cons = []
            if max_n:
                n = 4
                nA = np.concatenate([np.ones(Ns), np.zeros(1)])
                cons.append(sp.optimize.LinearConstraint(nA, lb=1, ub=n))
            
            if max_m:
                m = 1
                mA = np.concatenate([np.zeros(Ns), np.ones(1)])
                cons.append(sp.optimize.LinearConstraint(mA, lb=1, ub=m))
            
            if min_sens:
                cons.append(sp.optimize.NonlinearConstraint(tm.sens_exp,
                                                            lb=min_sens,
                                                            ub=1.0))
            
            if min_spec:
                cons.append(sp.optimize.NonlinearConstraint(tm.spec_exp,
                                                            lb=0.8,
                                                            ub=1.0))
            
            # Running the program
            opt = sp.optimize.minimize(
                fun=tm.j_exp,
                x0=init,
                args=(xp, xn),
                constraints=cons,
                bounds=bnds
            )
            self.opt = opt
            rounded = opt.x.round()
            self.z = rounded[:-1]
            self.m = rounded[-1]
            self.j = tm.j_lin(self.z, xp, xn, self.m)
            
            # Writing the output message
            var_ids = np.where(self.z == 1)[0]
            z_vars = ' '.join(self.var_names[var_ids])
            best_mess = str(int(self.m)) + ' of (' + z_vars + ')'
            print('The best combo was ' + best_mess)
            print('The combo has a J of ' + str(round(self.j, 2)))
            return
        else:
            # Now trying the compound program
            Nc = num_bags
            z_bnds = ((0, 1),) * Ns * Nc
            m_bnds = ((0, Ns),) * Nc
            bnds = z_bnds + m_bnds
            
            def m_morethan_n(z):
                m = z[-Nc:]
                z = z[:-Nc]
                z = z.reshape((Ns, Nc), order='F')
                z = smash_log(z - .5, B=15)
                nvals = z.sum(0)
                diffs = smash_log(nvals - m + .5)
                return (Nc - diffs.sum()) 
            
            # Constraint so that no symptom appears in more than one combo
            z_con_mat = np.concatenate([np.identity(Ns)] * Nc, axis=1)
            m_con_mat = np.zeros((Ns, Nc))
            nmax_mat = np.concatenate([z_con_mat, m_con_mat], axis=1)
            nmax_cons = sp.optimize.LinearConstraint(nmax_mat, lb=0, ub=1)
            
            # Constraint so that m <= n for any combo
            z_c_rows = [np.ones(Ns)] * Nc
            z_c_mat = np.zeros((Nc, Ns * Nc))
            for i, r in enumerate(z_c_rows):
                start = i * Ns
                end = start + Ns
                z_c_mat[i, start:end] = r
            
            z_c_mat = np.concatenate([z_c_mat, np.identity(Nc) * -1],
                                     axis=1)
            mn_cons = sp.optimize.LinearConstraint(z_c_mat, 
                                                   lb=0, 
                                                   ub=np.inf)
            mn_cons = sp.optimize.NonlinearConstraint(m_morethan_n, 
                                                      lb=-np.inf, 
                                                      ub=0.999)
            
            # Constraint that at least one combo must have m >= 1
            m_sum = np.concatenate([np.zeros(Ns * Nc),
                                    np.ones(Nc)])
            m_sum_cons = sp.optimize.LinearConstraint(m_sum, lb=1, ub=np.inf)
            
            # Running the optimization
            init = np.zeros(len(bnds))
            opt = sp.optimize.minimize(
                fun=tm.j_exp_comp,
                x0=init,
                args=(xp, xn, Nc),
                bounds=bnds,
                method='trust-constr',
                constraints=[nmax_cons, m_sum_cons, mn_cons]
            )
            self.opt = opt
            self.solution = opt.x.round()
            self.m = self.solution[-Nc:]
            self.z = self.solution[:-Nc].reshape((Ns, Nc), order='F')
            return tm.j_lin_comp(self.z, self.m, X, y)
    
    def predict(self, X):
        pass


class IntegerProgram:
    """The integer program. Always optimal, never fast. Go enjoy a cup of tea, 
    or five.
    """
    def __init__(self):
        return
    
    def fit(self, X, y,
            max_n=None,
            solver='CP-SAT'):
        """Fits the optimizer.
        
        Parameters
        ----------
          max_n : int, default=5
            The maximum allowable combination size.
          solver : str, default='CP-SAT'
            Which ortools solver to use. 'SCIP' will also work, but can be 
            really slow.
          
        """
        self.var_names = X.columns.values
        X = X.values
        
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        
        X = np.concatenate([X[pos], X[neg]], axis=0)
        y = np.concatenate([y[pos], y[neg]], axis=0)
        
        xp = X[:len(pos), :]
        xn = X[len(pos):, :]
        
        N = X.shape[0]
        Ns = X.shape[1]
        H = 100
        
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
        divs = [1 / (xp.shape[0] / X.shape[0])] * xp.shape[0]
        divs += [1 / (xn.shape[0] / X.shape[0])] * xn.shape[0]
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
        solver = pywraplp.Solver.CreateSolver('CP-SAT')
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
        
        z = np.round([x[i].solution_value()
                      for i in range(Ns)]).astype(np.int8)
        m = np.round(x[Ns].solution_value()).astype(np.int8)
        
        self.vars = x
        self.solver = solver
        self.objective = objective
        self.constraint = constraint
        self.z = z
        self.m = m
        
    
    def predict(self, X):
        pass
    
                