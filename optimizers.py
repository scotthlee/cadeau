"""Wrapper classes for different optimization functions."""
import pandas as pd
import numpy as np
import scipy as sp

from multiprocessing import Pool
from itertools import combinations

import tools.optimization as to
import tools.inference as ti
import tools.metrics as tm


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
            top_n=100,
            batch_keep_n=15,
            return_results=True):
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
            n_combos = [list(combinations(range(n_symp), i)) 
                        for i in range(1, max_n + 1)]

            # Dropping impossible symptom pairings
            if use_reverse:
                clashes = [[i, i + 15] for i in range(n_symp)]
                keepers = [[np.sum([c[0] in l and c[1] in l
                                    for c in clashes]) == 0
                            for l in combos]
                         for combos in n_combos]
                n_combos = [[c for j, c in enumerate(combos) if keepers[i][j]]
                            for i, combos in enumerate(n_combos)]
                symptom_list += ['no_' + s for s in symptom_list]

            # Running the search loop
            symp_out = []
            for i, combos in enumerate(n_combos):
                c_out = []
                X_combos = [X[:, c] for c in combos]
                for m in range(len(combos[0])):
                    inputs = [(y, np.array(np.array(np.sum(x, axis=1) > m,
                                                      dtype=np.uint8) > 0, 
                                             dtype=np.uint8))
                               for x in X_combos]
                    with Pool(processes=self.n_jobs) as p:
                        res = pd.concat(p.starmap(ti.clf_metrics, inputs),
                                   axis=0)
                    res['m'] = m
                    res['n'] = i
                    c_out.append(res)
                symp_out.append(c_out)
            
            # Getting the combo names
            combo_names = [[' '.join([symptom_list[i] for i in c])
                            for c in combos]
                           for combos in n_combos]
            
            # Filling in the combo names
            for i, dfs in enumerate(symp_out):
                for j in range(len(dfs)):
                    dfs[j]['rule'] = [str(j + 1) + ' of ' + s
                                      for s in combo_names[i]]    
            
            results = pd.concat(symp_out, axis=0)
            
        
        def predict(self, X):
            pass


class SmoothApproximator:
    """A nonlinear approximation to the linear program. Ride the wave, brah."""
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
            # Setting up the optional constraints for m and n
            cons = []
            if max_n:
                n = 4
                nA = np.concatenate([np.ones(Ns),
                                     np.zeros(1)])
                cons.append(sp.optimize.LinearConstraint(nA, lb=1, ub=n))
            
            if max_m:
                m = 1
                mA = np.concatenate([np.zeros(Ns),
                                     np.ones(1)])
                cons.append(optimize.LinearConstraint(mA, lb=1, ub=m))
            
            if min_sens:
                cons.append(sp.optimize.NonlinearConstraint(to.sens_exp,
                                                            lb=min_sens,
                                                            ub=1.0))
            
            if min_spec:
                cons.append(sp.optimize.NonlinearConstraint(to.spec_exp,
                                                            lb=0.8,
                                                            ub=1.0))
            
            # Running the program
            self.opt = sp.optimize.minimize(
                fun=to.j_exp,
                x0=init,
                args=(xp, xn),
                constraints=cons,
                bounds=bnds
            )
            
            good = opt.x.round()[:-1]
            #good_cols = np.where(good == 1)[0]
            #good_s = [var_list[i] for i in good_cols]
            #good_s
            return tm.j_lin(good, xp, xn, opt.x.round()[-1])
        else:
            # Now trying the compound program
            Nc = num_bags
            z_bnds = ((0, 1),) * Ns * Nc
            m_bnds = ((0, Ns),) * Nc
            bnds = z_bnds + m_bnds
            
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
            mn_cons = sp.optimize.NonlinearConstraint(to.m_morethan_n, 
                                                      lb=-np.inf, 
                                                      ub=0.999)
            
            # Constraint that at least one combo must have m >= 1
            m_sum = np.concatenate([np.zeros(Ns * Nc),
                                    np.ones(Nc)])
            m_sum_cons = sp.optimize.LinearConstraint(m_sum, lb=1, ub=np.inf)
            
            # Running the optimization
            init = np.zeros(len(bnds))
            opt = sp.optimize.minimize(
                fun=to.j_exp_comp,
                x0=init,
                args=(xp, xn, Nc),
                bounds=bnds,
                method='trust-constr',
                constraints=[nmax_cons, m_sum_cons, mn_cons]
            )
            
            solution = opt.x.round()
            mvals = solution[-Nc:]
            good = solution[:-Nc].reshape((Ns, Nc), order='F')
            return to.j_lin_comp(good, mvals, X, y)
    

    
                