"""Wrapper classes for different optimization functions."""
import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns
import sklearn
import math
import tqdm
import ortools

from multiprocessing import Pool, Array, shared_memory
from copy import deepcopy
from matplotlib import pyplot as plt
from ortools.linear_solver import pywraplp
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from matplotlib import pyplot as plt
from itertools import combinations, permutations, islice

from . import metrics, tools


metric_dict = {
    'j': {
        'fn1': 'sens',
        'fn2': 'spec',
        'xlab': '1 - Specificity',
        'ylab': 'Sensitivity'
    },
    'f1': {
        'fn1': 'sens',
        'fn2': 'ppv',
        'xlab': 'PPV',
        'ylab': 'Sensitivity'
    },
    'mcc': {
        'fn1': 'j',
        'fn2': 'mk',
        'xlab': 'Markedness',
        'ylab': 'Informedness'
    }
}



class FeaturePruner:
    """A wrapper for sklearn models that can be used to whittle down the 
    feature space for large problems.
    """
    def __init__(self,
                 model_type='rf',
                 n_jobs=-1, 
                 n_estimators=1000,
                 l1_ratio=0.5,
                 other_args=None):
        """Initializes the pruner.
        
        Parameters
        ----------
        model_type : str, default='rf'
          The kind of model to use as the pruner. Options are a random forest 
          ('rf'), a gradient boosting classifier ('gbc'), and a logistic
          regression with L1 ('l1'), L2 ('l2'), and L1+L2 ('elasticnet') 
          penalties.
        n_jobs : int, default=-1
          n_jobs parameter to pass to sklearn models. -1 uses all processes.
        n_estimators : int, default=1000
          n_estimators parameter to pass to sklearn.ensemble models.
        l1_ratio : float in (0, 1), default=0.5
          Blending parameter for the L!/L2 penalties in elasticnet.
        other_args : dict, default=None
          Dictionary of other args for the base model.
        
        Returns
        ----------
        A FeaturePruner object with a base model ready to fit.
        """
        tree_dict = {'rf': 'RandomForestClassifier',
                    'gbc': 'GradientBoostingClassifier'}
        self.tree_mods = ['rf', 'gbc']
        if model_type in self.tree_mods:
            args = {'n_estimators': n_estimators,
                    'n_jobs': n_jobs}
            if other_args:
                args = {**args, **other_args}
            mod_fn = getattr(sklearn.ensemble, 
                             tree_dict[model_type])
            self.mod = mod_fn(**args)
        else:
            args = {'penalty': model_type,
                    'solver': 'saga',
                    'l1_ratio': l1_ratio}
            if other_args:
                args = {**args, **other_args}
            self.mod = LogisticRegression(**args)
        
        self.mod_type = model_type
        
    def fit(self, X, y, 
            factor=0.5,
            return_x=False, 
            other_args=None):
        """Fits the FeaturePruner to a dataset.
        
        Parameters
        ----------
        X : pd.DataFrame
          The array of predictors.
        y : array-like
          The array of targets for prediction.
        factor : float or int, default=0.5
          Either the proportion or number of original features to keep.
        return_x : bool, default=False
          Whether to return the pruned predictors after fitting.
        other_args : dict, default=None
          A dict of other args to pass to the base model fit() function.
        
        Returns
        ----------
        None or a pd.DataFrame of the pruned predictors.
        """
        self.factor = factor
        self.var_names = X.columns.values
        if self.factor < 1:
            top_n = int(self.factor * X.shape[1])
        else:
            top_n = self.factor
        if other_args:
            self.mod.fit(X, y, **kwargs)
        else:
            self.mod.fit(X, y)
        if self.mod_type in self.tree_mods:
            imps = self.mod.feature_importances_
        else:
            imps = self.mod.coef_[0]
        
        sorted_imps = np.argsort(imps)[::-1]
        self.top_feature_ids = sorted_imps[:top_n]
        self.sorted_var_names = [self.var_names[f] for f in sorted_imps]  
        self.top_var_names = [self.var_names[f] for f in self.top_feature_ids]  
        self.X= X.iloc[:, self.top_feature_ids]
        if return_x:
            self.X
    
    def predict(self, X):
        """Returns a pruned version of a dataset.
        
        Parameters
        ----------
        X : pd.DataFrame or np.array
        
        Returns
        ----------
        """
        if type(X) == type(pd.DataFrame([0])):
            return X[self.top_var_names]
        else:
            return X[:, self.top_feature_ids]



class FullEnumeration:
    """A brute-force combinatorial optimizer. Hulk smash!"""
    def __init__(self, n_jobs=None, share_memory=False):
        self.n_jobs = n_jobs
        self.share_memory = share_memory
        return
    
    def fit(self, X, y,
            max_n=5,
            metric='j',
            compound=False,
            use_reverse=False,
            top_n=None,
            chunksize=1000,
            batch_keep_n=1000,
            write_full=False,
            csv_name='fe_results.csv',
            prune=True,
            tol=.05,
            progress_bars=True,
            verbose=False):
        """Fits the optimizer.
        
        Parameters
        ----------
        max_n : int, default=5
            The maximum allowable combination size.
        metric : str, default='j'
            The classification metric to be optimized. Options are 'j', 'f1',
            or 'mcc'.
        compound : bool, default=False
            Whether to search compound combinations. Performance will \
            probably be higher
        use_reverse : bool, default=False
            Whether to include reversed symptoms (e.g., 'not X'); this will \
            double the size of the feature space.
        top_n : int, default=100
            Number of top-performing combinations to save.
        batch_keep_n : int, default=15
            Number of top combos to keep from each batch. Only the top \
            combinations will be kept as candidates for the final cut. Usually \
            only applies for compound combinations.
        write_full : bool, default=True
            Whether to write the full (pre-pruned) set of results to disk. If 
            prune is on, only the pruned results will be saved when this is off.
        prune : bool, default=True
            Whether to discard combinatinos from each batch that are worse than 
            the best result from previous batches.
        tol : float, default=0.05,
            Maximum amount of a metric a combination in a batch can lag the max 
            from previous batches before being discarded. Only applies when 
            prune is set to True. 
        
        Returns
        ----------
        
        """
        # Separating the column names and values
        symptom_list = X.columns.values
        self.var_names = symptom_list
        X = X.values.astype(np.uint8)
        y = y.astype(np.uint8)
        self.X = X
        self.y = y
                
        # Setting up a lookup dictionary for the metric names
        self.metric = metric
        fn1 = metric_dict[metric]['fn1']
        fn2 = metric_dict[metric]['fn2']
        
        # Optionally adding reversed versions of the columns
        if use_reverse:
            X_rev = -X + 1
            X = np.concatenate((X, X_rev), axis=1)
        
        # Setting max_n, if not already specified
        if not max_n:
            max_n = n_symp
        
        # Optionally copying the data into shared memory
        if self.share_memory:
            shm_x = shared_memory.SharedMemory(name='predictors',
                                               create=True, 
                                               size=X.nbytes)
            shm_y = shared_memory.SharedMemory(name='outcome',
                                               create=True,
                                               size=y.nbytes)
            X_sh = np.ndarray(X.shape, 
                              dtype=X.dtype, 
                              buffer=shm_x.buf)
            y_sh = np.ndarray(y.shape,
                              dtype=y.dtype,
                              buffer=shm_y.buf)
            X_sh[:] = X[:]
            y_sh[:] = y[:]
            
        # Setting up the combinations
        n_symp = X.shape[1]
        n_vals = list(range(1, max_n + 1))
        
        # Dropping impossible symptom pairings
        """
        if use_reverse:
            clashes = [[i, i + n_symp] for i in range(n_symp)]
            keepers = [[np.sum([c[0] in l and c[1] in l
                                for c in clashes]) == 0
                        for l in combos]
                     for combos in col_combos]
            col_combos = [[c for j, c in enumerate(combos) if keepers[i][j]]
                        for i, combos in enumerate(col_combos)]
            symptom_list += ['no_' + s for s in symptom_list]
        """
        
        # Running the simple search loop
        symp_out = []
        if verbose:
            print('Evaluating the simple combinations.')
        
        with Pool(processes=self.n_jobs) as p:
            for n_val in n_vals:
                for m in range(n_val):
                    c_out = []
                    combo_iter = combinations(range(n_symp), n_val)
                    combos_remaining = True
                    while combos_remaining:
                        combos = list(islice(combo_iter, chunksize))
                        if len(combos) == 0:
                            combos_remaining = False
                        else:
                            if self.share_memory:
                                inputs = [(c, m, X.shape, metric) 
                                          for c in combos]
                                res = p.starmap(metrics.shared_mem_score, 
                                                inputs)
                            else:
                                slices = [np.array(X[:, c].sum(1) > m, 
                                                   dtype=np.uint8) 
                                          for c in combos]
                                inputs = [(y, s, metric) for s in slices]
                                res = p.starmap(metrics.score_set, inputs)
                            res = np.array(res)
                            combo_names = [' '.join([symptom_list[i]
                                                     for i in np.array(c)])
                                           for c in combos]
                            mn = np.array([[m + 1, n_val]] * res.shape[0])
                            cols = np.array(combo_names).reshape(-1, 1)
                            c_out.append(np.concatenate([mn, cols, res], 
                                                        axis=1))
                    symp_out.append(np.concatenate(c_out, axis=0))
            p.close()
            p.join()
        
        if self.share_memory:
            self.clear_shared_memory()
        
        results = pd.DataFrame(np.concatenate(symp_out),
                               columns=['m1', 'n1', 'rule1',
                                         fn1, fn2, metric])
        results.sort_values(metric,
                            ascending=False,
                            inplace=True)
        results[['m2', 'n2']] = 0
        results[['rule2', 'link']] = ''
        results = results[['m1', 'rule1', 'link',
                           'm2', 'rule2', 'n1',
                           'n2', fn1, fn2, metric]]
        float_cols = [fn1, fn2, metric]
        int_cols = ['m1', 'm2', 'n1', 'n2']
        results[float_cols] = results[float_cols].astype(float)
        results[int_cols] = results[int_cols].astype(int)

        if top_n:
            results = results.iloc[:top_n, :]
            results.reset_index(inplace=True, drop=True)

        if write_full:
            results.to_csv('data/' + csv_name, index=False)

        self.results = results

        if compound:
            # Make the initial list of combinations of the column combinations
            col_combos = [list(combinations(range(n_symp), i)) 
                        for i in range(1, max_n + 1)]
            col_combos = tools.flatten(col_combos)
            meta_iter = combinations(col_combos, 2)
            
            with Pool(processes=self.n_jobs) as p:
                if verbose:
                    print('Building the list of compound combinations.')
                
                metacombos = p.map(tools.unique_combo, meta_iter)
                p.close()
                p.join()
            
            metacombos = [c for c in metacombos if c is not None]
            
            # Combos of m for m-of-n
            mcs = [pair for pair in permutations(range(1, max_n + 1), 2)]
            mcs += [(i, i) for i in range(1, max_n + 1)]
            
            # Sums for at least 2 
            raw_pairsums = [[(X, c, mc, y, metric) 
                             for c in metacombos] for mc in mcs]
            
            # Weeding out combinations that don't make sense
            pairsum_input = []
            for i, pairs in enumerate(raw_pairsums):
                good_pairs = [pair for pair in pairs
                              if len(pair[1][0]) >= pair[2][0] 
                              and len(pair[1][1]) >= pair[2][1]]
                pairsum_input.append(good_pairs)
            pairsum_input = [input for input in pairsum_input
                             if len(input) > 0]
            
            # Max number of combos to consider from 'and', 'or', and 'any'
            if verbose:
                print('Running the evaluation loop.')
            
            num_runs = str(len(pairsum_input))
            with Pool(processes=self.n_jobs) as p:
                for run_num, input in enumerate(pairsum_input):
                    batch_n = np.min([len(input), batch_keep_n])
                    if verbose:
                        batch_mess = str(run_num + 1) + ' of ' + num_runs
                        print('')
                        print('Running batch ' + batch_mess)
                    
                    if progress_bars:
                        input = tqdm.tqdm(input)
                    
                    # Calculating f1 score for each of the combo sums
                    scores = p.starmap(metrics.pair_score, input)
                    and_scores = pd.DataFrame([s[0] for s in scores])
                    or_scores = pd.DataFrame([s[1] for s in scores])
                    all_scores = pd.concat([and_scores, or_scores], axis=0)
                    all_scores.columns = [fn1, fn2, metric]
                    
                    # Getting the names of the rules as strings
                    or_input = [(pair, 'or', self.var_names) 
                                 for pair in input]
                    or_info = p.starmap(tools.pair_info, or_input)
                    or_df = pd.DataFrame(or_info,
                                         columns=['m1', 'rule1', 'link',
                                                  'm2', 'rule2', 'n1', 'n2'])
                    and_df = deepcopy(or_df)
                    and_df['link'] = 'and'
                    info_df = pd.concat([and_df, or_df], axis=0)
                    info_df = pd.concat([info_df, all_scores], axis=1)
                    batch_res = info_df.sort_values(metric,
                                                    ascending=False)
                    batch_res = batch_res.iloc[0:batch_n, :]
                    
                    if write_full:
                        batch_res.to_csv('data/' + csv_name,
                                         mode='a',
                                         header=False,
                                         index=False)
                    if prune:
                        best_max = self.results[metric].max()
                        current_max = batch_res[metric].max()
                        if (current_max - best_max) >= -tol:
                            self.results = pd.concat([batch_res,
                                                      self.results], axis=0)
                        else:
                            print('Previous max was ' + str(best_max))
                            print('Current max is only ' + str(current_max))
                            print('Discarding current batch of rules.')
                    else:
                        self.results = pd.concat([batch_res,
                                                  self.results], aixs=0)

            p.close()
            p.join()
        
        self.results.sort_values(metric, ascending=False, inplace=True)
        self.results.reset_index(inplace=True, drop=True)
        self.results['total_n'] = self.results.n1 + self.results.n2
        
        return
    
    def clear_shared_memory(self):
        shm_x = shared_memory.SharedMemory(name='predictors',
                                           create=False, 
                                           size=self.X.nbytes)
        shm_y = shared_memory.SharedMemory(name='outcome',
                                           create=False,
                                           size=self.y.nbytes)
        shm_x.close()
        shm_x.unlink()
        shm_y.close()
        shm_y.unlink()
    
    def plot(self, 
             metric=None, 
             mark_best=True,
             separate_n=False,
             hue=None,
             grid_style='darkgrid',
             palette='crest',
             font_scale=1,
             add_hull=False):
        """Plots combinations in the selected metric's space."""
        sns.set_style('darkgrid')
        sns.set(font_scale=font_scale)
        cp = sns.color_palette(palette)
        
        if not metric:
            metric = self.metric
        
        if not hue:
            hue = metric
        
        md = metric_dict[self.metric]
        fn1, fn2 = md['fn1'], md['fn2']
        xlab, ylab = md['xlab'], md['ylab']
        
        y = self.results[fn1].values
        x = self.results[fn2].values
        
        if metric == 'j':
            x = 1 - self.results[fn2].values
        
        rp_col = 'total_n' if separate_n else None
        
        rp = sns.relplot(x=x, 
                         y=y, 
                         hue=self.results[metric], 
                         col=rp_col, 
                         data=self.results,
                         kind='scatter',
                         palette=palette)
        
        if mark_best and not separate_n:
            plt.scatter(x[0], y[0], 
                        color='black',
                        marker='x')
            
        rp.set(xlim=(0, 1), ylim=(0, 1))
        rp.fig.set_tight_layout(True)
        rp.set_xlabels(xlab)
        rp.set_ylabels(ylab)        
        
        plt.show()
        
        
    def predict(self, X):
        return tools.rule_to_y(X, self.results.loc[0])


class NonlinearApproximation:
    """A nonlinear approximation to the integer program. Playing it fast and 
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
          compound : bool, default=False
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
            # Setting up the optional constraints
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
                cons.append(sp.optimize.NonlinearConstraint(metrics.sens_exp,
                                                            lb=min_sens,
                                                            ub=1.0))
            if min_spec:
                cons.append(sp.optimize.NonlinearConstraint(metrics.spec_exp,
                                                            lb=0.8,
                                                            ub=1.0))
            # Running the program
            opt = sp.optimize.minimize(
                fun=metrics.j_exp,
                x0=init,
                args=(xp, xn),
                constraints=cons,
                bounds=bnds
            )
            self.opt = opt
            rounded = opt.x.round()
            self.z = rounded[:-1]
            self.m = rounded[-1]
            
            # Calculating the result's scores
            rule_cols = tools.zm_to_rule(self.z, self.m, self.var_names)
            y_ = self.predict(X)
            score_cols = metrics.score_set(y, y_, 'j', True)
            self.results = pd.concat([rule_cols, score_cols], axis=1)
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
                z = tools.smash_log(z - .5, B=15)
                nvals = z.sum(0)
                diffs = tools.smash_log(nvals - m + .5)
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
                fun=metrics.j_exp_comp,
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
            return metrics.j_lin_comp(self.z, self.m, X, y)
    
    def predict(self, X):
        """Returns the optimal guesses for a new set of data."""
        return tools.zm_to_y(self.z, self.m, X)


class IntegerProgram:
    """The integer program. Always optimal, and usually fast."""
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
        rule_cols = tools.zm_to_rule(z, m, self.var_names)
        y_ = self.predict(X)
        score_cols = metrics.score_set(y, y_, 'j', True)
        self.results = pd.concat([rule_cols, score_cols], axis=1)
    
    def predict(self, X):
        """Returns the optimal guesses for a new set of data."""
        return tools.zm_to_y(self.z, self.m, X)
    
                