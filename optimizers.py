"""Wrapper classes for different optimization functions."""
import pandas as pd
import numpy as np
import scipy as sp

from multiprocessing import Pool
from itertools import combinations

import tools.optimization as to
import tools.inference as ti


class ComboCruncher:
    """The sledgehammer: A brute-force combinatorial optimizer."""
    def __init__(self):
        return
    
    def fit(self, X, y,
            max_n=5,
            metric='j',
            metric_mode='max',
            compound=False,
            use_reverse=False,
            n_jobs=-1,
            top_n=100,
            batch_keep_n=15):
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
                    with Pool() as p:
                        res = pd.concat(p.starmap(ti.clf_metrics, inputs),
                                   axis=0)
                    res['m'] = m
                    res['n'] = i
                    res['type'] = 'symptoms_only'
                    c_out.append(res)
                symp_out.append(c_out)
            
            # Getting the combo names
            combo_names = [[' '.join([symptom_list[i] for i in c])
                            for c in combos]
                           for combos in n_combos]
            
            # Filling in the combo names
            for out in [symp_out, ant_out, exp_out]:
                for i, dfs in enumerate(out):
                    for j in range(len(dfs)):
                        dfs[j]['rule'] = [str(j + 1) + ' of ' + s
                                          for s in combo_names[i]]    
            
            return symp_out
        
        def predict(self, X):
            pass
            