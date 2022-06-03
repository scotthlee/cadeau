"""Generic support functions"""
import numpy as np
import pandas as pd
import math


def prop_table(y, pred, axis=0, round=2):
    tab = pd.crosstab(y, pred)
    if axis == 1:
        tab = tab.transpose()
        out = tab / np.sum(tab, axis=0)
        out = out.transpose()
    else:
        out = tab / np.sum(tab, axis=0)
    if round is not None:
        out = np.round(out, round)
    return out


def total_combos(Ns, max_n=None, max_m=None):
    if not max_n:
        max_n = Ns
    n_vals = list(range(1, max_n + 1))
    total = 0
    for n_val in n_vals:
        m_vals = list(range(1, n_val + 1))
        num_chunks = math.comb(Ns, n_val)
        total += num_chunks * len(m_vals)
        print([n_val, m_vals])
    return total
    

def risk_ratio(y, pred, round=2):
    props = np.array(prop_table(y, pred, round=None))
    rr = props[1, 1] / props[1, 0]
    if round is not None:
        rr = np.round(rr, round)
    return rr


def odds_ratio(y, pred, round=2):
    tab = np.array(pd.crosstab(y, pred))
    OR = (tab[0, 0]*tab[1, 1]) / (tab[1, 0]*tab[0, 1])
    if round is not None:
        OR = np.round(OR, round)
    return OR


def onehot_matrix(y, sparse=False):
    if not sparse:
        y_mat = np.zeros((y.shape[0], len(np.unique(y))))
        for row, col in enumerate(y):
            y_mat[row, col] = 1
    return y_mat


def max_probs(arr, maxes=None, axis=1):
    if maxes is None:
        maxes = np.argmax(arr, axis=axis)
    out = [arr[i, maxes[i]] for i in range(arr.shape[0])]
    return np.array(out)


def flatten(l):
    '''Flattens a list.'''
    return [item for sublist in l for item in sublist]


def unique_combo(c):
    """Determines if a combination of symptoms is unique."""
    if len(np.intersect1d(c[0], c[1])) == 0:
        return c
    else:
        return None

def smash_log(x, B=10, d=0):
    """Logistic function with a little extra kick."""
    return 1 / (1 + np.exp(-x * B)) - d


def zm_to_y(z, m, X):
    """Converts a variable choice vector, minimum count, and variables 
    to a binary guess vector.
    """
    return np.array(np.dot(X, z) >= m, dtype=np.uint8)



