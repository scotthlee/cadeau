"""Generic support functions"""
import numpy as np
import pandas as pd
import math


def prop_table(y, pred, axis=0, round=2):
    """Makes a proportions table for a vector of binary predictions."""
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


def onehot_matrix(y, sparse=False):
    """Converts a 1-D vector of class guesses to a one-hot matrix."""
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


def smash_log(x, B=10, d=0):
    """Logistic function with a little extra kick."""
    return 1 / (1 + np.exp(-x * B)) - d


def sparsify(col, 
             reshape=True, 
             return_df=True,
             long_names=False):
    """Makes a sparse array out of data frame of discrete variables."""
    levels = np.unique(col)
    out = np.array([col == level for level in levels],
                   dtype=np.uint8).transpose()
    if long_names:
        var = col.name + '.'
        levels = [var + level for level in levels]
    columns = [col.lower() for col in levels]
    if return_df:
        out = pd.DataFrame(out, columns=columns)
    return out


def row_sums(m, min=1):
    sums = np.sum(m, axis=1)
    return np.array(sums >= min, dtype=np.uint8)


def pair_sum(X, c, min=(1, 1)):
    a = row_sums(X[:, c[0]], min=min[0])
    b = row_sums(X[:, c[1]], min=min[1])
    return (a, b)


def combo_sum(ctup):
    sums = np.sum(ctup, axis=0)
    and_y = np.array(sums == 2, dtype=np.uint8)
    or_y = np.array(sums >= 1, dtype=np.uint8)
    out = np.concatenate([and_y.reshape(-1, 1),
                          or_y.reshape(-1, 1)],
                         axis=1)
    return out


def flatten(l):
    '''Flattens a list.'''
    return [item for sublist in l for item in sublist]


def unique_combo(c):
    """Determines if a combination of symptoms is unique."""
    if len(np.intersect1d(c[0], c[1])) == 0:
        return c
    else:
        return None


def pair_info(pair, link, col_list):
    """Turns lists of column strings and a prefix into a compound rule."""
    pair_cols = pair[1]
    pair_m = pair[2]
    m1 = pair_m[0]
    m2 = pair_m[1]
    n1 = len(pair_cols[0])
    n2 = len(pair_cols[1])
    
    # Making the string specifying the condition
    cnames = []
    for cols in pair_cols:
        cnames.append(' '.join([str(col_list[c]) 
                                     for c in cols]))
    out = [
        m1, cnames[0], link,
        m2, cnames[1], n1, n2
    ]
    return out


def zm_to_y(z, m, X):
    """Converts a variable choice vector, minimum count, and variables 
    to a binary guess vector.
    """
    return np.array(np.dot(X, z) >= m, dtype=np.uint8)


def zm_to_rule(z, m, cols, rule_num=1, return_df=True):
    """Converts a choice vector and minimum count to a rule string."""
    var_names = [cols[i] for i in np.where(z == 1)[0]]
    n = len(var_names)
    rule = ' '.join(var_names)
    m_col = 'm' + str(rule_num)
    n_col = 'n' + str(rule_num)
    if return_df:
        out = pd.DataFrame([m, rule, n]).transpose()
        out.columns = [m_col, rule, n_col]
    else:
        out = m, rule, n
    return out
    
