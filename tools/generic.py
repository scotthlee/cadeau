import numpy as np
import pandas as pd


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
