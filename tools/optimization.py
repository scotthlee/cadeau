import numpy as np
import scipy as sp
import pandas as pd

from scipy.special import expit, erf


def unique_combo(c):
    if len(np.intersect1d(c[0], c[1])) == 0:
        return c
    else:
        return None


def smash_log(x, B=10, d=0):
    return 1 / (1 + np.exp(-x * B)) - d


def sens(z, xp=xp, B=100):
    m = z[-1]
    z = z[:-1]
    return smash_log(np.dot(xp, z) - m, B=B).sum() / xp.shape[0]


def spec(z, xn=xn, B=100):
    m = z[-1]
    z = z[:-1]
    return 1 - smash_log(np.dot(xn, z) - m, B=B).sum() / xn.shape[0]


def j_lin(z, xp, xn, m):
    z = np.round(z)
    tpr = np.sum(np.dot(xp, z) >= m) / xp.shape[0]
    fpr = np.sum(np.dot(xn, z) >= m) / xn.shape[0]
    print(tpr, fpr)
    return tpr - fpr


def j_exp(z, xp, xn, a=1, b=1):
    m = z[-1]
    z = smash_log(z[:-1] - .5)
    tpr = smash_log(np.dot(xp, z) - m + .5).sum() / xp.shape[0]
    fpr = smash_log(np.dot(xn, z) - m + .5).sum() / xn.shape[0]
    return -1 * (a*tpr - b*fpr)


def j_exp_comp(z, xp, xn, c=2, a=1, b=1, th=0):
    # Setting things up
    s = xp.shape[1]
    m = z[-c:]
    z = z[:-c]
    z = z.reshape((s, c), order='F')
    z = smash_log(z - .5, B=15)
    
    # Penalizing bins where m > n
    nvals = z.sum(0)
    diffs = smash_log(nvals - m - .5)
    mn_penalty = th * (c - diffs.sum())
    
    # Now calculating the hits
    p_hits = smash_log(smash_log(np.dot(xp, z) - m + .5).sum(1) - .5).sum()
    n_hits = smash_log(smash_log(np.dot(xn, z) - m + .5).sum(1) - .5).sum()
    
    tpr = p_hits / xp.shape[0]
    fpr = n_hits / xn.shape[0] 
    weighted_j = a*tpr - b*fpr
    
    return -1 * weighted_j + mn_penalty


def j_lin_comp(n_mat, m_vec, X, y):
    counts = np.array([np.dot(X, v) for v in n_mat.T]).T
    diffs = np.array([counts[:, i] - m_vec[i] >= 0 
                      for i in range(len(m_vec))])
    guesses = np.array(np.sum(diffs, 0) > 0, dtype=np.uint8)
    j = tools.clf_metrics(y, guesses).j.values[0]
    return j


def m_morethan_n(z, c=Nc, s=Ns):
    m = z[-c:]
    z = z[:-c]
    z = z.reshape((s, c), order='F')
    z = smash_log(z - .5, B=15)
    nvals = z.sum(0)
    diffs = smash_log(nvals - m + .5)
    return (c - diffs.sum())
