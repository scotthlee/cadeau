"""Functions and classes for statistical inference, mostly by way of 
bootstrapping.
"""
import pandas as pd
import numpy as np
import os

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import binom, chi2, norm, percentileofscore
from multiprocessing import Pool
from copy import deepcopy

from .tools import *
from .metrics import *


def threshold(probs, cutoff=0.5):
    """Applies a decision threshold to a vector of probabilities.
    
    Parameters
    ----------
    probs : array-like
      The vector of probabilities.
    cutoff : float, default=0.5
      The decision threshold.
    
    Returns
    ----------
    A np.uint8 vector of binary class predictions.
    """
    return np.array(probs >= cutoff).astype(np.uint8)


def mcnemar_test(targets, guesses, cc=True):
    """Performs McNemar's chi-squared test.
    
    Parameters
    ----------
    targets : array-like
      The true binary class labels.
    guesses : array-like
      The predicted binary class labels.
    cc : bool, default=True
      Whether to perform the test with a continuity correction.
    
    Returns
    ----------
    A pandas DataFrame with the counts for the off-diagonals ('b', 'c'),
    the value of the test statistic ('stat'), and the p-value ('pval').
    """
    cm = confusion_matrix(true, pred)
    b = int(cm[0, 1])
    c = int(cm[1, 0])
    if cc:
        stat = (abs(b - c) - 1)**2 / (b + c)
    else:
        stat = (b - c)**2 / (b + c)
    p = 1 - chi2(df=1).cdf(stat)
    outmat = np.array([b, c, stat, p]).reshape(-1, 1)
    out = pd.DataFrame(outmat.transpose(),
                       columns=['b', 'c', 'stat', 'pval'])
    return out


def average_pvals(p_vals, 
                  w=None, 
                  method='harmonic',
                  smooth_val=None):
    """Uses either a harmonic mean or Fisher's method to average p-values.
    
    Parameters
    ----------
    p_vals : array-like
      The p-values to be averaged.
    w : array-like, default=None
      An optional array of weights to use for averaging.
    method : str, default='harmonic'
      Method for averaging the p-values. Options are 'harmonic', which 
      takes their harmonic mean, or 'fisher', which uses Fisher's method.
    smooth : float, default=None
      Optional smoothing value to be added to the p-values before averaging.
    
    Returns
    ----------
    The average p-value according to the specified method.
    """
    if smooth_val:
        p = p_vals + smooth_val
    else:
        p = deepcopy(p_vals)
    if method == 'harmonic':
        if w is None:
            w = np.repeat(1 / len(p), len(p))
        p_avg = 1 / np.sum(w / p)
    elif method == 'fisher':
        stat = -2 * np.sum(np.log(p))
        p_avg = 1 - chi2(df=1).cdf(stat)
    return p_avg


def jackknife_sample(X, by=None):
    """Returns list of a jackknife samples of a dataset.
    
    Parameters
    ----------
    X : array-like
      The dataset to be sampled.
    by : array-like, default=None
      A vector specifying which group an observation belongs to. Used for 
      generating cluster/block bootstrap samples.
    
    Returns
    ----------
    The jackknife samples, i.e., a list of list of row numbers, where each 
    sublist sl[i] is the full list of rows numbers minus the ith row number.
    """
    if by:
        groups = np.unique(by)
        rows = np.where([by == g for g in groups])[0]
        rows = np.array(flatten(rows))
    else:
        rows = np.array(list(range(targets.shape[0])))
    
    j_rows = [np.delete(rows, row) for row in rows]

    rows = np.array(list(range(X.shape[0])))
    j_rows = [np.delete(rows, row) for row in rows]
    
    return j_rows


def boot_sample(X,
                by=None,
                size=None,
                seed=None:
    """Samples row indices with replacement, with the option to sample 
    by a separate variable.
    
    Parameters
    ----------
    X : array-like
      The dataset to be sampled.
    by : array-like, default=None
      A vector specifying which group an observation belongs to. Used for 
      generating cluster/block bootstrap samples.
    size : int, default=None
      The size of the sample. By default, this is the number of observations
      in X (e.g., len(X) for lists, or X.shape[0] for np.arrays).
    seed : int, default=None
      Seed for the random number generator. If unspecified, one will be 
      randomly selected from the range (1, 1e6).
    
    Returns
    ----------
    An np.array of row indices specifying the bootstrap sample.
    """
    # Setting the random states for the samples
    if seed is None:
        seed = np.random.randint(1, 1e6, 1)[0]
    np.random.seed(seed)
    
    # Getting the sample size
    if size is None:
        size = X.shape[0]
    
    # Sampling across groups, if group is unspecified
    if by is None:
        np.random.seed(seed)
        idx = range(size)
        boot = np.random.choice(idx,
                                size=size,
                                replace=True)
    
    # Sampling by group, if group has been specified
    else:
        levels = np.unique(by)
        n_levels = len(levels)
        level_rows = [np.where(by == level)[0]
                     for level in levels]
        row_dict = dict(zip(levels, level_rows))
        boot = np.random.choice(levels,
                                size=n_levels, 
                                replace=True)
        obs = flatten([row_dict[b] for b in boot])
        boot = np.array(obs)
    
    return boot


class boot_cis:
    def __init__(
        self,
        targets,
        guesses,
        n=100,
        a=0.05,
        sample_by=None,
        method="bca",
        interpolation="nearest",
        average='weighted',
        mcnemar=False,
        seed=10221983):
        # Converting everything to NumPy arrays, just in case
        stype = type(pd.Series([0]))
        if type(targets) == stype:
            targets = targets.values
        if type(guesses) == stype:
            guesses = guesses.values

        # Getting the point estimates
        stat = clf_metrics(targets,
                           guesses,
                           average=average,
                           mcnemar=mcnemar).transpose()

        # Pulling out the column names to pass to the bootstrap dataframes
        colnames = list(stat.index.values)

        # Making an empty holder for the output
        scores = pd.DataFrame(np.zeros(shape=(n, stat.shape[0])),
                              columns=colnames)
        
        # Setting the seed
        if seed is None:
            seed = np.random.randint(0, 1e6, 1)
        np.random.seed(seed)
        seeds = np.random.randint(0, 1e6, n)

        # Generating the bootstrap samples and metrics
        boots = [boot_sample(targets, 
                             by=sample_by,
                             seed=seed) for seed in seeds]
        score_input = [(targets[b], guesses[b]) for b in boots]
        with Pool() as p:
            scores = p.starmap(clf_metrics, score_input)
        
        scores = pd.concat(scores, axis=0)

        # Calculating the confidence intervals
        lower = (a / 2) * 100
        upper = 100 - lower

        # Making sure a valid method was chosen
        methods = ["pct", "diff", "bca"]
        assert method in methods, "Method must be pct, diff, or bca."

        # Calculating the CIs with method #1: the percentiles of the
        # bootstrapped statistics
        if method == "pct":
            cis = np.nanpercentile(scores,
                                   q=(lower, upper),
                                   interpolation=interpolation,
                                   axis=0)
            cis = pd.DataFrame(cis.transpose(),
                               columns=["lower", "upper"],
                               index=colnames)

        # Or with method #2: the percentiles of the difference between the
        # obesrved statistics and the bootstrapped statistics
        elif method == "diff":
            stat_vals = stat.transpose().values.ravel()
            diffs = stat_vals - scores
            percents = np.nanpercentile(diffs,
                                        q=(lower, upper),
                                        interpolation=interpolation,
                                        axis=0)
            lower_bound = pd.Series(stat_vals + percents[0])
            upper_bound = pd.Series(stat_vals + percents[1])
            cis = pd.concat([lower_bound, upper_bound], axis=1)
            cis = cis.set_index(stat.index)

        # Or with method #3: the bias-corrected and accelerated bootstrap
        elif method == "bca":
            # Calculating the bias-correction factor
            stat_vals = stat.transpose().values.ravel()
            n_less = np.sum(scores < stat_vals, axis=0)
            p_less = n_less / n
            z0 = norm.ppf(p_less)

            # Fixing infs in z0
            z0[np.where(np.isinf(z0))[0]] = 0.0

            # Estiamating the acceleration factor
            j = jackknife_metrics(targets, guesses)
            diffs = j[1] - j[0]
            numer = np.sum(np.power(diffs, 3))
            denom = 6 * np.power(np.sum(np.power(diffs, 2)), 3 / 2)

            # Getting rid of 0s in the denominator
            zeros = np.where(denom == 0)[0]
            for z in zeros:
                denom[z] += 1e-6

            # Finishing up the acceleration parameter
            acc = numer / denom
            self.jack = j

            # Calculating the bounds for the confidence intervals
            zl = norm.ppf(a / 2)
            zu = norm.ppf(1 - (a / 2))
            lterm = (z0 + zl) / (1 - acc * (z0 + zl))
            uterm = (z0 + zu) / (1 - acc * (z0 + zu))
            ql = norm.cdf(z0 + lterm) * 100
            qu = norm.cdf(z0 + uterm) * 100
            
            # Passing things back to the class
            self.acc = acc.values
            self.b = z0
            self.ql = ql
            self.qu = qu

            # Returning the CIs based on the adjusted quintiles
            cis = [
                np.nanpercentile(
                    scores.iloc[:, i],
                    q=(ql[i], qu[i]),
                    interpolation=interpolation,
                    axis=0,
                ) for i in range(len(ql))
            ]
            cis = pd.DataFrame(cis, 
                               columns=["lower", "upper"], 
                               index=colnames)

        # Putting the stats with the lower and upper estimates
        cis = pd.concat([stat, cis], axis=1)
        cis.columns = ["stat", "lower", "upper"]

        # Passing the results back up to the class
        self.cis = cis
        self.scores = scores

        return


class diff_boot_cis:
    def __init__(self,
                 ref,
                 comp,
                 a=0.05,
                 abs_diff=False,
                 method='bca',
                 interpolation='nearest'):
        # Quick check for a valid estimation method
        methods = ['pct', 'diff', 'bca']
        assert method in methods, 'Method must be pct, diff, or bca.'
        
        # Pulling out the original estiamtes
        ref_stat = pd.Series(ref.cis.stat.drop('true_prev').values)
        ref_scores = ref.scores.drop('true_prev', axis=1)
        comp_stat = pd.Series(comp.cis.stat.drop('true_prev').values)
        comp_scores = comp.scores.drop('true_prev', axis=1)
        
        # Optionally Reversing the order of comparison
        diff_scores = comp_scores - ref_scores
        diff_stat = comp_stat - ref_stat
            
        # Setting the quantiles to retrieve
        lower = (a / 2) * 100
        upper = 100 - lower
        
        # Calculating the percentiles 
        if method == 'pct':
            cis = np.nanpercentile(diff_scores,
                                   q=(lower, upper),
                                   interpolation=interpolation,
                                   axis=0)
            cis = pd.DataFrame(cis.transpose())
        
        elif method == 'diff':
            diffs = diff_stat.values.reshape(1, -1) - diff_scores
            percents = np.nanpercentile(diffs,
                                        q=(lower, upper),
                                        interpolation=interpolation,
                                        axis=0)
            lower_bound = pd.Series(diff_stat + percents[0])
            upper_bound = pd.Series(diff_stat + percents[1])
            cis = pd.concat([lower_bound, upper_bound], axis=1)
        
        elif method == 'bca':
            # Removing true prevalence from consideration to avoid NaNs
            ref_j_means = ref.jack[1].drop('true_prev')
            ref_j_scores = ref.jack[0].drop('true_prev', axis=1)
            comp_j_means = comp.jack[1].drop('true_prev')
            comp_j_scores = comp.jack[0].drop('true_prev', axis=1)
            
            # Calculating the bias-correction factor
            n = ref.scores.shape[0]
            stat_vals = diff_stat.transpose().values.ravel()
            n_less = np.sum(diff_scores < stat_vals, axis=0)
            p_less = n_less / n
            z0 = norm.ppf(p_less)
            
            # Fixing infs in z0
            z0[np.where(np.isinf(z0))[0]] = 0.0
            
            # Estiamating the acceleration factor
            j_means = comp_j_means - ref_j_means
            j_scores = comp_j_scores - ref_j_scores
            diffs = j_means - j_scores
            numer = np.sum(np.power(diffs, 3))
            denom = 6 * np.power(np.sum(np.power(diffs, 2)), 3/2)
            
            # Getting rid of 0s in the denominator
            zeros = np.where(denom == 0)[0]
            for z in zeros:
                denom[z] += 1e-6
            
            acc = numer / denom
            
            # Calculating the bounds for the confidence intervals
            zl = norm.ppf(a / 2)
            zu = norm.ppf(1 - (a/2))
            lterm = (z0 + zl) / (1 - acc*(z0 + zl))
            uterm = (z0 + zu) / (1 - acc*(z0 + zu))
            ql = norm.cdf(z0 + lterm) * 100
            qu = norm.cdf(z0 + uterm) * 100
                                    
            # Returning the CIs based on the adjusted quantiles
            cis = [np.nanpercentile(diff_scores.iloc[:, i], 
                                    q=(ql[i], qu[i]),
                                    interpolation=interpolation,
                                    axis=0) 
                   for i in range(len(ql))]
            cis = pd.DataFrame(cis, columns=['lower', 'upper'])
                    
        cis = pd.concat([ref_stat, comp_stat, diff_stat, cis], 
                        axis=1)
        cis = cis.set_index(ref_scores.columns.values)
        cis.columns = ['ref', 'comp', 'd', 
                       'lower', 'upper']
        
        # Passing stuff back up to return
        self.cis = cis
        self.scores = diff_scores
        self.b = z0
        self.acc = acc
        
        return


def jackknife_metrics(targets, 
                      guesses,
                      sample_by=None, 
                      average='weighted'):
    # Replicates of the dataset with one row missing from each
    j_rows = jackknife_sample(targets, by=sample_by)
    
    # using a pool to get the metrics across each
    score_input = [(targets[idx], guesses[idx]) for idx in j_rows]
    with Pool() as p:
        scores = p.starmap(clf_metrics, score_input)
    
    scores = pd.concat(scores, axis=0)
    means = scores.mean()
    
    return scores, means



def merge_cis(c, round=4, mod_name=''):
    str_cis = c.round(round).astype(str)
    str_paste = pd.DataFrame(str_cis.stat + ' (' + str_cis.lower + 
                                 ', ' + str_cis.upper + ')',
                                 columns=[mod_name]).transpose()
    return str_paste


def merge_ci_list(l, mod_names=None, round=4):
    if type(l[0] != type(pd.DataFrame())):
        l = [c.cis for c in l]
    if mod_names is not None:
        merged_cis = [merge_cis(l[i], round, mod_names[i])
                      for i in range(len(l))]
    else:
        merged_cis = [merge_cis(c, round=round) for c in l]
    
    return pd.concat(merged_cis, axis=0)

