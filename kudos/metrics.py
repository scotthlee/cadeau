"""Various metrics for measuring classification performance."""
import numpy as np
import pandas as pd

from scipy.special import expit, erf
from sklearn.metrics import confusion_matrix
from multiprocessing import shared_memory

from .tools import smash_log, zm_to_y, pair_sum, combo_sum


fn_dict = {'j': ['sens', 'spec'],
           'f1': ['sens', 'ppv'],
           'mcc': ['j', 'mk']}


def brier_score(y, y_):
    """Calculates Brier score.
    
    Parameters
    ----------
    y : 1d array-like
      A vector of class labels.
    y_ : 1d array-like
      A vector of predicted probabilities.
    
    Returns
    ----------
    bs : float
        The Brier score.
    """
    n_classes = len(np.unique(y))
    assert n_classes > 1
    if n_classes == 2:
        bs = np.sum((y_ - y)**2) / y.shape[0]
    else:
        y = onehot_matrix(y)
        row_diffs = np.diff((y_, y), axis=0)[0]
        squared_diffs = row_diffs ** 2
        row_sums = np.sum(squared_diffs, axis=1) 
        bs = row_sums.mean()
    return bs


def mcc(y, y_, undef_val=0):
    """Calculates Matthews Correlation Coefficient. There are two ways of doing
    this: one based on the counts in the confusion matrix, and one based on two
    component metrics, Youden's J index and markedness. This function 
    implements the latter, returning the score as the product of the component 
    metrics on the given problem (y, y_) and on its dual (y_, y).
    
    Parameters
    ----------
    y : 1d array-like
        A binary vector of class labels.
    y_ : 1d array-like
        A binary vector of predicted class labels.
    undef_val : float, default=0.0
        What to return when the metric is undefined. Options are 0 or NaN.
    
    Returns
    -------
    mcc : float
        Matthews correlation coefficient, or phi.
    """
    prod = j(y, y_) * j(y_, y) * mk(y, y_) * mk(y_, y)
    return prod ** (1/4) if prod != 0 else undef_val


def f1(y, y_):
    """Alternative call for f_score() where b equals 1.
    
    Parameters
    ----------
    y : 1d array-like
        A 1d binary vector of class labels.
    y_ : array-like
        A binary vector of predicted class labels.
    
    Returns
    -------
    f1 : float
        The F1 score.
    """
    return f_score(y, y_, b=1)


def f_score(y, y_, b=1, undef_val=0):
    """Calculates F-score.
    
    Parameters
    ----------
    y : 1d array-like
        A binary vector of class labels.
    y_ : 1d array-like
        A binary vector of predicted class labels.
    b : int, default=1
        The degree of f-score.
    undef_val : float, default=0.0
        What to return when the score is undefined. Options are 0.0 and NaN.
    
    Returns
    -------
    f : float
        The f-score.
    """
    se = sens(y, y_)
    pv = ppv(y, y_)
    if se + pv != 0:
        return (1 + b**2) * (se * pv) / ((b**2 * pv) + se)
    else:
        return undef_val


def sens(y, y_):
    """Calculates sensitivity, or recall.
    
    Parameters
    ----------
    y : 1d array-like
        A binary vector of class labels.
    y_ : 1d array-like
        A binary vector of predicted class labels.
    
    Returns
    -------
    sens : float.
        Sensitivity, AKA recall or the true positive rate (TPR).
    """
    tp = np.sum((y ==1) & (y_ == 1))
    return tp / y.sum()


def spec(y, y_):
    """Calculates specificity, or 1 - FPR.
    
    Parameters
    ----------
    y : 1d array-like
        A binary vector of class labels.
    y_ : 1d array-like
        A binary vector of predicted class labels.
    
    Returns
    -------
    spec : float
        Specificity, or 1 - FPR.
    """
    tn = np.sum((y == 0) & (y_ == 0))
    return tn / np.sum(y == 0)


def ppv(y, y_):
    """Calculates positive predictive value, or precision.
    
    Parameters
    ----------
    y : 1d array-like
        A binary vector of class labels.
    y_ : 1d array-like
        A binary vector of predicted class labels.
    
    Returns
    -------
    ppv : float
        Positive predictive value.
    """
    tp = np.sum((y == 1) & (y_ == 1))
    return tp / y_.sum()


def npv(y, y_):
    """Calculates negative predictive value.
    
    Parameters
    ----------
    y : 1d array-like
        A binary vector of class labels.
    y_ : 1d array-like
        A binary vector of predicted class labels.
    
    Returns
    -------
    npv : float
        Negative predictive value.
    """
    tn = np.sum((y == 0) & (y_ == 0))
    return tn / np.sum(y_ == 0)


def mk(y, y_):
    """Calculates markedness, or PPV + NPV - 1. One of the two component 
    metrics for Matthews correlation coefficient, along with the J index.
    
    Parameters
    ----------
    y : 1d array-like
        A binary vector of class labels.
    y_ : 1d array-like
        A binary vector of predicted class labels.
    
    Returns
    -------
    mk : float.
        Markedness, or PPV + NPV minus 1.
    """
    return ppv(y, y_) + npv(y, y_) - 1


def j(y, y_, a=1, b=1):
    """Calculates Youden's J index from two binary vectors.
    
    Parameters
    ----------
    y : 1d array-like
        Binary vector of true class labels.
    y_ : 1d array-like
        Binary vector of predicted class labels.
    a : float, default=1.0
        (Optional) weight for sensitivity.
    b : float, default=1.0
        (Optional) weight for specificity.
    
    Returns
    -------
    j : float
        Youden's J index.
    """
    c = a + b
    a = a / c * 2
    b = b / c * 2
    sens = np.sum((y == 1) & (y_ == 1)) / y.sum()
    spec = np.sum((y == 0) & (y_ == 0)) / (len(y) - y.sum())
    return a*sens + b*spec - 1


def sens_exp(z, xp, B=100):
    """Approximates sensitivity, or true positive rate, using the smash_log()
    function.
    
    Parameters
    ----------
    z : 1d array-like
        A binary variable choice vector.
    xp : 2d array-like
        A binary matrix of observations for the true positives.
    B : int, default=100
        The "smash factor", i.e., the multiplier for x in the logistic funciton.
    
    Returns
    -------
    sens (approx) : float
        Sensitivity, AKA recall or true positive rate (TPR), approximately.
    """
    m = z[-1]
    z = z[:-1]
    return smash_log(np.dot(xp, z) - m, B=B).sum() / xp.shape[0]


def spec_exp(z, xn, B=100):
    """Approximates specificity, or 1 minus FPR, using the smash_log() 
    function.
    
    Parameters
    ----------
    z : 1d array-like
        A binary variable choice vector.
    xn : 2d array-like
        A binary matrix of observations for the true negatives.
    B : int, default=100
        The "smash factor", i.e., the multiplier for x in the logistic funciton.
    
    Returns
    -------
    spec (approx) : float
        Specificity, or 1 - FPR, approximately.
    """
    m = z[-1]
    z = z[:-1]
    return 1 - smash_log(np.dot(xn, z) - m, B=B).sum() / xn.shape[0]


def j_lin(z, m, X, y):
    """Calculates Youden's J index as a linear combination of a variable 
    choice vector, two variable matrices, and m.
    
    Parameters
    ----------
    z : 1d array-like
        A binary variable choice vector.
    m : int
        Minimum row sum for X needed for the corresponding predicted class
        labels y_ to be 1.
    X : 2d array-like
        A binary matrix of observations.
    y : 1d array-like
        A binary vector of class labels. 
      
    Returns
    -------
    j : float
        Youden's J index.
    """
    z = np.round(z)
    y_ = zm_to_y(z, m, X)
    return j(y, y_)


def j_lin_comp(z_mat, m_vec, X, y):
    """Calculates Youden's J index from the N matrix and M vector specifying
    a given compound rule. Both must be binary.
    
    Parameters
    ----------
    z_mat : 2d array-like
        An array of 1d variable choice vectors.
    m_vec : int
        A vector with the minimum row sums for X needed for the predicted
        class labels y_ to be 1. Must correspond to the variable ordering in 
        z_mat.
    X : 2d array-like
        A binary matrix of observations.
    y : 1d array-like
        A binary vector of class labels. 
    
    Returns
    -------
    j : float
        Youden's J index.
    """
    guesses = np.array([zm_to_y(z_mat[:, i], m_vec[i], X)
                        for i in range(z_mat.shape[1])])
    y_ = np.array(np.sum(guesses, 0) > 0, dtype=np.uint8)
    stat = j(y, y_)
    return stat


def j_exp(z, xp, xn, a=1, b=1):
    """Approximates Youden's J index for a single m-of-n rule using the 
    parameters of a solved LP (z) and the smash_log() function.
    
    Parameters
    ----------
    z : 1d array-like
        A binary variable choice vector.
    xp : 2d array-like
        Binary matrix of observations for the true positives.
    xn : 2d array-like
        Binary matrix of observatinos for the true negatives.
    a : float, default=1.0
        Weight for sensitivity (TPR) in calculating J.
    b : float, default=1.0
        Weight for specificity (1 - FPR) in calculating J.
    
    Returns
    -------
    j : float
        Youden's J index, approximately.
    """
    m = z[-1]
    z = smash_log(z[:-1] - .5)
    tpr = smash_log(np.dot(xp, z) - m + .5).sum() / xp.shape[0]
    fpr = smash_log(np.dot(xn, z) - m + .5).sum() / xn.shape[0]
    return -1 * (a*tpr - b*fpr)


def j_exp_comp(z, xp, xn, c=2, a=1, b=1, th=0):
    """Approximates Youden's J index for a compound m-of-n rule using the
    parameters of a solved LP (z) and the smash_log() function.
    
    Parameters
    ----------
    z : 1d array-like
        A binary variable choice vector.
    xp : 2d array-like
        Binary matrix of observations for the true positives.
    xn : 2d array-like
        Binary matrix of observatinos for the true negatives.
    c : int, default=2
        Number of sub-rules in the compound rule.
    a : float, default=1.0
        Weight for sensitivity (TPR) in calculating J.
    b : float, default=1.0
        Weight for specificity (1 - FPR) in calculating J.
    th : float, default=0.0
        No idea.
    
    Returns
    -------
    j : float
        Youden's J Index, approximately.
    """
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


def pair_score(X, combo_cols, combo_mins, y, metric):
    """Scores a pair of sets of columns.
    
    Parameters
    ----------
    X : 2d array-like
        Binary matrix of observations.
    combo_cols : list of 1d array-like
        A list of sets of column numbers specifying the component variables of a
        sub-rule.
    combo_mins : 1d array-like
        The minimum 
    y : 1d array-like
        A binary vector of true class labels.
    metric : {'j', 'f1', 'mcc'}
        Which metric to use as the base metric. 
    
    Returns
    -------
    and_scores, or_scores : list of float arrays
        The ``and_scores`` are the metrics for when both combos must be met, and
        the ``or_scores`` are for when only one of the combos must be met. 
    """
    ps = pair_sum(X, combo_cols, combo_mins)
    cs = combo_sum(ps)
    and_scores = score_set(y, cs[:, 0], metric)
    or_scores = score_set(y, cs[:, 1], metric)
    return [and_scores, or_scores]


def score_set(y, y_, metric, return_df=False):
    """Provides a small set of scores for a set of predictions.
    
    Parameters
    ----------
    y : 1d array-like
        A binary vector of the true class labels.
    y_ : 1d array-like
        A binary vector of the predicted class labels.
    metric : {'j', 'f1', 'mcc'}
        The base metric for evaluating the predictions.
    return_df : bool, default=False
        Whether to return the scores as a pandas DataFrame.
    
    Returns
    -------
    scores : list of floats
        The first component, second component, and base metric scores for the 
        predictions.
    """
    fn1 = globals()[fn_dict[metric][0]]
    fn2 = globals()[fn_dict[metric][1]]
    fn3 = globals()[metric]
    score_fns = [fn1, fn2, fn3]
    scores = [fn(y, y_) for fn in score_fns]
    if return_df:
        out = pd.DataFrame(scores).transpose()
        out.columns = fn_dict[metric] + [metric]
        return out
    else:
        return scores


def shared_mem_score(combo_cols, m, xshape, metric='j'):
    """Pulls X and y from shared memory and then scores a combination.
    
    Parameters
    ----------
    combo_cols : 1d
    
    Returns
    -------
    scores : list of floats
        The first component, second component, and base metric scores for the 
        predictions.
    """
    X_buf = shared_memory.SharedMemory(name='predictors')
    y_buf = shared_memory.SharedMemory(name='outcome')
    X = np.ndarray(xshape, dtype=np.uint8, buffer=X_buf.buf)
    y = np.ndarray((xshape[0],), dtype=np.uint8, buffer=y_buf.buf)
    y_ = np.array(X[:, combo_cols].sum(1) > m, dtype=np.uint8)
    scores = score_set(y, y_, metric)
    X_buf.close()
    y_buf.close()
    return scores


def clf_metrics(y, y_,
                average='weighted',
                preds_are_probs=False,
                cutpoint=0.5,
                mod_name=None,
                round=4,
                round_pval=False,
                mcnemar=False,
                argmax_axis=1,
                undef_val=0):
    """Generates a panel of classification metrics for a predictor.
    
    Parameters
    ----------
    y : 1d array-like
        A vector of true class labels.
    y_ : 1d or 2d array-like
        A vector of predicted class labels or class probabilities.
    average : {'weighted', 'macro', 'micro'}, default='weighted'
        How to average metrics for multiclass problems.
    preds_are_probs : bool, default=False
        Whether the predictions are labels or probabilities.
    cutpoint : float, default=0.5
        Decision threshold to apply to the probabilities.
    mod_name : str, default=None
        (Optional) name for the rule or model that generated the predictions.
    round : int, default=4
        Rounding parameter for the results.
    round_pval : bool, default=False
        Whether to round p-values from McNemar's chi-squared test.
    mcnemar : bool, default=False
        Whether to display p-values form McNemar's chi-squared test.
    argmax_axis : int, default=1
        Axis parameter for argmaxing multiclass probabilities.
    undef_val : float, default=0.0
        Return value for metrics when they're undefined.
    
    Returns
    -------
    results : pd.DataFrame
        A pandas DataFrame of classification metrics.
    """
    # Converting pd.Series to np.array
    stype = type(pd.Series([0]))
    if type(y_) == stype:
        y_ = y_.values
    if type(y) == stype:
        y = y.values
    
    # Optional exit for doing averages with multiclass/label inputs
    if len(np.unique(y)) > 2:
        # Getting binary metrics for each set of results
        codes = np.unique(y)
        
        # Argmaxing for when we have probabilities
        if preds_are_probs:
            auc = roc_auc_score(y, y_,
                                average=average,
                                multi_class='ovr')
            brier = brier_score(y, y_)
            y_ = np.argmax(y_, axis=argmax_axis)
        
        # Making lists of the binary predictions (OVR)    
        y = [np.array([doc == code for doc in y], dtype=np.uint8)
             for code in codes]
        y_ = [np.array([doc == code for doc in y_], dtype=np.uint8)
              for code in codes]
        
        # Getting the stats for each set of binary predictions
        stats = [clf_metrics(y[i], y_[i], round=16) for i in range(len(y))]
        stats = pd.concat(stats, axis=0)
        stats.fillna(0, inplace=True)
        cols = stats.columns.values
        
        # Calculating the averaged metrics
        if average == 'weighted':
            weighted = np.average(stats, 
                                  weights=stats.true_prev,
                                  axis=0)
            out = pd.DataFrame(weighted).transpose()
            out.columns = cols
        elif average == 'macro':
            out = pd.DataFrame(stats.mean()).transpose()
        elif average == 'micro':
            out = clf_metrics(np.concatenate(y),
                              np.concatenate(y_))
        
        # Adding AUC and AP for when we have probabilities
        if preds_are_probs:
            out.auc = auc
            out.brier = brier
        
        # Rounding things off
        out = out.round(round)
        count_cols = [
            'tp', 'fp', 'tn', 'fn', 'true_prev',
            'pred_prev', 'prev_diff'
        ]
        out[count_cols] = out[count_cols].round()
        
        if mod_name is not None:
            out['model'] = mod_name
        
        return out
    
    # Thresholding the probabilities, if provided
    if preds_are_probs:
        auc = roc_auc_score(y, y_)
        brier = brier_score(y, y_)
        ap = average_precision_score(y, y_)
        pred = threshold(y_, cutpoint)
    else:
        brier = np.round(brier_score(y, y_), round)
    
    # Constructing the 2x2 table
    confmat = confusion_matrix(y, y_)
    tp = confmat[1, 1]
    fp = confmat[0, 1]
    tn = confmat[0, 0]
    fn = confmat[1, 0]
    
    # Calculating the main binary metrics
    ppv = np.round(tp / (tp + fp), round) if tp + fp > 0 else undef_val
    sens = np.round(tp / (tp + fn), round) if tp + fn > 0 else undef_val
    spec = np.round(tn / (tn + fp), round) if tn + fp > 0 else undef_val
    npv = np.round(tn / (tn + fn), round) if tn + fn > 0 else undef_val
    if sens + ppv != 0:
        f1 = np.round(2*(sens*ppv) / (sens+ppv), round)  
    else:
        f1 = undef_val
    
    # Calculating the Matthews correlation coefficient
    mcc_num = ((tp * tn) - (fp * fn))
    mcc_denom = np.sqrt(((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    mcc = mcc_num / mcc_denom if mcc_denom != 0 else undef_val
    
    # Calculating Youden's J and the Brier score
    j = sens + spec - 1
    
    # Rolling everything so far into a dataframe
    outmat = np.array([tp, fp, tn, fn,
                       sens, spec, ppv,
                       npv, j, f1, mcc, brier]).reshape(-1, 1)
    out = pd.DataFrame(outmat.transpose(),
                       columns=['tp', 'fp', 'tn', 
                                'fn', 'sens', 'spec', 'ppv',
                                'npv', 'j', 'f1', 'mcc', 'brier'])
    
    # Optionally tacking on stats from the raw probabilities
    if preds_are_probs:
        out['auc'] = auc
        out['ap'] = ap
    else:
        out['auc'] = 0
        out['ap'] = 0
    
    # Calculating some additional measures based on positive calls
    true_prev = int(np.sum(y == 1))
    pred_prev = int(np.sum(y_ == 1))
    abs_diff = (true_prev - pred_prev) * -1
    rel_diff = np.round(abs_diff / true_prev, round)
    if mcnemar:
        pval = mcnemar_test(y, y_).pval[0]
        if round_pval:
            pval = np.round(pval, round)
    count_outmat = np.array([true_prev, pred_prev, abs_diff, 
                             rel_diff]).reshape(-1, 1)
    count_out = pd.DataFrame(count_outmat.transpose(),
                             columns=['true_prev', 'pred_prev', 
                                      'prev_diff', 'rel_prev_diff'])
    out = pd.concat([out, count_out], axis=1)
    
    # Optionally dropping the mcnemar p-val
    if mcnemar:
        out['mcnemar'] = pval
    
    # And finally tacking on the model name
    if mod_name is not None:
        stat_cols = list(out.columns.values)
        out['model'] = mod_name
        out = out[['model'] + stat_cols]
    
    return out

