import numpy as np
import pandas as pd

from scipy.special import expit, erf
from sklearn.metrics import confusion_matrix

from .generic import smash_log, zm_to_y


def brier_score(targets, guesses):
    """Calculates Brier score."""
    n_classes = len(np.unique(targets))
    assert n_classes > 1
    if n_classes == 2:
        bs = np.sum((guesses - targets)**2) / targets.shape[0]
    else:
        y = onehot_matrix(targets)
        row_diffs = np.diff((guesses, y), axis=0)[0]
        squared_diffs = row_diffs ** 2
        row_sums = np.sum(squared_diffs, axis=1) 
        bs = row_sums.mean()
    return bs


def mcc(y, y_):
    """Calculates Matthews Correlation Coefficient."""
    tp = np.sum((y == 1) & (y_ == 1))
    fp = np.sum((y == 0) & (y_ == 1))
    fn = np.sum((y == 1) & (y_ == 0))
    tn = np.sum((y == 0) & (y_ == 0))
    num = ((tp * tn) - (fp * fn))
    denom = np.sqrt(((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    return num / denom if denom != 0 else 0


def f1(y, y_):
    """Alternative call for f_score()."""
    return f_score(y, y_, b=1)


def f_score(y, y_, b=1):
    """Calculates F-score from two binary vectors."""
    tp = np.sum((y == 1) & (y_ == 1))
    sens = tp / y.sum()
    ppv = tp / y_.sum()
    if sens + ppv != 0:
        return (1 + b**2) * (sens * ppv) / ((b**2 * ppv) + sens)
    else:
        return 0


def sens_exp(z, xp, B=100):
    """Approximates sensitivity, or true positive rate, using the smash_log()
    function.
    """
    m = z[-1]
    z = z[:-1]
    return smash_log(np.dot(xp, z) - m, B=B).sum() / xp.shape[0]


def spec_exp(z, xn, B=100):
    """Approximates specificity, or 1 minus FPR, using the smash_log() 
    function.
    """
    m = z[-1]
    z = z[:-1]
    return 1 - smash_log(np.dot(xn, z) - m, B=B).sum() / xn.shape[0]


def j(y, y_, a=1, b=1):
    """Calculates Youden's J index from two binary vectors."""
    c = a + b
    a = a / c * 2
    b = b / c * 2
    sens = np.sum((y == 1) & (y_ == 1)) / y.sum()
    spec = np.sum((y == 0) & (y_ == 0)) / (len(y) - y.sum())
    return a*sens + b*spec - 1


def j_lin(z, m, X, y):
    """Calculates Youden's J index as a linear combination of a variable 
    choice vector, two variable matrices, and m."""
    z = np.round(z)
    y_ = zm_to_y(z, m, X)
    return j(y, y_)


def j_lin_comp(z_mat, m_vec, X, y):
    """Calculates Youden's J index from the N matrix and M vector specifying
    a given compound rule. Both must be binary.
    """
    guesses = np.array([zm_to_y(z_mat[:, i], m_vec[i], X)
                        for i in range(z_mat.shape[1])])
    y_ = np.array(np.sum(guesses, 0) > 0, dtype=np.uint8)
    stat = j(y, y_)
    return stat


def j_exp(z, xp, xn, a=1, b=1):
    """Calculates Youden's J index for a single m-of-n rule using the 
    parameters of a solved LP (z) and the smash_log() function.
    """
    m = z[-1]
    z = smash_log(z[:-1] - .5)
    tpr = smash_log(np.dot(xp, z) - m + .5).sum() / xp.shape[0]
    fpr = smash_log(np.dot(xn, z) - m + .5).sum() / xn.shape[0]
    return -1 * (a*tpr - b*fpr)


def j_exp_comp(z, xp, xn, c=2, a=1, b=1, th=0):
    """Calculates Youden's J index for a compound m-of-n rule using the
    parameters of a solved LP (z) and the smash_log() function.
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


def clf_metrics(true, 
                pred,
                average='weighted',
                preds_are_probs=False,
                cutpoint=0.5,
                mod_name=None,
                round=4,
                round_pval=False,
                mcnemar=False,
                argmax_axis=1):
    '''Runs basic diagnostic stats on binary (only) predictions'''
    # Converting pd.Series to np.array
    stype = type(pd.Series([0]))
    if type(pred) == stype:
        pred = pred.values
    if type(true) == stype:
        true = true.values
    
    # Optional exit for doing averages with multiclass/label inputs
    if len(np.unique(true)) > 2:
        # Getting binary metrics for each set of results
        codes = np.unique(true)
        
        # Argmaxing for when we have probabilities
        if preds_are_probs:
            auc = roc_auc_score(true,
                                pred,
                                average=average,
                                multi_class='ovr')
            brier = brier_score(true, pred)
            pred = np.argmax(pred, axis=argmax_axis)
        
        # Making lists of the binary predictions (OVR)    
        y = [np.array([doc == code for doc in true], dtype=np.uint8)
             for code in codes]
        y_ = [np.array([doc == code for doc in pred], dtype=np.uint8)
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
        auc = roc_auc_score(true, pred)
        brier = brier_score(true, pred)
        ap = average_precision_score(true, pred)
        pred = threshold(pred, cutpoint)
    else:
        brier = np.round(brier_score(true, pred), round)
    
    # Constructing the 2x2 table
    confmat = confusion_matrix(true, pred)
    tp = confmat[1, 1]
    fp = confmat[0, 1]
    tn = confmat[0, 0]
    fn = confmat[1, 0]
    
    # Calculating the main binary metrics
    ppv = np.round(tp / (tp + fp), round) if tp + fp > 0 else 0
    sens = np.round(tp / (tp + fn), round) if tp + fn > 0 else 0
    spec = np.round(tn / (tn + fp), round) if tn + fp > 0 else 0
    npv = np.round(tn / (tn + fn), round) if tn + fn > 0 else 0
    f1 = np.round(2*(sens*ppv) / (sens+ppv), round) if sens + ppv != 0 else 0 
    
    # Calculating the Matthews correlation coefficient
    mcc_num = ((tp * tn) - (fp * fn))
    mcc_denom = np.sqrt(((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    mcc = mcc_num / mcc_denom if mcc_denom != 0 else 0
    
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
    true_prev = int(np.sum(true == 1))
    pred_prev = int(np.sum(pred == 1))
    abs_diff = (true_prev - pred_prev) * -1
    rel_diff = np.round(abs_diff / true_prev, round)
    if mcnemar:
        pval = mcnemar_test(true, pred).pval[0]
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
        out['model'] = mod_name
    
    return out

