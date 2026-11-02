"""DeLong tests for (paired) AUROC, with an optional participant-clustered variant.

* Standard DeLong (DeLong, DeLong & Clarke-Pearson, Biometrics 1988), computed with
  the fast mid-rank algorithm of Sun & Xu (IEEE Signal Process. Lett. 2014).
  Windows are treated as independent observations.
* Clustered DeLong (Obuchowski, Biometrics 1997): the same structural components
  are summed within participant, so the variance accounts for correlation between
  windows of the same participant. With one window per participant it reduces
  exactly to the standard DeLong variance.

Ties in scores receive half credit (mid-ranks), matching roc_auc_score.

The manuscript's dAUROC P values are the standard (unclustered) paired test,
delong_paired_test(y, a, b, groups=None). The clustered variant is reported by
bootstrap_auc.py only as a sensitivity analysis (clustered_delong_* columns).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata


def _placements(y: np.ndarray, scores: np.ndarray):
    """Return AUCs (K,), V10 (K, M) for positives and V01 (K, N) for negatives."""
    pos, neg = y == 1, y == 0
    m, n = int(pos.sum()), int(neg.sum())
    if m == 0 or n == 0:
        raise ValueError("DeLong requires both classes.")
    k = scores.shape[0]
    v10, v01, aucs = np.empty((k, m)), np.empty((k, n)), np.empty(k)
    for r in range(k):
        s = scores[r]
        tz = rankdata(s)                 # mid-ranks among all observations
        tx = rankdata(s[pos])            # among positives
        ty = rankdata(s[neg])            # among negatives
        v10[r] = (tz[pos] - tx) / n      # P(score_pos > score_neg) for each positive
        v01[r] = 1.0 - (tz[neg] - ty) / m
        aucs[r] = v10[r].mean()
    return aucs, v10, v01, pos, neg


def delong_covariance(y, scores, groups=None):
    """AUCs and their DeLong covariance matrix.

    y      : (n_obs,) binary labels
    scores : (K, n_obs) scores of K models on the same observations
    groups : optional (n_obs,) cluster ids (participants) for the clustered variant
    """
    y = np.asarray(y).astype(int)
    scores = np.atleast_2d(np.asarray(scores, dtype=float))
    aucs, v10, v01, pos, neg = _placements(y, scores)
    m, n = v10.shape[1], v01.shape[1]

    if groups is None:
        s10 = np.atleast_2d(np.cov(v10))
        s01 = np.atleast_2d(np.cov(v01))
        return aucs, s10 / m + s01 / n

    groups = np.asarray(groups)
    codes, _ = _factorize(groups)
    n_clusters = codes.max() + 1
    gp, gn = codes[pos], codes[neg]
    m_i = np.bincount(gp, minlength=n_clusters)
    n_i = np.bincount(gn, minlength=n_clusters)
    # Per-cluster sums of structural components, centred: (K, I)
    c10 = np.stack([np.bincount(gp, weights=v10[r], minlength=n_clusters) for r in range(len(aucs))]) - np.outer(aucs, m_i)
    c01 = np.stack([np.bincount(gn, weights=v01[r], minlength=n_clusters) for r in range(len(aucs))]) - np.outer(aucs, n_i)
    i10, i01, i_all = int((m_i > 0).sum()), int((n_i > 0).sum()), n_clusters
    if min(i10, i01) < 2:
        raise ValueError("Clustered DeLong needs >= 2 clusters containing each class.")
    s10 = i10 / ((i10 - 1) * m) * (c10 @ c10.T)
    s01 = i01 / ((i01 - 1) * n) * (c01 @ c01.T)
    s11 = i_all / (i_all - 1) * (c10 @ c01.T)
    return aucs, s10 / m + s01 / n + (s11 + s11.T) / (m * n)


def _factorize(a):
    uniq, codes = np.unique(a, return_inverse=True)
    return codes.reshape(-1), uniq


def delong_auc_ci(y, score, groups=None, alpha=0.05):
    """AUC, standard error and normal-approximation CI (clipped to [0, 1])."""
    auc, cov = delong_covariance(y, np.atleast_2d(score), groups)
    se = float(np.sqrt(max(cov[0, 0], 0.0)))
    z = norm.ppf(1 - alpha / 2)
    return float(auc[0]), se, max(0.0, auc[0] - z * se), min(1.0, auc[0] + z * se)


def delong_paired_test(y, score_a, score_b, groups=None, alpha=0.05):
    """Paired DeLong test of AUC_a - AUC_b on the same observations.

    Returns dict(delta, se, z, p_value, ci_low, ci_high); two-sided P value.
    """
    aucs, cov = delong_covariance(y, np.vstack([score_a, score_b]), groups)
    delta = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    se = float(np.sqrt(max(var, 0.0)))
    if se == 0.0:
        z, p = (0.0, 1.0) if delta == 0 else (np.sign(delta) * np.inf, 0.0)
    else:
        z = delta / se
        p = float(2 * norm.sf(abs(z)))
    q = norm.ppf(1 - alpha / 2)
    return {"delta": delta, "se": se, "z": float(z), "p_value": p,
            "ci_low": delta - q * se, "ci_high": delta + q * se}
