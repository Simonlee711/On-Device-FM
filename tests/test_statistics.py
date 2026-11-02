"""Synthetic fixtures with known answers for thresholds, operating points, bootstrap pairing and DeLong."""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from _bootstrap import bootstrap_indices, paired_p_value, wide_by_model
from apply_operating_points import apply_thresholds, load_thresholds
from bootstrap_auc import compare_aurocs
from bootstrap_operating_points import bootstrap_operating_points
from conftest import REPO
from delong import delong_covariance, delong_paired_test
from prediction_schema import validate_predictions
from select_validation_thresholds import select_thresholds


def rows(model, split, person, sample, y, s, task="PVC", regime="Frozen"):
    return pd.DataFrame({"task": task, "model": model, "regime": regime, "split": split,
                         "person_id": person, "sample_id": sample, "y_true": y, "y_score": s})


# --- validation-only threshold selection ---------------------------------------------------

def val_fixture() -> pd.DataFrame:
    # 20 negatives at 0.00, 0.05, ..., 0.95 and 10 positives at 0.525, 0.575, ..., 0.975.
    neg = np.round(np.arange(20) * 0.05, 3)
    pos = np.round(0.525 + np.arange(10) * 0.05, 3)
    y = np.r_[np.zeros(20, int), np.ones(10, int)]
    s = np.r_[neg, pos]
    return rows("A", "val", np.arange(30) // 5, np.arange(30) % 5, y, s)


def test_threshold_known_answer_and_tie_break():
    thr = select_thresholds(validate_predictions(val_fixture()), [0.90])
    r = thr.iloc[0]
    # Specificity 0.90 is reached at t = 0.875 (sens 3/10) and at t = 0.90 (sens 2/10);
    # the tie is broken toward higher validation sensitivity.
    assert r.threshold == pytest.approx(0.875)
    assert r.validation_specificity == pytest.approx(0.90)
    assert r.validation_sensitivity == pytest.approx(0.30)
    assert r.n_validation == 30


def test_thresholds_ignore_test_rows():
    val = val_fixture()
    test_a = rows("A", "test", [9] * 4, range(4), [0, 0, 1, 1], [0.99, 0.98, 0.01, 0.02])
    test_b = rows("A", "test", [9] * 4, range(4), [1, 1, 0, 0], [0.5, 0.6, 0.7, 0.8])
    t1 = select_thresholds(validate_predictions(pd.concat([val, test_a])), [0.95, 0.90, 0.80])
    t2 = select_thresholds(validate_predictions(pd.concat([val, test_b])), [0.95, 0.90, 0.80])
    t0 = select_thresholds(validate_predictions(val), [0.95, 0.90, 0.80])
    pd.testing.assert_frame_equal(t1, t0)
    pd.testing.assert_frame_equal(t2, t0)
    with pytest.raises(ValueError, match="validation split"):
        select_thresholds(validate_predictions(test_a), [0.9])


def test_apply_known_confusion_matrix():
    test = rows("A", "test", [1, 1, 1, 2, 2, 2, 2], range(7),
                [1, 1, 0, 0, 0, 1, 0], [0.9, 0.4, 0.7, 0.2, 0.5, 0.5, 0.1])
    thr = load_thresholds(pd.DataFrame({"task": ["PVC"], "model": ["A"], "regime": ["Frozen"],
                                        "target_specificity": [0.9], "threshold": [0.5]}))
    r = apply_thresholds(validate_predictions(test), thr).iloc[0]
    # >= 0.5 is positive: TP {0.9, 0.5}, FN {0.4}, FP {0.7, 0.5}, TN {0.2, 0.1}
    assert (r.tp, r.fp, r.tn, r.fn) == (2, 2, 2, 1)
    assert r.sensitivity == pytest.approx(2 / 3) and r.specificity == pytest.approx(0.5)
    assert r.ppv == pytest.approx(0.5) and r.npv == pytest.approx(2 / 3)
    assert r.youden_index == pytest.approx(2 / 3 + 0.5 - 1)


# --- bootstrap at fixed operating points ----------------------------------------------------

N_PART = 6


def paired_test_fixture() -> pd.DataFrame:
    """Participant p has p + 1 positive and 3 negative windows.

    Reference "R" scores every positive 0.9 -> TP_p = p + 1 at threshold 0.5.
    Comparator "C" scores one positive per participant 0.2 -> TP_p = p.
    Negatives score 0.1 (R) and 0.3 / 0.3 / 0.6 (C: one FP per participant).
    So dTP_p = 1 for every participant while TP_p itself varies across participants.
    """
    out = []
    for p in range(N_PART):
        n_pos = p + 1
        y = [1] * n_pos + [0, 0, 0]
        s_r = [0.9] * n_pos + [0.1, 0.1, 0.1]
        s_c = [0.2] + [0.9] * (n_pos - 1) + [0.3, 0.3, 0.6]
        sid = range(len(y))
        out += [rows("R", "test", p, sid, y, s_r), rows("C", "test", p, sid, y, s_c)]
    return validate_predictions(pd.concat(out, ignore_index=True))


def fixed_thresholds(value=0.5) -> pd.DataFrame:
    return load_thresholds(pd.DataFrame({"task": "PVC", "model": ["R", "C"], "regime": "Frozen",
                                         "target_specificity": 0.9, "threshold": value}))


def test_paired_tp_uses_identical_participant_multiplicities():
    pred, thr = paired_test_fixture(), fixed_thresholds()
    n_boot, seed = 300, 11
    pts, paired, reps = bootstrap_operating_points(pred, thr, "R", n_boot, seed, return_replicates=True)
    rep = reps[("PVC", "Frozen", 0.9)]

    # TP of each model varies across replicates (participants differ)...
    assert np.std(rep["R"]["tp"]) > 0 and np.std(rep["C"]["tp"]) > 0
    # ...but the paired difference is always the number of sampled participants (= N_PART),
    # which holds only if both models see the same participants with the same multiplicities.
    np.testing.assert_array_equal(rep["R"]["tp"] - rep["C"]["tp"], np.full(n_boot, N_PART))
    r = paired.iloc[0]
    assert (r.tp_reference, r.tp_comparator, r.delta_tp) == (21, 15, 6)
    assert (r.ci_low, r.ci_high) == (6, 6)
    assert r.p_value == pytest.approx(2 / (n_boot + 1))


def test_replicates_apply_the_fixed_validation_threshold():
    pred, thr = paired_test_fixture(), fixed_thresholds()
    n_boot, seed = 200, 3
    pts, _, reps = bootstrap_operating_points(pred, thr, "R", n_boot, seed, return_replicates=True)
    rep = reps[("PVC", "Frozen", 0.9)]

    # Recompute every replicate from per-participant counts at the frozen threshold 0.5,
    # using the multiplicities produced by the same participant resampler.
    w, _ = wide_by_model(pred[pred.split == "test"])
    person = w.person_id.to_numpy()
    per_person = {m: {p: int(((w[m] >= 0.5) & (w.y_true == 1))[person == p].sum()) for p in range(N_PART)}
                  for m in ("R", "C")}
    fp_c = {p: int(((w["C"] >= 0.5) & (w.y_true == 0))[person == p].sum()) for p in range(N_PART)}
    for b, idx in enumerate(bootstrap_indices(person, n_boot, seed)):
        sampled = person[idx]
        mult = {p: int((sampled == p).sum()) // int((person == p).sum()) for p in range(N_PART)}
        assert sum(mult.values()) == N_PART
        for m in ("R", "C"):
            assert rep[m]["tp"][b] == sum(mult[p] * per_person[m][p] for p in range(N_PART))
        assert rep["C"]["fp"][b] == sum(mult[p] * fp_c[p] for p in range(N_PART))

    # Point estimates equal a single application of the same thresholds.
    direct = apply_thresholds(pred, thr).set_index("model")
    for m in ("R", "C"):
        got = pts.set_index("model").loc[m]
        assert (got.tp, got.fp, got.tn, got.fn) == tuple(direct.loc[m, ["tp", "fp", "tn", "fn"]])


def test_threshold_rows_without_test_predictions_are_rejected():
    pred = paired_test_fixture()
    thr = load_thresholds(pd.concat([fixed_thresholds(), pd.DataFrame(
        {"task": ["PVC"], "model": ["Ghost"], "regime": ["Frozen"], "target_specificity": [0.9], "threshold": [0.5]})]))
    with pytest.raises(ValueError, match="without test predictions"):
        bootstrap_operating_points(pred, thr, "R", 10, 0)


def test_paired_p_value_finite_replicate_correction():
    assert paired_p_value(np.ones(999)) == pytest.approx(2 / 1000)
    assert paired_p_value(np.zeros(10)) == 1.0
    assert paired_p_value(np.r_[np.ones(5), -np.ones(5)]) == 1.0


# --- DeLong and AUROC provenance -----------------------------------------------------------

def direct_delong(y, a, b):
    """O(M*N) DeLong (1988) for the paired difference AUC_a - AUC_b."""
    def psi(x, z):
        return (x > z) + 0.5 * (x == z)
    pos, neg = y == 1, y == 0
    comps = []
    for s in (a, b):
        k = psi(s[pos][:, None], s[neg][None, :])
        comps.append((k.mean(), k.mean(1), k.mean(0)))
    v10 = np.vstack([c[1] for c in comps])
    v01 = np.vstack([c[2] for c in comps])
    cov = np.cov(v10) / pos.sum() + np.cov(v01) / neg.sum()
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    return comps[0][0] - comps[1][0], np.sqrt(var)


def test_delong_matches_direct_formula_with_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    a = np.round(y + rng.normal(0, 1, 300), 1)  # rounding creates ties
    b = np.round(0.5 * y + rng.normal(0, 1, 300), 1)
    delta, se = direct_delong(y, a, b)
    r = delong_paired_test(y, a, b)
    assert r["delta"] == pytest.approx(delta, abs=1e-12)
    assert r["se"] == pytest.approx(se, rel=1e-10)
    assert r["delta"] == pytest.approx(roc_auc_score(y, a) - roc_auc_score(y, b), abs=1e-12)


def test_clustered_delong_reduces_to_standard_with_singleton_clusters():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    s = np.vstack([y + rng.normal(0, 1, 200), y + rng.normal(0, 1.5, 200)])
    _, plain = delong_covariance(y, s)
    _, clustered = delong_covariance(y, s, groups=np.arange(200))
    np.testing.assert_allclose(clustered, plain, rtol=1e-10)


def auc_fixture() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    out = []
    for p in range(12):
        y = rng.integers(0, 2, 30)
        y[:2] = [0, 1]
        base = rng.normal(0, 1, 30)
        out += [rows("Ref", "test", p, range(30), y, y * 1.5 + base),
                rows("Cmp", "test", p, range(30), y, y * 0.7 + base + rng.normal(0, 0.5, 30))]
    return validate_predictions(pd.concat(out, ignore_index=True))


def test_reported_auroc_columns_come_from_the_stated_methods():
    pred = auc_fixture()
    n_boot, seed = 200, 2024
    summary, paired = compare_aurocs(pred, "Ref", n_boot, seed)
    r = paired.iloc[0]
    w, _ = wide_by_model(pred)
    y, a, b, person = w.y_true.to_numpy(), w["Ref"].to_numpy(), w["Cmp"].to_numpy(), w.person_id.to_numpy()

    # Manuscript P value: ordinary (unclustered) paired DeLong on all test windows.
    assert r.paired_delong_p_value == pytest.approx(delong_paired_test(y, a, b)["p_value"], rel=1e-12)
    assert r.clustered_delong_p_value == pytest.approx(delong_paired_test(y, a, b, person)["p_value"], rel=1e-12)

    # Manuscript CI: percentile interval of paired participant-bootstrap differences.
    d = np.array([roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i])
                  for i in bootstrap_indices(person, n_boot, seed)])
    assert (r.participant_bootstrap_ci_low, r.participant_bootstrap_ci_high) == pytest.approx(
        tuple(np.percentile(d, [2.5, 97.5])))
    assert r.participant_bootstrap_p_value == pytest.approx(paired_p_value(d))
    assert r.delta_auroc == pytest.approx(roc_auc_score(y, a) - roc_auc_score(y, b))

    # No ambiguous legacy column names remain.
    for old in ("delong_p_value", "boot_p_value", "boot_ci_low", "delong_cluster_p_value"):
        assert old not in paired.columns and old not in summary.columns


# --- end-to-end workflow on the released canonical example ----------------------------------

def test_statistical_workflow_runs_on_released_example(tmp_path):
    env = {**os.environ, "PYTHON": sys.executable}
    subprocess.run(["bash", str(REPO / "analysis" / "run_statistical_workflow.sh"),
                    "--predictions", str(REPO / "examples" / "synthetic_predictions.csv"),
                    "--out", str(tmp_path), "--fig3-regime", "Frozen"], check=True, env=env,
                   capture_output=True)
    for name in ("auroc_summary", "auroc_paired_differences", "validation_thresholds", "operating_points",
                 "operating_points_bootstrap", "operating_points_paired_tp", "fig3_clinical"):
        assert (tmp_path / f"{name}.csv").exists(), name
    summary = pd.read_csv(tmp_path / "auroc_summary.csv").set_index("model")
    assert summary.loc["HiMAE (1.2M)", "auroc"] == pytest.approx(0.9093, abs=5e-4)
