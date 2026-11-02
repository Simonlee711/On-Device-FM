"""Device-log summarizer (latency/throughput only, supplied logs) and analytic model complexity."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from summarize_device_benchmarks import efficiency_table, summarize


def device_log() -> pd.DataFrame:
    return pd.DataFrame({
        "model": ["HiMAE (1.2M)"] * 4 + ["Other"] * 4,
        "run": list(range(4)) * 2,
        "latency_s": [0.90, 0.91, 0.89, 0.90, 1.40, 1.45, 1.42, 1.43],
        "throughput_wps": [1.20, 1.19, 1.21, 1.20, 0.70, 0.71, 0.69, 0.70],
        "energy_j": [1.0] * 8, "battery_start_pct": [100] * 8,  # present in old logs; must be ignored
    })


def test_device_summary_ignores_energy_and_labels_source(capsys):
    summary, tests, models = summarize(device_log(), "HiMAE (1.2M)")
    assert "ignoring columns" in capsys.readouterr().out
    assert set(summary.metric) == {"latency_s", "throughput_wps"}
    assert summary.source.str.contains("external device log").all()
    eff = efficiency_table(summary, models, pd.DataFrame({"model": ["HiMAE (1.2M)"], "params_m": [1.2]}))
    text = " ".join(map(str, [*summary.columns, *tests.columns, *eff.columns])).lower()
    for word in ("energy", "battery", "discharge"):
        assert word not in text
    assert list(eff.columns) == ["model", "cpu_latency", "cpu_throughput", "params"]


def test_welch_test_matches_scipy():
    log = device_log()
    _, tests, _ = summarize(log, "HiMAE (1.2M)")
    r = tests[(tests.model == "Other") & (tests.metric == "latency_s")].iloc[0]
    ref = stats.ttest_ind(log.latency_s[4:], log.latency_s[:4], equal_var=False)
    assert r.p_value == pytest.approx(ref.pvalue) and r.t_stat == pytest.approx(ref.statistic)
    assert r.mean_diff == pytest.approx(np.mean(log.latency_s[4:]) - np.mean(log.latency_s[:4]))


def test_model_complexity_scopes():
    pytest.importorskip("torch")
    from model_complexity import complexity_table, parse_args

    t = complexity_table(parse_args([])).set_index("scope")
    assert list(t.index) == ["pretraining_autoencoder", "downstream_classifier"]
    assert round(t.loc["pretraining_autoencoder", "params_m"] * 1e6) == 1_245_953
    assert round(t.loc["downstream_classifier", "params_m"] * 1e6) == 700_929  # encoder 700,672 + head 257
    assert t.loc["pretraining_autoencoder", "memory_mib"] == pytest.approx(1_245_953 * 4 / 2**20)
