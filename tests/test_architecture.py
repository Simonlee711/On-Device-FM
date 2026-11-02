"""Checks that the released code matches the main-text HiMAE configuration.

Run with:  python -m pytest tests/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pvc.utils.model_arch.himae import HiMAE, PAPER_CONFIG  # noqa: E402

CKPT = REPO / "himae_synth.ckpt"


def paper_model() -> HiMAE:
    return HiMAE({"source": "ppg", "sampling_freq": PAPER_CONFIG["sampling_freq"],
                  "seg_len": PAPER_CONFIG["seg_len"], "model_params": {}})


def test_paper_config_values():
    # Methods: channel hierarchy [16, 32, 64, 128, 256], patch size 5, kernel 5, stride 2,
    # masking ratio 0.8; 10-second windows at 100 Hz (1,000 samples).
    assert PAPER_CONFIG["channels"] == (16, 32, 64, 128, 256)
    assert PAPER_CONFIG["patch_len"] == 5
    assert PAPER_CONFIG["kernel_size"] == 5
    assert PAPER_CONFIG["stride"] == 2
    assert PAPER_CONFIG["mask_ratio"] == 0.8
    assert PAPER_CONFIG["sampling_freq"] * PAPER_CONFIG["seg_len"] == 1000


def test_default_model_is_paper_config():
    m = paper_model()
    assert m.channels == PAPER_CONFIG["channels"]
    assert m.cfg.patch_len == PAPER_CONFIG["patch_len"]
    assert m.seq_len == 1000
    convs = [mod for mod in m.encoder_layers.modules() if isinstance(mod, torch.nn.Conv1d) and mod.kernel_size[0] > 1]
    assert {c.kernel_size[0] for c in convs} == {5}
    assert {c.stride[0] for c in convs} == {1, 2}
    assert sum(p.numel() for p in m.parameters()) == 1_245_953  # "1.2M"


def test_multiresolution_embeddings():
    m = paper_model().eval()
    x = torch.randn(3, 1, 1000)
    feats = m.encode(x)
    assert [f.shape[1] for f in feats] == list(PAPER_CONFIG["channels"])
    assert [f.shape[2] for f in feats] == [500, 250, 125, 63, 32]
    embs = m.embed_multiresolution(x)
    assert [e.shape for e in embs] == [(3, c) for c in PAPER_CONFIG["channels"]]
    rf = [g["receptive_field"] for g in m.stage_geometry()]
    assert rf == sorted(rf) and rf[0] == 13
    assert m(ppg=x)["ppg_reconstructed"].shape == (3, 1, 1000)


@pytest.mark.skipif(not CKPT.exists(), reason="released checkpoint not present")
def test_released_checkpoint_matches_paper_config():
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    hp = state["hyper_parameters"]
    assert hp["f_s"] == PAPER_CONFIG["sampling_freq"]
    assert hp["T"] == PAPER_CONFIG["seg_len"]
    assert hp["patch"] == PAPER_CONFIG["patch_len"]
    assert tuple(hp["channels"]) == PAPER_CONFIG["channels"]
    assert hp["mask_ratio"] == PAPER_CONFIG["mask_ratio"]
    sd = {k[len("model."):]: v for k, v in state["state_dict"].items() if k.startswith("model.")}
    paper_model().load_state_dict(sd, strict=True)


def test_pretraining_model_shares_definition():
    pytest.importorskip("lightning")
    for mod in ("s3fs", "wandb", "boto3"):
        pytest.importorskip(mod)
    sys.path.insert(0, str(REPO / "pretrain"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("himae_pretrain", REPO / "pretrain" / "himae.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    a, b = mod.HiMAE(), paper_model()
    assert {k: v.shape for k, v in a.state_dict().items()} == {k: v.shape for k, v in b.state_dict().items()}
    assert a(torch.randn(2, 1, 1000)).shape == (2, 1, 1000)
