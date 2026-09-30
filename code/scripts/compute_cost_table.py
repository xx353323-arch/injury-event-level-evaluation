import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from baselines_comparison import LSTMModel, StandardGAT
from graph_backend_v2 import make_model_v2
from new_baselines import TransformerEncoder, ProtoNet
from run_dann_10seed import DANNModel


class LogitOnly(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        out = self.m(x)
        return out[0] if isinstance(out, tuple) else out


class ProtoStep(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.y = None

    def forward(self, x):
        z = self.m.embed(x)
        y = self.y if self.y is not None else torch.zeros(x.shape[0])
        c1 = z[y > 0.5].mean(0) if (y > 0.5).any() else z.mean(0)
        c0 = z[y <= 0.5].mean(0) if (y <= 0.5).any() else z.mean(0)
        return ((z - c0) ** 2).sum(-1) - ((z - c1) ** 2).sum(-1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference(model, x, n_warmup=5, n_repeat=50):
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
        t0 = time.perf_counter()
        for _ in range(n_repeat):
            _ = model(x)
        t1 = time.perf_counter()
    return (t1 - t0) / n_repeat * 1000.0


def measure_train_step(model, x, y, n_warmup=3, n_repeat=20):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(n_warmup):
        opt.zero_grad()
        logit = model(x)
        loss = bce(logit, y)
        loss.backward()
        opt.step()
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        opt.zero_grad()
        logit = model(x)
        loss = bce(logit, y)
        loss.backward()
        opt.step()
    t1 = time.perf_counter()
    return (t1 - t0) / n_repeat * 1000.0


def measure_xgboost(seed, n_train=500, n_feat=140):
    from sklearn.ensemble import GradientBoostingClassifier
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_train, n_feat))
    y = rng.integers(0, 2, n_train)
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                     subsample=0.8, random_state=seed)
    t0 = time.perf_counter()
    clf.fit(X, y)
    train_t = (time.perf_counter() - t0) * 1000.0
    n_eval = 100
    X_te = rng.standard_normal((n_eval, n_feat))
    t0 = time.perf_counter()
    n_repeat = 20
    for _ in range(n_repeat):
        _ = clf.predict_proba(X_te)
    infer_t = (time.perf_counter() - t0) / n_repeat * 1000.0
    n_params = sum(t.tree_.node_count for est in clf.estimators_.ravel() for t in [est])
    return dict(params=int(n_params), train_ms=float(train_t), infer_ms=float(infer_t))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    print("=" * 60)
    print("Computational Cost Comparison")
    print("=" * 60)

    torch.manual_seed(42)
    B, T, C = 64, 14, 10
    x = torch.randn(B, T, C)
    y = torch.randint(0, 2, (B,)).float()

    results = []

    print("\n[1] XGBoost (GradientBoosting)")
    xgb_stat = measure_xgboost(seed=42)
    print(f"    Params (tree nodes): {xgb_stat['params']:,}")
    print(f"    Train time (full fit): {xgb_stat['train_ms']:.2f} ms")
    print(f"    Inference (B=100):    {xgb_stat['infer_ms']:.2f} ms")
    results.append(dict(method="XGBoost", params=xgb_stat["params"],
                        train_ms_per_step=None, train_full_ms=xgb_stat["train_ms"],
                        infer_ms=xgb_stat["infer_ms"],
                        note="GradientBoosting, params=tree nodes; train_full_ms=full fit"))

    print("\n[2] LSTM")
    lstm = LSTMModel(input_dim=10, hidden=64, n_layers=2, dropout=0.3)
    p_lstm = count_params(lstm)
    t_lstm_train = measure_train_step(lstm, x, y)
    t_lstm_infer = measure_inference(lstm, x)
    print(f"    Params: {p_lstm:,}")
    print(f"    Train step (B=64): {t_lstm_train:.2f} ms")
    print(f"    Inference (B=64):  {t_lstm_infer:.2f} ms")
    results.append(dict(method="LSTM", params=p_lstm,
                        train_ms_per_step=float(t_lstm_train), train_full_ms=None,
                        infer_ms=float(t_lstm_infer),
                        note="2-layer LSTM hidden=64, dropout=0.3"))

    print("\n[3] GAT (StandardGAT)")
    gat = StandardGAT(n_ch=10, d_model=32, n_heads=2, T=14)
    p_gat = count_params(gat)
    t_gat_train = measure_train_step(gat, x, y)
    t_gat_infer = measure_inference(gat, x)
    print(f"    Params: {p_gat:,}")
    print(f"    Train step (B=64): {t_gat_train:.2f} ms")
    print(f"    Inference (B=64):  {t_gat_infer:.2f} ms")
    results.append(dict(method="GAT", params=p_gat,
                        train_ms_per_step=float(t_gat_train), train_full_ms=None,
                        infer_ms=float(t_gat_infer),
                        note="StandardGAT d_model=32, 2 heads, channel-wise attention"))

    print("\n[4] NVG (TCAGNN with fixed natural-visibility-graph)")
    nvg = make_model_v2(backend="nvg", n_ch=10, T=14)
    p_nvg = count_params(nvg)
    t_nvg_train = measure_train_step(nvg, x, y)
    t_nvg_infer = measure_inference(nvg, x)
    print(f"    Params: {p_nvg:,}")
    print(f"    Train step (B=64): {t_nvg_train:.2f} ms")
    print(f"    Inference (B=64):  {t_nvg_infer:.2f} ms")
    results.append(dict(method="NVG", params=p_nvg,
                        train_ms_per_step=float(t_nvg_train), train_full_ms=None,
                        infer_ms=float(t_nvg_infer),
                        note="TCAGNN + fixed natural visibility graph"))

    print("\n[5] LMVG (TCAGNN with learnable multi-view graph)")
    lmvg = make_model_v2(backend="lmvg", n_ch=10, T=14)
    p_lmvg = count_params(lmvg)
    t_lmvg_train = measure_train_step(lmvg, x, y)
    t_lmvg_infer = measure_inference(lmvg, x)
    print(f"    Params: {p_lmvg:,}")
    print(f"    Train step (B=64): {t_lmvg_train:.2f} ms")
    print(f"    Inference (B=64):  {t_lmvg_infer:.2f} ms")
    results.append(dict(method="LMVG", params=p_lmvg,
                        train_ms_per_step=float(t_lmvg_train), train_full_ms=None,
                        infer_ms=float(t_lmvg_infer),
                        note="TCAGNN + learnable multi-view graph (3 views with softmax fusion)"))

    extra = [
        ("Transformer", LogitOnly(TransformerEncoder(n_channels=10, T=14)), "2-layer Transformer encoder d_model=32, 2 heads"),
        ("DANN", LogitOnly(DANNModel(input_dim=10)), "LSTM encoder with classification and domain heads; supervised step on the classification logit"),
        ("ProtoNet", ProtoStep(ProtoNet(n_channels=10, T=14)), "Transformer encoder with projection head; prototype logits from the batch"),
    ]
    for k, (name, model, note) in enumerate(extra, start=6):
        print(f"\n[{k}] {name}")
        if isinstance(model, ProtoStep):
            model.y = y
        p_m = count_params(model)
        t_tr = measure_train_step(model, x, y)
        t_in = measure_inference(model, x)
        print(f"    Params: {p_m:,}")
        print(f"    Train step (B=64): {t_tr:.2f} ms")
        print(f"    Inference (B=64):  {t_in:.2f} ms")
        results.append(dict(method=name, params=p_m, train_ms_per_step=float(t_tr), train_full_ms=None,
                            infer_ms=float(t_in), note=note))

    print("\n" + "=" * 60)
    print("Summary Table")
    print("=" * 60)
    print(f"{'Method':<12s} {'Params':>12s} {'Train/step(ms)':>18s} {'Infer(ms)':>14s}")
    print("-" * 60)
    for r in results:
        train_str = f"{r['train_ms_per_step']:.2f}" if r['train_ms_per_step'] is not None else f"{r['train_full_ms']:.0f}(full)"
        print(f"{r['method']:<12s} {r['params']:>12,} {train_str:>18s} {r['infer_ms']:>14.2f}")

    out_dir = os.environ.get("COST_OUT", here)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "computational_cost_results.json"), "w") as f:
        json.dump(dict(
            results=results,
            env=dict(
                batch_size=B, T=T, n_channels=C,
                torch_version=torch.__version__,
                device="cpu",
                n_warmup_infer=5, n_repeat_infer=50,
                n_warmup_train=3, n_repeat_train=20,
            ),
        ), f, indent=2)

    pd.DataFrame(results).to_csv(os.path.join(out_dir, "computational_cost_table.csv"), index=False)
    print(f"\nSaved: computational_cost_results.json and computational_cost_table.csv")


if __name__ == "__main__":
    main()
