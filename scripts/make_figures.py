"""Make README figures from logs/results. Each figure is skipped if its inputs are missing.

    python scripts/make_figures.py --runs runs/seed0 runs/seed1 runs/seed2 \
        --evals results/seed0_eval.json results/seed1_eval.json results/seed2_eval.json \
        --ablation results/ablation_steps.json --uncertainty results/uncertainty.json --out results/figures
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Reported in Chi et al. 2023 (arXiv v5, Table I/II), Push-T, max / avg of last 10 ckpts,
# mean of 3 training seeds x 50 initializations. Copied from the paper; not produced by this repo.
PAPER_PUSHT = {
    "state (Table I)": {"DiffusionPolicy-C": (0.95, 0.91), "DiffusionPolicy-T": (0.95, 0.79),
                        "IBC": (0.90, 0.84), "BET": (0.79, 0.70), "LSTM-GMM": (0.67, 0.61)},
    "image (Table II)": {"DiffusionPolicy-C": (0.91, 0.84), "DiffusionPolicy-T": (0.78, 0.66),
                         "IBC": (0.75, 0.64), "LSTM-GMM": (0.69, 0.54)},
}


def load_jsonl(path):
    return [json.loads(line) for line in open(path)]


def fig_training(runs, out):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    for r in runs:
        log = os.path.join(r, "train_log.jsonl")
        if not os.path.exists(log):
            continue
        rec = load_jsonl(log)
        ax[0].plot([x["epoch"] for x in rec], [x["loss"] for x in rec], label=os.path.basename(r))
        ev = [x for x in rec if "quick_eval_mean_score" in x]
        ax[1].plot([x["epoch"] for x in ev], [x["quick_eval_mean_score"] for x in ev], marker=".",
                   label=os.path.basename(r))
    ax[0].set(xlabel="epoch", ylabel="noise-prediction MSE", yscale="log", title="Training loss")
    ax[1].set(xlabel="epoch", ylabel="mean score (10 eps, monitor only)", ylim=(0, 1), title="Quick eval")
    for a in ax:
        a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "training_curves.png"), dpi=150); plt.close(fig)


def table_vs_paper(evals, out):
    rows = []
    for e in evals:
        s = json.load(open(e))
        rows.append((s["max_over_ckpts"], s["avg_over_ckpts"], s["n_ckpts"], s["per_ckpt"][0]["n_episodes"]))
    if not rows:
        return
    mx, av = np.mean([r[0] for r in rows]), np.mean([r[1] for r in rows])
    lines = ["| Method | Push-T max | Push-T avg (last 10) | Source |", "|---|---|---|---|"]
    for name, (pm, pa) in PAPER_PUSHT["state (Table I)"].items():
        lines.append(f"| {name} | {pm:.2f} | {pa:.2f} | paper, Table I |")
    lines.append(f"| **This repo (CNN, 5-D state obs)** | **{mx:.2f}** | **{av:.2f}** | "
                 f"measured: {len(rows)} seed(s) x {rows[0][2]} ckpts x {rows[0][3]} inits |")
    open(os.path.join(out, "results_table.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def fig_ablation(path, out):
    d = json.load(open(path))["rows"]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for r in d:
        ax.scatter(r["latency_ms_median"], r["mean_score"], color="C0" if r["method"] == "ddim" else "C3")
        ax.annotate(f"{r['method'].upper()}-{r['steps']}", (r["latency_ms_median"], r["mean_score"]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set(xscale="log", xlabel="median latency per replan (ms, log)", ylabel="mean score (50 seeds)",
           title="Sampler steps: score vs latency"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "ablation_steps.png"), dpi=150); plt.close(fig)


def fig_uncertainty(path, out):
    d = json.load(open(path))
    sc = np.array([e["score"] for e in d["episodes"]]); dm = np.array([e["disp_mean"] for e in d["episodes"]])
    fail = sc < d["fail_below"]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.scatter(dm[~fail], sc[~fail], s=12, label="success", color="C0")
    ax.scatter(dm[fail], sc[fail], s=12, label="failure", color="C3")
    a = d["auroc"]["disp_mean"]
    ax.set(xlabel="mean action-sample dispersion (px)", ylabel="episode score",
           title=f"AUROC={a['auroc']:.2f} [{a['ci95'][0]:.2f}, {a['ci95'][1]:.2f}], n={len(sc)}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "uncertainty_vs_score.png"), dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="*", default=[])
    p.add_argument("--evals", nargs="*", default=[])
    p.add_argument("--ablation")
    p.add_argument("--uncertainty")
    p.add_argument("--out", default="results/figures")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.runs:
        fig_training(a.runs, a.out)
    table_vs_paper(a.evals, a.out)
    if a.ablation and os.path.exists(a.ablation):
        fig_ablation(a.ablation, a.out)
    if a.uncertainty and os.path.exists(a.uncertainty):
        fig_uncertainty(a.uncertainty, a.out)


if __name__ == "__main__":
    main()
