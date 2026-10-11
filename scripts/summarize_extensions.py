"""Paired bootstrap CIs for extension 1 and AUROC summary for extension 2 (reads results/*.json)."""
import json, numpy as np
d = json.load(open('results/ablation_steps.json'))['rows']
base = np.array(d[0]['per_episode']); rng = np.random.default_rng(0)
print(f"{'setting':10s} {'score':>6s} {'diff vs DDPM-100':>17s} {'95% CI':>18s} {'ms':>7s}")
for r in d:
    x = np.array(r['per_episode']); diff = x - base
    bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(5000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"{r['method']+'-'+str(r['steps']):10s} {x.mean():6.3f} {diff.mean():+17.3f} [{lo:+.3f}, {hi:+.3f}] {r['latency_ms_median']:7.1f}")
u = json.load(open('results/uncertainty.json'))
print({k: u[k] for k in ['K', 'method', 'steps', 'fail_below', 'n_episodes', 'n_failures']})
for k, v in u['auroc'].items(): print(f"{k:16s} AUROC={v['auroc']:.3f}  95% CI [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")
