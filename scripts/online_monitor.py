"""Is the dispersion signal usable ONLINE, and is it just episode length?
For each replan step t, take episodes still running at t and predict eventual failure
from the running median of dispersion over replans 1..t (only information available at t)."""
import json, sys, numpy as np
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), ".."))
from scripts.uncertainty_failure import auroc, bootstrap_ci
u = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'results/uncertainty.json'))
eps = u['episodes']; thr = u['fail_below']
L = np.array([len(e['disp']) for e in eps]); fail = np.array([e['score'] < thr for e in eps])
print(f"replans per episode: failures median {np.median(L[fail]):.0f}, successes median {np.median(L[~fail]):.0f}")
print(f"{'t':>3s} {'running':>8s} {'fail':>5s} {'AUROC':>6s}  95% CI")
for t in [2, 4, 6, 8, 10, 15, 20, 25, 30]:
    idx = [i for i, e in enumerate(eps) if len(e['disp']) >= t]
    if len(idx) < 10: break
    s = [float(np.median(eps[i]['disp'][:t])) for i in idx]; y = fail[idx]
    lo, hi = bootstrap_ci(s, y)
    print(f"{t:3d} {len(idx):8d} {int(y.sum()):5d} {auroc(s, y):6.3f}  [{lo:.3f}, {hi:.3f}]")
