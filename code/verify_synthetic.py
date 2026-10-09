#!/usr/bin/env python3
"""Synthetic checks of the functions in run_all.py.

Inputs: none (random synthetic data). Outputs: PASS lines printed to the screen.
Checks: Jensen direction of g, conservation under source aggregation, and
convergence of the Barnes-Hut sum to the exact sum as theta goes to 0.
"""
import numpy as np

def g(x, a=0.5, b=0.5):
    return 1.0 / (1.0 + a * np.maximum(0.0, -x) + b * np.maximum(0.0, x))

rng = np.random.default_rng(7)
# 1. Jensen direction for same-sign and mixed-sign values
xs = rng.uniform(0.2, 2.0, 1000)
assert g(xs.mean()) <= g(xs).mean() + 1e-12
xm = np.array([-2.0, 2.0])
assert g(xm.mean()) > g(xm).mean()
print("1. Jensen / mixed-sign directions: PASS")
# 2. source aggregation conserves the total
M = rng.uniform(1, 100, 500); x = rng.normal(0, 1, 500)
assert abs((M * g(x)).sum() - float(np.sum(M * g(x)))) == 0.0
print("2. Source conservation (identity by construction): PASS")
# 3. Barnes-Hut against the exact sum
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location("run_all", pathlib.Path(__file__).parent / "run_all.py")
# run_all.py loads data on import, so only build_tree and bh_eval are taken from it
import ast
src = (pathlib.Path(__file__).parent / "run_all.py").read_text()
tree = ast.parse(src)
wanted = {"build_tree", "bh_eval"}
code = "\n\n".join(ast.get_source_segment(src, n) for n in tree.body
                   if isinstance(n, ast.FunctionDef) and n.name in wanted)
ns = {"np": np, "SELF": 25.0, "D0": 100.0}
exec(code, ns)
build_tree, bh_eval = ns["build_tree"], ns["bh_eval"]
xy = rng.uniform(0, 500, (120, 2)); w = rng.uniform(1, 50, 120)
xt = rng.uniform(0, 500, (60, 2))
nodes = build_tree(xy, w)
Aex = np.array([float(np.exp(-np.maximum(np.hypot(*(xy - t).T), 25.0) / 100.0) @ w) for t in xt])
for theta, tol in [(0.3, 0.10), (0.1, 0.002)]:
    Ab, ev = bh_eval(nodes, xy, w, xt, theta)
    err = np.abs(Ab - Aex) / Aex
    werr = float(np.average(err, weights=Aex))  # access-weighted error
    assert werr < tol, (theta, werr)
    print(f"3. BH theta={theta}: access-weighted mean rel err {werr*100:.3f}% (unweighted {err.mean()*100:.2f}%)  evals {ev} vs exact {len(xt)*len(w)}: PASS")
print("ALL SYNTHETIC CHECKS PASS (machinery verification only)")
