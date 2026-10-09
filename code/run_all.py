#!/usr/bin/env python3
"""Runs experiments E0 to E4 on the data in ../data/.

Inputs: cells.csv, tracts.csv, spei_cells.npz, state_boundaries.geojson.
Outputs: ../results/ tables and figures, ../run_manifest.json.
Access: A_i(t) = sum_j W_ij * M_j / m_j(t), W = exp(-distance/d0), self distance 25 km,
m_j = 1 + alpha*max(0,-SPEI_j) + beta*max(0,SPEI_j), alpha = beta = 0.5, d0 = 100 km.
"""
import json, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA, RES = ROOT / "data", ROOT / "results"
ALPHA = BETA = 0.5; D0 = 100.0; SELF = 25.0; R = 6371.0
rng_global = np.random.default_rng(20261006)

def g(x, a=ALPHA, b=BETA):
    return 1.0 / (1.0 + a * np.maximum(0.0, -x) + b * np.maximum(0.0, x))

def hav(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dla = la1[:, None] - la2[None, :]; dlo = lo1[:, None] - lo2[None, :]
    h = np.sin(dla/2)**2 + np.cos(la1[:, None]) * np.cos(la2[None, :]) * np.sin(dlo/2)**2
    d = 2 * R * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
    return d

def wrms(err, w):
    return float(np.sqrt(np.sum(w * err**2) / np.sum(w)))

# load data
cells = pd.read_csv(DATA / "cells.csv")
tracts = pd.read_csv(DATA / "tracts.csv")
z = np.load(DATA / "spei_cells.npz", allow_pickle=True)
months = list(z["months"]); X = z["spei"]            # (528, 564), NaN = no data
T, N = X.shape
M = cells["population"].values.astype(float)
lat, lon = cells["lat"].values, cells["lon"].values
state = cells["state"].values
finite = np.isfinite(X)
Wm = np.exp(-np.maximum(hav(lat, lon, lat, lon), 0) / D0); np.fill_diagonal(Wm, np.exp(-SELF / D0))
# set the diagonal to the self-distance term
np.fill_diagonal(Wm, np.exp(-SELF / D0))
Mw = M / M.sum()

def access_series(w_series):
    """Access from effective sources, (T,N) to (T,N)."""
    return w_series @ Wm.T

w_true = np.where(finite, M[None, :] * g(np.where(finite, X, 0.0)), 0.0)
A_true = access_series(w_true)
A_static = Wm @ M
loss_true = A_true / A_static[None, :] - 1.0
anom_true = A_true / A_true.mean(0)[None, :] - 1.0

def rep_metrics(A_rep, A0_rep):
    loss = A_rep / A0_rep[None, :] - 1.0
    anom = A_rep / A_rep.mean(0)[None, :] - 1.0
    return dict(
        mean_loss_pct=float(np.sum(Mw[None, :] * loss.mean(0)) * 100),
        bias_loss_pp=float(np.sum(Mw[None, :] * (loss - loss_true).mean(0)) * 100),
        rmse_loss_pp=wrms((loss - loss_true) * 100, Mw[None, :].repeat(T, 0) if False else np.broadcast_to(Mw, (T, N))),
        rmse_anom_pp=wrms((anom - anom_true) * 100, np.broadcast_to(Mw, (T, N))),
        bias_anom_pp=float(np.average(anom - anom_true, weights=np.broadcast_to(Mw, (T, N))) * 100),
    )

# E0 checks
def e0():
    rows = []
    # zero anomaly check
    A0s = access_series(M[None, :].repeat(T, 0) * 1.0)
    rows.append(dict(check="zero_anomaly", value=float(np.abs(A0s / A_static[None, :] - 1).max()),
                     unit="max |loss| (fraction)", expected="0"))
    # mean loss vs static over all months
    rows.append(dict(check="all_months_mean_loss_vs_static_pct",
                     value=float(np.sum(Mw[None, :] * loss_true.mean(0)) * 100), unit="%", expected="-26.6"))
    # random weather: permute finite SPEI values across cells within each month
    Xr = X.copy()
    for t in range(T):
        idx = np.where(finite[t])[0]
        Xr[t, idx] = rng_global.permutation(X[t, idx])
    wr = np.where(finite, M[None, :] * g(np.where(finite, Xr, 0.0)), 0.0)
    Ar = access_series(wr)
    rows.append(dict(check="random_weather_mean_loss_vs_static_pct",
                     value=float((np.sum(Mw[None, :] * (Ar / A_static[None, :] - 1).mean(0))) * 100),
                     unit="%", expected="-25.3"))
    # climatology-relative access
    Abar = A_true.mean(0); D = A_true / Abar[None, :] - 1.0
    aggD = D @ Mw
    dom_spei = np.array([np.sum(Mw[finite[t]] * X[t, finite[t]]) / np.sum(Mw[finite[t]]) for t in range(T)])
    abs_spei = np.array([np.sum(Mw[finite[t]] * np.abs(X[t, finite[t]])) / np.sum(Mw[finite[t]]) for t in range(T)])
    order = np.argsort(dom_spei); dry, wet = order[:53], order[-53:]
    rows.append(dict(check="climrel_agg_driest_decile_pct", value=float(aggD[dry].mean() * 100), unit="%", expected="-4.50"))
    rows.append(dict(check="climrel_agg_wettest_decile_pct", value=float(aggD[wet].mean() * 100), unit="%", expected="-7.53"))
    rows.append(dict(check="climrel_percell_absD_popwtd_mean_pct",
                     value=float(np.mean((np.abs(D) @ Mw)) * 100), unit="%", expected="14.1"))
    # random weather 2: resample from the pooled finite SPEI values
    pooled = X[finite]; Xr2 = np.full_like(X, np.nan)
    Xr2[finite] = rng_global.choice(pooled, size=int(finite.sum()), replace=True)
    wr2 = np.where(finite, M[None, :] * g(np.where(finite, Xr2, 0.0)), 0.0)
    Ar2 = access_series(wr2)
    rows.append(dict(check="random_weather_pooled_resample_mean_loss_vs_static_pct",
                     value=float(np.sum(Mw[None, :] * (Ar2 / A_static[None, :] - 1).mean(0)) * 100),
                     unit="%", expected="-25.3 (different random construction)"))
    rows.append(dict(check="corr_aggD_vs_absSPEI", value=float(np.corrcoef(aggD, abs_spei)[0, 1]), unit="r", expected="-0.987"))
    rows.append(dict(check="corr_aggD_vs_signedSPEI", value=float(np.corrcoef(aggD, dom_spei)[0, 1]), unit="r", expected="-0.08"))
    for s in ["CA", "FL", "TX", "IL"]:
        ms = state == s
        ss = np.array([np.sum(M[ms & finite[t]] * X[t, ms & finite[t]]) / np.sum(M[ms & finite[t]]) for t in range(T)])
        od = np.argsort(ss)[:53]
        rows.append(dict(check=f"climrel_own_driest_decile_instate_{s}_pct",
                         value=float((D[:, ms] @ (M[ms] / M[ms].sum()))[od].mean() * 100), unit="%", expected="-18.6..-22.0"))
    # case month IL 1988-07, in-state
    ti = months.index("1988-07"); ms = state == "IL"
    rows.append(dict(check="climrel_case_IL_1988-07_instate_pct",
                     value=float(D[ti, ms] @ (M[ms] / M[ms].sum()) * 100), unit="%", expected="-37.5"))
    rows.append(dict(check="climrel_case_IL_1988-07_domain_pct", value=float(aggD[ti] * 100), unit="%", expected="-2.0"))
    return pd.DataFrame(rows)

# partitions
def leaf_index(groups):
    """Leaf label for each cell and the number of leaves."""
    lab = np.empty(N, int)
    for k, gr in enumerate(groups):
        lab[gr] = k
    return lab, len(groups)

def block_partition(bdeg):
    if bdeg is None:
        return leaf_index([np.array([i]) for i in range(N)])
    bi = np.floor((lat - 25.0) / bdeg).astype(int); bj = np.floor((lon + 125.0) / bdeg).astype(int)
    keys = {}
    for i in range(N):
        keys.setdefault((bi[i], bj[i]), []).append(i)
    return leaf_index([np.array(v) for v in keys.values()])

def rep_access(lab, L, rep, a=ALPHA, b=BETA):
    """Access at the native cells under a representation, and its static version."""
    if rep == "field":
        xh = np.full_like(X, np.nan)
        for l in range(L):
            m = lab == l
            with np.errstate(invalid="ignore"):
                xh[:, m] = np.nanmean(np.where(finite[:, m], X[:, m], np.nan), axis=1)[:, None]
        w = np.where(finite, M[None, :] * g(np.where(finite, xh, 0.0), a, b), 0.0)
        return access_series(w), A_static.copy()
    # leaf-level sources (source / joint)
    S = np.zeros((T, L)); cx = np.zeros(L); cy = np.zeros(L)
    if rep == "source":
        w = np.where(finite, M[None, :] * g(np.where(finite, X, 0.0), a, b), 0.0)
        for l in range(L):
            m = lab == l; S[:, l] = w[:, m].sum(1)
            tot = w.sum(0)[m].sum()
            if tot > 0:
                cx[l] = (lon[m] * w.sum(0)[m]).sum() / tot; cy[l] = (lat[m] * w.sum(0)[m]).sum() / tot
            else:
                cx[l], cy[l] = lon[m].mean(), lat[m].mean()
        w0 = M.copy()
    elif rep == "joint":
        for l in range(L):
            m = lab == l
            with np.errstate(invalid="ignore"):
                xb = np.nanmean(np.where(finite[:, m], X[:, m], np.nan), axis=1)
            Ml = np.where(finite[:, m], M[m][None, :], 0.0).sum(1)
            S[:, l] = Ml * g(np.where(np.isfinite(xb), xb, 0.0), a, b)
            cx[l] = (lon[m] * M[m]).sum() / M[m].sum(); cy[l] = (lat[m] * M[m]).sum() / M[m].sum()
        w0 = np.array([M[lab == l].sum() for l in range(L)])
        cx0, cy0 = cx.copy(), cy.copy()
    else:
        raise ValueError(rep)
    d = hav(lat, lon, cy, cx); K = np.exp(-np.maximum(d, SELF) / D0) if False else np.exp(-d / D0)
    K = np.where(d < SELF, np.exp(-SELF / D0), K)  # distance floor of 25 km
    A = S @ K.T
    if rep == "source":
        # static access: aggregate M at the mass-weighted centroid
        S0 = np.array([M[lab == l].sum() for l in range(L)])
        cx0 = np.array([ (lon[lab == l] * M[lab == l]).sum() / max(M[lab == l].sum(), 1e-12) for l in range(L)])
        cy0 = np.array([ (lat[lab == l] * M[lab == l]).sum() / max(M[lab == l].sum(), 1e-12) for l in range(L)])
        d0m = hav(lat, lon, cy0, cx0); K0 = np.exp(-d0m / D0); K0 = np.where(d0m < SELF, np.exp(-SELF / D0), K0)
        A0 = S0 @ K0.T
    else:
        A0 = w0 @ K.T
    return A, A0

# E1: resolution ladder and coherence
def e1():
    rows, coh_rows = [], []
    for bdeg, lab_name in [(None, "native 0.5deg"), (1.0, "1deg"), (2.0, "2deg"), (4.0, "4deg")]:
        lab, L = block_partition(bdeg)
        for rep in ["field", "joint", "source"]:
            A, A0 = rep_access(lab, L, rep)
            rows.append(dict(experiment="E1_ladder", block_deg=bdeg if bdeg else 0.5, leaves=L, rep=rep,
                             alpha=ALPHA, **rep_metrics(A, A0)))
        # coherence diagnostics for field averaging at this block size
        if bdeg:
            for l in range(L):
                m = lab == l
                if m.sum() < 2: continue
                xf = np.where(finite[:, m], X[:, m], np.nan)
                with np.errstate(invalid="ignore"):
                    xbar = np.nanmean(xf, axis=1); absmean = np.nanmean(np.abs(xf), axis=1)
                    coh = np.where(absmean > 1e-9, np.abs(xbar) / np.abs(absmean), np.nan)
                    # mass-weighted source bias in g units: g(xbar) - mean_M g(x)
                    gg = g(np.where(finite[:, m], X[:, m], 0.0))
                    gm = (gg * np.where(finite[:, m], M[m][None, :], 0.0)).sum(1) / np.where(finite[:, m], M[m][None, :], 0.0).sum(1)
                    bias = g(np.where(np.isfinite(xbar), xbar, 0.0)) - gm
                    has_both = np.array([np.any((xf[t] < -1) & np.isfinite(xf[t])) and np.any((xf[t] > 1) & np.isfinite(xf[t])) for t in range(T)])
                coh_rows.append(pd.DataFrame(dict(block_deg=bdeg, coherence=coh, source_bias_g=bias,
                                                  xbar=xbar, boundary=has_both,
                                                  leaf_mass=float(M[m].sum()))))
    # alpha sensitivity, field averaging at 2 deg
    for a in [0.25, 1.0]:
        lab, L = block_partition(2.0)
        # field averaging with alpha=a
        xh = np.full_like(X, np.nan)
        for l in range(L):
            m = lab == l
            with np.errstate(invalid="ignore"):
                xh[:, m] = np.nanmean(np.where(finite[:, m], X[:, m], np.nan), axis=1)[:, None]
        w = np.where(finite, M[None, :] * g(np.where(finite, xh, 0.0), a, a), 0.0)
        A = access_series(w)
        wt = np.where(finite, M[None, :] * g(np.where(finite, X, 0.0), a, a), 0.0)
        At = access_series(wt)
        loss = A / A_static[None, :] - 1; losst = At / A_static[None, :] - 1
        rows.append(dict(experiment="E1_alpha_sens", block_deg=2.0, leaves=L, rep="field", alpha=a,
                         mean_loss_pct=float(np.sum(Mw[None, :] * loss.mean(0)) * 100),
                         bias_loss_pp=float(np.sum(Mw[None, :] * (loss - losst).mean(0)) * 100),
                         rmse_loss_pp=wrms((loss - losst) * 100, np.broadcast_to(Mw, (T, N))),
                         rmse_anom_pp=np.nan, bias_anom_pp=np.nan,
                         true_mean_loss_pct=float(np.sum(Mw[None, :] * losst.mean(0)) * 100)))
    coh = pd.concat(coh_rows, ignore_index=True)
    bins = pd.cut(coh["coherence"], [0, .25, .5, .75, .9, 1.0001], labels=["0-.25", ".25-.5", ".5-.75", ".75-.9", ".9-1"])
    summ = coh.assign(bin=bins).groupby("bin", observed=True).agg(
        n=("source_bias_g", "size"), mean_source_bias_g=("source_bias_g", "mean"),
        share_boundary=("boundary", "mean"), mean_coherence=("coherence", "mean")).reset_index()
    summ.to_csv(RES / "e1_coherence_bins.csv", index=False)
    coh.to_csv(RES / "e1_coherence_leafmonths.csv.gz", index=False)
    return pd.DataFrame(rows), summ

# E2: equal budget
def leaf_scores(kind, idx):
    xs = X[:, idx]; fs = finite[:, idx]; mm = M[idx]
    if kind == "mass":
        return float(mm.sum())
    if kind == "range_static":
        with np.errstate(invalid="ignore"):
            return float(np.nanmean(np.nanmax(np.where(fs, xs, np.nan), axis=1) - np.nanmin(np.where(fs, xs, np.nan), axis=1)))
    if kind == "l1g_static":
        with np.errstate(invalid="ignore"):
            xbar = np.nanmean(np.where(fs, xs, np.nan), axis=1)
        gg = g(np.where(fs, xs, 0.0))
        ind = (np.where(fs, mm[None, :], 0.0) * np.abs(g(np.where(np.isfinite(xbar), xbar, 0.0))[:, None] - gg)).sum(1)
        return float(np.mean(ind))
    if kind == "uniform":
        return float((lon[idx].max() - lon[idx].min() + 0.5) * (lat[idx].max() - lat[idx].min() + 0.5))
    raise ValueError(kind)

def build_partition(kind, budget):
    leaves = [np.arange(N)]
    unsplittable = set()
    while len(leaves) < budget:
        cand = [(leaf_scores(kind, idx), i) for i, idx in enumerate(leaves) if i not in unsplittable and len(idx) >= 2]
        if not cand: break
        _, i = max(cand); idx = leaves.pop(i)
        done = False
        for axis in [0, 1]:  # 0=lon, 1=lat
            pass
        spans = [(lon[idx].max() - lon[idx].min(), 0), (lat[idx].max() - lat[idx].min(), 1)]
        for _, axis in sorted(spans, reverse=True):
            v = (lon if axis == 0 else lat)[idx]; mid = (v.min() + v.max()) / 2
            a_, b_ = idx[v <= mid], idx[v > mid]
            if len(a_) and len(b_):
                leaves += [a_, b_]; done = True; break
        if not done:
            leaves.append(idx); unsplittable.add(len(leaves) - 1)
    return leaf_index(leaves)

def e2():
    rows = []
    for budget in [64, 128, 256]:
        for kind in ["uniform", "range_static", "mass", "l1g_static"]:
            lab, L = build_partition(kind, budget)
            for rep in ["field", "source"]:
                A, A0 = rep_access(lab, L, rep)
                rows.append(dict(experiment="E2_equal_budget", budget=budget, leaves=L, partition=kind, rep=rep,
                                 **rep_metrics(A, A0)))
    return pd.DataFrame(rows)

# E4: Barnes-Hut
def build_tree(xy, w, max_leaf=4, max_depth=14):
    nodes = []  # node: bounding box, total weight W, centroid cx, cy, children or cells
    def rec(idx, depth):
        x0, x1 = xy[idx, 0].min(), xy[idx, 0].max(); y0, y1 = xy[idx, 1].min(), xy[idx, 1].max()
        node = dict(x0=x0, x1=x1, y0=y0, y1=y1, W=float(w[idx].sum()),
                    cx=float((xy[idx, 0] * w[idx]).sum() / max(w[idx].sum(), 1e-300)),
                    cy=float((xy[idx, 1] * w[idx]).sum() / max(w[idx].sum(), 1e-300)),
                    size=float(max(x1 - x0, y1 - y0)), cells=None, children=None)
        nid = len(nodes); nodes.append(node)
        if len(idx) <= max_leaf or depth >= max_depth:
            nodes[nid]["cells"] = idx; return nid
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        kids = []
        for qx in [0, 1]:
            for qy in [0, 1]:
                sel = idx[((xy[idx, 0] > mx) == bool(qx)) & ((xy[idx, 1] > my) == bool(qy))]
                if len(sel): kids.append(rec(sel, depth + 1))
        if len(kids) <= 1:
            nodes[nid]["cells"] = idx; nodes[nid]["children"] = None
        else:
            nodes[nid]["children"] = kids
        return nid
    rec(np.arange(len(w)), 0)
    return nodes

def bh_eval(nodes, xy_s, w, xy_t, theta, d0=D0):
    A = np.zeros(len(xy_t)); evals = 0
    t_idx = np.arange(len(xy_t)); n_idx = np.zeros(len(xy_t), int)  # root = 0
    active_t, active_n = t_idx, n_idx
    while len(active_t):
        nd = [nodes[i] for i in active_n]
        size = np.array([d["size"] for d in nd]); cx = np.array([d["cx"] for d in nd]); cy = np.array([d["cy"] for d in nd])
        Wt = np.array([d["W"] for d in nd])
        dx = xy_t[active_t, 0] - cx; dy = xy_t[active_t, 1] - cy; dist = np.hypot(dx, dy)
        inside = np.array([d["x0"] - 1e-9 <= xy_t[t, 0] <= d["x1"] + 1e-9 and d["y0"] - 1e-9 <= xy_t[t, 1] <= d["y1"] + 1e-9
                           for t, d in zip(active_t, nd)])
        is_leaf = np.array([d["children"] is None for d in nd])
        accept = (~inside) & (size / np.maximum(dist, 1e-9) < theta) & (~is_leaf) & (Wt > 0)
        A[active_t[accept]] += np.exp(-np.maximum(dist[accept], SELF) / d0) * Wt[accept]
        evals += int(accept.sum())
        leafmask = is_leaf
        if leafmask.any():
            for t, d in zip(active_t[leafmask], [nd[i] for i in np.where(leafmask)[0]]):
                cc = d["cells"]; dd = np.hypot(xy_t[t, 0] - xy_s[cc, 0], xy_t[t, 1] - xy_s[cc, 1])
                A[t] += float(np.exp(-np.maximum(dd, SELF) / d0) @ w[cc]); evals += len(cc)
        expand = ~(accept | leafmask)
        if expand.any():
            nt, nn = [], []
            for t, d in zip(active_t[expand], [nd[i] for i in np.where(expand)[0]]):
                for k in d["children"]:
                    nt.append(t); nn.append(k)
            active_t, active_n = np.array(nt), np.array(nn, dtype=int) if nt else np.array([], dtype=int)
        else:
            active_t = np.array([], dtype=int)
    return A, evals

def e4():
    rows = []
    cosf = 111.32 * np.cos(np.radians(34.0))
    xy_s = np.column_stack([lon * cosf, lat * 110.9])
    xy_t = np.column_stack([tracts["lon"].values * cosf, tracts["lat"].values * 110.9])
    tpop = tracts["population"].values
    for mon in ["1998-06", "1988-07", "1994-06"]:
        t = months.index(mon)
        w = np.where(finite[t], M * g(np.where(finite[t], X[t], 0.0)), 0.0)
        # exact sum in planar coordinates
        Aex = np.zeros(len(xy_t))
        for s in range(0, len(xy_t), 2000):
            dd = np.hypot(xy_t[s:s+2000, 0][:, None] - xy_s[None, :, 0], xy_t[s:s+2000, 1][:, None] - xy_s[None, :, 1])
            Aex[s:s+2000] = np.exp(-np.maximum(dd, SELF) / D0) @ w
        exact_evals = len(xy_t) * N
        nodes = build_tree(xy_s, w)
        for theta in [0.15, 0.2, 0.3, 0.5, 0.8]:
            Ab, ev = bh_eval(nodes, xy_s, w, xy_t, theta)
            rel = np.abs(Ab - Aex) / np.maximum(Aex, 1e-9)
            rows.append(dict(experiment="E4_barnes_hut", month=mon, theta=theta, n_targets=len(xy_t), n_sources=N,
                             exact_kernel_evals=int(exact_evals), bh_kernel_evals=int(ev),
                             cost_ratio=float(ev / exact_evals),
                             mean_rel_err_pct=float(np.average(rel, weights=tpop) * 100),
                             max_rel_err_pct=float(rel.max() * 100),
                             mean_signed_rel_err_pct=float(np.average((Ab - Aex) / np.maximum(Aex, 1e-9), weights=tpop) * 100)))
    return pd.DataFrame(rows)

FIG_FILES = ["fig1_ladder.png", "fig1_ladder.pdf", "fig2_equal_budget.png", "fig2_equal_budget.pdf",
             "fig3_representation.png", "fig3_representation.pdf", "fig4_barnes_hut.png", "fig4_barnes_hut.pdf",
             "fig5_case_anomaly_maps.png", "fig5_case_anomaly_maps.pdf"]
CB = dict(field="#0072B2", joint="#E69F00", source="#009E73", uniform="#666666",
          range_static="#CC79A7", mass="#E69F00", l1g_static="#0072B2")

def _save(fig, name):
    fig.savefig(RES / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(RES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)

def make_figures(e1df, e2df, e4df):
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                         "legend.fontsize": 7, "figure.dpi": 150, "pdf.fonttype": 42})
    lad = e1df[e1df.experiment == "E1_ladder"]
    # Figure 1
    fig, ax = plt.subplots(2, 1, figsize=(4.5, 5.2), sharex=True)
    for rep, mk in [("field", "o"), ("joint", "s"), ("source", "^")]:
        d = lad[lad.rep == rep]
        ax[0].plot(d.block_deg, d.bias_loss_pp, marker=mk, color=CB[rep], label=rep)
        ax[1].plot(d.block_deg, d.rmse_anom_pp, marker=mk, color=CB[rep], label=rep)
    ax[0].axhline(0, color="#999999", lw=0.8); ax[0].set_xlabel("Block size (°)"); ax[0].set_ylabel("Mean-loss bias (pp)")
    ax[0].set_title("a)", loc="left", fontweight="bold"); ax[0].legend(frameon=False)
    ax[1].set_xlabel("Block size (°)"); ax[1].set_ylabel("Anomaly RMSE (pp)")
    ax[1].set_title("b)", loc="left", fontweight="bold")
    for a in ax: a.grid(alpha=0.25, lw=0.5)
    fig.tight_layout(); _save(fig, "fig1_ladder")
    # Figure 2
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for kind, mk, ls in [("uniform", "s", "--"), ("range_static", "x", ":"), ("mass", "^", "-."), ("l1g_static", "o", "-")]:
        d = e2df[(e2df.partition == kind) & (e2df.rep == "field")]
        lab = {"uniform": "uniform", "range_static": "range (static)", "mass": "mass only",
               "l1g_static": "response-error $I_\\ell$"}[kind]
        ax.plot(d.leaves, d.rmse_anom_pp, marker=mk, ls=ls, color=CB[kind], label=lab)
    ax.set_xlabel("Leaves (equal budget)"); ax.set_ylabel("Anomaly RMSE, field averaging (pp)")
    ax.legend(frameon=False); ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout(); _save(fig, "fig2_equal_budget")
    # Figure 3
    fig, ax = plt.subplots(2, 1, figsize=(4.5, 6.2))
    reps = lad[lad.block_deg > 0.5]
    x = np.arange(3); wbar = 0.38
    ax[0].bar(x - wbar/2, reps[reps.rep == "field"].rmse_anom_pp, wbar, color=CB["field"], label="field averaging")
    ax[0].bar(x + wbar/2, reps[reps.rep == "source"].rmse_anom_pp, wbar, color=CB["source"], label="source aggregation")
    ax[0].set_xticks(x, ["1°\n181 leaves", "2°\n63 leaves", "4°\n28 leaves"]); ax[0].set_ylabel("Anomaly RMSE (pp)")
    ax[0].set_title("a)", loc="left", fontweight="bold"); ax[0].legend(frameon=False)
    e2p = e2df.copy(); e2p["ratio"] = np.nan
    for (bud, kind), grp in e2p.groupby(["budget", "partition"]):
        f = grp[grp.rep == "field"].rmse_anom_pp.iloc[0]; s = grp[grp.rep == "source"].rmse_anom_pp.iloc[0]
        e2p.loc[grp.index, "ratio"] = s / f
    rat = e2p[e2p.rep == "field"].sort_values(["budget", "partition"])
    ax[1].barh(np.arange(len(rat)), rat.ratio, color="#56B4E9")
    disp = {"uniform": "uniform", "range_static": "range", "mass": "mass", "l1g_static": "response-error"}
    ax[1].set_yticks(np.arange(len(rat)), [f"{disp[p]}, {b}" for p, b in zip(rat.partition, rat.budget)], fontsize=6.5)
    ax[1].axvline(0.2, color="#CC79A7", ls="--", lw=1); ax[1].text(0.21, len(rat) - 0.4, "H3 criterion (0.20)", fontsize=7, color="#CC79A7")
    ax[1].set_xlabel("Source / field anomaly RMSE"); ax[1].set_title("b)", loc="left", fontweight="bold")
    for a in ax: a.grid(alpha=0.25, lw=0.5, axis="x" if a is ax[1] else "both")
    fig.tight_layout(); _save(fig, "fig3_representation")
    # Figure 4
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((0.0, 0.005), 0.2, 0.995, facecolor="#009E73", alpha=0.10, zorder=0))
    ax.text(0.012, 0.05, "H4 criterion region:\nerror < 1% and cost < 0.20", fontsize=6.5, color="#00674b")
    for mon, mk in [("1998-06", "o"), ("1988-07", "s"), ("1994-06", "^")]:
        d = e4df[e4df.month == mon]
        ax.plot(d.cost_ratio, d.mean_rel_err_pct, marker=mk, color="#0072B2", label=f"month {mon}")
        if mon == "1998-06":
            for _, r in d.iterrows():
                ax.text(r.cost_ratio + 0.008, r.mean_rel_err_pct * 1.18, f"θ={r.theta:g}", fontsize=6.5)
    ax.axhline(1.0, color="#CC79A7", ls="--", lw=1)
    ax.set_yscale("log"); ax.set_ylim(0.005, 30); ax.set_xlim(0, 0.45)
    ax.set_xlabel("Kernel evaluations / exact summation"); ax.set_ylabel("Mean |relative error| (%)")
    ax.legend(frameon=False); ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout(); _save(fig, "fig4_barnes_hut")
    # Figure 5: case-month maps in EPSG:5070
    import geopandas as gpd
    from shapely.geometry import box
    Abar = A_true.mean(0); D = (A_true / Abar[None, :] - 1.0) * 100
    polys = [box(lo - 0.25, la - 0.25, lo + 0.25, la + 0.25) for la, lo in zip(lat, lon)]
    cells_g = gpd.GeoDataFrame({"Dholder": np.zeros(N)}, geometry=polys, crs="EPSG:4326").to_crs("EPSG:5070")
    bnd = gpd.read_file(DATA / "state_boundaries.geojson").to_crs("EPSG:5070")
    cases = [("1988-07", "IL", "Illinois driest month"), ("2020-10", "CA", "California driest month"),
             ("1998-06", "TX", "Texas driest month"), ("1998-02", "FL", "Florida wettest month")]
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=-40, vcenter=0, vmax=40)
    fig, axes = plt.subplots(2, 2, figsize=(4.5, 4.4), gridspec_kw=dict(wspace=0.02, hspace=0.12))
    bx0, by0, bx1, by1 = bnd.total_bounds
    pad = 8e4
    sm = None
    for ax, (mon, st, label) in zip(axes.ravel(), cases):
        t = months.index(mon); cells_g["Dholder"] = D[t]
        sm = cells_g.plot(column="Dholder", cmap="RdBu_r", norm=norm, ax=ax, edgecolor="none")
        bnd.plot(ax=ax, facecolor="none", edgecolor="#8a8a8a", linewidth=0.5)
        ms = state == st
        instate = float(D[t, ms] @ (M[ms] / M[ms].sum())); agg = float(D[t] @ Mw)
        ax.set_title(f"{label}, {mon}\nin-state {instate:.1f}%, domain {agg:.1f}%", fontsize=6.5)
        ax.set_axis_off(); ax.set_xlim(bx0 - pad, bx1 + pad); ax.set_ylim(by0 - pad, by1 + pad)
    # scale bar and north arrow on the first panel (projected metres)
    ax0 = axes[0, 0]; x0, y0 = bx0 + 1e4, by0 + 1.2e5; L = 1e6
    from matplotlib.patches import Rectangle as _Rect
    ax0.add_patch(_Rect((x0, y0), L/2, 2.0e4, facecolor="white", edgecolor="black", lw=0.7))
    ax0.add_patch(_Rect((x0 + L/2, y0), L/2, 2.0e4, facecolor="black", edgecolor="black", lw=0.7))
    ax0.text(x0, y0 - 1.0e5, "0", fontsize=6, ha="center", va="top"); ax0.text(x0 + L, y0 - 1.0e5, "1000 km", fontsize=6, ha="center", va="top")
    ax0.annotate("", xy=(0.93, 0.95), xytext=(0.93, 0.75), xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="-|>", lw=1.0, color="black"))
    ax0.text(0.93, 0.96, "N", transform=ax0.transAxes, ha="center", va="bottom", fontsize=7, fontweight="bold")
    import matplotlib as mpl
    smap = mpl.cm.ScalarMappable(norm=norm, cmap="RdBu_r"); smap.set_array([])
    cb = fig.colorbar(smap, ax=axes, shrink=0.6, orientation="horizontal", pad=0.02, fraction=0.035, aspect=35)
    cb.set_label("Access deviation from climatology (%)", fontsize=7); cb.ax.tick_params(labelsize=6.5)
    _save(fig, "fig5_case_anomaly_maps")

def main():
    RES.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); log = []
    e0df = e0(); e0df.to_csv(RES / "e0_gate_replication.csv", index=False); log.append("E0 done")
    e1df, coh = e1(); e1df.to_csv(RES / "e1_ladder.csv", index=False); log.append("E1 done")
    e2df = e2(); e2df.to_csv(RES / "e2_equal_budget.csv", index=False); log.append("E2 done")
    e4df = e4(); e4df.to_csv(RES / "e4_barnes_hut.csv", index=False); log.append("E4 done")
    make_figures(e1df, e2df, e4df)
    manifest = dict(run="p3_scale_amr_real", wall_clock_seconds=time.time() - t0,
                    expected_outputs=["results/e0_gate_replication.csv", "results/e1_ladder.csv",
                                      "results/e1_coherence_bins.csv", "results/e1_coherence_leafmonths.csv.gz",
                                      "results/e2_equal_budget.csv", "results/e4_barnes_hut.csv"] +
                                     [f"results/{f}" for f in FIG_FILES],
                    months=[months[0], months[-1]], n_months=T, n_cells=N,
                    total_population=float(M.sum()), n_tracts=int(len(tracts)),
                    alpha=ALPHA, beta=BETA, d0_km=D0, log=log)
    (ROOT / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(e0df.to_string()); print(e1df.to_string()); print(e2df.to_string()); print(e4df.to_string()); print(coh.to_string())
    print("wall", time.time() - t0)

if __name__ == "__main__":
    main()
