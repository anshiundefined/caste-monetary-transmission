"""
Heterogeneous Monetary Policy Transmission in Segmented Credit Markets: The Role of Caste in India
Replication + extensions.   Author: Meetanshi Gaba (Symbiosis School of Economics)

Design: continuous-treatment difference-in-differences. Every state faces the same RBI repo rate;
exposure differs by SC/ST population share (Census 2011). With state and year fixed effects,
identification comes from Repo_t x SCST_s: do high-SC/ST states' credit growth respond differently?

Part A reproduces the paper (M1-M3, lagged, random effects).
Part B adds inference and robustness a referee would ask for with 34 clusters:
  wild cluster bootstrap, randomisation inference, leave-one-state-out,
  non-linear dose-response by SC/ST quartile, and tightening-vs-easing asymmetry.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

from utils import ROOT, DATA, HIGHLIGHT, MUTED, PALETTE, md_table, savefig, setup, write_results

SRC = "RBI (credit, repo rate, NSDP, CPI via Economic Survey / DBIE); Census 2011 (SC/ST shares). Panel: 34 states/UTs, 2011–2025"
RNG = np.random.default_rng(2025)
B_BOOT, B_RI = 1999, 2000


# ---------------------------------------------------------------------------------- data
def ccol(y: int) -> str:
    return f"Credit {y}" if y < 2020 else f"Credit_{y}"


def load() -> pd.DataFrame:
    src = DATA / "data_clean.csv" if (DATA / "data_clean.csv").exists() else ROOT / "data_clean.csv"
    w = pd.read_csv(src).rename(columns={"SC/ST Share (SC+ST shares/Population * 100)": "scst"})
    rows = []
    for _, r in w.iterrows():
        for y in range(2011, 2026):
            rows.append({
                "state": r["STATE"], "year": y, "scst": r["scst"],
                "credit": r[ccol(y)],
                "g": 100 * (r[ccol(y)] / r[ccol(y - 1)] - 1),                       # credit growth, %
                "repo": r[f"Repo_{y}"],                                              # FY time-weighted average
                "repo_l": r[f"Repo_{y - 1}"] if y > 2011 else 5.887,                 # FY10-11 TWA
                "cpi": r.get(f"CPI {y - 1}-{str(y)[2:]}", np.nan),
                "nsdp": r.get(f"NSDP {y - 1}-{str(y)[2:]}", np.nan),
            })
    p = pd.DataFrame(rows)
    p["lnnsdp"] = np.log(p["nsdp"])
    p["repo_x_scst"] = p["repo"] * p["scst"]
    p.to_csv(DATA / "panel_long.csv", index=False)
    return p


def cl(formula: str, d: pd.DataFrame):
    return smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["state"])[0]})


# ------------------------------------------------------------------ wild cluster bootstrap (WCR, Webb weights)
def fwl(d: pd.DataFrame, y: str, x: str, controls: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partial out fixed effects and controls from y and x (Frisch-Waugh-Lovell)."""
    my = smf.ols(f"{y} ~ {controls}", data=d).fit().resid.values
    mx = smf.ols(f"{x} ~ {controls}", data=d).fit().resid.values
    return my, mx, pd.factorize(d["state"])[0]


def cluster_t(my, mx, g) -> tuple[float, float]:
    b = (mx @ my) / (mx @ mx)
    e = my - b * mx
    G = g.max() + 1
    s = np.bincount(g, weights=mx * e, minlength=G)
    n, k = len(my), 1
    adj = G / (G - 1) * (n - 1) / (n - k)
    se = np.sqrt(adj * (s @ s)) / (mx @ mx)
    return b, b / se


def wild_bootstrap(d: pd.DataFrame, x: str, controls: str) -> dict:
    d = d.dropna(subset=["g", x] + [c for c in ["lnnsdp", "cpi"] if c in controls]).copy()
    my, mx, g = fwl(d, "g", x, controls)
    b, t = cluster_t(my, mx, g)
    webb = np.array([-np.sqrt(1.5), -1, -np.sqrt(0.5), np.sqrt(0.5), 1, np.sqrt(1.5)])
    G = g.max() + 1
    ts, bs = np.empty(B_BOOT), np.empty(B_BOOT)
    for i in range(B_BOOT):
        v = RNG.choice(webb, G)[g]
        ys = my * v                                    # restricted (H0: beta = 0) residuals, flipped by cluster
        bs[i], ts[i] = cluster_t(ys, mx, g)
    p = float(np.mean(np.abs(ts) >= abs(t)))
    # bootstrap-t confidence interval (unrestricted residuals) for the wasted multiplier
    e = my - b * mx
    bb = np.empty(B_BOOT)
    for i in range(B_BOOT):
        v = RNG.choice(webb, G)[g]
        bb[i] = (mx @ (b * mx + e * v)) / (mx @ mx)
    return {"b": b, "t": t, "p_wild": p, "ci": (np.percentile(bb, 2.5), np.percentile(bb, 97.5)), "n": len(my)}


# ------------------------------------------------------------------ randomisation inference (balanced M1)
def ri(p: pd.DataFrame) -> dict:
    """Permute SC/ST shares across states; closed-form two-way-FE estimator on the balanced panel."""
    Y = p.pivot(index="state", columns="year", values="g")
    R = p.groupby("year")["repo"].first().loc[Y.columns].values
    s = p.groupby("state")["scst"].first().loc[Y.index].values
    Yd = Y.values - Y.values.mean(1, keepdims=True) - Y.values.mean(0, keepdims=True) + Y.values.mean()
    rd = R - R.mean()

    def est(shares):
        sd = shares - shares.mean()
        X = np.outer(sd, rd)
        return (X * Yd).sum() / (X * X).sum()

    b0 = est(s)
    draws = np.array([est(RNG.permutation(s)) for _ in range(B_RI)])
    return {"b": b0, "draws": draws, "p": float(np.mean(np.abs(draws) >= abs(b0)))}


# ---------------------------------------------------------------------------------- main
def main() -> None:
    setup()
    p = load()
    bal = p.dropna(subset=["g", "repo"])
    md = []
    FE = "C(state) + C(year)"

    # ---------------- Part A: reproduce the paper
    specs = [
        ("M1 Baseline", f"g ~ repo_x_scst + {FE}", ["g"], "repo_x_scst"),
        ("M2 + ln NSDP", f"g ~ repo_x_scst + lnnsdp + {FE}", ["lnnsdp"], "repo_x_scst"),
        ("M2' + ln NSDP + CPI", f"g ~ repo_x_scst + lnnsdp + cpi + {FE}", ["lnnsdp", "cpi"], "repo_x_scst"),
        ("M3 No year FE", "g ~ repo + repo_x_scst + lnnsdp + cpi + C(state)", ["lnnsdp", "cpi"], "repo_x_scst"),
        ("Lagged repo", f"g ~ repo_l:scst + {FE}", ["repo_l"], "repo_l:scst"),
        ("Lagged repo + ln NSDP + CPI", f"g ~ repo_l:scst + lnnsdp + cpi + {FE}", ["repo_l", "lnnsdp", "cpi"], "repo_l:scst"),
    ]
    rowsA, fits = [], {}
    for name, f, need, term in specs:
        d = p.dropna(subset=["g"] + need)
        m = cl(f, d)
        fits[name] = (m, term)
        ci = m.conf_int().loc[term]
        rowsA.append({"Model": name, "Repo × SC/ST": m.params[term], "SE (cluster)": m.bse[term],
                      "p": m.pvalues[term], "95% CI low": ci[0], "95% CI high": ci[1], "N": int(m.nobs)})
    A = pd.DataFrame(rowsA)
    m3 = fits["M3 No year FE"][0]

    # random effects (paper Table 2): sign reversal is the point
    try:
        from linearmodels.panel import RandomEffects
        d = p.dropna(subset=["g", "lnnsdp", "cpi"]).set_index(["state", "year"])
        X = d[["repo", "repo_x_scst", "lnnsdp", "cpi"]].assign(const=1.0)
        re = RandomEffects(d["g"], X).fit(cov_type="clustered", cluster_entity=True)
        re_row = f"Random effects (for comparison): Repo × SC/ST = **{re.params['repo_x_scst']:+.4f}** (p = {re.pvalues['repo_x_scst']:.3f}). " \
                 "RE assumes state effects are uncorrelated with SC/ST share; the sign flip versus FE is evidence that assumption fails."
    except Exception as e:  # pragma: no cover
        re_row = f"Random effects not estimated ({e})."

    # ---------------- Part B: inference with 34 clusters
    wb = wild_bootstrap(bal, "repo_x_scst", FE)
    wb2 = wild_bootstrap(p, "repo_x_scst", f"lnnsdp + {FE}")
    r = ri(bal)
    mean_s = p.groupby("state")["scst"].first().mean()
    wasted = wb["b"] * mean_s
    wasted_ci = (wb["ci"][0] * mean_s, wb["ci"][1] * mean_s)

    # leave-one-state-out
    loo = []
    for s in bal["state"].unique():
        d = bal[bal["state"] != s]
        m = cl(f"g ~ repo_x_scst + {FE}", d)
        loo.append({"dropped": s, "b": m.params["repo_x_scst"], "lo": m.conf_int().loc["repo_x_scst", 0],
                    "hi": m.conf_int().loc["repo_x_scst", 1]})
    loo = pd.DataFrame(loo).sort_values("b")

    # non-linear dose-response: repo x quartile of SC/ST share (Q1 = reference)
    shares = p.groupby("state")["scst"].first()
    q = pd.qcut(shares, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    p["q"] = p["state"].map(q).astype(str)
    for k in ["Q2", "Q3", "Q4"]:
        p[f"repo_{k}"] = p["repo"] * (p["q"] == k)
    mq = cl(f"g ~ repo_Q2 + repo_Q3 + repo_Q4 + {FE}", p.dropna(subset=["g"]))
    qrange = shares.groupby(q, observed=True).agg(["min", "max"])

    # tightening vs easing (asymmetry)
    yr = p.groupby("year")["repo"].first()
    tight = (yr.diff() > 0).rename("tight")
    tight.loc[yr.index[0]] = True                      # FY11-12 was a tightening year (repo rose through 2011)
    p["tight"] = p["year"].map(tight).astype(float)
    p["rxs_tight"] = p["repo_x_scst"] * p["tight"]
    p["rxs_ease"] = p["repo_x_scst"] * (1 - p["tight"])
    ma = cl(f"g ~ rxs_tight + rxs_ease + {FE}", p.dropna(subset=["g"]))
    diff = ma.t_test("rxs_tight - rxs_ease")

    # ---------------- figures
    # 1. descriptive DiD picture: high vs low SC/ST states
    med = shares.median()
    p["group"] = np.where(p["scst"] > med, "High SC/ST share", "Low SC/ST share")
    gm = p.groupby(["year", "group"])["g"].median().unstack()
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(gm.index, gm["High SC/ST share"], color=HIGHLIGHT, lw=2.4, marker="o", ms=4, label="High SC/ST share states (median)")
    ax1.plot(gm.index, gm["Low SC/ST share"], color=PALETTE[0], lw=2.4, marker="o", ms=4, label="Low SC/ST share states (median)")
    ax1.set_ylabel("Credit growth (%)")
    ax2 = ax1.twinx()
    ax2.step(yr.index, yr.values, where="mid", color=MUTED, lw=2, label="Repo rate (rhs)")
    ax2.set_ylabel("Repo rate, FY time-weighted (%)", color="#777"); ax2.grid(False)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper right")
    ax1.set_title("Same policy rate, different credit responses")
    f1 = savefig(fig, "01_descriptive", SRC)

    # 2. coefficient stability incl. bootstrap / RI
    fig, ax = plt.subplots(figsize=(10, 4.8))
    y = np.arange(len(A))[::-1]
    ax.errorbar(A["Repo × SC/ST"], y, xerr=[A["Repo × SC/ST"] - A["95% CI low"], A["95% CI high"] - A["Repo × SC/ST"]],
                fmt="o", color=PALETTE[0], capsize=4, ms=7, label="Cluster-robust 95% CI")
    ax.errorbar([wb["b"]], [y[0] + 0.25], xerr=[[wb["b"] - wb["ci"][0]], [wb["ci"][1] - wb["b"]]], fmt="s", color=HIGHLIGHT,
                capsize=4, ms=6, label="M1, wild cluster bootstrap 95% CI")
    ax.axvline(0, color="#333", lw=1)
    ax.set_yticks(y, A["Model"])
    ax.set_xlabel("Repo × SC/ST share coefficient (pp of credit growth per pp repo per pp SC/ST share)")
    ax.set_title("Coefficient stability across specifications")
    ax.legend(fontsize=8.5, loc="lower left")
    f2 = savefig(fig, "02_coefficients", SRC)

    # 3. randomisation inference
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.hist(r["draws"], bins=60, color=MUTED, edgecolor="white")
    ax.axvline(r["b"], color=HIGHLIGHT, lw=2.5, label=f"Actual estimate {r['b']:.4f}")
    ax.set_xlabel("M1 coefficient when SC/ST shares are randomly reassigned across states")
    ax.set_title(f"Randomisation inference ({B_RI:,} permutations): p = {r['p']:.3f}")
    ax.legend()
    f3 = savefig(fig, "03_randomisation_inference", SRC)

    # 4. leave-one-out
    fig, ax = plt.subplots(figsize=(9, 9))
    yy = np.arange(len(loo))
    ax.errorbar(loo["b"], yy, xerr=[loo["b"] - loo["lo"], loo["hi"] - loo["b"]], fmt="o", color=PALETTE[0], ms=4, capsize=2)
    ax.axvline(wb["b"], color=HIGHLIGHT, ls="--", lw=1.5, label="Full sample")
    ax.axvline(0, color="#333", lw=1)
    ax.set_yticks(yy, loo["dropped"], fontsize=8)
    ax.set_xlabel("M1 coefficient with one state dropped")
    ax.set_title("Leave-one-state-out")
    ax.legend()
    f4 = savefig(fig, "04_leave_one_out", SRC)

    # 5. dose-response + asymmetry
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    qs = ["Q2", "Q3", "Q4"]
    b = [0] + [mq.params[f"repo_{k}"] for k in qs]
    lo = [0] + [mq.conf_int().loc[f"repo_{k}", 0] for k in qs]
    hi = [0] + [mq.conf_int().loc[f"repo_{k}", 1] for k in qs]
    labels = [f"{k}\n{qrange.loc[k, 'min']:.0f}–{qrange.loc[k, 'max']:.0f}%" for k in ["Q1"] + qs]
    ax.errorbar(range(4), b, yerr=[np.array(b) - np.array(lo), np.array(hi) - np.array(b)], fmt="o-", color=PALETTE[0], capsize=5, ms=8)
    ax.axhline(0, color="#333", lw=1)
    ax.set_xticks(range(4), labels)
    ax.set_ylabel("Repo effect on credit growth vs Q1 (pp per pp repo)")
    ax.set_title("Dose-response by SC/ST share quartile")
    ax = axes[1]
    vals = [ma.params["rxs_tight"], ma.params["rxs_ease"]]
    cis = [ma.conf_int().loc["rxs_tight"], ma.conf_int().loc["rxs_ease"]]
    ax.bar([0, 1], vals, color=[HIGHLIGHT, PALETTE[0]], width=0.5)
    ax.errorbar([0, 1], vals, yerr=[[v - c[0] for v, c in zip(vals, cis)], [c[1] - v for v, c in zip(vals, cis)]], fmt="none", ecolor="#333", capsize=6)
    ax.axhline(0, color="#333", lw=1)
    ax.set_xticks([0, 1], ["Tightening years", "Easing / hold years"])
    ax.set_title(f"Asymmetry (difference p = {float(diff.pvalue):.2f})")
    ax.set_ylabel("Repo × SC/ST coefficient")
    f5 = savefig(fig, "05_nonlinearity_asymmetry", SRC)

    # ---------------- write-up
    m1 = A.iloc[0]
    md.append("### Headline numbers\n")
    md.append(f"- **Baseline (M1):** Repo × SC/ST = **{m1['Repo × SC/ST']:.4f}** (cluster SE {m1['SE (cluster)']:.4f}, p = {m1['p']:.3f}; N = {int(m1['N'])}).")
    md.append(f"- **Wasted multiplier:** at the mean SC/ST share ({mean_s:.1f}%), a 1 pp higher repo rate is associated with credit growth "
              f"**{abs(wasted):.2f} pp** lower than in a state with no SC/ST population (wild-bootstrap 95% CI {wasted_ci[0]:.2f} to {wasted_ci[1]:.2f}).")
    md.append(f"- **Wild cluster bootstrap** (Webb weights, {B_BOOT:,} draws, 34 clusters): p = **{wb['p_wild']:.3f}** for M1; "
              f"**{wb2['p_wild']:.3f}** with ln NSDP (M2).")
    md.append(f"- **Randomisation inference:** only **{100 * r['p']:.1f}%** of {B_RI:,} random reassignments of SC/ST shares produce an estimate as large as the real one.")
    md.append(f"- **Leave-one-state-out:** estimates range from **{loo['b'].min():.4f}** to **{loo['b'].max():.4f}**; "
              f"{int((loo['hi'] < 0).sum())} of {len(loo)} stay significant at 5%.")
    md.append(f"- **Dose-response:** relative to the lowest-share quartile, the repo effect is "
              + ", ".join(f"{k} {mq.params[f'repo_{k}']:+.2f} (p = {mq.pvalues[f'repo_{k}']:.2f})" for k in qs) + ".")
    md.append(f"- **Asymmetry:** tightening years {ma.params['rxs_tight']:.4f} vs easing/hold years {ma.params['rxs_ease']:.4f} (difference p = {float(diff.pvalue):.2f}).")
    md.append(f"- {re_row}\n")
    md.append("### Part A — replication of the paper\n")
    md.append(md_table(A, "{:.4f}"))
    md.append(f"\nM3 also identifies the level effect of the repo rate: **{m3.params['repo']:+.3f}** (p = {m3.pvalues['repo']:.3f}).\n")
    md.append("### Figures\n")
    for f, cap in [(f1, "High vs low SC/ST share states"), (f2, "Coefficient stability"), (f3, "Randomisation inference"),
                   (f4, "Leave-one-state-out"), (f5, "Dose-response and tightening/easing asymmetry")]:
        md.append(f"**{cap}**\n\n![{cap}]({f})\n")
    write_results("\n".join(md))
    print("done")


if __name__ == "__main__":
    main()
