"""
Re-analysis for UBJBAS-D-26-00004 (revision).

Produces every statistic the reviewers requested, from the item-level data:
  - KMO, Bartlett, EFA (principal axis, oblimin), explained variance, loadings   (R1 5.5, R3 #3)
  - Cronbach's alpha, composite reliability, AVE, Fornell-Larcker, HTMT          (R1 5.5, R3 #3)
  - CFA fit (chi2/df, CFI, TLI, RMSEA, SRMR) and Harman's single-factor test     (R1 8, R3 #3, #14)
  - Hierarchical regressions with controls: B, SE, beta, t, p, 95% CI,
    R2, adj. R2, delta R2, F(df), N                                              (R1 5.3, 5.6, R3 #8, #10)
  - Diagnostics: VIF, tolerance, Durbin-Watson, Breusch-Pagan, Shapiro-Wilk      (R1 5.6, R3 #10)
  - Wald test comparing the AI vs analytics vs automation coefficients           (R3 #9)
  - Bootstrapped mediation (5,000 percentile resamples): a, b, c, c', a*b,
    boot SE, 95% CI, proportion mediated, mediation type                        (R1 6.7, R3 #6)
  - LMS activity vs self-reported engagement correlation                         (R3 #11)
  - Robustness: models by academic level and for accounting students only        (R1 5.3, 9)
  - Figures 4-7 regenerated without titles embedded in the image                 (R1 p.13 #4)

Usage:  python reanalysis.py data.xlsx   (or .csv / .sav)
Requires: pandas numpy scipy statsmodels factor_analyzer semopy matplotlib openpyxl
          (pyreadstat for .sav)
Edit the CONFIG block if your column names differ.
"""
import sys, warnings
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------- CONFIG
CONSTRUCTS = {          # construct -> item columns
    "AI_EXP": ["AI1", "AI2", "AI3", "AI4", "AI5"],
    "AN_EXP": ["AN1", "AN2", "AN3", "AN4", "AN5"],
    "AU_EXP": ["AU1", "AU2", "AU3", "AU4", "AU5"],
    "ENG":    ["ENG1", "ENG2", "ENG3", "ENG4", "ENG5", "ENG6"],
    "CT":     ["CT1", "CT2", "CT3", "CT4"],
    "DS":     ["DS1", "DS2", "DS3", "DS4"],
    "SAT":    ["SAT1", "SAT2", "SAT3", "SAT4"],
}
PERF = "PERF"            # 0-100 from institutional records
LMS = "LMS"              # optional; set to None if absent
GENDER = "GEN"           # 1 = male, 2 = female
LEVEL = "ACL"            # 1 = UG, 2 = PG, 3 = PhD
FIELD = "FOS"            # 1 = accounting, 2 = commerce/finance
PRIOR = ["PRIOR_AI", "PRIOR_AN", "PRIOR_AU"]   # 1 = yes, 0/2 = no; set [] if absent
N_BOOT = 5000
SEED = 2026
# -------------------------------------------------------------------------


def load(path):
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".sav"):
        return pd.read_spss(path)
    return pd.read_excel(path)


def controls_frame(df):
    c = pd.DataFrame(index=df.index)
    c["PG"] = (df[LEVEL] == 2).astype(float)
    c["PhD"] = (df[LEVEL] == 3).astype(float)
    if FIELD in df:
        c["CommFin"] = (df[FIELD] == 2).astype(float)
    if GENDER in df:
        c["Female"] = (df[GENDER] == 2).astype(float)
    for p in PRIOR:
        if p in df:
            c[p] = (df[p] == 1).astype(float)
    return c.loc[:, c.std() > 0]


def zscore(s):
    return (s - s.mean()) / s.std(ddof=1)


# ---------------------------------------------------------- measurement
def measurement(df, out):
    from factor_analyzer import FactorAnalyzer
    from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
    import factor_analyzer.factor_analyzer as _fa_mod   # shim for newer scikit-learn
    _orig = _fa_mod.check_array
    def _check_array(*a, **k):
        if "force_all_finite" in k:
            v = k.pop("force_all_finite")
            try:
                return _orig(*a, ensure_all_finite=v, **k)
            except TypeError:
                return _orig(*a, force_all_finite=v, **k)
        return _orig(*a, **k)
    _fa_mod.check_array = _check_array
    items = [i for v in CONSTRUCTS.values() for i in v]
    X = df[items].astype(float)
    chi2, p = calculate_bartlett_sphericity(X)
    _, kmo = calculate_kmo(X)
    k = len(CONSTRUCTS)
    fa = FactorAnalyzer(n_factors=k, rotation="oblimin", method="principal")
    fa.fit(X)
    ev, _ = fa.get_eigenvalues()
    var = fa.get_factor_variance()
    load = pd.DataFrame(fa.loadings_, index=items, columns=[f"F{i+1}" for i in range(k)])
    nd = len(items) * (len(items) - 1) / 2
    out["efa_summary"] = pd.DataFrame({
        "KMO": [kmo], "Bartlett_chi2": [chi2], "Bartlett_df": [nd], "Bartlett_p": [p],
        "factors_eigen_gt1": [int((ev > 1).sum())],
        "cum_var_explained_%": [100 * var[2][-1]]})
    out["efa_loadings"] = load.round(3)

    # Harman single factor (unrotated, 1 factor)
    h = FactorAnalyzer(n_factors=1, rotation=None, method="principal").fit(X)
    out["harman"] = pd.DataFrame({"single_factor_var_%": [100 * h.get_factor_variance()[1][0]]})

    # CFA
    import semopy
    desc = "\n".join(f"{c} =~ " + " + ".join(v) for c, v in CONSTRUCTS.items())
    m = semopy.Model(desc)
    m.fit(X)
    st = semopy.calc_stats(m).T
    est = m.inspect(std_est=True)
    lam = est[(est["op"] == "~") & (est["rval"].isin(CONSTRUCTS.keys()))][["lval", "rval", "Est. Std"]]
    lam.columns = ["item", "construct", "std_loading"]
    out["cfa_loadings"] = lam.reset_index(drop=True)
    obs = m.vars["observed"]
    sig = pd.DataFrame(m.calc_sigma()[0], index=obs, columns=obs).loc[items, items]
    S_ = X.cov(ddof=0)
    resid = S_ - sig
    sd = np.sqrt(np.diag(S_))
    srmr = np.sqrt(np.mean(np.square((resid.values / np.outer(sd, sd))[np.tril_indices(len(items))])))
    out["cfa_fit"] = pd.DataFrame({
        "chi2": st.loc["chi2"].values, "df": st.loc["DoF"].values,
        "chi2/df": st.loc["chi2"].values / st.loc["DoF"].values,
        "CFI": st.loc["CFI"].values, "TLI": st.loc["TLI"].values,
        "RMSEA": st.loc["RMSEA"].values, "SRMR": [srmr]})

    rows = []
    scores = {}
    for c, its in CONSTRUCTS.items():
        sub = X[its]
        kk = len(its)
        alpha = kk / (kk - 1) * (1 - sub.var(ddof=1).sum() / sub.sum(axis=1).var(ddof=1))
        l = lam[lam.construct == c]["std_loading"].astype(float).values
        cr = l.sum() ** 2 / (l.sum() ** 2 + (1 - l ** 2).sum())
        ave = (l ** 2).mean()
        rows.append({"construct": c, "items": kk, "alpha": alpha, "CR": cr, "AVE": ave,
                     "sqrt_AVE": np.sqrt(ave), "loading_min": l.min(), "loading_max": l.max()})
        scores[c] = sub.mean(axis=1)
    rel = pd.DataFrame(rows).set_index("construct")
    out["reliability_validity"] = rel.round(3)

    S = pd.DataFrame(scores)
    corr = S.corr()
    fl = corr.copy()
    for c in fl.index:
        fl.loc[c, c] = rel.loc[c, "sqrt_AVE"]
    out["fornell_larcker"] = fl.round(3)

    ic = X.corr().abs()
    names = list(CONSTRUCTS)
    htmt = pd.DataFrame(np.nan, index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[:i]:
            het = ic.loc[CONSTRUCTS[a], CONSTRUCTS[b]].values.mean()
            def mono(cn):
                m_ = ic.loc[CONSTRUCTS[cn], CONSTRUCTS[cn]].values
                return m_[np.triu_indices(len(CONSTRUCTS[cn]), 1)].mean()
            htmt.loc[a, b] = het / np.sqrt(mono(a) * mono(b))
    out["htmt"] = htmt.round(3)
    return S


# ---------------------------------------------------------- regression
def ols(y, X):
    return sm.OLS(y, sm.add_constant(X, has_constant="add")).fit()


def hier(df, ctrl, dv, ivs, name, out_rows, diag_rows, std=True):
    y = df[dv]
    m1 = ols(y, ctrl)
    X2 = pd.concat([ctrl, df[ivs]], axis=1)
    m2 = ols(y, X2)
    sy = y.std(ddof=1)
    ci = m2.conf_int()
    for v in ivs:
        out_rows.append({
            "model": name, "DV": dv, "predictor": v, "B": m2.params[v], "SE": m2.bse[v],
            "beta": m2.params[v] * df[v].std(ddof=1) / sy, "t": m2.tvalues[v], "p": m2.pvalues[v],
            "CI_low": ci.loc[v, 0], "CI_high": ci.loc[v, 1]})
    for v in ctrl.columns:
        out_rows.append({"model": name, "DV": dv, "predictor": f"(control) {v}", "B": m2.params[v],
                         "SE": m2.bse[v], "beta": np.nan, "t": m2.tvalues[v], "p": m2.pvalues[v],
                         "CI_low": ci.loc[v, 0], "CI_high": ci.loc[v, 1]})
    Xc = sm.add_constant(X2, has_constant="add")
    vifs = [variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])]
    bp = het_breuschpagan(m2.resid, Xc)
    sw = stats.shapiro(m2.resid)
    f_change = ((m2.rsquared - m1.rsquared) / len(ivs)) / ((1 - m2.rsquared) / m2.df_resid)
    diag_rows.append({
        "model": name, "DV": dv, "N": int(m2.nobs), "R2_step1": m1.rsquared, "R2": m2.rsquared,
        "adj_R2": m2.rsquared_adj, "delta_R2": m2.rsquared - m1.rsquared,
        "F_change": f_change, "F_change_p": 1 - stats.f.cdf(f_change, len(ivs), m2.df_resid),
        "F": m2.fvalue, "df1": int(m2.df_model), "df2": int(m2.df_resid), "F_p": m2.f_pvalue,
        "max_VIF": max(vifs), "min_tolerance": 1 / max(vifs), "Durbin_Watson": durbin_watson(m2.resid),
        "BP_LM": bp[0], "BP_p": bp[1], "Shapiro_W": sw.statistic, "Shapiro_p": sw.pvalue})
    return m2


def wald_compare(df, ctrl, dv, ivs):
    """Compare standardized coefficients of exposures within the PERF model."""
    Z = df.copy()
    for v in ivs + [dv]:
        Z[v] = zscore(Z[v])
    m = ols(Z[dv], pd.concat([ctrl, Z[ivs]], axis=1))
    rows = []
    for i, a in enumerate(ivs):
        for b in ivs[i + 1:]:
            r = m.f_test(f"{a} - {b} = 0")
            rows.append({"DV": dv, "comparison": f"{a} vs {b}", "beta_diff": m.params[a] - m.params[b],
                         "F": float(np.squeeze(r.fvalue)), "df1": int(r.df_num), "df2": int(r.df_denom),
                         "p": float(r.pvalue)})
    return rows


# ---------------------------------------------------------- mediation
def mediation(df, ctrl, x, m, y, rng):
    Z = df[[x, m, y]].apply(zscore)
    def paths(idx):
        C = ctrl.iloc[idx].reset_index(drop=True)
        d = Z.iloc[idx].reset_index(drop=True)
        a = ols(d[m], pd.concat([C, d[[x]]], axis=1)).params[x]
        mb = ols(d[y], pd.concat([C, d[[x, m]]], axis=1))
        c = ols(d[y], pd.concat([C, d[[x]]], axis=1)).params[x]
        return a, mb.params[m], c, mb.params[x]
    n = len(Z)
    idx = np.arange(n)
    a, b, c, cp = paths(idx)
    C = ctrl.reset_index(drop=True)
    d = Z.reset_index(drop=True)
    ma = ols(d[m], pd.concat([C, d[[x]]], axis=1))
    mb = ols(d[y], pd.concat([C, d[[x, m]]], axis=1))
    mc = ols(d[y], pd.concat([C, d[[x]]], axis=1))
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        s = rng.integers(0, n, n)
        aa, bb, _, _ = paths(s)
        boots[i] = aa * bb
    lo, hi = np.percentile(boots, [2.5, 97.5])
    sig_ind = not (lo <= 0 <= hi)
    mtype = ("partial" if mb.pvalues[x] < .05 else "full") if sig_ind else "none"
    return {"X": x, "M": m, "Y": y,
            "a": a, "a_p": ma.pvalues[x], "b": b, "b_p": mb.pvalues[m],
            "c_total": c, "c_p": mc.pvalues[x], "c_prime_direct": cp, "c_prime_p": mb.pvalues[x],
            "indirect_ab": a * b, "boot_SE": boots.std(ddof=1), "CI95_low": lo, "CI95_high": hi,
            "proportion_mediated": a * b / c if c != 0 else np.nan, "mediation_type": mtype}


# ---------------------------------------------------------- figures
def figures(S, df, reg, med, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def bar(labels, vals, ylabel, fname):
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ax.bar(labels, vals, color="#1f77b4")
        ax.set_ylabel(ylabel)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout(); fig.savefig(f"{outdir}/{fname}", dpi=300); plt.close(fig)
    means = [S[c].mean() for c in CONSTRUCTS]
    bar(list(CONSTRUCTS), means, "Mean score (1-5)", "Figure4_means.png")
    r1 = reg[(reg.model == "M1") & (~reg.predictor.str.startswith("("))]
    bar(r1.predictor.tolist(), r1.beta.tolist(), "Standardized coefficient (β)", "Figure5_PERF_betas.png")
    labs = [f"{r.X}→{r.Y}" for r in med.itertuples()]
    bar(labs, med.c_total.tolist(), "Total effect (β)", "Figure6_direct_effects.png")
    bar(labs, med.c_prime_direct.tolist(), "Direct effect with ENG (β)", "Figure7_effects_with_mediator.png")


# ---------------------------------------------------------- main
def run(df, label, out, full=True):
    rng = np.random.default_rng(SEED)
    if full:
        S = measurement(df, out)
    else:
        S = pd.DataFrame({c: df[v].mean(axis=1) for c, v in CONSTRUCTS.items()})
    d = pd.concat([S, df[[PERF]]], axis=1)
    ctrl = controls_frame(df)
    if not full:          # robustness subsets: drop constant controls
        ctrl = ctrl.loc[:, ctrl.std() > 0]
    reg_rows, diag_rows = [], []
    hier(d, ctrl, PERF, ["AI_EXP", "AN_EXP", "AU_EXP"], "M1", reg_rows, diag_rows)   # H1
    hier(d, ctrl, "CT", ["AN_EXP"], "M2", reg_rows, diag_rows)                        # H2
    hier(d, ctrl, "DS", ["AU_EXP"], "M3", reg_rows, diag_rows)                        # H3
    hier(d, ctrl, "SAT", ["AI_EXP", "AN_EXP", "AU_EXP"], "M4", reg_rows, diag_rows)   # H7
    hier(d, ctrl, PERF, ["ENG"], "M5", reg_rows, diag_rows)                           # H8
    hier(d, ctrl, "SAT", ["ENG"], "M6", reg_rows, diag_rows)                          # H9
    reg = pd.DataFrame(reg_rows)
    out[f"{label}regression"] = reg.round(4)
    out[f"{label}diagnostics"] = pd.DataFrame(diag_rows).round(4)
    if full:
        out["coef_comparison"] = pd.DataFrame(wald_compare(d, ctrl, PERF, ["AI_EXP", "AN_EXP", "AU_EXP"])).round(4)
        med = pd.DataFrame([mediation(d, ctrl, "AI_EXP", "ENG", PERF, rng),        # H4
                            mediation(d, ctrl, "AN_EXP", "ENG", "CT", rng),        # H5
                            mediation(d, ctrl, "AU_EXP", "ENG", "DS", rng)])       # H6
        out["mediation"] = med.round(4)
        desc = d.describe().T[["min", "max", "mean", "std"]]
        out["descriptives"] = desc.round(2)
        out["correlations"] = d.corr().round(3)
        if LMS and LMS in df:
            r, p = stats.pearsonr(df[LMS], S["ENG"])
            out["lms_vs_eng"] = pd.DataFrame({"r": [r], "p": [p], "N": [len(df)]})
        return reg, med, S
    return reg, None, S


def main(path):
    df = load(path)
    need = [i for v in CONSTRUCTS.values() for i in v] + [PERF, LEVEL]
    miss = [c for c in need if c not in df]
    if miss:
        sys.exit(f"Missing columns: {miss}. Edit CONFIG at the top of this script.")
    df = df.dropna(subset=need).reset_index(drop=True)
    out = {}
    reg, med, S = run(df, "", out)
    # robustness: by academic level, and accounting students only
    for lvl, nm in [(1, "UG"), (2, "PG"), (3, "PhD")]:
        sub = df[df[LEVEL] == lvl].reset_index(drop=True)
        if len(sub) >= 25:
            run(sub, f"rob_{nm}_", out, full=False)
    if FIELD in df:
        run(df[df[FIELD] == 1].reset_index(drop=True), "rob_ACC_", out, full=False)
    with pd.ExcelWriter("reanalysis_results.xlsx") as xw:
        for k, v in out.items():
            v.to_excel(xw, sheet_name=k[:31])
    figures(S, df, reg, med, ".")
    print("Wrote reanalysis_results.xlsx and Figure4-7 PNGs. Sheets:", ", ".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
