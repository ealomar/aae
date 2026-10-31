"""
Parameter recovery and sensitivity analysis of the evaluation protocol (UBJBAS-D-26-00004, R2).

For each scenario the generator in simulate_dataset.py is re-run R times at sample size n, the
protocol's models (same controls as reanalysis.py) are fitted, and for every hypothesis test we record:
  true    = population value of the standardized estimand under the scenario
            (the same model fitted to one very large simulated sample, N_POP records)
  mean    = mean standardized estimate over R replications
  bias    = mean - true
  emp_sd  = standard deviation of the estimate over replications
  cover   = proportion of nominal 95% CIs that contain the true value (direct coefficients only)
  reject  = proportion of replications with p < .05 (indirect effects: joint significance of a and b)
Under the null scenario 'reject' is the Type I error rate; elsewhere it is power.

Scenarios
  calibrated      path values in simulate_dataset.PARAMS (the values used for the released dataset)
  null            every structural path = 0
  small           every specified path = 0.10
  medium          every specified path = 0.30
  low_reliability calibrated paths, target alpha = 0.70 for every construct
  corr_exposure   calibrated paths, correlation among the three exposures = 0.40
  confounded      calibrated paths + an unmeasured variable U (loading 0.30) on all exposures and outcomes
  confounded_null null paths + the same unmeasured U
Usage: python sensitivity_analysis.py      (writes sensitivity_results.csv, sensitivity_summary.csv)
"""
import os, sys, copy
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simulate_dataset as S

R = 500
N_POP = 200_000
SIZES = [150, 200]
BASE = copy.deepcopy(S.PARAMS)
TESTS = ["H1 AI→PERF", "H1 AN→PERF", "H1 AU→PERF", "H2 AN→CT", "H3 AU→DS",
         "H4 AI→ENG→PERF", "H5 AN→ENG→CT", "H6 AU→ENG→DS",
         "H7 AI→SAT", "H7 AN→SAT", "H7 AU→SAT", "H8 ENG→PERF", "H9 ENG→SAT"]


def scenario(name):
    P = copy.deepcopy(BASE)
    conf = 0.0
    if name in ("null", "confounded_null"):
        P["paths"] = {y: {x: 0.0 for x in d} for y, d in P["paths"].items()}
        P["level_effect_on"] = {k: 0.0 for k in P["level_effect_on"]}
    if name == "small":
        P["paths"] = {y: {x: 0.10 for x in d} for y, d in P["paths"].items()}
    if name == "medium":
        P["paths"] = {y: {x: 0.30 for x in d} for y, d in P["paths"].items()}
    if name == "low_reliability":
        P["target_alpha"] = {c: 0.70 for c in P["target_alpha"]}
    if name == "corr_exposure":
        P["exposure_correlation"] = 0.40
    if name in ("confounded", "confounded_null"):
        conf = 0.30
    return P, conf


def latent(n, rng, lvl, P, conf):
    """Same structural model as simulate_dataset.latent, with an optional unmeasured confounder U."""
    rho = P["exposure_correlation"]
    C = np.full((3, 3), rho) + np.eye(3) * (1 - rho)
    X = rng.multivariate_normal(np.zeros(3), C, n)
    U = rng.normal(size=n)
    if conf:
        X = np.sqrt(1 - conf ** 2) * X + conf * U[:, None]
    L = {"AI_EXP": X[:, 0], "AN_EXP": X[:, 1], "AU_EXP": X[:, 2]}
    for y in ["ENG", "PERF", "CT", "DS", "SAT"]:
        mu = sum(b * L[x] for x, b in P["paths"][y].items())
        mu = mu + P["level_effect_on"].get(y, 0) * (lvl - lvl.mean()) + conf * U
        resid_var = max(1 - np.var(mu), 0.05)
        L[y] = mu + rng.normal(0, np.sqrt(resid_var), n)
        L[y] = (L[y] - L[y].mean()) / L[y].std()
    return L


def make(n, rng, P, conf, calib):
    acl = rng.permutation(np.repeat([1, 2, 3], [int(n * .4), int(n * .4), n - 2 * int(n * .4)]))
    lvl = (acl - 1).astype(float)
    L = latent(n, rng, lvl, P, conf)
    sc = {c: S.items_from(L[c], c, rng, *calib[c]).mean(axis=1).values for c in S.CONS}
    sc["PERF"] = L["PERF"]

    def topk(score, frac):
        k = int(round(frac * n)); z = np.zeros(n); z[np.argsort(-score)[:k]] = 1; return z
    sd = P["prior_exposure_noise_sd"]
    prior = [topk(L[c] + rng.normal(0, sd, n), f)
             for c, f in [("AI_EXP", 92 / 150), ("AN_EXP", 87 / 150), ("AU_EXP", 69 / 150)]]
    Cv = np.column_stack([(acl == 2), (acl == 3), rng.random(n) < 48 / 150, rng.random(n) < 72 / 150]
                         + prior).astype(float)
    return sc, Cv


def ols(y, X):
    """Standardized OLS: returns coefficients, SEs, p-values, df for the first columns of X."""
    z = lambda v: (v - v.mean()) / v.std()
    y = z(y); X = np.column_stack([z(c) if c.std() > 0 else c for c in X.T])
    X = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b; df = len(y) - X.shape[1]
    se = np.sqrt(np.diag(res @ res / df * np.linalg.inv(X.T @ X)))
    return b[1:], se[1:], 2 * stats.t.sf(np.abs(b[1:] / se[1:]), df), df


def fit(sc, Cv):
    out = {}
    def put(name, b, se, p, df, i):
        out[name] = (b[i], se[i], p[i], df)
    r = ols(sc["PERF"], np.column_stack([sc["AI_EXP"], sc["AN_EXP"], sc["AU_EXP"], Cv]))
    for i, h in enumerate(["H1 AI→PERF", "H1 AN→PERF", "H1 AU→PERF"]): put(h, *r, i)
    put("H2 AN→CT", *ols(sc["CT"], np.column_stack([sc["AN_EXP"], Cv])), 0)
    put("H3 AU→DS", *ols(sc["DS"], np.column_stack([sc["AU_EXP"], Cv])), 0)
    for h, x, y in [("H4 AI→ENG→PERF", "AI_EXP", "PERF"), ("H5 AN→ENG→CT", "AN_EXP", "CT"),
                    ("H6 AU→ENG→DS", "AU_EXP", "DS")]:
        ba, _, pa, _ = ols(sc["ENG"], np.column_stack([sc[x], Cv]))
        bb, _, pb, _ = ols(sc[y], np.column_stack([sc["ENG"], sc[x], Cv]))
        out[h] = (ba[0] * bb[0], np.nan, max(pa[0], pb[0]), None)   # joint-significance test
    r = ols(sc["SAT"], np.column_stack([sc["AI_EXP"], sc["AN_EXP"], sc["AU_EXP"], Cv]))
    for i, h in enumerate(["H7 AI→SAT", "H7 AN→SAT", "H7 AU→SAT"]): put(h, *r, i)
    put("H8 ENG→PERF", *ols(sc["PERF"], np.column_stack([sc["ENG"], Cv])), 0)
    put("H9 ENG→SAT", *ols(sc["SAT"], np.column_stack([sc["ENG"], Cv])), 0)
    return out


def run(name, rng):
    P, conf = scenario(name)
    S.PARAMS.clear(); S.PARAMS.update(P)                       # items_from / calibrate read S.PARAMS
    calib = {c: S.calibrate(c, np.random.default_rng(7)) for c in S.CONS}
    true = {h: v[0] for h, v in fit(*make(N_POP, rng, P, conf, calib)).items()}
    rows = []
    for n in SIZES:
        reps = [fit(*make(n, rng, P, conf, calib)) for _ in range(R)]
        for h in TESTS:
            b = np.array([r[h][0] for r in reps]); p = np.array([r[h][2] for r in reps])
            row = dict(scenario=name, N=n, test=h, true=true[h], mean=b.mean(), bias=b.mean() - true[h],
                       emp_sd=b.std(ddof=1), reject=(p < .05).mean())
            if reps[0][h][3] is not None:
                se = np.array([r[h][1] for r in reps]); t = stats.t.ppf(.975, reps[0][h][3])
                row["cover"] = ((b - t * se <= true[h]) & (true[h] <= b + t * se)).mean()
            rows.append(row)
    S.PARAMS.clear(); S.PARAMS.update(copy.deepcopy(BASE))
    print("done", name, flush=True)
    return rows


if __name__ == "__main__":
    rng = np.random.default_rng(2027)
    names = ["calibrated", "null", "small", "medium", "low_reliability", "corr_exposure",
             "confounded", "confounded_null"]
    res = pd.DataFrame([r for nm in names for r in run(nm, rng)])
    res.round(3).to_csv("sensitivity_results.csv", index=False)
    summ = res.groupby(["scenario", "N"]).agg(
        mean_abs_bias=("bias", lambda v: v.abs().mean()), max_abs_bias=("bias", lambda v: v.abs().max()),
        min_cover=("cover", "min"), mean_cover=("cover", "mean"),
        min_reject=("reject", "min"), max_reject=("reject", "max")).reindex(names, level=0)
    summ.round(3).to_csv("sensitivity_summary.csv")
    print(summ.round(3).to_string())
