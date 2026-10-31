"""Monte Carlo power (same control variables as the main analysis) of the evaluation protocol: re-generate the simulated dataset R times at
several sample sizes and record how often each hypothesis is supported (p < .05; indirect
effects by the joint-significance test of paths a and b). Uses the generator in simulate_dataset.py."""
import sys, os, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simulate_dataset as S

R = 500
SIZES = [100, 150, 200, 300, 400]

def ols_p(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    df = len(y) - X.shape[1]
    s2 = res @ res / df
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return b[1:], 2 * stats.t.sf(np.abs(b[1:] / se[1:]), df)

def one(n, rng, calib):
    acl = rng.permutation(np.repeat([1, 2, 3], [int(n * .4), int(n * .4), n - 2 * int(n * .4)]))
    lvl = (acl - 1).astype(float)
    L = S.latent(n, rng, lvl)
    sc = {c: S.items_from(L[c], c, rng, *calib[c]).mean(axis=1).values for c in S.CONS}
    sc["PERF"] = L["PERF"]
    def topk(score, frac):
        k = int(round(frac * n)); z = np.zeros(n); z[np.argsort(-score)[:k]] = 1; return z
    sd = S.PARAMS["prior_exposure_noise_sd"]
    prior = [topk(L[c] + rng.normal(0, sd, n), f) for c, f in [("AI_EXP", 92 / 150), ("AN_EXP", 87 / 150), ("AU_EXP", 69 / 150)]]
    C = np.column_stack([(acl == 2), (acl == 3), rng.random(n) < 48 / 150, rng.random(n) < 72 / 150] + prior).astype(float)
    out = {}
    _, p = ols_p(sc["PERF"], np.column_stack([sc["AI_EXP"], sc["AN_EXP"], sc["AU_EXP"], C]))
    out["H1 AI→PERF"], out["H1 AN→PERF"], out["H1 AU→PERF"] = p[0] < .05, p[1] < .05, p[2] < .05
    _, p = ols_p(sc["CT"], np.column_stack([sc["AN_EXP"], C])); out["H2 AN→CT"] = p[0] < .05
    _, p = ols_p(sc["DS"], np.column_stack([sc["AU_EXP"], C])); out["H3 AU→DS"] = p[0] < .05
    for h, x, y in [("H4 AI→ENG→PERF", "AI_EXP", "PERF"), ("H5 AN→ENG→CT", "AN_EXP", "CT"), ("H6 AU→ENG→DS", "AU_EXP", "DS")]:
        _, pa = ols_p(sc["ENG"], np.column_stack([sc[x], C]))
        _, pb = ols_p(sc[y], np.column_stack([sc["ENG"], sc[x], C]))
        out[h] = (pa[0] < .05) and (pb[0] < .05)
    _, p = ols_p(sc["SAT"], np.column_stack([sc["AI_EXP"], sc["AN_EXP"], sc["AU_EXP"], C]))
    out["H7 AI→SAT"], out["H7 AN→SAT"], out["H7 AU→SAT"] = p[0] < .05, p[1] < .05, p[2] < .05
    _, p = ols_p(sc["PERF"], np.column_stack([sc["ENG"], C])); out["H8 ENG→PERF"] = p[0] < .05
    _, p = ols_p(sc["SAT"], np.column_stack([sc["ENG"], C])); out["H9 ENG→SAT"] = p[0] < .05
    return out

if __name__ == "__main__":
    calib = {c: S.calibrate(c, np.random.default_rng(7)) for c in S.CONS}
    rng = np.random.default_rng(99)
    rows = {}
    for n in SIZES:
        rows[n] = pd.DataFrame([one(n, rng, calib) for _ in range(R)]).mean()
    tab = pd.DataFrame(rows).round(3)
    tab.columns = [f"N={n}" for n in SIZES]
    tab.to_csv("power_analysis.csv")
    print(tab.to_string())
