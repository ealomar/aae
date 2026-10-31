"""
Robustness check (UBJBAS-D-26-00004, R2): the complete framework estimated as one structural
equation model with latent constructs (semopy, maximum likelihood), on the released dataset.
Controls: academic level (PG, PhD dummies), field of study, gender, prior exposure (AI, AN, AU).
Indirect effects a*b are computed from the standardized paths; their 95% CIs use a percentile
bootstrap over records (B resamples).
Usage: python sem_robustness.py simulated_dataset.csv   -> sem_paths.csv, sem_fit.csv, sem_indirect.csv
"""
import sys, warnings
import numpy as np, pandas as pd
import semopy
warnings.filterwarnings("ignore")
B = 1000

ITEMS = {"AI_EXP": "AI", "AN_EXP": "AN", "AU_EXP": "AU", "ENG": "ENG", "CT": "CT", "DS": "DS", "SAT": "SAT"}
K = {"AI_EXP": 5, "AN_EXP": 5, "AU_EXP": 5, "ENG": 6, "CT": 4, "DS": 4, "SAT": 4}
CTRL = "PG + PhD + COMMFIN + FEMALE + PRIOR_AI + PRIOR_AN + PRIOR_AU"
meas = "\n".join(f"{c} =~ " + " + ".join(f"{ITEMS[c]}{j + 1}" for j in range(K[c])) for c in K)
struct = f"""
ENG ~ AI_EXP + AN_EXP + AU_EXP + {CTRL}
PERF ~ AI_EXP + AN_EXP + AU_EXP + ENG + {CTRL}
CT ~ AN_EXP + ENG + {CTRL}
DS ~ AU_EXP + ENG + {CTRL}
SAT ~ AI_EXP + AN_EXP + AU_EXP + ENG + {CTRL}
"""
DESC = meas + struct
IND = [("H4", "AI_EXP", "PERF"), ("H5", "AN_EXP", "CT"), ("H6", "AU_EXP", "DS")]


def prep(df):
    d = df.copy()
    d["PG"] = (d.ACL == 2).astype(float); d["PhD"] = (d.ACL == 3).astype(float)
    d["COMMFIN"] = (d.FOS == 2).astype(float); d["FEMALE"] = (d.GEN == 2).astype(float)
    d["PERF"] = (d.PERF - d.PERF.mean()) / d.PERF.std()
    return d


def fit(d):
    m = semopy.Model(DESC); m.fit(d, obj="MLW")
    est = m.inspect(std_est=True)
    return m, est[est.op == "~"]


def std_path(paths, y, x):
    r = paths[(paths.lval == y) & (paths.rval == x)]
    return float(r["Est. Std"].iloc[0])


if __name__ == "__main__":
    d = prep(pd.read_csv(sys.argv[1]))
    m, paths = fit(d)
    keep = paths[paths.lval.isin(["ENG", "PERF", "CT", "DS", "SAT"]) & paths.rval.isin(list(K))]
    keep[["lval", "rval", "Estimate", "Std. Err", "p-value", "Est. Std"]].round(3).to_csv("sem_paths.csv", index=False)
    st = semopy.calc_stats(m).T
    fitt = st.loc[["DoF", "chi2", "CFI", "TLI", "RMSEA"]].T
    fitt.round(3).to_csv("sem_fit.csv", index=False)
    rng = np.random.default_rng(2026)
    point = {h: std_path(paths, "ENG", x) * std_path(paths, y, "ENG") for h, x, y in IND}
    boots = {h: [] for h, _, _ in IND}
    for _ in range(B):
        db = d.iloc[rng.integers(0, len(d), len(d))].reset_index(drop=True)
        try:
            _, pb = fit(db)
            for h, x, y in IND:
                boots[h].append(std_path(pb, "ENG", x) * std_path(pb, y, "ENG"))
        except Exception:
            pass
    ind = pd.DataFrame([dict(hyp=h, X=x, Y=y, indirect=point[h], ci_low=np.percentile(boots[h], 2.5),
                             ci_high=np.percentile(boots[h], 97.5), boot_ok=len(boots[h])) for h, x, y in IND])
    ind.round(3).to_csv("sem_indirect.csv", index=False)
    print(keep[["lval", "rval", "Est. Std", "p-value"]].round(3).to_string())
    print(fitt.round(3).to_string()); print(ind.round(3).to_string())
