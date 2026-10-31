"""
Simulated dataset for UBJBAS-D-26-00004 (AI, analytics and automation in accounting education).

Generates N = 150 synthetic student records that follow the data description in the manuscript:
  - sample composition (Table 4 of the R2 manuscript): 60 UG / 60 PG / 30 PhD, 78 male / 72 female, age groups,
    prior exposure to AI (92 yes), analytics (87 yes), automation/RPA (69 yes), mode of learning,
    field of study (102 accounting / 48 commerce-finance);
  - 33 five-point Likert items for seven constructs (Table 5, Appendix A: 5/5/5/6/4/4/4 items),
    with item loadings set to give the reported Cronbach's alpha values;
  - construct means and SDs set to target values (target_mean, target_sd below; observed values in Table 9);
  - a structural model among the latent constructs (Section 6, H1-H9). The path values are effect sizes
    specified by the authors: they reproduce the coefficients reported in the first version of the
    manuscript, which analysed an earlier simulated dataset whose data file and generator settings
    were lost (manuscript Section 3.2; specified values listed in Table 17);
  - two variables anchored on the public dataset of Hasan et al. (2021), Data 6(11):110:
      LMS  = empirical distribution of Moodle activity (Online C + Online O, 326 students),
             expressed as a 0-100 percentile score;
      PERF = empirical distribution shape of coursework marks ((CW1 + CW2) / 2, 326 students),
             linearly rescaled to mean 78.45 and SD 8.62, bounded to 0-100.
    Both are linked to the latent constructs through a Gaussian copula (rank mapping).

All parameters are in PARAMS below and are written to simulation_parameters.json, so the
procedure can be reported in full and reproduced exactly (fixed seed).

Usage: python simulate_dataset.py <folder with the Hasan et al. CSV files>
Output: simulated_dataset.csv, simulation_parameters.json
"""
import sys, glob, json, os
import numpy as np, pandas as pd

SEED = 2026
N = 150

PARAMS = {
    "seed": SEED,
    "N": N,
    "exposure_correlation": 0.116,         # correlation among AI_EXP, AN_EXP, AU_EXP latents
    # Standardized latent path values (author-specified effect sizes). They were solved (least squares)
    # so that the implied population regression coefficients reproduce those reported in the first
    # version of the manuscript, after allowing for measurement attenuation of 0.92.
    "paths": {
        "ENG":  {"AI_EXP": 0.406, "AN_EXP": 0.373, "AU_EXP": 0.432},
        "PERF": {"AI_EXP": 0.336, "AN_EXP": 0.161, "AU_EXP": 0.090, "ENG": 0.269},
        "CT":   {"AN_EXP": 0.323, "ENG": 0.384},
        "DS":   {"AU_EXP": 0.340, "ENG": 0.425},
        "SAT":  {"AI_EXP": 0.186, "AN_EXP": 0.164, "AU_EXP": 0.091, "ENG": 0.318},
    },
    "prior_exposure_noise_sd": 2.0,        # prior exposure (yes/no) = top-ranked of latent + N(0, 2.0)
    "level_effect_on": {"CT": 0.10, "DS": 0.08},   # per academic level step (UG=0, PG=1, PhD=2)
    "lms_eng_copula_r": 0.45,              # correlation of LMS activity rank with ENG latent
    "items": {"AI_EXP": 5, "AN_EXP": 5, "AU_EXP": 5, "ENG": 6, "CT": 4, "DS": 4, "SAT": 4},
    "target_alpha": {"AI_EXP": 0.84, "AN_EXP": 0.82, "AU_EXP": 0.86, "ENG": 0.88,
                     "CT": 0.80, "DS": 0.83, "SAT": 0.85},
    "target_mean": {"AI_EXP": 3.68, "AN_EXP": 3.54, "AU_EXP": 3.29, "ENG": 3.74,
                    "CT": 3.62, "DS": 3.70, "SAT": 3.81},
    "target_sd": {"AI_EXP": 0.82, "AN_EXP": 0.79, "AU_EXP": 0.85, "ENG": 0.76,
                  "CT": 0.71, "DS": 0.74, "SAT": 0.69},
    "perf_target_mean": 78.45, "perf_target_sd": 8.62,
    "composition": {
        "ACL": {"1": 60, "2": 60, "3": 30},
        "GEN": {"1": 78, "2": 72},
        "FOS": {"1": 102, "2": 48},
        "PRIOR_AI": 92, "PRIOR_AN": 87, "PRIOR_AU": 69,
        "MODE": {"1_online": 46, "2_blended": 64, "3_classroom": 40},
        "AGE": {"18-21": 38, "22-25": 64, "26-30": 32, "30+": 16},
    },
}
CONS = list(PARAMS["items"])


def latent(n, rng, lvl):
    p = PARAMS
    rho = p["exposure_correlation"]
    C = np.full((3, 3), rho) + np.eye(3) * (1 - rho)
    X = rng.multivariate_normal(np.zeros(3), C, n)
    L = {"AI_EXP": X[:, 0], "AN_EXP": X[:, 1], "AU_EXP": X[:, 2]}
    for y in ["ENG", "PERF", "CT", "DS", "SAT"]:
        mu = sum(b * L[x] for x, b in p["paths"][y].items())
        mu = mu + p["level_effect_on"].get(y, 0) * (lvl - lvl.mean())
        resid_var = max(1 - np.var(mu), 0.05)
        L[y] = mu + rng.normal(0, np.sqrt(resid_var), n)
        L[y] = (L[y] - L[y].mean()) / L[y].std()
    return L


def loading_for_alpha(alpha, k):
    r = alpha / (k - alpha * (k - 1))       # average inter-item correlation
    return min(np.sqrt(r) * 1.04, 0.95)     # small uplift offsets loss from 5-point rounding


def items_from(lat, c, rng, mu, s):
    k = PARAMS["items"][c]
    lam = loading_for_alpha(PARAMS["target_alpha"][c], k)
    cols = {}
    for j in range(k):
        z = lam * lat + np.sqrt(1 - lam ** 2) * rng.normal(size=len(lat))
        cols[f"{c.split('_')[0]}{j + 1}"] = np.clip(np.rint(mu + s * z), 1, 5).astype(int)
    return pd.DataFrame(cols)


def calibrate(c, rng):
    """Find item location/scale giving the target construct mean and SD (large-sample search)."""
    lat = rng.normal(size=200_000)
    tm, ts = PARAMS["target_mean"][c], PARAMS["target_sd"][c]
    mu, s = tm, ts
    for _ in range(40):
        sc = items_from(lat, c, np.random.default_rng(1), mu, s).mean(axis=1)
        mu += 0.8 * (tm - sc.mean())
        s *= (ts / sc.std()) ** 0.8
    return mu, s


def assign_exact(n, counts, rng, score=None):
    """Assign categories with exact counts; if score given, higher score -> first category."""
    labels = np.concatenate([[k] * v for k, v in counts.items()])
    if score is None:
        return rng.permutation(labels)
    order = np.argsort(-score)
    out = np.empty(n, dtype=object)
    out[order] = labels
    return out


def copula_map(latent_z, source_values, r, rng):
    """Rank-map a variable correlated r with latent_z onto the empirical source distribution."""
    z = r * latent_z + np.sqrt(1 - r ** 2) * rng.normal(size=len(latent_z))
    q = (pd.Series(z).rank().values - 0.5) / len(z)
    return np.quantile(source_values, q)


def main(src):
    rng = np.random.default_rng(SEED)
    res = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(src, "Result*.csv"))])
    act = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(src, "Activity*.csv"))])
    cw = ((pd.to_numeric(res["CW1"], errors="coerce") + pd.to_numeric(res["CW2"], errors="coerce")) / 2).dropna().values
    lms_raw = (act["Online C"] + act["Online O"]).dropna().values
    lms_pct = pd.Series(lms_raw).rank(pct=True).values * 100

    comp = PARAMS["composition"]
    acl = assign_exact(N, {int(k): v for k, v in comp["ACL"].items()}, rng).astype(int)
    lvl = (acl - 1).astype(float)
    L = latent(N, rng, lvl)

    df = pd.DataFrame({"ID": [f"S{i + 1:03d}" for i in range(N)], "ACL": acl})
    df["GEN"] = assign_exact(N, {1: 78, 2: 72}, rng).astype(int)
    df["FOS"] = assign_exact(N, {1: 102, 2: 48}, rng).astype(int)
    # age group follows academic level (younger UG, older PhD), exact Table 3 counts
    age_score = lvl + rng.normal(0, 0.6, N)
    df["AGE_GROUP"] = assign_exact(N, {"30+": 16, "26-30": 32, "22-25": 64, "18-21": 38}, rng, age_score)
    df["MODE"] = assign_exact(N, {1: 46, 2: 64, 3: 40}, rng).astype(int)
    for v, lat_name in [("PRIOR_AI", "AI_EXP"), ("PRIOR_AN", "AN_EXP"), ("PRIOR_AU", "AU_EXP")]:
        sc = L[lat_name] + rng.normal(0, PARAMS["prior_exposure_noise_sd"], N)          # prior exposure related to current exposure
        df[v] = assign_exact(N, {1: comp[v], 0: N - comp[v]}, rng, sc).astype(int)

    calib = {}
    for c in CONS:
        mu, s = calibrate(c, np.random.default_rng(7))
        calib[c] = {"item_location": round(float(mu), 4), "item_scale": round(float(s), 4),
                    "loading": round(float(loading_for_alpha(PARAMS["target_alpha"][c], PARAMS["items"][c])), 4)}
        df = pd.concat([df, items_from(L[c], c, rng, mu, s)], axis=1)

    perf_shape = copula_map(L["PERF"], cw, 0.999, rng)
    perf = (perf_shape - perf_shape.mean()) / perf_shape.std()
    df["PERF"] = np.clip(np.round(PARAMS["perf_target_mean"] + PARAMS["perf_target_sd"] * perf, 1), 0, 100)
    df["LMS"] = np.round(copula_map(L["ENG"], lms_pct, PARAMS["lms_eng_copula_r"], rng), 1)

    df.to_csv("simulated_dataset.csv", index=False)
    meta = dict(PARAMS)
    meta["item_calibration"] = calib
    meta["source_dataset"] = {
        "citation": "Hasan, Palaniappan, Mahmood, Abbas, Sarker (2021), Data 6(11):110, doi:10.3390/data6110110",
        "PERF_anchor": f"empirical quantiles of (CW1+CW2)/2, n={len(cw)}, mean={cw.mean():.2f}, sd={cw.std():.2f}",
        "LMS_anchor": f"empirical percentile of Online C + Online O, n={len(lms_raw)}",
    }
    json.dump(meta, open("simulation_parameters.json", "w"), indent=2, default=str)
    print("Wrote simulated_dataset.csv", df.shape, "and simulation_parameters.json")


if __name__ == "__main__":
    main(sys.argv[1])
