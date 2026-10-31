"""Synthetic credit-registry generator for the pilot (requirements R1-R6, partial).
Conditioned only on aggregates published in Doko et al. 2021 (Tables 1-5); everything
else (purpose codes, age law, coefficients) is an explicit assumption."""
import numpy as np, pandas as pd, duckdb

# published aggregates (Doko et al. 2021): bank size (Table 3), interest-rate type and loan type (Table 2)
BANK = {"E": 49636, "G": 669363, "M": 42070, "S": 238931}
INT_T = {"P": 924435, "F": 59966, "V": 15599}
LOAN_T = {"A": 983415, "E": 16585}
TARGET_RATE = 167_000 / 1_000_000 + 0.001   # overall risky share implied by Table 2 (~16.8%)
PURPOSES = ["p1", "p2", "p3", "p4", "p5", "p6"]                # ASSUMPTION
PURP_P = np.array([0.30, 0.25, 0.20, 0.12, 0.08, 0.05])         # ASSUMPTION
DELAY0 = dict(p_zero=0.78, mean=20.0)                           # non-risky: delay law (ASSUMPTION)
DELAY1 = dict(p_zero=0.25, mu=np.log(70.0), sd=0.9)             # risky: delay law (ASSUMPTION)
CANARY_FLIP = 0.03

def _norm(d): 
    k = list(d); p = np.array([d[x] for x in k], float); return k, p / p.sum()

def _logit_part(df):
    b = np.where(df.interest_type == "F", -1.1, np.where(df.interest_type == "V", 0.07, 0.0))
    b = b + np.where(df.loan_type == "E", 0.85, 0.0)
    b = b - 0.5 * df.paid_loans + 0.15 * (df.n_loans - 1) + 0.05 * (df.duration_years - 5.5)
    b = b + 0.4 * np.exp(-((df.age - 32) / 6.0) ** 2)
    b = b + np.where(df.bank_size == "S", 0.15, 0.0) + np.where(df.purpose.isin(["p4", "p5"]), 0.2, 0.0)
    return b

_B0 = {}
def _intercept():
    if "b0" in _B0: return _B0["b0"]
    df = _covariates(400_000, np.random.default_rng(12345)); base = _logit_part(df).values
    lo, hi = -6.0, 2.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(base + mid)))).mean() > TARGET_RATE: hi = mid
        else: lo = mid
    _B0["b0"] = (lo + hi) / 2; return _B0["b0"]

def _covariates(n, rng):
    kb, pb = _norm(BANK); ki, pi = _norm(INT_T); kl, pl = _norm(LOAN_T)
    it = rng.choice(ki, n, p=pi)
    rate_mu = np.where(it == "P", 6.0, np.where(it == "F", 7.0, 5.0))
    df = pd.DataFrame({
        "bank_size": rng.choice(kb, n, p=pb), "loan_type": rng.choice(kl, n, p=pl), "interest_type": it,
        "purpose": rng.choice(PURPOSES, n, p=PURP_P),
        "n_loans": np.minimum(11, 1 + rng.poisson(0.238, n)),
        "duration_years": np.clip(np.round(np.exp(rng.normal(1.61, 0.42, n))), 1, 31),
        "paid_loans": np.minimum(5, rng.poisson(0.086, n)),
        "age": np.clip(np.round(rng.normal(42, 13, n)), 20, 80),
        "interest_rate": np.round(rng.normal(rate_mu, 1.5, n), 2)})
    return df

def oracle_post(p_x, d):
    f0 = np.where(d == 0, DELAY0["p_zero"], (1 - DELAY0["p_zero"]) * np.exp(-d / DELAY0["mean"]) / DELAY0["mean"])
    dd = np.maximum(d, 1.0)
    ln = np.exp(-(np.log(dd) - DELAY1["mu"]) ** 2 / (2 * DELAY1["sd"] ** 2)) / (dd * DELAY1["sd"] * np.sqrt(2 * np.pi))
    f1 = np.where(d == 0, DELAY1["p_zero"], (1 - DELAY1["p_zero"]) * ln)
    return p_x * f1 / (p_x * f1 + (1 - p_x) * f0)

def generate(n, seed, faults=True):
    rng = np.random.default_rng(seed)
    df = _covariates(n, rng)
    p_x = 1 / (1 + np.exp(-(_logit_part(df).values + _intercept())))
    y = (rng.random(n) < p_x).astype(int)
    nz = rng.random(n)
    d0 = np.ceil(np.maximum(1, rng.exponential(DELAY0["mean"], n)))
    d1 = np.ceil(np.maximum(1, np.exp(rng.normal(DELAY1["mu"], DELAY1["sd"], n))))
    df["delay_days"] = np.where(y == 0, np.where(nz < DELAY0["p_zero"], 0, d0), np.where(nz < DELAY1["p_zero"], 0, d1))
    flip = rng.random(n) < CANARY_FLIP
    df["collection_flag"] = np.where(flip, 1 - y, y)            # R6 leakage canary (post-outcome proxy of label)
    df["risky"] = y
    df["_p_x"] = p_x                                            # evaluator-held (R3)
    df["_oracle"] = oracle_post(p_x, df.delay_days.values)
    if faults:                                                  # R5 (nulls and outliers only in the pilot)
        m = rng.random(n) < 0.03; df.loc[m, "duration_years"] = np.nan
        o = rng.random(n) < 0.005; df.loc[o, "age"] = rng.integers(150, 200, o.sum())
    df.insert(0, "client_id", np.arange(n))
    return df

def write_parquet(df, path):
    con = duckdb.connect(); con.register("d", df)
    con.execute(f"COPY d TO '{path}' (FORMAT PARQUET)"); con.close()

if __name__ == "__main__":
    df = generate(300_000, 1)
    print("default rate", df.risky.mean().round(4), "(published ~0.168)")
    print(df.groupby("interest_type").risky.mean().round(3).to_dict(), "(pub P .174 F .066 V .186)")
    print(df.groupby("loan_type").risky.mean().round(3).to_dict(), "(pub A .165 E .328)")
    print("mean n_loans", df.n_loans.mean().round(3), "(1.238)  mean duration", df.duration_years.mean().round(2), "(5.7)  mean paid", df.paid_loans.mean().round(3), "(0.086)")
    print("bank share S", (df.bank_size=="S").mean().round(3), "(0.239)")
    from sklearn.metrics import roc_auc_score
    print("oracle AUC", round(roc_auc_score(df.risky, df._oracle),4), " canary agree", (df.collection_flag==df.risky).mean().round(3))
