"""Reference pipeline (reduced) and scripted pipeline variants for the protocol pilot.
Pipelines see only the training pool (labels available) and hidden-test FEATURES; the evaluator scores the predictions."""
import time, numpy as np, pandas as pd, duckdb
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, roc_curve

CAT = ["bank_size", "loan_type", "interest_type", "purpose"]
NUM = ["n_loans", "duration_years", "paid_loans", "age", "interest_rate", "delay_days"]
W = {"D1": 3, "D2": 1, "D3": 1, "D4": 2, "D5": 1, "D6": 1, "D7": 2, "D8": 3, "D9": 2, "D10": 2, "D11": 2}

def ks(y, s):
    f, t, _ = roc_curve(y, s); return float(np.max(t - f))

def iv_of(x, y, bins=10):
    if x.dtype.kind in "fi" and x.nunique() > bins:
        x = pd.qcut(x, bins, duplicates="drop")
    t = pd.crosstab(x, y).astype(float) + 0.5
    pg, pb = t[0] / t[0].sum(), t[1] / t[1].sum()
    return float(((pg - pb) * np.log(pg / pb)).sum())

class Prep:
    """Fits preprocessing on `train` only (SQL for statistics), then transforms any frame."""
    def __init__(self, cfg): self.cfg = cfg
    def fit(self, tr):
        con = duckdb.connect(); con.register("tr", tr); c = self.cfg
        self.fill = (con.execute("SELECT mode(duration_years) FROM tr").fetchone()[0] if c["impute"] == "mode"
                     else con.execute("SELECT median(duration_years) FROM tr").fetchone()[0])
        self.fence = {}
        for col in ["age", "duration_years", "interest_rate"]:
            q1, q3, p01, p99 = con.execute(f"SELECT quantile_cont({col},0.25), quantile_cont({col},0.75), quantile_cont({col},0.01), quantile_cont({col},0.99) FROM tr").fetchone()
            self.fence[col] = (q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1), p01, p99)
        a = con.execute("SELECT quantile_cont(age, [0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95]) FROM tr WHERE age<100").fetchone()[0]
        self.age_edges = np.unique(a)
        self.rate_edges = np.unique(con.execute("SELECT quantile_cont(interest_rate, [0.25,0.5,0.75]) FROM tr").fetchone()[0])
        con.close(); return self
    def transform(self, df, train=False):
        d = df.copy(); c = self.cfg
        d["duration_years"] = d.duration_years.fillna(self.fill)
        if train and c["outliers"] == "remove":
            m = pd.Series(True, index=d.index)
            for col, (lo, hi, _, _) in self.fence.items(): m &= d[col].between(lo, hi)
            d = d[m]
        if c["outliers"] == "winsor":
            for col, (_, _, p01, p99) in self.fence.items(): d[col] = d[col].clip(p01, p99)
        d["age"] = np.searchsorted(self.age_edges, d.age.values)               # D5 quantile bins (20)
        d["interest_rate"] = np.searchsorted(self.rate_edges, d.interest_rate.values)   # 4 bins
        return d

def select(tr, cfg):
    cand = CAT + NUM + (["collection_flag"] if cfg["canary"] else [])
    iv = {f: iv_of(tr[f], tr.risky) for f in cand}
    keep = [f for f in cand if iv[f] >= 0.02]
    X = pd.get_dummies(tr[keep], columns=[f for f in keep if f in CAT], dtype=float)
    r = X.corr().abs()
    drop = set()
    for i, a in enumerate(X.columns):
        for b in X.columns[i + 1:]:
            if a not in drop and b not in drop and r.loc[a, b] > 0.8:
                fa = next((f for f in keep if a == f or a.startswith(f + "_")), a); fb = next((f for f in keep if b == f or b.startswith(f + "_")), b)
                drop.add(b if iv[fa] >= iv[fb] else a)
    cols = [c for c in X.columns if c not in drop]
    sel = sorted({next(f for f in keep if c == f or c.startswith(f + "_")) for c in cols})
    return sel, cols, iv

def design(df, sel, cols):
    X = pd.get_dummies(df[sel], columns=[f for f in sel if f in CAT], dtype=float)
    return X.reindex(columns=cols, fill_value=0.0)

def models(seed):
    return {"LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=300)),
            "DT": DecisionTreeClassifier(max_depth=8, min_samples_leaf=100, random_state=seed),
            "RF": RandomForestClassifier(n_estimators=30, max_depth=10, min_samples_leaf=50, n_jobs=1, random_state=seed)}

def oversample(df, rng):
    pos, neg = df[df.risky == 1], df[df.risky == 0]
    extra = pos.sample(len(neg) - len(pos), replace=True, random_state=int(rng.integers(1e9)))
    return pd.concat([df, extra]).sample(frac=1, random_state=int(rng.integers(1e9))).reset_index(drop=True)

VEC = {  # decision vectors by construction: choice is 'ref', 'adm' (admissible deviation) or 'bad'
 "ref": {}, "canary": {"D4": "bad"}, "impute_winsor": {"D2": "adm", "D3": "adm"},
 "select_all": {"D8": "bad"}, "oversample_first": {"D8": "bad", "D9": "bad"}}

def fidelity(variant):
    v = VEC[variant]; tot = sum(W.values())
    strict = sum(w for k, w in W.items() if v.get(k, "ref") == "ref") / tot
    adm = sum(w for k, w in W.items() if v.get(k, "ref") in ("ref", "adm")) / tot
    return strict, adm

def run(path, data_seed, rep_seed, variant="ref"):
    t0 = time.time(); rng = np.random.default_rng(rep_seed)
    cfg = {"impute": "median" if variant == "impute_winsor" else "mode",
           "outliers": "winsor" if variant == "impute_winsor" else "remove", "canary": variant == "canary"}
    con = duckdb.connect()
    con.execute(f"CREATE TABLE reg AS SELECT *, (hash(client_id + {data_seed}*1000003) % 100) < 20 AS is_test FROM read_parquet('{path}')")
    pool = con.execute("SELECT * FROM reg WHERE NOT is_test").df(); test = con.execute("SELECT * FROM reg WHERE is_test").df(); con.close()
    feats_test = test.drop(columns=["risky", "_p_x", "_oracle"])      # hidden labels
    # ---- internal validation (D8): 80/20 hold-out inside the training pool
    pool_p = pool
    if variant == "oversample_first": pool_p = oversample(pool, rng)           # leak: duplicates cross the split
    if variant == "select_all":
        prep0 = Prep(cfg).fit(pool); tr_all = prep0.transform(pool, train=True); sel0, cols0, _ = select(tr_all, cfg)
    idx = rng.permutation(len(pool_p)); cut = int(0.8 * len(pool_p))
    fit_raw, val_raw = pool_p.iloc[idx[:cut]], pool_p.iloc[idx[cut:]]
    if variant == "select_all":
        prep = prep0; fit = prep.transform(fit_raw, train=True); val = prep.transform(val_raw); sel, cols = sel0, cols0
    else:
        prep = Prep(cfg).fit(fit_raw); fit = prep.transform(fit_raw, train=True); val = prep.transform(val_raw); sel, cols, _ = select(fit, cfg)
    dup = float(val_raw.drop(columns=["client_id"]).merge(fit_raw.drop(columns=["client_id"]).drop_duplicates(), how="left", indicator=True, on=[c for c in fit_raw.columns if c not in ("client_id","_p_x","_oracle")]).eval("_merge=='both'").mean()) if variant == "oversample_first" else 0.0
    best, bauc = None, -1; per = {}
    for name, m in models(rep_seed).items():
        m.fit(design(fit, sel, cols), fit.risky); a = roc_auc_score(val.risky, m.predict_proba(design(val, sel, cols))[:, 1]); per[name] = a
        if a > bauc: best, bauc = name, a
    # ---- refit on the whole pool, predict hidden test
    prep = Prep(cfg).fit(pool_p); full = prep.transform(pool_p, train=True)
    if variant != "select_all": sel, cols, iv = select(full, cfg)
    m = models(rep_seed)[best]; m.fit(design(full, sel, cols), full.risky)
    tt = prep.transform(feats_test); s = m.predict_proba(design(tt, sel, cols))[:, 1]
    y = test.risky.values
    return {"variant": variant, "data_seed": data_seed, "rep_seed": rep_seed, "model": best, "features": sel,
            "val_auc": float(bauc), "auc": float(roc_auc_score(y, s)), "ks": ks(y, s), "pr_auc": float(average_precision_score(y, s)),
            "f1": float(f1_score(y, s > 0.5)), "acc": float(accuracy_score(y, s > 0.5)),
            "oracle_auc": float(roc_auc_score(y, test._oracle.values)), "holdout_dup_rate": dup,
            "n_train_pool": int(len(pool)), "n_test": int(len(test)), "seconds": time.time() - t0}

if __name__ == "__main__":
    import sys, gen
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    p = f"/tmp/reg_{n}_1.parquet"; gen.write_parquet(gen.generate(n, 1), p)
    for v in sys.argv[2:] or ["ref"]:
        r = run(p, 1, 1, v); print({k: (round(x, 4) if isinstance(x, float) else x) for k, x in r.items()})
