"""Aggregate agent runs into the tables used in the paper. Reads runs/*.json, ../results.jsonl (reference) and optional audit.csv."""
import glob, json, itertools, os, numpy as np, pandas as pd
W = {"D1": 3, "D2": 1, "D3": 1, "D4": 2, "D5": 1, "D6": 1, "D7": 2, "D8": 3, "D9": 2, "D10": 2, "D11": 2}
DELTA = 0.005
runs = [json.load(open(f)) for f in sorted(glob.glob("runs/*.json"))]
df = pd.DataFrame(runs)
ref = pd.DataFrame([json.loads(l) for l in open("../results.jsonl")]); ref = ref[(ref.stage == "main") & (ref.variant == "ref")]
refm = ref.groupby("data_seed").auc.mean()
df["ref_auc"] = df.data_seed.map(refm); df["dauc"] = df.auc - df.ref_auc
df["feats"] = df.manifest.map(lambda m: tuple(sorted(m.get("features_used", []))) if isinstance(m, dict) else ())
if os.path.exists("audit.csv"):
    a = pd.read_csv("audit.csv").set_index("run_id"); tot = sum(W.values())
    df["F_strict"] = df.run_id.map(lambda r: sum(w for k, w in W.items() if a.loc[r, k] == "ref") / tot if r in a.index else np.nan)
    df["F_adm"] = df.run_id.map(lambda r: sum(w for k, w in W.items() if a.loc[r, k] in ("ref", "adm")) / tot if r in a.index else np.nan)
def jac(sets):
    s = [set(x) for x in sets]; p = list(itertools.combinations(range(len(s)), 2)); return np.mean([len(s[i] & s[j]) / len(s[i] | s[j]) for i, j in p]) if p else np.nan
def tar(g):
    if "audit" not in globals(): return np.nan
    cols = [k for k in W if k != "D1"]; sub = audit.loc[audit.index.isin(g.run_id), cols]
    return float((sub.nunique() == 1).mean()) if len(sub) > 1 else np.nan
audit = pd.read_csv("audit.csv").set_index("run_id") if os.path.exists("audit.csv") else None
rows = []
for (model, level), g in df.groupby(["model", "level"]):
    ok = g[g.ok == True]
    rows.append({"model": model, "level": level, "runs": len(g), "ok": len(ok), "auc": ok.auc.mean(), "auc_sd": ok.auc.std(ddof=1), "dauc": ok.dauc.mean(),
                 "equiv_share": float((ok.dauc.abs() <= DELTA).mean()) if len(ok) else np.nan, "above_oracle": int(ok.above_oracle.sum()),
                 "canary_used": int(ok.canary_in_features.sum()), "canary_flagged": int(ok.canary_flagged.sum()), "feat_jaccard": jac(ok.feats),
                 "F_strict": ok.F_strict.mean() if "F_strict" in ok else np.nan, "F_adm": ok.F_adm.mean() if "F_adm" in ok else np.nan,
                 "TAR": tar(g) if audit is not None else np.nan, "tokens_in": g.tokens_in.mean(), "tokens_out": g.tokens_out.mean(), "seconds": g.seconds.mean(), "turns": g.turns.mean()})
t = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(t.round(4).to_string(index=False)); t.to_csv("agent_summary.csv", index=False)
print("\nreference mean AUC by data seed:", refm.round(4).to_dict(), " margin delta =", DELTA)
