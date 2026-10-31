import json, itertools, numpy as np, pandas as pd, pipe
R = [json.loads(l) for l in open("results.jsonl")]
df = pd.DataFrame(R)
out = {}
cal = df[df.stage == "calibration"]
for m in ["auc", "ks", "pr_auc"]:
    out[f"cal_{m}_mean"] = cal[m].mean(); out[f"cal_{m}_sd"] = cal[m].std(ddof=1)
delta = {m: max(2 * out[f"cal_{m}_sd"], 0.005) for m in ["auc", "ks", "pr_auc"]}
out["delta"] = delta; out["cal_n"] = len(cal)
main = df[df.stage == "main"]; ref = main[main.variant == "ref"]
for m in ["auc", "ks", "pr_auc", "f1"]:
    g = ref.groupby("data_seed")[m]; J = g.size().iloc[0]
    sw2 = g.var(ddof=1).mean(); sm = g.mean(); sb2 = max(0.0, sm.var(ddof=1) - sw2 / J)
    out[f"ref_{m}_mean"] = ref[m].mean(); out[f"ref_{m}_sd_within"] = np.sqrt(sw2); out[f"ref_{m}_sd_between"] = np.sqrt(sb2)
out["ref_oracle_mean"] = ref.oracle_auc.mean(); out["ref_gap_to_oracle"] = ref.oracle_auc.mean() - ref.auc.mean()
out["ref_models"] = ref.model.value_counts().to_dict()
def jac(sets): 
    s = [set(x) for x in sets]; pairs = list(itertools.combinations(range(len(s)), 2))
    return float(np.mean([len(s[i] & s[j]) / len(s[i] | s[j]) for i, j in pairs]))
out["ref_J_within"] = float(np.mean([jac(g.features) for _, g in ref.groupby("data_seed")]))
out["ref_J_all"] = jac(ref.features)
out["ref_featsets"] = ref.features.map(lambda x: "+".join(x)).value_counts().to_dict()
out["ref_seconds"] = ref.seconds.mean()
rm = ref.groupby("data_seed")[["auc", "ks", "pr_auc"]].mean()
mut = {}
for v in ["canary", "impute_winsor", "select_all", "oversample_first"]:
    d = main[main.variant == v]; dm = d.groupby("data_seed")[["auc", "ks", "pr_auc"]].mean()
    diff = dm - rm; s, a = pipe.fidelity(v)
    mut[v] = {"auc": d.auc.mean(), "auc_diff_mean": diff.auc.mean(), "auc_diff_min": diff.auc.min(), "auc_diff_max": diff.auc.max(),
              "ks": d.ks.mean(), "pr_auc": d.pr_auc.mean(), "val_auc": d.val_auc.mean(), "val_test_gap": (d.val_auc - d.auc).mean(),
              "oracle": d.oracle_auc.mean(), "above_oracle_runs": int((d.auc > d.oracle_auc + 0.005).sum()), "n_runs": len(d),
              "canary_used": int(d.features.map(lambda f: "collection_flag" in f).sum()), "dup_rate": d.holdout_dup_rate.mean(),
              "within_margin_all_seeds": bool((diff.auc.abs() <= delta["auc"]).all()), "F_strict": s, "F_adm": a,
              "models": d.model.value_counts().to_dict(), "featsets": d.features.map(lambda x: "+".join(x)).value_counts().to_dict()}
mut["ref"] = {"val_auc": ref.val_auc.mean(), "val_test_gap": (ref.val_auc - ref.auc).mean(), "auc": ref.auc.mean(), "oracle": ref.oracle_auc.mean()}
out["mut"] = mut
t6 = df[df.stage == "tier1e6"]
if len(t6): out["t6"] = {"auc_mean": t6.auc.mean(), "auc_sd": t6.auc.std(ddof=1), "ks": t6.ks.mean(), "sec_mean": t6.seconds.mean(), "gen_seconds": float(t6.gen_seconds.iloc[0]), "n_pool": int(t6.n_train_pool.iloc[0]), "n_test": int(t6.n_test.iloc[0]), "n": len(t6), "oracle": t6.oracle_auc.mean()}
t7 = df[df.stage == "tier1e7_sql"]
if len(t7): out["t7"] = t7.iloc[0].dropna().to_dict()
json.dump(out, open("summary.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
