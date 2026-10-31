import json, time, os, glob, numpy as np, duckdb, gen, pipe
OUT = "results.jsonl"
def log(r):
    with open(OUT, "a") as f: f.write(json.dumps(r) + "\n")
if os.path.exists(OUT): os.remove(OUT)
N = 200_000
# 1. calibration set (frozen margin) -------------------------------------------------
p = f"/tmp/reg_cal.parquet"; gen.write_parquet(gen.generate(N, 100), p)
for r in range(1, 11): log({"stage": "calibration", **pipe.run(p, 100, r, "ref")})
# 2. main pilot: 5 data seeds x 5 replicate seeds, reference and 4 scripted variants ---
for ds in range(1, 6):
    p = f"/tmp/reg_main_{ds}.parquet"; gen.write_parquet(gen.generate(N, ds), p)
    for v in ["ref", "canary", "impute_winsor", "select_all", "oversample_first"]:
        for r in range(1, 6): log({"stage": "main", **pipe.run(p, ds, r, v)})
    print("seed", ds, "done", flush=True)
# 3. scale tier 1e6 (reference only) ---------------------------------------------------
p = "/tmp/reg_1e6.parquet"; t = time.time(); df = gen.generate(1_000_000, 1); gt = time.time() - t; gen.write_parquet(df, p); del df
for r in range(1, 4): log({"stage": "tier1e6", "gen_seconds": gt, **pipe.run(p, 1, r, "ref")})
# 4. scale tier 1e7: SQL preparation only (10 partitions x 1e6) --------------------------
os.makedirs("/tmp/reg1e7", exist_ok=True); t = time.time()
for k in range(10):
    gen.write_parquet(gen.generate(1_000_000, 1000 + k), f"/tmp/reg1e7/part{k}.parquet")
gen_s = time.time() - t
con = duckdb.connect(); t = time.time()
res = con.execute("""SELECT count(*) n, avg(risky) rate, avg((duration_years IS NULL)::int) null_dur,
    quantile_cont(age, 0.25) q1, quantile_cont(age, 0.75) q3, mode(duration_years) mode_dur
  FROM read_parquet('/tmp/reg1e7/*.parquet')""").fetchone()
res2 = con.execute("""SELECT interest_type, count(*) n, avg(risky) rate FROM read_parquet('/tmp/reg1e7/*.parquet') GROUP BY 1 ORDER BY 1""").fetchall()
log({"stage": "tier1e7_sql", "rows": int(res[0]), "default_rate": float(res[1]), "gen_seconds": gen_s, "sql_seconds": time.time() - t,
     "by_interest": [[a, int(b), float(c)] for a, b, c in res2]})
print("ALL DONE", flush=True)
