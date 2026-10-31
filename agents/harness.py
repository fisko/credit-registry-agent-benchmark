"""Run LLM agents on the pilot benchmark.  Usage examples:
  python harness.py --provider mock --model mock-ref --levels S3 --seeds 1 --reps 1          # offline test
  python harness.py --provider openai --model gpt-5.4-mini --levels S1 S3 S3p --seeds 1 --reps 5
Results: runs/<run_id>.json (metrics, tokens, seconds, final code); work dirs in work/<run_id>/ ."""
import argparse, json, os, re, subprocess, sys, time, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gen, duckdb, numpy as np, pandas as pd
from specs import SPECS
import llm as LLM
from evaluate import evaluate

ROOT = os.path.dirname(os.path.abspath(__file__)); N_ROWS = 200_000
SYSTEM = "You are a careful, honest data scientist. Follow the instructions exactly."

def prepare(seed):
    """Same hidden-test split as the pilot (hash of client_id); labels and oracle never leave the evaluator."""
    d = f"{ROOT}/data/seed{seed}"; 
    if os.path.exists(f"{d}/eval.parquet"): return d
    os.makedirs(d, exist_ok=True); df = gen.generate(N_ROWS, seed); gen.write_parquet(df, f"{d}/full.parquet")
    con = duckdb.connect(); con.execute(f"CREATE TABLE r AS SELECT *, (hash(client_id + {seed}*1000003) % 100) < 20 AS is_test FROM read_parquet('{d}/full.parquet')")
    drop = ["_p_x", "_oracle", "is_test"]
    con.execute(f"COPY (SELECT * EXCLUDE ({','.join(drop)}) FROM r WHERE NOT is_test) TO '{d}/train.parquet' (FORMAT PARQUET)")
    con.execute(f"COPY (SELECT * EXCLUDE (risky,{','.join(drop)}) FROM r WHERE is_test) TO '{d}/test_features.parquet' (FORMAT PARQUET)")
    con.execute(f"COPY (SELECT client_id, risky, _oracle FROM r WHERE is_test) TO '{d}/eval.parquet' (FORMAT PARQUET)"); con.close()
    return d

def extract_code(text):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S); return m.group(1) if m else None

def run_one(provider, model, level, seed, rep, max_turns=8, timeout=600):
    run_id = f"{model.replace('/','_')}_{level}_s{seed}_r{rep}"; out = f"{ROOT}/runs/{run_id}.json"
    if os.path.exists(out): print("skip", run_id); return
    os.makedirs(f"{ROOT}/runs", exist_ok=True); data = prepare(seed)
    wd = f"{ROOT}/work/{run_id}"; os.makedirs(wd, exist_ok=True)
    for f in ["train.parquet", "test_features.parquet"]:
        if not os.path.exists(f"{wd}/{f}"): os.symlink(f"{data}/{f}", f"{wd}/{f}")
    client = LLM.make(provider, model); msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": SPECS[level]}]
    tin = tout = 0; codes = []; t0 = time.time(); status = "max_turns"; log = []
    for turn in range(max_turns):
        text, u = client.generate(msgs); tin += u["in"]; tout += u["out"]; msgs.append({"role": "assistant", "content": text})
        code = extract_code(text)
        if code is None:
            if "DONE" in text: status = "done"; break
            msgs.append({"role": "user", "content": "Reply with one complete Python script in a ```python block, or DONE if finished."}); continue
        path = f"{wd}/step_{turn}.py"; open(path, "w").write(code); codes.append(code)
        try:
            p = subprocess.run([sys.executable, path], cwd=wd, capture_output=True, text=True, timeout=timeout); o = (p.stdout + "\n" + p.stderr)[-3000:]
        except subprocess.TimeoutExpired: o = f"TIMEOUT after {timeout}s"
        have = {f: os.path.exists(f"{wd}/{f}") for f in ["predictions.csv", "manifest.json"]}
        msgs.append({"role": "user", "content": f"Output:\n{o}\nFiles present: {have}\nContinue, or reply DONE if both deliverables are final."}); log.append(o[-500:])
    res = evaluate(wd, data)
    json.dump({"run_id": run_id, "provider": provider, "model": model, "level": level, "data_seed": seed, "rep": rep, "status": status, "turns": len(codes),
               "tokens_in": tin, "tokens_out": tout, "seconds": time.time() - t0, "code": "\n\n# ---- next step ----\n".join(codes), **res}, open(out, "w"), indent=1)
    print(run_id, res.get("ok"), res.get("auc"), flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--provider", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--levels", nargs="+", default=["S1", "S3", "S3p"]); ap.add_argument("--seeds", nargs="+", type=int, default=[1]); ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    for seed in a.seeds:
        for level in a.levels:
            for rep in range(1, a.reps + 1): run_one(a.provider, a.model, level, seed, rep)
