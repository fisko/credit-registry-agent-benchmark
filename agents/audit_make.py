"""Create audit.csv: one row per run, columns D1..D11. Two raters fill each cell with ref / adm / bad (read the run's `code` and manifest).
ref = same as reference choice; adm = different but admissible (see Table 1 of the paper); bad = not admissible. D1 is pre-filled (target is provided)."""
import glob, json, csv, os
rows = [json.load(open(f)) for f in sorted(glob.glob("runs/*.json"))]
with open("audit.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["run_id"] + [f"D{i}" for i in range(1, 12)])
    for r in rows: w.writerow([r["run_id"], "ref"] + [""] * 10)
print("wrote audit.csv with", len(rows), "runs; fill D2..D11 with ref/adm/bad (and have a second rater fill a copy for agreement)")
