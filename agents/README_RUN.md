# Running the agent pilot (S1 / S3 / S3') -- step by step

All commands are run from the repository's `agents/` folder. Python 3.10+.

## 0. Set up (5 min)
    pip install numpy pandas scikit-learn scipy duckdb pyarrow openai anthropic
Use a disposable machine or container: the harness executes model-written Python on that machine (it is NOT a hardened sandbox).

## 1. Offline smoke test (2 min) -- no API key needed
    python harness.py --provider mock --model mock-ref    --levels S3 --seeds 1 --reps 1
    python harness.py --provider mock --model mock-canary --levels S1 --seeds 1 --reps 1
    python aggregate.py
Expect AUC about 0.846 for mock-ref and about 0.990 (above oracle, canary used) for mock-canary. Then delete the test outputs:
    rm -rf runs work audit.csv agent_summary.csv      (Windows: rmdir /s runs work)

## 2. Configure your models (10 min)
Azure AI Foundry / OpenAI-compatible (Responses API):
    export LLM_BASE_URL="https://<your-resource>.openai.azure.com/openai/v1/"
    export LLM_API_KEY="<key>"
    (if your deployment only supports chat completions:  export LLM_API=chat)
Anthropic (optional third model):   export ANTHROPIC_API_KEY="<key>"
Model names are your DEPLOYMENT names (for Foundry) or model ids. Write down exact model ids, provider and the date -- the paper must report them.
Do one live test run first:
    python harness.py --provider openai --model <deployment> --levels S1 --seeds 1 --reps 1
Open runs/<id>.json: check `ok`, `auc`, `turns`, tokens, and read `code`. If it failed for an infrastructure reason (key, quota, endpoint), fix and delete that json; do not change the prompts.

## 3. Pre-build the data once (1 min)
    python -c "import harness; [harness.prepare(s) for s in (1,2,3)]"
(Do this before launching parallel processes so they do not race.)

## 4. Run the study
Minimum viable (about 2-3 hours, ~30 runs): two models x {S1, S3, S3p} x 5 repetitions x data seed 1
    python harness.py --provider openai --model <modelA> --levels S1 S3 S3p --seeds 1 --reps 5
    python harness.py --provider openai --model <modelB> --levels S1 S3 S3p --seeds 1 --reps 5
Open one terminal per model (or per level) to run in parallel. Better if time allows: add `--seeds 1 2 3` (more data seeds), `--levels S1 S2 S3 S3p`.
If you have only about 1 hour: one model, `--levels S1 S3 S3p --reps 3` (9 runs).
Rules that keep the study honest:
 - Do not edit specs.py after the first real run.
 - Report ALL runs. A failed run (no valid predictions) counts as not accepted; re-run only for infrastructure errors (API/quota), and note it.
 - Do not tune prompts per model. Same budget (max 8 turns, 600 s per script) for every condition.

## 5. Audit decisions (30-60 min)
    python audit_make.py
Open audit.csv. For each run read `code` + `manifest` in runs/<id>.json and fill D2..D11 with:
  ref = same as the reference choice, adm = different but admissible (see Table 1 in the paper), bad = not admissible
(D1 is pre-filled: the target is provided). Examples: median instead of mode imputation = adm; fitting selection or scaling on data that includes the validation rows = bad; using collection_flag as a feature = D4 bad; no held-out validation = D8 bad.
Have a second person (e.g. your co-author) code a copy of the file independently; agreement = share of identical cells. Report it.

## 6. Aggregate and send me the results
    python aggregate.py
This writes agent_summary.csv (per model x level: runs, ok, AUC mean/SD, difference to reference, share within the 0.005 margin, runs above the oracle bound, canary used / flagged, feature-set Jaccard, F_strict, F_adm, decision agreement TAR, tokens, seconds, turns).
Send me agent_summary.csv, the audit agreement figure, and 2-3 qualitative notes (e.g. did S3 runs actually follow the IV threshold? did any agent mention collection_flag as risky?). I will write the agent-pilot section, update the abstract and report H1-H6 as supported / not supported / not tested.

## What each hypothesis needs
H1 (gap shrinks S1->S3): AUC difference by level.   H2 (adm. fidelity vs strict): audit.csv.   H3 (stability): feat_jaccard, TAR.
H4 (canary detection): canary_used / canary_flagged by level.   H5 (S3 vs S3p fidelity): needs the audit for S3 vs S3p.   H6 (cost): tokens, seconds per accepted run.
With the minimum plan you can speak to H1, H3, H4, H5 and H6 descriptively; say clearly that n is small.
