# credit-registry-agent-benchmark

Code and logs for the paper *Can an LLM Agent Reproduce an Expert Credit-Risk Analytics Pipeline? An Evaluation Protocol and Synthetic Benchmark for Registry-Scale Analytics* (F. Doko, B. Fetaji).

**Status.** The feasibility pilot with scripted pipelines is complete and its raw logs are included. The LLM-agent study is planned future work: the agent harness is provided and was tested offline with a mock agent, but **no LLM agent results exist yet**.

## Contents
| Path | What it is |
|---|---|
| `gen.py` | Synthetic credit-registry generator, conditioned only on aggregates published in Doko, Kalajdziski, Mishkovski (2021), J. Risk Financ. Manag. 14(3):138. Everything else (age law, loan purposes, delay-day laws, coefficients) is an explicit assumption in the code. Exports the true default probability (oracle). |
| `pipe.py` | Reduced reference pipeline (LR, DT, RF; information-value selection; SQL statistics in DuckDB) and four scripted variants (canary used, median + winsorise, selection before split, oversample before split). |
| `run_pilot.py` | Runs calibration, the main pilot (5 data seeds x 5 replicate seeds), the 1e6-row tier and the 1e7-row SQL step. |
| `analyze.py` | Produces `summary.json` from `results.jsonl` (the numbers in the paper's pilot section). |
| `results.jsonl`, `summary.json` | Raw run logs and summary of the pilot reported in the paper. |
| `agents/` | Agent harness: task specifications S1/S2/S3/S3', LLM adapters, runner, evaluator, audit template, aggregation, and `README_RUN.md` (step-by-step guide). |

## Reproduce the pilot
    pip install numpy pandas scikit-learn scipy duckdb pyarrow
    python run_pilot.py     # about 20 minutes on one CPU core
    python analyze.py

## Run the agent study (future work)
See `agents/README_RUN.md`. Start with the offline smoke test (`--provider mock`) before using any API key.

## Limitations
Synthetic data (no real registry records are included or needed); reduced reference (no SVM, neural network or SMOTE comparison); scripted variants, not agents; the harness executes model-written code on the host machine and is not a hardened sandbox.

## Citation
Please cite the paper once published. License: MIT.
