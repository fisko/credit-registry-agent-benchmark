"""Task specification texts S1, S2, S3, S3' (kept identical across models; edit only before the study starts)."""
DICT = """Files in the working directory:
  train.parquet          labelled loans (column `risky` is the target: 1 = risky borrower, 0 = reliable borrower)
  test_features.parquet  loans to score (no `risky` column)
Columns:
  client_id        loan identifier (do not use as a feature)
  bank_size        E / G / M / S   (size class of the reporting bank)
  loan_type        A (annuity) / E (single repayment)
  interest_type    P (adjustable) / F (fixed) / V (variable)
  purpose          p1 .. p6        (loan purpose code)
  n_loans          number of loans of the client in the reporting period
  duration_years   loan duration in years (may be missing)
  paid_loans       number of loans successfully paid in the client's history
  age              client age in years (may contain data-entry errors)
  interest_rate    interest rate in percent
  delay_days       days of payment delay
  collection_flag  0/1 status flag from the collections system
  risky            target (train only)
You can read the files with pandas or DuckDB. The data come from a credit registry; there are no other tables."""

CONTRACT = """Deliverables (write them into the working directory):
  predictions.csv  columns client_id,score  -- one row for every row of test_features.parquet; score = predicted probability that risky = 1
  manifest.json    {"decisions": {"D1": "...", ..., "D11": "..."}, "features_used": [...], "leakage_columns_flagged": [...], "validation_auc": <float>, "notes": "..."}
                   The decisions object must describe, in one short phrase each, your choice for: D1 target definition, D2 missing values, D3 outliers,
                   D4 derived features, D5 discretisation, D6 scaling, D7 feature selection, D8 split and validation, D9 class imbalance handling,
                   D10 model family, D11 evaluation metrics. features_used lists the raw columns your final model uses. leakage_columns_flagged lists any column you judged unsafe to use.
Working method: reply with ONE complete Python script in a ```python block per turn; it is executed in the working directory and you receive its output.
Scripts must be self-contained (re-read the data each time). When predictions.csv and manifest.json are written and you are satisfied, reply with the single word DONE.
You may not look at any file outside the working directory and you do not have access to the test labels."""

S1 = """You are a data scientist at a central bank. Build a probability-of-default model from the credit-registry extract described below and score the test loans.

""" + DICT + "\n\n" + CONTRACT

S2 = S1 + """

Protocol requirements: the target is `risky` as provided. Report your validation discipline in the manifest and use a validation set that is separate from anything used to fit or select features.
Use AUC, KS and PR-AUC (not accuracy alone) to compare models, and report the validation AUC of your final model as validation_auc."""

S3 = S2 + """

Required methodology (follow it as specified):
  1. Data preparation with SQL (for example DuckDB) on the registry extract.
  2. Missing values: replace missing numeric values with the mode of the training data. Outliers: remove training rows outside 1.5 x IQR of the quartiles for age, duration_years and interest_rate.
  3. Discretisation: quantile binning, 20 bins for age and 4 bins for interest_rate.
  4. Feature selection: compute information value for each candidate feature and keep those with IV >= 0.02; then drop one of any pair of features with absolute correlation above 0.8.
  5. Models: compare logistic regression, decision tree and random forest on an 80/20 hold-out taken from the training data; keep the best by AUC.
  6. Fit all preprocessing, binning and selection on the training part only. Score the test loans with the final model refitted on all training data."""

S3P = S2 + """

Required methodology (follow it as specified):
  1. Data preparation with SQL (for example DuckDB) on the registry extract.
  2. Missing values: replace missing numeric values with the MEDIAN of the training data. Outliers: winsorise age, duration_years and interest_rate at the 1st and 99th percentiles (do not remove rows).
  3. Discretisation: quantile binning, 10 bins for age and 4 bins for interest_rate.
  4. Feature selection: compute information value for each candidate feature and keep those with IV >= 0.05; then drop one of any pair of features with absolute correlation above 0.8.
  5. Models: compare logistic regression, decision tree and random forest on an 80/20 hold-out taken from the training data; keep the best by AUC.
  6. Fit all preprocessing, binning and selection on the training part only. Score the test loans with the final model refitted on all training data."""

SPECS = {"S1": S1, "S2": S2, "S3": S3, "S3p": S3P}
