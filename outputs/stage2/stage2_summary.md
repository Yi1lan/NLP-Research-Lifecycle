# Stage 2 Summary: Data Acquisition, Exploration, and Preprocessing

## Data Acquisition and Preprocessing Overview

- Raw dataset downloaded to: `data/raw/dontpatronizeme_pcl.tsv`
- Practice splits downloaded to: `data/raw/practice_splits/`
- Official test set downloaded to: `data/raw/task4_test.tsv`
- Preprocessing summary JSON: `data/processed/preprocessing_summary.json`
- Rows removed during cleanup: **1**
- Duplicate paragraph IDs detected: **0**

---

## EDA Technique 1: Class Distribution

### Visual/Tabular Evidence

- Figure: `outputs/stage2/figures/class_distribution.png`
- Table: `outputs/stage2/tables/class_distribution.csv`

### Analysis

- Train positive (PCL) rate: **9.48%**
- Dev positive (PCL) rate: **9.51%**
- The class ratio is imbalanced toward **No PCL**, so accuracy alone would be misleading.

### Impact Statement

- Use class-sensitive evaluation (positive-class F1 as the primary metric).
- Prefer class-weighted loss and/or sampling strategies in Stage 3/4 to improve recall on the PCL class.

---

## EDA Technique 2: Token-Length Profiling

### Visual/Tabular Evidence

- Figure: `outputs/stage2/figures/length_distribution.png`
- Table: `outputs/stage2/tables/length_summary.csv`

### Analysis

- Train p95 token length: **114.00**
- Mean tokens (Train, PCL): **53.52**
- Mean tokens (Train, No PCL): **48.17**
- Length spread indicates that truncation settings will materially affect model behavior.

### Impact Statement

- Set model `max_length` using observed percentile statistics instead of arbitrary defaults.
- Evaluate truncation trade-offs during Stage 4 tuning to avoid losing critical context in long examples.
