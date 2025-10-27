# Storico — SAP Payment Pattern Analytics Platform

**Executive Summary**: End-to-end ML platform for analyzing client payment behavior from SAP accounting exports. Combines batch data pipelines (ingestion → feature engineering → training) with an interactive Streamlit dashboard for pattern visualization, ranking metrics, SHAP explainability, clustering, and policy simulation. Built for credit analysts and data scientists to understand "which invoices will clients pay next" and optimize collection strategies.

**Key Capabilities**:
- Ingest multi-file SAP CSV/XLSX exports with automatic schema normalization
- Generate pointwise training data (invoice-event pairs) with balanced negative sampling
- Train CatBoost/AutoGluon binary classifiers with temporal validation splits
- Compute 10+ heuristic payment policies (FIFO, LIFO, same_period, eomfirst, largest, etc.)
- Interactive Pattern Card UI: calendar heatmaps, ACF plots, SHAP top-3 features, what-if policy gating
- KPI Dashboard: precision@k, hit rate, MRR, calibration curves (Brier/ECE/MCE)
- HDBSCAN clustering on client embeddings with interactive parameter tuning
- Mock data generator for sandbox/demo environments (50 synthetic clients)
- Deep-link URL sharing for filters/page state restoration

**Target Users**: Credit managers, data scientists, ML engineers, DevOps (deployment)

---

## Quickstart

### Requirements
- **OS**: Windows 10/11, Linux (Ubuntu 20.04+), macOS 11+
- **Python**: 3.11+ (tested with 3.11.5)
- **RAM**: 8GB minimum, 16GB recommended for AutoGluon training
- **GPU**: Optional (CatBoost supports GPU, AutoGluon CPU-only)
- **Storage**: ~2GB for dependencies, ~500MB for data/models (varies with dataset size)

### Setup

```powershell
# 1. Clone repository
git clone https://github.com/adrian-iancu_ktcsrl/storico.git
cd storico

# 2. Create virtual environment (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# On Linux/macOS:
# python3 -m venv .venv
# source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify installation
python -m compileall src app
python --version  # Should show 3.11+
pip list | Select-String "catboost|streamlit|pandas"
```

### Run (Quick)

**Option A: Use mock data (no SAP files needed)**
```powershell
# Generate synthetic dataset
python -m scripts.make_mock_data --output data_mock/raw/sap_mock.xlsx

# Run full pipeline with mock data
python -m src.pipeline.run_all --input data_mock/raw/*.xlsx --model catboost

# Launch Streamlit dashboard
streamlit run app/streamlit_app.py
# Navigate to http://localhost:8501, enable "Mock Mode" toggle in sidebar
```

**Option B: Use real SAP exports**
```powershell
# 1. Place CSV/XLSX files in data/raw/ folder
# 2. Run pipeline
python -m src.pipeline.run_all --input data/raw/*.csv --model catboost

# 3. Launch dashboard
streamlit run app/streamlit_app.py
# Access at http://localhost:8501
```

**Run Individual Steps** (for debugging):
```powershell
# Ingestion only
python -m src.data.ingest_multi --input data/raw/*.csv --output data/curated/line_items.parquet

# Feature engineering
python -m src.features.build_events
python -m src.features.make_labels
python -m src.features.make_features

# Training
python -m src.models.train --model catboost --out models/

# Analytics
python -m src.analytics.pattern_detectors
python -m src.analytics.kpi_ranking
```

---

## Project Structure (Directory Tree)

```
storico/
├── data/                      # Production data directory
│   ├── raw/                   # SAP exports (CSV/XLSX) - NOT committed
│   └── curated/               # Parquet artifacts (events, features, predictions)
├── data_mock/                 # Mock mode isolated data (synthetic dataset)
│   ├── raw/                   # Generated sap_mock.xlsx
│   └── curated/               # Mock mode parquet outputs
├── models/                    # Trained model artifacts
│   ├── catboost.cbm          # CatBoost binary classifier
│   ├── calibrator.pkl        # Isotonic calibrator
│   ├── autogluon/            # AutoGluon model directory
│   └── champion.json         # Model metadata (AUC-PR, match rate, seed)
├── logs/                      # JSONL logs (run_YYYYMMDD_HHMMSS.jsonl)
├── reports/                   # HTML/PDF pattern reports (future)
├── src/                       # Backend Python package
│   ├── data/                  # Ingestion modules
│   │   ├── ingest.py         # Single-file CSV/XLSX ingestion
│   │   └── ingest_multi.py   # Multi-file ingestion with deduplication
│   ├── features/              # Feature engineering
│   │   ├── build_events.py   # Event aggregation (clearing events)
│   │   ├── make_labels.py    # Pointwise labels (positive/negative sampling)
│   │   └── make_features.py  # Feature engineering (800+ lines, calendar/lags/ratios)
│   ├── models/                # Training modules
│   │   ├── train.py          # Unified training orchestrator
│   │   ├── train_catboost.py # CatBoost pipeline
│   │   └── train_autogluon.py# AutoGluon pipeline
│   ├── analytics/             # Post-training analytics
│   │   ├── pattern_detectors.py # Policy recall computation (FIFO/LIFO/eomfirst/etc.)
│   │   ├── kpi_ranking.py    # Ranking metrics (precision@k, MRR, nDCG)
│   │   ├── calibration.py    # Calibration diagnostics (Brier/ECE/MCE)
│   │   ├── make_client_embeddings.py # PCA client embeddings
│   │   ├── run_hdbscan.py    # HDBSCAN clustering
│   │   ├── cluster_profile.py # Cluster profiling
│   │   └── rules_backtest.py # Policy backtest simulator
│   ├── report/                # NLG report generation
│   │   ├── nlg_patterns.py   # Pattern text summarization
│   │   └── make_report.py    # HTML report builder
│   ├── pipeline/              # Orchestration
│   │   └── run_all.py        # Unified CLI runner (ingest → train → analytics)
│   └── utils/                 # Utilities
│       └── logging_setup.py  # JSONL logging config
├── app/                       # Streamlit UI
│   ├── streamlit_app.py      # Main entry point (2049 lines)
│   ├── page_components/       # UI modules
│   │   ├── pattern_card_v2.py # Pattern Card page (988 lines)
│   │   ├── kpi_dashboard.py  # KPI Dashboard (442 lines)
│   │   ├── clusters.py       # Clusters page (692 lines)
│   │   └── policy_iq.py      # Policy IQ simulator
│   └── utils/
│       └── export.py         # CSV export, deep-link generation, JSON serialization
├── scripts/                   # Utilities
│   └── make_mock_data.py     # Mock data generator (227 lines)
├── tests/                     # Unit tests
│   └── test_mock_isolation.py # Mock mode isolation tests
├── docs/                      # Documentation supplements
├── requirements.txt           # Python dependencies
├── copilot-instructions.md    # Agent prompt rules
└── README.md                  # This file
```

### Key Directories Explained

| Path | Purpose | Critical Files |
|------|---------|----------------|
| `data/curated/` | Parquet artifacts (events, features, predictions) | `line_items.parquet`, `train_features.parquet`, `per_event_policy_scores.parquet` |
| `models/` | Trained models + metadata | `catboost.cbm`, `calibrator.pkl`, `champion.json` |
| `logs/` | JSONL execution logs | `run_YYYYMMDD_HHMMSS.jsonl` |
| `src/data/` | Ingestion (SAP CSV→Parquet) | `ingest_multi.py` |
| `src/features/` | Event/feature/label builders | `make_features.py` (853 lines) |
| `src/models/` | Training pipelines | `train.py`, `train_catboost.py` |
| `src/analytics/` | Policy scoring, ranking, calibration | `pattern_detectors.py`, `kpi_ranking.py` |
| `app/` | Streamlit dashboard | `streamlit_app.py`, `page_components/*.py` |
| `scripts/` | Utilities (mock data gen) | `make_mock_data.py` |

---

## Module Inventory & API Reference

### 1. Data Ingestion (`src/data/`)

#### `src/data/ingest.py`
**Purpose**: Single-file SAP export ingestion with schema normalization.

**Key Functions**:
- `normalise_dataframe(df: pd.DataFrame) -> pd.DataFrame` — Standardize column names, parse dates, validate schema
- `read_raw_file(path: Path) -> pd.DataFrame` — Read CSV/XLSX with encoding detection
- `collect_raw_files(pattern: str) -> List[Path]` — Glob expansion for input files
- `log_dataset_summary(df: pd.DataFrame) -> None` — Print row count, date range, client count

**CLI Entry Point**: 
```powershell
python -m src.data.ingest --input data/raw/estratto_2023.xlsx --output data/curated/line_items.parquet
```

**Flags**:
- `--input PATH` — Input CSV/XLSX file
- `--output PATH` — Output parquet path (default: `data/curated/line_items.parquet`)

---

#### `src/data/ingest_multi.py`
**Purpose**: Multi-file ingestion with deduplication across sources.

**Key Functions**:
- `parse_args(argv) -> argparse.Namespace` — CLI argument parser
- `run(patterns, output_path, docmap_path, dedupe_keys, keep_first) -> None` — Main execution logic

**CLI Entry Point**:
```powershell
python -m src.data.ingest_multi \
  --input data/raw/*.csv \
  --input data/raw/extra/*.xlsx \
  --output data/curated/line_items.parquet \
  --docmap data/curated/doc_map.parquet \
  --dedupe-on cliente numero_documento doc_pareggio data_documento
```

**Flags**:
- `--input PATTERN` — Glob pattern (repeatable)
- `--output PATH` — Destination parquet for line items
- `--docmap PATH` — Optional document map parquet
- `--dedupe-on COL1 COL2 ...` — Deduplication key columns
- `--keep-first` — Keep first occurrence (default: keep last)

**Output Schema** (`line_items.parquet`):
| Column | Type | Description |
|--------|------|-------------|
| `cliente` | string | Client ID (e.g., "100123") |
| `numero_documento` | string | Invoice number |
| `doc_pareggio` | string | Clearing document ID |
| `data_documento` | datetime64[ns] | Invoice date |
| `data_pareggio` | datetime64[ns] | Clearing date (null if open) |
| `data_registrazione` | datetime64[ns] | Registration date |
| `imp_div_int` | float64 | Amount in EUR |
| `tipo_documento` | string | Document type (DR/DZ/KR) |
| `periodo_contabile` | string | Fiscal period (1-12) |
| `esercizio` | string | Fiscal year |

---

### 2. Feature Engineering (`src/features/`)

#### `src/features/build_events.py`
**Purpose**: Aggregate line items into clearing events.

**Key Functions**:
- `build_event_key_column(line_items: pd.DataFrame) -> pd.Series` — Construct `event_key` = `Cliente|YYYY-MM-DD|DocPareggio`
- `compute_events(event_items: pd.DataFrame) -> pd.DataFrame` — Aggregate event-level metrics (amount, inter-event gap)

**CLI Entry Point**:
```powershell
python -m src.features.build_events
```

**Outputs**:
- `data/curated/event_items.parquet` — Line items with `event_key` column
- `data/curated/events.parquet` — Event-level aggregates

**Event Schema** (`events.parquet`):
| Column | Type | Description |
|--------|------|-------------|
| `cliente` | string | Client ID |
| `event_key` | string | Unique event identifier |
| `data_pareggio` | datetime64[ns] | Clearing date |
| `importo_evento` | float64 | Total event amount |
| `n_righe_evento` | int64 | Number of invoices in event |
| `inter_event_gap_days` | float64 | Days since previous event |

---

#### `src/features/make_labels.py`
**Purpose**: Generate pointwise training dataset with balanced negatives.

**Key Functions**:
- `build_pointwise_labels(project_root: Path) -> pd.DataFrame` — Create (invoice, event) pairs with labels
- `append_labels_for_event(event_row, customer_items, results)` — Sample negatives for single event

**Logic**:
1. For each event: all invoices in that event → `label=1` (positives)
2. All open invoices at event date NOT in event → candidates for `label=0` (negatives)
3. Sample up to 20 negatives per event, prioritizing same `tipo_documento` and `periodo_contabile`

**CLI Entry Point**:
```powershell
python -m src.features.make_labels
```

**Output**: `data/curated/train_pointwise.parquet`

**Schema**:
| Column | Type | Description |
|--------|------|-------------|
| `cliente` | string | Client ID |
| `event_key` | string | Event identifier |
| `numero_documento` | string | Invoice number |
| `label` | Int64 | 1=paid in event, 0=negative sample |
| *(all columns from line_items)* | — | Inherited from source |

**Acceptance Criteria**: Print `#eventi, #positivi, #negativi, ratio` (expect ratio ~15-20)

---

#### `src/features/make_features.py`
**Purpose**: Comprehensive feature engineering (853 lines, 50+ features).

**Key Functions**:
- `load_curated_datasets(project_root) -> Dict[str, pd.DataFrame]` — Load train_pointwise, events, event_items
- `_compute_client_cadence_windows(events, window_days) -> pd.DataFrame` — Rolling 180-day cadence ratios (weekly/EOM)
- `_build_temporal_features(train, events) -> pd.DataFrame` — Calendar features (EOM, quarter-end, DOW, lag_reg_vs_doc)
- `_build_amount_features(train, events) -> pd.DataFrame` — Amount ratios, shares, Gini coefficient
- `_build_lag_features(train) -> pd.DataFrame` — Historical lags (docnum delta, inter-event gaps)
- `run_pipeline(project_root) -> None` — Orchestrate all feature engineering

**Feature Categories**:
1. **Calendar Features** (10 features):
   - `is_eom_event` — Event within 3 days of month-end
   - `days_to_eom` — Days to next month-end
   - `is_quarter_end_event` — Event within 3 days of quarter-end
   - `dow_event` — Day of week (0=Monday)
   - `same_fiscal_year`, `same_quarter_as_event` — Invoice/event alignment
   - `weekly_cadence_client`, `eom_cadence_client` — Rolling 180-day ratios

2. **Amount Features** (4 features):
   - `amt_ratio_B` — Invoice amount / event amount
   - `amt_share_rank_event` — Rank of invoice amount within event
   - `event_gini_amounts_open` — Gini coefficient of open amounts at event date
   - `event_amount_scale` — Log10(event amount)

3. **Lag Features** (3 features):
   - `age_days` — Invoice age at event date
   - `lag_reg_vs_doc` — Days between registration and document date
   - `docnum_delta_prev` — Document number delta vs previous invoice

4. **Categorical Features**:
   - `tipo_documento` — Document type (DR/DZ/KR)
   - `periodo_contabile` — Fiscal period (1-12)
   - `same_period` — Invoice period == event period (binary)

**CLI Entry Point**:
```powershell
python -m src.features.make_features
```

**Output**: `data/curated/train_features.parquet` (all features + label)

---

### 3. Model Training (`src/models/`)

#### `src/models/train.py`
**Purpose**: Unified training orchestrator with champion metadata.

**Key Functions**:
- `parse_args(argv) -> argparse.Namespace` — CLI parser
- `run_catboost(project_root, train_features_path, output_dir) -> ChampionMetrics` — CatBoost pipeline wrapper
- `run_autogluon(project_root, train_features_path, output_dir) -> ChampionMetrics` — AutoGluon pipeline wrapper
- `persist_champion_metadata(metadata, output_dir)` — Write `champion.json`

**CLI Entry Point**:
```powershell
# Train CatBoost
python -m src.models.train --model catboost --out models/ --seed 42

# Train AutoGluon
python -m src.models.train --model autogluon --out models/ --seed 42
```

**Flags**:
- `--model {catboost|autogluon}` — Model family (required)
- `--train-features PATH` — Override default train_features.parquet
- `--out PATH` — Output directory for models (default: `models/`)
- `--seed INT` — Random seed (default: 42)

**Outputs**:
- `models/catboost.cbm` — Trained CatBoost model
- `models/calibrator.pkl` — Isotonic calibrator (sklearn)
- `models/champion.json` — Metadata JSON

**Champion JSON Schema**:
```json
{
  "model": "catboost",
  "trained_at": "2025-10-27T14:23:45Z",
  "seed": 42,
  "split": {"train_cutoff": "2024-06-30", "test_cutoff": "2024-12-31"},
  "metrics": {
    "auc_pr": 0.847,
    "match_rate": 0.723
  }
}
```

---

#### `src/models/train_catboost.py`
**Purpose**: CatBoost binary classifier with calibration.

**Key Functions**:
- `load_curated_tables(project_root) -> Dict[str, pd.DataFrame]` — Load train_features, events, event_items
- `temporal_split(features, events) -> Tuple[pd.DataFrame, pd.DataFrame]` — Split on event dates (60/40)
- `select_features(df) -> Tuple[pd.DataFrame, pd.Series, List[str], List[int]]` — Feature/target separation, categorical indices
- `train_model(X, y, cat_indices, split) -> catboost.CBoostClassifier` — Train with early stopping
- `calibrate_probabilities(model, X, y, cat_indices, split) -> IsotonicRegression` — Fit calibrator
- `evaluate_invoice_metrics(probs, targets) -> Dict` — Compute AUC-PR, precision@k
- `evaluate_event_metrics(test_df, probs) -> Tuple[pd.DataFrame, pd.DataFrame]` — Per-event and per-client match rates

**CatBoost Hyperparameters**:
- `iterations=1000`, `learning_rate=0.05`, `depth=6`, `l2_leaf_reg=3`
- `border_count=254`, `early_stopping_rounds=50`
- `loss_function="Logloss"`, `eval_metric="AUC"`

**Temporal Split Logic**:
- Sort events by `data_pareggio`
- Train: first 60% of events (chronological)
- Test: last 40% of events
- Prevents leakage: test events always occur AFTER train events

**Calibration**:
- Fit `IsotonicRegression(out_of_bounds="clip")` on validation fold
- Input: raw CatBoost probabilities
- Output: calibrated probabilities (better reliability)

**CLI Entry Point** (via `train.py`):
```powershell
python -m src.models.train --model catboost
```

---

#### `src/models/train_autogluon.py`
**Purpose**: AutoGluon TabularPredictor with ensemble.

**Key Functions**:
- `run_autogluon_training(project_root, output_dir, seed) -> ChampionMetrics` — Main pipeline
- `_load_and_prepare(project_root) -> Tuple[pd.DataFrame, pd.DataFrame]` — Load + temporal split
- `_fit_predictor(train_df, output_dir) -> TabularPredictor` — Fit AutoGluon with preset="medium_quality"

**AutoGluon Config**:
- `label="label"`, `problem_type="binary"`, `eval_metric="average_precision"`
- `preset="medium_quality"` (trains GBM, NN, RF, ensemble)
- `time_limit=3600` (1 hour)

**CLI Entry Point** (via `train.py`):
```powershell
python -m src.models.train --model autogluon
```

**Outputs**:
- `models/autogluon/` — Full AutoGluon model directory
- `models/champion.json` — Updated with AutoGluon metrics

---

### 4. Analytics (`src/analytics/`)

#### `src/analytics/pattern_detectors.py`
**Purpose**: Compute policy recall scores for 10+ heuristic strategies.

**Key Functions**:
- `prepare_doc_map(line_items) -> pd.DataFrame` — Enrich with amount ranks, sequence blocks
- `compute_policy_rankings(cliente, event_key, event_date, doc_map) -> PolicyRanking` — Score all policies
- `_score_fifo(candidates, event_date) -> List[str]` — FIFO (oldest first) ranking
- `_score_lifo(candidates, event_date) -> List[str]` — LIFO (newest first)
- `_score_same_period(candidates, event_period) -> List[str]` — Same fiscal period
- `_score_largest(candidates) -> List[str]` — Largest amount first
- `_score_eomfirst(candidates) -> List[str]` — EOM invoices first
- `run_policy_detection(project_root) -> None` — Compute per-event recalls, save to parquet

**Policy Definitions**:
| Policy | Description | Ranking Logic |
|--------|-------------|---------------|
| `fifo` | First-In First-Out | Sort by `data_documento` ASC |
| `lifo` | Last-In First-Out | Sort by `data_documento` DESC |
| `same_period` | Same fiscal period | Filter `periodo_contabile` == event period, then FIFO |
| `largest` | Largest amount first | Sort by `abs(imp_div_int)` DESC |
| `smallest` | Smallest amount first | Sort by `abs(imp_div_int)` ASC |
| `eomfirst` | EOM invoices first | Filter invoices within 3 days of month-end, then FIFO |
| `regmonth` | Registration month match | Filter `data_registrazione.month` == `data_pareggio.month` |
| `quarter` | Same quarter | Filter invoices in same quarter as event |
| `seqblock` | Sequential docnum blocks | Prioritize consecutive document numbers (≥3) |
| `typeprio` | Dominant type priority | Prioritize most frequent `tipo_documento` in last 90 days |

**Recall Calculation**:
- For each event: compute top-k ranking per policy
- Recall@k = |intersection(predicted_top_k, actual_paid)| / |actual_paid|
- Store recalls in `per_event_policy_scores.parquet`

**CLI Entry Point**:
```powershell
python -m src.analytics.pattern_detectors
```

**Output**: `data/curated/per_event_policy_scores.parquet`

**Schema**:
| Column | Type | Description |
|--------|------|-------------|
| `cliente` | string | Client ID |
| `event_key` | string | Event identifier |
| `data_evento` | datetime64[ns] | Event date |
| `fifo_score` | float64 | FIFO recall@k |
| `lifo_score` | float64 | LIFO recall@k |
| *(9 more policy columns)* | — | Recall scores [0.0, 1.0] |

---

#### `src/analytics/kpi_ranking.py`
**Purpose**: Compute ranking metrics (precision@k, MRR, nDCG) for model predictions.

**Key Functions**:
- `compute_ranking_metrics(predictions, ground_truth, k_values) -> Dict` — Precision@k, recall@k, nDCG@k, MRR
- `aggregate_by_segment(metrics_df, segment_key) -> pd.DataFrame` — Group by client/cardinality/cluster
- `run_kpi_pipeline(project_root) -> None` — Compute and save all ranking KPIs

**Metrics**:
- **Precision@k**: Fraction of top-k predictions that are correct
- **Recall@k (Hit Rate)**: Fraction of ground-truth items found in top-k
- **Coverage@k**: Fraction of events with ≥1 correct item in top-k
- **nDCG@k**: Normalized Discounted Cumulative Gain (position-aware)
- **MRR**: Mean Reciprocal Rank (average 1/rank of first correct item)

**CLI Entry Point**:
```powershell
python -m src.analytics.kpi_ranking
```

**Output**: `data/curated/kpi_ranking.parquet`

**Schema**:
| Column | Type | Description |
|--------|------|-------------|
| `aggregation_key` | string | Segment type (global/cliente/cardinality_bucket) |
| `aggregation_value` | string | Segment value (e.g., "C001", "k=2") |
| `n_events` | int64 | Number of events in segment |
| `precision@1` | float64 | Precision at rank 1 |
| `precision@3` | float64 | Precision at rank 3 |
| `hit_rate@3` | float64 | Recall at rank 3 |
| `coverage@3` | float64 | Coverage at rank 3 |
| `ndcg@3` | float64 | nDCG at rank 3 |
| `mrr` | float64 | Mean Reciprocal Rank |

---

#### `src/analytics/calibration.py`
**Purpose**: Calibration diagnostics (Brier score, ECE, MCE, reliability curves).

**Key Functions**:
- `compute_calibration_metrics(y_true, y_pred, n_bins=10) -> Dict` — Brier/ECE/MCE scores
- `compute_reliability_curve(y_true, y_pred, n_bins=10) -> pd.DataFrame` — Binned calibration data
- `run_calibration_pipeline(project_root) -> None` — Compute calibration for all models/segments

**Metrics**:
- **Brier Score**: MSE of predicted probabilities vs actual labels (lower = better)
- **ECE (Expected Calibration Error)**: Weighted average |confidence - accuracy| across bins
- **MCE (Max Calibration Error)**: Maximum |confidence - accuracy| across bins

**CLI Entry Point**:
```powershell
python -m src.analytics.calibration
```

**Outputs**:
- `data/curated/calibration_summary.parquet` — Per-model/segment metrics
- `data/curated/calibration_bins.parquet` — Reliability curve data

---

#### `src/analytics/make_client_embeddings.py`
**Purpose**: Generate PCA embeddings from client policy profiles.

**Key Functions**:
- `compute_client_profiles(per_event_scores) -> pd.DataFrame` — Aggregate policy recalls per client
- `fit_pca_embeddings(profiles, n_components=20) -> pd.DataFrame` — PCA dimensionality reduction
- `run_embedding_pipeline(project_root) -> None` — Compute and save embeddings

**CLI Entry Point**:
```powershell
python -m src.analytics.make_client_embeddings
```

**Output**: `data/curated/client_embeddings.parquet` (cliente + 20 PCA dimensions)

---

#### `src/analytics/run_hdbscan.py`
**Purpose**: HDBSCAN clustering on client embeddings.

**Key Functions**:
- `run_hdbscan_clustering(embeddings, min_cluster_size=5, min_samples=3) -> np.ndarray` — Fit HDBSCAN
- `run_hdbscan_pipeline(project_root, min_cluster_size, min_samples) -> None` — Cluster and save

**CLI Entry Point**:
```powershell
python -m src.analytics.run_hdbscan --min-cluster-size 5 --min-samples 3
```

**Output**: `data/curated/client_clusters.parquet` (cliente + cluster_label)

---

### 5. Pipeline Orchestration (`src/pipeline/`)

#### `src/pipeline/run_all.py`
**Purpose**: Unified CLI to run entire pipeline (ingest → train → analytics).

**Key Functions**:
- `parse_args(argv) -> PipelineConfig` — Parse CLI flags
- `run_pipeline(config) -> None` — Execute all stages

**CLI Entry Point**:
```powershell
python -m src.pipeline.run_all \
  --input data/raw/*.csv \
  --model catboost \
  --make-embeddings \
  --run-hdbscan \
  --make-kpi \
  --make-calibration \
  --make-policy-metrics
```

**Flags**:
- `--input PATTERN` — Input file glob (repeatable)
- `--docmap PATH` — Optional document map output
- `--model {catboost|autogluon}` — Model to train
- `--make-embeddings` — Generate PCA embeddings
- `--run-hdbscan` — Run HDBSCAN clustering
- `--embedding-neighbors INT` — KNN neighbors for UMAP (default: 15)
- `--use-umap` — Use UMAP instead of PCA
- `--min-cluster-size INT` — HDBSCAN param (default: 5)
- `--min-samples INT` — HDBSCAN param (default: 3)
- `--make-kpi` — Compute ranking KPIs
- `--make-calibration` — Compute calibration metrics
- `--make-policy-metrics` — Compute policy confusion matrix

**Execution Order**:
1. Ingestion (`ingest_multi.run`)
2. Event building (`build_events.build_events_tables`)
3. Label generation (`make_labels.build_pointwise_labels`)
4. Feature engineering (`make_features.run_pipeline`)
5. Training (`train.run_{catboost|autogluon}`)
6. Policy detection (`pattern_detectors.run_policy_detection`)
7. (Optional) Embeddings + clustering
8. (Optional) KPI ranking
9. (Optional) Calibration
10. (Optional) Policy metrics

**Logging**: All stages log to `logs/run_YYYYMMDD_HHMMSS.jsonl`

---

### 6. Streamlit Dashboard (`app/`)

#### `app/streamlit_app.py`
**Purpose**: Main UI entry point with 5 pages (Pattern Card, KPI Dashboard, Data Explorer, Clusters, Policy IQ).

**Key Functions**:
- `_data_root() -> Path` — Return `data_mock/` if mock_mode active, else `data/`
- `_load_parquet(path) -> pd.DataFrame` — Cached parquet loader
- `_load_champion_metadata() -> Dict` — Read champion.json
- `_render_developer_panel(metrics) -> None` — Developer mode overlay (JSON log viewer, timing, memory)
- `main() -> None` — App entry point with query param parsing

**Pages** (sidebar navigation):
1. **Pattern Card** — Client-level pattern visualization
2. **KPI Dashboard** — Ranking/calibration metrics
3. **Data Explorer** — Raw data tables
4. **Clusters** — HDBSCAN cluster explorer
5. **Policy IQ** — Policy simulator

**Global Features**:
- **Mock Mode Toggle**: Switch between `data/` and `data_mock/` (preserves isolation)
- **Developer Mode**: Expander with JSON log tail, timing breakdown, memory usage
- **Deep-Link Support**: Parse `?page=X&cliente=Y` query params to restore state
- **Mock Data Generator Button**: "📦 Crea dataset demo" runs `make_mock_data.py` + ingestion

**Launch Command**:
```powershell
streamlit run app/streamlit_app.py
```

**Default Port**: `http://localhost:8501`

---

#### `app/page_components/pattern_card_v2.py`
**Purpose**: Pattern Card page (988 lines) — client pattern visualization with explainability.

**Components**:
1. **Client Selector** — Dropdown with search
2. **Pattern Summary Card** — Dominant policy, cardinality, periodicity badges
3. **Calendar Heatmap** — 12-month event calendar (day-of-week × month)
4. **Cadence Histogram** — Inter-event gap distribution with periodicity markers (7/14/30 days)
5. **ACF Plot** — Autocorrelation function to detect periodicity
6. **Explainability Panel** — SHAP top-3 features, what-if policy gating with α slider

**Key Functions**:
- `render_pattern_card_v2(data_root_func, hide_regmonth) -> None` — Main render function
- `plot_calendar_heatmap_12mo(events) -> go.Figure` — Plotly heatmap
- `plot_cadence_histogram_enhanced(gaps) -> go.Figure` — Histogram with vlines
- `plot_acf_enhanced(gaps, max_lag=10) -> go.Figure` — ACF bars + confidence bands
- `render_explainability_panel(event, model, features, hide_regmonth) -> None` — SHAP + what-if

**Data Inputs**:
- `data/curated/events.parquet` — Event history
- `data/curated/per_event_policy_scores.parquet` — Policy recalls
- `data/curated/client_profiles.parquet` — Aggregated client metrics
- `models/catboost.cbm` — Trained model for SHAP

**Columns Used**:
- Events: `cliente`, `data_pareggio`, `event_key`, `importo_evento`, `n_righe_evento`, `inter_event_gap_days`
- Policy scores: `fifo_score`, `lifo_score`, `same_period_score`, `eomfirst_score`, etc.

**Explainability**:
- **SHAP**: Use `model.get_feature_importance(type="ShapValues")` for top-3 features
- **What-If**: User adjusts policy boost factors (α) → recompute rank → show delta (green/red)
- **Regmonth Toggle**: Hide `regmonth` checkbox if `hide_regmonth=True` (global toggle)

---

#### `app/page_components/kpi_dashboard.py`
**Purpose**: KPI Dashboard (442 lines) — ranking and calibration metrics with filters.

**Tabs**:
1. **Ranking** — Precision@k, hit rate, MRR tables (global/cardinality/per-client)
2. **Calibration** — Brier/ECE/MCE cards, reliability curves (plotly line charts)

**Key Functions**:
- `render_kpi_dashboard_page(data_root_func) -> None` — Main render
- `_render_ranking_tab(kpi_df, filtered_df) -> None` — Ranking metrics display
- `_render_calibration_tab(calibration_summary, calibration_bins, filtered_summary) -> None` — Calibration curves

**Filters** (sidebar):
- Model selection (catboost/autogluon)
- Client multi-select
- Cardinality bucket (k=1/k=2/k≥3)
- Segment type (global/cliente/cardinality)

**Data Inputs**:
- `data/curated/kpi_ranking.parquet` — Ranking metrics
- `data/curated/calibration_summary.parquet` — Calibration metrics
- `data/curated/calibration_bins.parquet` — Reliability curve bins

**CSV Downloads**:
- "📥 Download Ranking CSV" — Filtered ranking data
- "📥 Download Calibration Summary CSV" — Calibration metrics
- "📥 Download Calibration Bins CSV" — Reliability curve data

---

#### `app/page_components/clusters.py`
**Purpose**: Clusters page (692 lines) — HDBSCAN cluster explorer with parameter tuning.

**Components**:
1. **UMAP/PCA 2D Scatter** — Client embeddings colored by cluster
2. **Cluster Profiles Table** — Dominant pattern, cardinality, periodicity per cluster
3. **Parameter Tuning Panel** (sidebar):
   - `min_cluster_size` slider (3-20)
   - `min_samples` slider (1-10)
   - "Preview" button (non-persistent, session state only)
   - "Apply & Save" button (persist to parquet)

**Key Functions**:
- `render_clusters_page(data_root_func) -> None` — Main render
- `_recompute_hdbscan(embeddings_df, min_cluster_size, min_samples) -> Tuple[np.ndarray, int, int]` — Run HDBSCAN
- `_generate_automatic_label(dominant_pattern, k_mode, periodicity_metrics) -> Tuple[str, float]` — Auto-label clusters

**Data Inputs**:
- `data/curated/client_embeddings.parquet` — PCA embeddings
- `data/curated/client_clusters.parquet` — Cluster assignments
- `data/curated/cluster_profiles.parquet` — Aggregated cluster metrics

**HDBSCAN Parameters**:
- `min_cluster_size`: Minimum points to form a cluster (default: 5)
- `min_samples`: Core point threshold (default: 3)
- `metric`: Distance metric (fixed: "cosine")

**Mock Mode Support**: Uses `data_root_func()` to resolve paths (data/ vs data_mock/)

---

#### `app/page_components/policy_iq.py`
**Purpose**: Policy IQ simulator — interactive what-if analysis for policy combinations.

**Features**:
- Select multiple policies (FIFO, LIFO, same_period, etc.)
- Adjust boost factors (α) for each policy
- Simulate ranking outcomes
- Compare hit rates across policy mixes

**Key Functions**:
- `render_policy_iq(data_root_func) -> None` — Main render
- `_simulate_policy_mix(policies, boosts, event_data) -> pd.DataFrame` — Compute combined scores

**Data Inputs**:
- `data/curated/per_event_policy_scores.parquet` — Policy recalls
- `data/curated/events.parquet` — Event metadata

---

### 7. Utilities (`scripts/`, `src/utils/`)

#### `scripts/make_mock_data.py`
**Purpose**: Generate synthetic SAP dataset for demo/sandbox.

**Key Functions**:
- `generate_mock_dataset(n_clients=50, min_months=12, max_months=18, seed=42) -> pd.DataFrame` — Generate full dataset
- `generate_client_data(client_id, n_months, cardinality, pattern_type, start_date, rng) -> pd.DataFrame` — Single client

**Synthetic Data Characteristics**:
- 50 clients (C001..C050)
- 12-18 months of history per client
- Mixed patterns: FIFO (33%), eomfirst (33%), same_period (33%)
- Cardinality k ∈ {1, 2, 3, 4, 5} (normal distribution, μ=3)
- Log-normal amounts: median ~1000 EUR, σ=500
- 85% paid, 15% open invoices
- Sequential document numbers (INV000001, INV000002, ...)

**CLI Entry Point**:
```powershell
python -m scripts.make_mock_data --output data_mock/raw/sap_mock.xlsx --n-clients 50 --seed 42
```

**Flags**:
- `--output PATH` — Output XLSX path (default: `data_mock/raw/sap_mock.xlsx`)
- `--n-clients INT` — Number of clients (default: 50)
- `--min-months INT` — Min months per client (default: 12)
- `--max-months INT` — Max months per client (default: 18)
- `--seed INT` — Random seed (default: 42)

**Integration with Streamlit**:
- Sidebar button "📦 Crea dataset demo" calls this script via `subprocess.run`
- After generation, runs `ingest_multi.run()` to populate `data_mock/curated/`

---

#### `src/utils/logging_setup.py`
**Purpose**: Centralized JSONL logging configuration.

**Key Functions**:
- `configure_logging(level=logging.INFO) -> Path` — Setup file + console handlers
- `get_current_log_file() -> Optional[Path]` — Return active log path
- `log_timed(stage_name) -> Callable` — Decorator for timing function execution

**Log Format**: JSONL (one JSON object per line)
```json
{"timestamp": "2025-10-27T14:23:45.123Z", "level": "INFO", "module": "src.features.make_features", "message": "Feature engineering complete", "duration_ms": 1234.56}
```

**Log Location**: `logs/run_YYYYMMDD_HHMMSS.jsonl`

---

## Data Flow Diagram (ASCII)

```
┌─────────────────────┐
│ SAP Exports (CSV)   │
│ data/raw/*.csv      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ INGESTION           │ src/data/ingest_multi.py
│ - Normalize schema  │ --input data/raw/*.csv
│ - Deduplicate       │ --output data/curated/line_items.parquet
│ - Parse dates       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ line_items.parquet  │ (Cliente, NumeroDocumento, DataDocumento,
│ data/curated/       │  DataPareggio, DocPareggio, ImpDivInt, etc.)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ EVENT BUILDING      │ src/features/build_events.py
│ - Aggregate by      │
│   (Cliente, Data-   │
│   Pareggio, Doc-    │
│   Pareggio)         │
└──────────┬──────────┘
           │
           ├─────────────────────────┐
           │                         │
           ▼                         ▼
┌──────────────────┐     ┌──────────────────┐
│ events.parquet   │     │ event_items.     │
│ (event-level     │     │ parquet          │
│  aggregates)     │     │ (line items +    │
└─────────┬────────┘     │  event_key)      │
          │              └─────────┬────────┘
          │                        │
          └────────────┬───────────┘
                       │
                       ▼
           ┌─────────────────────┐
           │ LABEL GENERATION    │ src/features/make_labels.py
           │ - Positive samples: │
           │   invoices in event │
           │ - Negative samples: │
           │   open invoices NOT │
           │   in event (max 20) │
           └──────────┬──────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │ train_pointwise.    │ (Cliente, EventKey, NumeroDoc,
           │ parquet             │  Label=0/1, all line_items cols)
           └──────────┬──────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │ FEATURE ENGINEERING │ src/features/make_features.py
           │ - Calendar (EOM,    │ - 50+ features
           │   DOW, quarter)     │ - 853 lines
           │ - Amount ratios     │
           │ - Lags, cadence     │
           └──────────┬──────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │ train_features.     │ (all features + label)
           │ parquet             │
           └──────────┬──────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │ MODEL TRAINING      │ src/models/train.py
           │ - Temporal split    │ --model catboost
           │   (60/40 by event   │ --out models/
           │   date)             │
           │ - CatBoost or       │
           │   AutoGluon         │
           │ - Isotonic calib.   │
           └──────────┬──────────┘
                      │
                      ├─────────────────────────┐
                      │                         │
                      ▼                         ▼
           ┌──────────────────┐     ┌──────────────────┐
           │ catboost.cbm     │     │ calibrator.pkl   │
           │ (trained model)  │     │ (Isotonic)       │
           └─────────┬────────┘     └─────────┬────────┘
                     │                        │
                     └────────────┬───────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │ champion.json       │ (model, trained_at,
                       │                     │  AUC-PR, match_rate)
                       └─────────┬───────────┘
                                 │
                                 ▼
                       ┌─────────────────────┐
                       │ ANALYTICS           │ src/analytics/
                       │ - Policy detection  │ pattern_detectors.py
                       │ - KPI ranking       │ kpi_ranking.py
                       │ - Calibration       │ calibration.py
                       │ - Embeddings        │ make_client_embeddings.py
                       │ - Clustering        │ run_hdbscan.py
                       └─────────┬───────────┘
                                 │
                                 ├────────┬────────┬────────┐
                                 │        │        │        │
                                 ▼        ▼        ▼        ▼
                  ┌──────────────┐ ┌─────┐ ┌─────┐ ┌─────────┐
                  │per_event_    │ │kpi_ │ │calib│ │client_  │
                  │policy_scores │ │rank │ │*    │ │clusters │
                  │.parquet      │ │.pqt │ │.pqt │ │.parquet │
                  └──────────────┘ └─────┘ └─────┘ └─────────┘
                                 │
                                 ▼
                       ┌─────────────────────┐
                       │ STREAMLIT DASHBOARD │ streamlit run
                       │ - Pattern Card      │ app/streamlit_app.py
                       │ - KPI Dashboard     │
                       │ - Clusters          │
                       │ - Policy IQ         │
                       └─────────────────────┘
```

---

## Configuration & Environment Variables

### Environment Variables (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_DATA_ROOT` | `data` | Root data directory (override for custom paths) |
| `STREAMLIT_SERVER_PORT` | `8501` | Streamlit server port |
| `STREAMLIT_SERVER_ADDRESS` | `localhost` | Streamlit bind address |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG/INFO/WARNING/ERROR) |

**Example `.env` file**:
```bash
APP_DATA_ROOT=data
STREAMLIT_SERVER_PORT=8501
LOG_LEVEL=INFO
```

### Mock Mode Path Resolution

**Logic** (`app/streamlit_app.py:_data_root()`):
- If `st.session_state["mock_mode"] == True`: return `data_mock/`
- Else: return `os.getenv("APP_DATA_ROOT", "data")`

**Impact**:
- All parquet loads use `_data_root() / "curated" / "filename.parquet"`
- Prevents mixing real and synthetic data
- HDBSCAN "Apply & Save" respects mock mode for persistence

---

## Testing

### Unit Tests

**Location**: `tests/test_mock_isolation.py`

**Run Tests**:
```powershell
pytest tests/ -v
```

**Coverage** (future):
```powershell
pytest --cov=src --cov=app --cov-report=html tests/
```

### Manual Testing Checklist

1. **Ingestion**:
   ```powershell
   python -m src.data.ingest_multi --input data/raw/*.csv
   # Verify: data/curated/line_items.parquet exists, row count printed
   ```

2. **Feature Engineering**:
   ```powershell
   python -m src.features.build_events
   python -m src.features.make_labels
   python -m src.features.make_features
   # Verify: events.parquet, train_pointwise.parquet, train_features.parquet exist
   ```

3. **Training**:
   ```powershell
   python -m src.models.train --model catboost
   # Verify: models/catboost.cbm, calibrator.pkl, champion.json exist
   # Check champion.json has auc_pr > 0.5
   ```

4. **Analytics**:
   ```powershell
   python -m src.analytics.pattern_detectors
   python -m src.analytics.kpi_ranking
   # Verify: per_event_policy_scores.parquet, kpi_ranking.parquet exist
   ```

5. **Streamlit**:
   ```powershell
   streamlit run app/streamlit_app.py
   # Navigate to http://localhost:8501
   # Test: Pattern Card loads, client selector works, charts render
   # Test: KPI Dashboard shows metrics, CSV download works
   # Test: Clusters page displays UMAP scatter, parameter sliders work
   # Test: Mock mode toggle switches data paths correctly
   ```

---

## Deployment

### Local Deployment (Production Mode)

```powershell
# 1. Prepare data
python -m src.pipeline.run_all --input data/raw/*.csv --model catboost --make-kpi --make-calibration

# 2. Launch Streamlit
streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

### Docker Deployment (Future)

**Note**: No Dockerfile currently in repo. Suggested template:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
```

**Build & Run**:
```powershell
docker build -t storico-app .
docker run -p 8501:8501 -v $(pwd)/data:/app/data storico-app
```

### Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name storico.company.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

---

## Observability & Troubleshooting

### Logging

**Log Format**: JSONL (logs/run_YYYYMMDD_HHMMSS.jsonl)

**View Logs**:
```powershell
# Tail latest log
Get-Content logs\run_*.jsonl -Tail 50

# Filter warnings/errors
Get-Content logs\run_*.jsonl | Select-String '"level":"(WARNING|ERROR)"'
```

**Log Levels**:
- `DEBUG`: Verbose (feature engineering details, row counts)
- `INFO`: Standard (pipeline stage completion, file saves)
- `WARNING`: Recoverable issues (missing columns, empty dataframes)
- `ERROR`: Failures (file not found, model load errors)

**Developer Mode** (Streamlit):
- Enable in sidebar: "⚙️ Developer Mode" expander
- Shows: JSON log tail (last 100 lines), timing breakdown (section durations), memory usage (dataframe sizes)

---

### Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `FileNotFoundError: data/curated/line_items.parquet` | Ingestion not run | Run `python -m src.data.ingest_multi --input data/raw/*.csv` |
| `KeyError: 'label'` in training | `make_labels.py` skipped | Run `python -m src.features.make_labels` before training |
| `ValueError: cat_features indices out of range` | Categorical columns missing | Check `train_features.parquet` has `tipo_documento`, `periodo_contabile` |
| `ImportError: No module named 'hdbscan'` | HDBSCAN not installed | Run `pip install hdbscan` (optional dependency) |
| `StreamlitAPIException: session_state['mock_mode'] not found` | First launch | Initialize mock_mode in sidebar toggle code |
| Streamlit "Connection error" | Port 8501 in use | Stop other Streamlit apps or use `--server.port 8502` |
| `champion.json` missing | Training not completed | Run `python -m src.models.train --model catboost` |
| Charts show "Nessun dato disponibile" | Empty parquet or wrong path | Check data_root (mock_mode toggle), verify parquet files exist |
| "Preview" button in Clusters page does nothing | HDBSCAN import failed | Install hdbscan: `pip install hdbscan` |

---

### Performance Tuning

**Ingestion** (large files >1GB):
- Use `--dedupe-on` with minimal columns (avoid `imp_div_int`)
- Consider chunked reading (modify `ingest_multi.py` to use `pd.read_csv(chunksize=10000)`)

**Feature Engineering** (>500k rows):
- Reduce `ROLLING_WINDOW_DAYS` from 180 to 90 in `make_features.py`
- Disable unused features (comment out in `run_pipeline()`)

**Training** (AutoGluon slow):
- Reduce `time_limit` in `train_autogluon.py` (default 3600s → 1800s)
- Use `preset="medium_quality"` instead of `"best_quality"`

**Streamlit** (slow dashboard):
- Increase `st.cache_data` TTL (add `ttl=3600` parameter)
- Filter large dataframes before display (use `st.dataframe(df.head(1000))`)
- Use `st.spinner()` for long operations

---

## Roadmap & Known Issues

### Known Issues
- [ ] AutoGluon GPU support not tested (CPU-only validated)
- [ ] UMAP embedding option in Clusters page requires manual install (`pip install umap-learn`)
- [ ] Deep-link restore for Clusters page parameters not implemented
- [ ] CSV exports from KPI Dashboard may timeout on very large datasets (>1M rows)
- [ ] Mock data generator does not simulate credit notes (DZ type underrepresented)

### Roadmap (Future Enhancements)
- [ ] Docker Compose setup with PostgreSQL backend
- [ ] CI/CD pipeline (GitHub Actions: lint → test → deploy)
- [ ] REST API wrapper for predictions (FastAPI)
- [ ] Real-time streaming ingestion (Kafka consumer)
- [ ] Multi-tenancy support (client data isolation)
- [ ] PDF report generation (replace HTML reports)
- [ ] A/B testing framework for policy evaluation
- [ ] Integration with SAP BW (direct database connector)

---

## References

**Core Dependencies**:
- [Pandas](https://pandas.pydata.org/docs/) — Data manipulation
- [CatBoost](https://catboost.ai/en/docs/) — Gradient boosting (categorical features)
- [AutoGluon](https://auto.gluon.ai/stable/index.html) — AutoML tabular models
- [Streamlit](https://docs.streamlit.io/) — Interactive dashboards
- [Plotly](https://plotly.com/python/) — Interactive charts
- [HDBSCAN](https://hdbscan.readthedocs.io/) — Density-based clustering
- [scikit-learn](https://scikit-learn.org/stable/) — IsotonicRegression calibration

**Algorithms & Concepts**:
- [Isotonic Regression (Calibration)](https://scikit-learn.org/stable/modules/calibration.html) — Sklearn calibration docs
- [SHAP (Explainability)](https://shap.readthedocs.io/) — CatBoost SHAP integration
- [Precision@k, nDCG](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)) — Ranking metrics
- [Brier Score, ECE](https://en.wikipedia.org/wiki/Brier_score) — Calibration metrics

**Best Practices**:
- [Temporal Validation Splits](https://otexts.com/fpp3/tscv.html) — Time series cross-validation
- [Negative Sampling](https://arxiv.org/abs/1310.4546) — Word2Vec paper (negative sampling technique)
- [PCA Embeddings](https://scikit-learn.org/stable/modules/decomposition.html#pca) — Dimensionality reduction

---

**Last Updated**: 2025-10-27  
**Version**: 1.0.0  
**Maintainer**: adrian-iancu_ktcsrl  
**Repository**: [github.com/adrian-iancu_ktcsrl/storico](https://github.com/adrian-iancu_ktcsrl/storico)
