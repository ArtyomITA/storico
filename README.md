# Storico K-pattern: Invoice Payment Pattern Learning System

By **Adrian Ciprian Iancu** · [GitHub](https://github.com/ArtyomITA) · [Hugging Face](https://huggingface.co/adriandj3) · [LinkedIn](https://www.linkedin.com/in/adrian-ciprian-iancu-a2a477391) · [site](https://artyomita.github.io/)

> **Supervised + unsupervised ML pipeline for open-invoice ranking.**
> Learns each client's payment behaviour from historical accounting data,
> clusters clients by behavioural profile, and assigns a **0-5 priority score**
> to every open invoice, deterministically, without hand-crafted rules.

**Stack:** Python 3.11 · CatBoost · AutoGluon · HDBSCAN · Streamlit
**Input:** Historical accounting export (closed + open line items, `.xlsx`/`.csv`)
**Output:** Ranked open invoices with `p_i ∈ [0,1]` and discrete score `0–5` · Interactive dashboard
**Last updated:** February 2026

---

## Architecture Overview

Multi-stage pipeline where each layer produces `.parquet` artefacts consumed by the next.
No hard-coded rules: everything derives from observed payment behaviour.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        STORICO — PIPELINE LAYERS                             │
├──────────────┬───────────────────────────────────────────────────────────────┤
│  LAYER       │  MODULE                        →  ARTEFACT                    │
├──────────────┼───────────────────────────────────────────────────────────────┤
│  [A] INGEST  │  normalize.py                  →  normalized/events.parquet   │
│              │  ingest_multi.py               →  curated/line_items.parquet  │
│              │  build_events.py               →  curated/events.parquet      │
├──────────────┼───────────────────────────────────────────────────────────────┤
│  [B] FEAT.   │  make_labels.py                →  curated/train_pointwise.pq  │
│  ENGINEERING │  make_features.py              →  curated/train_features.pq   │
│              │                                →  curated/client_profiles.pq  │
├──────────────┼───────────────────────────────────────────────────────────────┤
│  [C] UNSU-   │  pattern_detectors.py          →  curated/client_patterns.pq  │
│   PERVISED   │  make_client_embeddings.py     →  curated/client_embed.pq     │
│              │  embedding_viz.py (PCA/UMAP)   →  curated/embed_2d.pq        │
│              │  run_hdbscan.py                →  curated/client_clusters.pq  │
│              │  cluster_profile.py            →  curated/cluster_priors.pq   │
├──────────────┼───────────────────────────────────────────────────────────────┤
│  [D] SUPER-  │  pointwise_trainer.py          →  models/champion.json        │
│   VISED      │  train_catboost.py             →  models/catboost.cbm         │
│              │  train_autogluon.py            →  models/autogluon/           │
│              │  calibration.py (Isotonic)     →  models/calibrator.pkl       │
├──────────────┼───────────────────────────────────────────────────────────────┤
│  [E] SCORING │  score_open_invoices.py        →  curated/open_scored.pq      │
│  & ANALYTICS │  rules_backtest.py             →  reports/backtest.csv        │
│              │  kpi_ranking.py                →  reports/kpi_ranking.csv     │
│              │  make_pattern_table.py         →  curated/pattern_table.pq    │
│              │  make_report.py                →  reports/*.html              │
├──────────────┼───────────────────────────────────────────────────────────────┤
│  [F] UI      │  streamlit_app.py              →  localhost:8501              │
│              │  overview_wizard.py            →  4-step wizard + FAQ         │
│              │  clusters.py                   →  HDBSCAN explorer + UMAP     │
│              │  pattern_card_v2.py            →  Per-client drill-down       │
│              │  kpi_dashboard.py              →  Ranking metrics view        │
│              │  policy_iq.py                  →  Confusion matrix + uplift   │
│              │  client_debugger.py            →  Single-client debug view    │
└──────────────┴───────────────────────────────────────────────────────────────┘
```

---

## Layer A: Ingest & Normalisation

Ingests raw accounting exports and normalises them into a canonical schema.

**Challenge:** Source files use inconsistent date formats (YYYYMMDD vs DD.MM.YYYY),
localised numeric separators, and client-specific column aliases. The normaliser
handles all variants without manual intervention.

| Module | Responsibility |
|---|---|
| `src/data/normalize.py` | Column mapping → canonical schema, date/amount parsing |
| `src/data/ingest.py` | Null token normalisation, schema validation |
| `src/data/ingest_multi.py` | Multi-file / multi-sheet ingestion with auto column-map |
| `src/data/generate_mock_scenarios.py` | Synthetic client generator (FIFO / Largest / Random) |
| `src/utils/sap_glossary.py` | Column alias dictionary for varying source schemas |

**Clearing event key:** `(client_id, clearing_date, clearing_doc)`, one event = one payment session where a client settles one or more open items simultaneously.

---

## Layer B: Feature Engineering

Constructs a pointwise feature matrix: one row per `(event, invoice)` pair.
Features capture both the invoice's attributes at that moment and the client's
aggregated historical behaviour.

### Per-Invoice Features (Pointwise)

| Feature | Description |
|---|---|
| `days_since_doc` | Invoice age in days at event time |
| `amount_rank` | Normalised amount rank within open set (0=lowest, 1=highest) |
| `is_oldest` | Is this the oldest open invoice in the current set? |
| `doc_period_lag` | Distance (periods) between invoice period and current period |
| `relative_amount` | Amount / mean open amounts for this client |

### Per-Client Aggregate Features (Behavioural Profile)

| Feature | Description |
|---|---|
| `avg_fifo_score` | Historical tendency to pay FIFO (oldest first) |
| `avg_largest_score` | Historical tendency to pay highest amounts first |
| `is_eom_payer` | Consistently pays in the last 5 days of the month |
| `k_entropy` | Unpredictability of open-invoice count per event |
| `avg_settlement_days` | Mean lag (days) from issue date to payment |
| `dominant_policy` | Most-followed policy: `fifo / lifo / largest / eom / random` |

---

## Layer C: Unsupervised Learning

**Core idea:** Group clients by *behaviour*, not by sector or revenue.  
HDBSCAN finds density-based clusters without a fixed K parameter.

```
Client profiles (float vector)
    → HDBSCAN
        → Cluster ID per client
            → Cluster Priors P(policy | cluster)    [Dirichlet smoothing]
                → Extra features injected into the supervised model
```

| Component | Detail |
|---|---|
| **Embedding** | SHAP values from a preliminary model → behavioural vector per client |
| **Projection** | PCA (default) or UMAP (optional flag) → 2D for visualisation |
| **HDBSCAN params** | `min_cluster_size=30`, `min_samples=15`, `cluster_selection_method=eom` |
| **Cluster Priors** | `P(policy=FIFO | cluster=0) = 0.78`, Dirichlet smoothed distributions |
| **Policy Detectors** | 8 policies evaluated per event: FIFO, LIFO, Largest, Smallest, EOM, Same-Period, Nearest-Due, Random |

---

## Layer D: Supervised Learning (Pointwise Ranker)

**Objective:** Given a set of open invoices for a client, assign each a probability
`p_i ∈ [0,1]` of being settled in the next payment event. Probability is then
mapped to a discrete score `0–5`.

### Model Pipeline

```
train_features.parquet
      +
client_patterns.parquet ─────────────────────────┐
      +                                           ▼
cluster_priors.parquet  ──→  enrich_features()  ──→  Feature Matrix
                                                        ↓
                                          ┌─────────────────────────┐
                                          │  CatBoost YetiRank      │  ← champion (default)
                                          │  AutoGluon Tabular      │  ← challenger
                                          └─────────────────────────┘
                                                        ↓
                                          Isotonic Regression Calibration
                                                        ↓
                                               p_i  →  score 0–5
```

| Model | Objective | Notes |
|---|---|---|
| **CatBoost YetiRank** | Pointwise learning-to-rank | Default champion; native categorical support |
| **AutoGluon Tabular** | Stacked ensemble AutoML | Challenger; slower, often stronger on small data |
| **Isotonic Regression** | Probability calibration | Guarantees `P(paid | p_i=0.7) ≈ 0.70` |

**Champion management:** `models/champion.json` tracks the production model.
Challenger is promoted only if it beats the champion on nDCG@10 on the held-out test set.

### Validation Metrics (target on deterministic mock data)

| Metric | Target | Description |
|---|---|---|
| AUC | ≥ 0.87 | Global discriminative power |
| Precision@5 | ≥ 0.74 | Of top-5 predicted invoices, how many were actually paid |
| nDCG@10 | ≥ 0.82 | Ranking quality at position 10 (position-discounted) |
| Brier Score | ≤ 0.20 | Calibration quality (0 = perfect) |
| ECE | ≤ 0.05 | Expected Calibration Error |

---

## Layer E: Scoring, Analytics & Reporting

1. **Score open invoices** (`score_open_invoices.py`) → `open_scored.parquet`
2. **Policy backtesting**, historical validation: how much value would ranking have recovered?
3. **Ranking KPIs**, Precision@k, MRR, nDCG@k per client and globally
4. **HTML report generation**, per-client pattern card with charts and score table
5. **NLG pattern summaries**, auto-generated textual descriptions of client behaviour

---

## Layer F: Streamlit Dashboard

Three logical sections mapped to operational roles:

| Page | Module | Content |
|---|---|---|
| **Home & Monitor** | `overview_wizard.py` | 4-step configuration wizard · pipeline status · error log |
| **Model Internals** | `clusters.py` + `kpi_dashboard.py` + `policy_iq.py` | HDBSCAN map · ML metrics · policy confusion matrix |
| **Client Debugger** | `client_debugger.py` | Client history · open invoices · score 0-5 · per-invoice "Why?" explanation |

---

## Repository Structure

```
storico/
├── app/                          # [F] Streamlit UI
│   ├── streamlit_app.py          #     Main router
│   └── page_components/          #     Page modules
├── src/
│   ├── data/                     # [A] Ingest & Normalisation
│   ├── features/                 # [B] Feature Engineering
│   ├── analytics/                # [C] Unsupervised + Analytics
│   ├── training/                 # [D] Model Training
│   ├── models/                   # [D] Model Definitions
│   ├── inference/                # [E] Scoring
│   ├── pipeline/                 #     Orchestrator (run_all.py)
│   ├── report/                   # [E] HTML Report Generator
│   └── utils/                    #     Logging, column glossary, etc.
├── data/
│   ├── raw/                      # ⛔ Source files — NOT in git
│   ├── raw_sample/               # ✅ Minimal CSV sample for CI tests
│   ├── normalized/               # ⛔ Post-normalise artefacts (generated)
│   ├── curated/                  # ⛔ Pipeline artefacts (generated)
│   └── mock/                     # ⛔ Synthetic data (generated)
├── models/                       # ⛔ Model weights — NOT in git
├── catboost_info/                # ⛔ CatBoost training logs — NOT in git
├── reports/                      # ⛔ Generated HTML reports — NOT in git
├── logs/                         # ⛔ Runtime JSONL logs — NOT in git
├── scripts/                      # Utility & integration scripts
├── tests/                        # pytest test suite
├── requirements.txt
└── stepinfo.md                   # Technical plan and progress tracker
```

---

## Tech Stack

| Category | Library | Usage |
|---|---|---|
| **Data** | `pandas ≥2.0`, `pyarrow ≥14` | Dataframe manipulation, Parquet I/O |
| **ML Core** | `scikit-learn ≥1.4`, `scipy ≥1.11` | Preprocessing, calibration, metrics |
| **Gradient Boosting** | `catboost ≥1.2` | Primary ranker (YetiRank objective) |
| **AutoML** | `autogluon.tabular ≥1.1` | Challenger stacked ensemble |
| **Boosting (bench.)** | `lightgbm ≥3.3`, `xgboost ≥2.0` | Alternative benchmark models |
| **Clustering** | `hdbscan ≥0.8.33` | Density-based behavioural segmentation |
| **Dimensionality Red.** | `umap-learn ≥0.5` | 2D projection (optional, recommended over PCA) |
| **UI** | `streamlit ≥1.35` | Operational dashboard |
| **Visualisation** | `plotly ≥5.20` | Interactive charts |
| **File I/O** | `openpyxl ≥3.1` | Excel ingestion |
| **Logging** | `rich ≥13.0` | Structured JSONL logs + coloured console |

**Runtime:** Python `3.11` · Windows / Linux · no cloud dependency

---

## Quick Start

```bash
# Clone & setup
git clone https://github.com/ArtyomITA/storico.git
cd storico
python -m venv .venv
# Windows:
.\.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt

# Copy your source data
cp /path/to/accounting_export.xlsx data/raw/

# Run full pipeline
python -m src.pipeline.run_all --model catboost

# Launch dashboard
streamlit run app/streamlit_app.py
```

**Individual pipeline steps:**
```bash
python -m src.pipeline.run_all --step ingest
python -m src.pipeline.run_all --step train
python -m src.pipeline.run_all --step score
python -m src.pipeline.run_all --make-embeddings --run-hdbscan
```

---

## Project Status

> **End goal:** A production-grade system that, given any set of open invoices for
> a client, predicts, with measurable precision, which invoices will be settled
> in the next payment run. No domain assumptions, no manual rules.

### Completed

| Period | Milestone |
|---|---|
| Nov 2024 | Multi-format data ingest pipeline (normalise, multi-sheet, null handling) |
| Nov 2024 | Full feature engineering (client profiles, pointwise features, temporal lags) |
| Nov 2024 | Pattern detectors (8 policies: FIFO, LIFO, Largest, EOM, Same-Period, etc.) |
| Nov 2024 | CatBoost YetiRank training + AutoGluon challenger + champion manager |
| Nov 2024 | Open invoice scoring + Isotonic calibration |
| Dec 2024 | HDBSCAN behavioural clustering + cluster priors (Dirichlet smoothing) |
| Dec 2024 | SHAP-based client embeddings + PCA/UMAP projection |
| Dec 2024 | Ranking KPIs (Precision@k, MRR, nDCG@k) + calibration diagnostics |
| Dec 2024 | Policy classifier metrics (confusion matrix, accuracy, uplift) |
| Dec 2024 | HTML per-client pattern card reports |
| Dec 2024 | Streamlit dashboard (Overview, Clusters, KPI, Policy IQ, Pattern Card) |
| Dec 2024 | Deterministic synthetic data generator (3 profiles: FIFO / Largest / Random) |
| Jan 2025 | Pre-training validation pipeline (date integrity, feature completeness) |
| Feb 2026 | Client Debugger page + Model Internals page refactoring |

### In Progress / Remaining

| Priority | Task | Layer |
|---|---|---|
| 🔴 | Validate `cluster_priors → enrich_features → CatBoost` flow on deterministic mock data | Training |
| 🔴 | Verify feature importance: `cluster_prior_fifo`, `cluster_prior_largest` are non-zero | Training |
| 🟡 | Streamlit UI refactor → 3 logical pages (Home / Model Internals / Client Debugger) | UI |
| 🟡 | Client Debugger: per-invoice SHAP-based "Why?" explanations | UI |
| 🟢 | NLG narrative integration in HTML reports | Report |
| 🟢 | End-to-end pipeline test on mock data (target: CatBoost accuracy > 95%) | Testing |
| ⬜ | REST API for external system integration | Future |

### Completion Estimate

```
Data Ingest & Feature Eng.     ████████████████████  100%
Unsupervised (HDBSCAN/embed.)  ████████████████░░░░   80%  (validation pending)
Training Pipeline              ████████████████░░░░   80%  (cluster→model flow unvalidated)
Scoring & Analytics            ████████████████████  100%
Streamlit Dashboard            ████████████░░░░░░░░   60%  (UI refactor in progress)
Testing & Validation           ████████░░░░░░░░░░░░   40%  (mock E2E incomplete)
─────────────────────────────  ██████████████░░░░░░   ~72%  overall
```

---

## Deployment Notes

- Fully local: no cloud dependency, runs on a standard Windows/Linux workstation
- Source data files must never be committed (covered by `.gitignore`)
- Model artefacts are too large for git, use DVC or local storage
- No API keys in source code; use `.env` (excluded from git) for any env-specific config

---

*ArtyomITA, Invoice Payment Intelligence System*
*Last updated: February 2026*
