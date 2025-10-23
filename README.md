# SAP Payment Pattern Analytics — Technical Runbook

This document is the end-to-end technical reference for the SAP payment-pattern analytics stack contained in this repository. It replaces legacy business-focused notes and dives into ingestion, feature engineering, modelling, analytics, embeddings, clustering, reporting, and delivery mechanics. Every section is written for engineers, data scientists, and DevOps specialists who need full traceability of the pipeline.

---

## 0. Pre-flight validation checklist

### 0.1 Quick health scripts (run before any full refresh)
1. **Check Python toolchain and dependency integrity**
   ```powershell
   python --version
   pip list | Select-String "catboost|autogluon|pandas|streamlit"
   ```
2. **Lint the codebase for accidental syntax regressions**
   ```powershell
   python -m compileall src app
   ```
3. **Verify curated storage is writable and not stale**
   ```powershell
   Get-ChildItem data/curated -ErrorAction SilentlyContinue
   ```
4. **Run the feature smoke test only (fast)**
   ```powershell
   python -m src.features.make_features
   ```
5. **Execute the unified dry-run (without training) to surface missing inputs**
   ```powershell
   python -m src.pipeline.run_all --input data/raw/*.csv --model catboost
   ```

### 0.2 Expected artefacts after smoke tests
- `data/curated/line_items.parquet`
- `data/curated/events.parquet`
- `data/curated/event_items.parquet`
- `data/curated/train_pointwise.parquet`
- `data/curated/train_features.parquet`
- `data/curated/client_profiles.parquet`
- `data/curated/per_event_policy_scores.parquet`
- `data/curated/event_policy_predictions.parquet`
- `data/curated/kpi_ranking.parquet` (se `--make-kpi` attivo)
- `data/curated/calibration_summary.parquet` + `calibration_bins.parquet` (se `--make-calibration` attivo)
- `data/curated/event_policy_metrics.parquet` + `event_policy_confusion.parquet` (se `--make-policy-metrics` attivo)
- Logs confirming the RandomForest smoke test executed (top feature list printed).

---

## 1. Repository topology

| Path | Description |
| --- | --- |
| `src/data/` | Ingestion and document-map utilities. |
| `src/features/` | Event construction, label creation, and feature matrix engineering. |
| `src/models/` | CatBoost and AutoGluon trainers plus champion orchestrator. |
| `src/analytics/` | Policy detectors, embeddings, clustering, rule backtests, KPI ranking, calibration diagnostics, policy classifier metrics. |
| `src/report/` | Generation of HTML pattern cards. |
| `src/pipeline/` | Unified command runner `run_all`. |
| `app/` | Streamlit application (Pattern Card + KPI Dashboard + Data Explorer + Clusters pages). |
| `app/pages/` | Modular page components (clusters.py, kpi_dashboard.py). |
| `data/raw/` | Input CSV/XLSX extracts from SAP (user supplied). |
| `data/curated/` | All intermediate and final Parquet artefacts. |
| `models/` | Persisted ML models, calibrators, and AutoGluon predictor bundle. |
| `reports/` | Generated HTML dossiers per client. |

---

## 2. Data contract (input layer)

### 2.1 Required columns and semantics
| Column (raw) | Type hint | Transformation | Notes |
| --- | --- | --- | --- |
| `Cliente` | string | Trim, normalise null tokens, cast to `string` dtype | Primary client identifier. |
| `Data di pareggio` | date | `to_datetime` with `%d/%m/%Y` fallback to Excel serials | Optional for open invoices. |
| `Doc. pareggio` | string | Trim + remove `.0` suffixes | Combined with `Cliente` to build event key. |
| `Esercizio` | string | Normalise to `string`; retains original format | Fiscal year in SAP nomenclature. |
| `Numero documento` | string | Normalise digits, drop trailing `.0` | Unique invoice number per client. |
| `Imp. div. int.` | numeric | Remove spaces, replace comma with decimal dot, cast to `float` | Invoice amount in company currency. |
| `Data registrazione` | date | Same parsing strategy as `Data di pareggio` | Posting date. |
| `Data documento` | date | Primary invoice date, mandatory | Must not be null; rows missing it are dropped upstream. |
| `Data di acquisizione` | date | Optional capture date | Fallback to `NaT`. |
| `Divisa` | string | Trim | ISO currency or SAP code. |
| `Tipo di documento` | string | Trim | e.g., `DR`, `DZ`, etc. |
| `Periodo contabile` | string | Normalised string; numeric coercion downstream | Month in SAP (1..12). |
| `Chiave contabile` | string | Trim | Accounting key. |

### 2.2 Null-handling policy
- Empty strings, `nan`, `NaN`, `None`, `<NA>`, and `NaT` are treated as null at ingestion.
- Date columns missing values remain `NaT`; downstream components either drop or impute depending on stage.
- Amounts failing numeric coercion are set to `NaN` and subsequently filtered out during label creation.

### 2.3 File types
- `.csv` with UTF-8 encoding (BOM tolerated).
- `.xlsx` processed through `openpyxl` reader.

---

## 3. Pipeline blueprint (high-level dependency graph)

1. `src/data/ingest_multi.run`
2. `src/features/build_events.build_events_tables`
3. `src/features/make_labels.build_pointwise_labels`
4. `src/features/make_features.build_feature_matrix`
5. `src.models.train.run` (CatBoost or AutoGluon)
6. `src.analytics.pattern_detectors.main`
7. `src.analytics.rules_backtest.main`
8. `src.analytics.make_pattern_table.build_pattern_table`
9. `src.analytics.make_client_embeddings.main`
10. `src.analytics.embedding_viz.main`
11. `src.analytics.run_hdbscan.main`
12. `src.analytics.cluster_profile.main`
13. `src.report.make_report.generate_reports`
14. `app/streamlit_app.py` (interactive consumption)

Each stage reads the artefacts produced by prior steps and fails fast if prerequisites are missing.

---

## 4. Stage walkthroughs

### 4.1 Ingestion (`src/data/ingest_multi.py`)
- **Purpose**: Merge heterogeneous SAP extracts, normalise schema, optionally materialise document map.
- **Key functions**:
  - `resolve_input_files`: expands glob patterns or defaults to `data/raw`.
  - `load_and_normalise`: applies renaming, date parsing, amount cleaning.
  - `deduplicate_dataset`: remove duplicates on configurable subset (default: `cliente`, `numero_documento`, `doc_pareggio`, `data_documento`).
  - `build_doc_map`: aggregate-level statistics per document.
- **Single-source mode**: `src.data.ingest.ingest()` offers a quick path that scans `data/raw`, logs schema/amount summaries, and writes `line_items.parquet` without CLI flags—handy for local smoke tests.
- **Outputs**:
  - `data/curated/line_items.parquet`
  - Optional `docmap_path` if `--docmap` CLI argument is supplied.
- **Doc map schema** (`cliente`, `numero_documento`, `record_count`, `importo_totale`, `data_documento_min/max`, `data_pareggio_min/max`, `divisa`, `tipo_documento`, `doc_pareggio_primo`) enables downstream reconciliation against SAP registries.
- **CLI usage**:
  ```powershell
  python -m src.data.ingest_multi \
    --input data/raw/2024/*.csv \
    --input data/raw/2023/*.xlsx \
    --docmap data/curated/doc_map.parquet \
    --dedupe-on cliente numero_documento doc_pareggio data_documento
  ```
- **Logging expectation**: file-level read logs, duplicate counts, summary of `tipo_documento` and `chiave_contabile` distribution.

### 4.2 Event reconstruction (`src/features/build_events.py`)
- **Objective**: Build event-level representation and attach event keys to each line item.
- **Process**:
  1. Load `line_items.parquet`.
  2. Enrich with `event_key = cliente|YYYY-MM-DD|doc_pareggio` when clearing date exists.
  3. `compute_events`: group by (`cliente`, `data_pareggio`, `doc_pareggio`, `event_key`) to calculate `importo_evento`, `n_righe_evento`, `inter_event_gap_days`.
  4. Persist both `event_items.parquet` and `events.parquet`.
- **Acceptance output**: console prints top 3 events showing `event_key`, `n_righe_evento`, `inter_event_gap_days`.

### 4.3 Label generation (`src/features/make_labels.py`)
- **Goal**: Create supervised training pairs (invoice vs clearing event) with positives + curated negatives.
- **Important knobs**:
  - `MAX_NEGATIVES_PER_EVENT = 20` keeps dataset balanced.
  - Negative candidate filter ensures invoice was open at event date.
  - Priority weighting on `tipo_documento` and `periodo_contabile` similarity before taking head.
- **Persisted file**: `data/curated/train_pointwise.parquet`.
- **Console summary**: prints counts of events, positives, negatives, and ratio.

### 4.4 Feature engineering (`src/features/make_features.py`)
- **Inputs**: `train_pointwise`, `events`, `event_items`.
- **Client profiles**: aggregated historical metrics such as `p_same_period_hist`, `age_med_hist`, ordinal day-of-week encodings, `gap_med_events`, ratio of event cardinalities.
- **Per-row features** include:
  - **Temporal alignment vs event**: `age_days`, `lag_reg_vs_doc`, `lag_acq_vs_doc`, `same_period`, `same_fiscal_year`, and `same_quarter_as_event` compare invoice timestamps to the clearing event.
  - **Calendar + cadence footprint** (derived from `events.parquet` with a 180-day sliding window):
    - Event-level markers `is_eom_event`, `days_to_eom`, `is_quarter_end_event`, `days_to_qe`, `dow_event`, `dow_run_is_thu`.
    - Rolling adherence ratios `weekly_cadence_client` and `eom_cadence_client` measuring how consistently a client clears on the same weekday or close to month-end.
    - Cyclical encodings `dow_sin`, `dow_cos`, `dom_sin`, `dom_cos`, `mon_sin`, `mon_cos`.
  - **Sequential document signatures** (built per client-month from `train_pointwise.parquet`): `docnum_delta_prev`, `docnum_is_consecutive_block`, `seq_block_len_client_month`, `close_to_recent_block` capture invoice numbering runs and their proximity to the event.
  - **Amount structure metrics**:
    - Event-centric: `amt_rank_open`, `amt_share_rank_event` (normalized rank across candidates), `amt_ratio_B`, `event_gini_amounts_open` (Gini coefficient over absolute invoice amounts), `event_amount_scale` (event total vs client median event amount).
    - Client-centric: `amt_decile_client`, `amt_z_client` (z-score vs client mean) from aggregated invoice histories.
  - **Categorical context** preserved as `string` dtypes: `tipo_documento`, `chiave_contabile`, `divisa`.

- **Input tables**: `train_pointwise.parquet`, `events.parquet`, and `event_items.parquet` (for client profiles). Missing optional columns (e.g., `data_acquisizione`) default to neutral values without halting the build.
- **Outputs**:
  - `train_features.parquet`
  - `client_profiles.parquet`
- **Smoke test**: fits a `RandomForestClassifier` (n=200, `random_state=42`) and prints top-10 feature importances.

### 4.5 Training orchestrator (`src/models/train.py`)
- **Roles**:
  - Parse CLI (`--model catboost|autogluon`, `--train-features`, `--out`, `--seed`).
  - Set RNG seeds, ensure output directories exist.
  - Dispatch to `run_catboost` or `run_autogluon`.
  - Persist `models/champion.json` with metrics and timestamp.
- **Common metrics**: `auc_pr`, `match_rate` aggregated over events.
- **Command**:
  ```powershell
  python -m src.models.train --model catboost --out models --seed 42
  ```

#### 4.5.1 CatBoost trainer (`src.models.train_catboost.py`)
- **Pipeline**:
  1. Load curated tables.
  2. `enrich_features`: add `periodo_contabile`, `macro_tipo` when available.
  3. Temporal split (70% train, 30% test) sorted by `data_pareggio`.
  4. Build `GroupKFold` split (or fallback) on `cliente` for early stopping.
  5. Train `CatBoostClassifier` with depth=6, lr=0.05, 1500 iterations plus early stopping.
  6. Fit isotonic calibrator; fallback to identity if labels degenerate.
  7. Evaluate invoice metrics (LogLoss/AUC-PR) and event-level metrics (recall@k, match_rate).
  8. Save model to `models/catboost.cbm`, calibrator to `models/calibrator.pkl`.
  9. Store client metrics to `data/curated/client_model_metrics_catboost.parquet`.
- **Run directly**:
  ```powershell
  python -m src.models.train_catboost
  ```

#### 4.5.2 AutoGluon trainer (`src.models.train_autogluon.py`)
- **Steps**:
  1. Determine compute environment with GPU detection.
  2. Temporal split identical to CatBoost.
  3. Fit `TabularPredictor` using `medium_quality_faster_train` presets, excluding neural nets (`NN_TORCH`, `FASTAI`).
  4. AutoGluon path: `models/autogluon/`.
  5. Evaluate metrics, save `client_model_metrics_autogluon.parquet`.
  6. Print leaderboard and comparison with CatBoost if metrics available.
- **Run**:
  ```powershell
  python -m src.models.train --model autogluon --out models --seed 123
  ```

### 4.6 Behaviour analytics (`src/analytics`) 

#### 4.6.1 Pattern detectors (`pattern_detectors.py`)
- Inputs: `events.parquet`, `event_items.parquet`, `line_items.parquet`, `train_pointwise.parquet`, `train_features.parquet`. The module rebuilds a candidate table with invoice metadata (`abs_amount`, document numbers, registration dates) to avoid leakage.
- Computes deterministic policy recalls per event:
  - `fifo` / `lifo`: sort on invoice date ascending/descending.
  - `same_period`: prioritise invoices whose fiscal period matches the event month.
  - `largest` / `smallest`: pick highest/lowest absolute amounts.
  - `eomfirst`: reward invoices within ±3 days of month-end when the event itself is month-end.
  - `regmonth`: favour invoices whose registration month equals the clearing month, breaking ties by proximity to the event date.
  - `quarter`: give priority to documents in the same calendar quarter as the event.
  - `seqblock`: surface invoices that belong to sequential document-number runs of length ≥3 within the client-month.
  - `typeprio`: prefer the document type that dominated the 90 days preceding the event (per client).
- Candidate tables join engineered features with raw invoice columns to compute the above heuristics and recover labels for recall evaluation.
- Periodicity kernel scores for weekly/biweekly/monthly cadence, day-of-week/day-of-month concentration metrics, and event cardinality distribution statistics (`P_k1`, `P_k2`, `P_kge3`, `k_mode`, `k_entropy`).
- Amount preference slope via point-biserial correlation and a stability index (1 / (1 + rolling std) ) combining dispersion of policy recalls.
- Outputs: `data/curated/client_pattern_scores.parquet` and `data/curated/per_event_policy_scores.parquet` (recall per policy with alias columns `policy_*`).

#### 4.6.2 Rules backtest (`rules_backtest.py`)
- Simulates deterministic policies (configured inside module) and compares to ground truth events.
- Produces `data/curated/client_rules_accuracy.parquet` with per-client match rate for each rule block.

#### 4.6.3 Pattern table (`make_pattern_table.py`)
- Inputs: `client_pattern_scores`, `client_rules_accuracy`, CatBoost + AutoGluon client metrics.
- Aggregation logic:
  - Derive `pattern_label`, `confidence`, and `stability_index` per client from policy and cardinality probabilities.
  - Left-join rule backtest match rates and model-level match rates (`model_match_rate_cat`, `model_match_rate_auto`).
- Persist `data/curated/pattern_table.parquet` (one row per client) used by Streamlit and HTML reports.

#### 4.6.4 Event policy classifier (`event_policy_classifier.py`)
- Builds event-level feature vectors directly from `events.parquet` and merges them with `per_event_policy_scores.parquet` labels. Predictors include `cliente_code`, `n_righe_evento`, `inter_event_gap_days`, `importo_evento`, calendar slices (`event_day`, `event_month`, `event_quarter`) and flags (`is_month_end`, `is_quarter_end`).
- Trains a multiclass `CatBoostClassifier` to predict the best deterministic policy per clearing event, with hold-out accuracy and mean recall@k metrics.
- Persists probability-calibrated inferences to `data/curated/event_policy_predictions.parquet` (one row per evaluated event, including fully populated `proba_<policy>` columns).

### 4.7 Embeddings & visualization

#### 4.7.1 Embedding builder (`make_client_embeddings.py`)
- Reads champion metadata to decide whether SHAP vectors come from CatBoost or the AutoGluon champion (falling back to the first tree-based base learner).
- Selects up to `--max-samples` positive rows per client (default 200) sorted by recency and computes mean absolute SHAP values per feature.
- Aggregates three feature blocks before projection:
  - **SHAP**: per-feature mean/abs contributions with prefix `shap_` (normalised per client when available).
  - **Policy scores**: cadence kernels (`periodicity_*`), deterministic recalls (`fifo`, `lifo`, `same_period`, `largest`, `smallest`, **`eomfirst`**, **`regmonth`**, **`quarter`**, **`seqblock`**, **`typeprio`**) and supporting metrics such as `policy_amount_pref_slope`, `policy_stability_index`, `policy_P_k*`.
  - **Client profile**: historical cadence and range indicators (weekly / month-end cadence ratios, `profile_same_fiscal_year_rate`, `profile_same_quarter_rate` when available), day-of-week embeddings, document type shares, inter-event gaps, etc.
- Applies `StandardScaler` to the full feature matrix, then runs PCA with up to 64 components (automatically capped by the number of clients/features). Stores the per-component and cumulative explained variance in the metadata payload.
- Persists:
  - `data/curated/client_embeddings.parquet` containing `cliente` plus `embedding_pcXX` columns.
  - `data/curated/client_embeddings_meta.json` with column ordering, raw feature list (pre-scaling), PCA dimensionality, explained-variance ratios, and SHAP availability flag.
- Console output now reports the effective embedding dimension and the cumulative PCA variance before printing the top 3 rows.

#### 4.7.2 Projection (`embedding_viz.py`)
- Inputs: `client_embeddings.parquet`.
- Steps:
  1. Build k-nearest-neighbour graph (default `--neighbors 10`).
  2. If `--umap` flag, compute 2D UMAP projection; fallback to PCA otherwise.
  3. Persist `client_embedding_2d.parquet` and `client_neighbors.parquet`.

#### 4.7.3 Clustering (`run_hdbscan.py` + `cluster_profile.py`)
- Applies HDBSCAN with CLI-configurable `--min-cluster-size` and `--min-samples`.
- Saves cluster assignments to `client_clusters.parquet`.
- `cluster_profile.py` aggregates behaviour metrics per cluster and stores `cluster_summary.parquet`.

### 4.8 Reporting (`src/report/make_report.py`)
- Generates HTML pattern cards per client.
- Integrates CatBoost & AutoGluon inference (if models available) to recommend top invoices alongside policy suggestions.
- Visual elements (Plotly): periodicity histogram, ACF bar chart, calendar heatmap, policy bars, cardinality distribution.
- CLI:
  ```powershell
  python -m src.report.make_report --output-dir reports
  ```
- Output naming: `reports/pattern_card_<cliente>.html`.

#### 4.8.1 PROMPT E6 — Claim + Evidence summaries
- `src.report.nlg_patterns.render_pattern_summary(cliente, tables=...)` now returns an ordered list of markdown strings capturing cadence, policy preference, cardinality, amount bias, and overall confidence.
- Each sentence blends the **claim** (behaviour headline) with the **evidence** (ratios, recalls, entropy, slopes) using resilient formatters (`_format_percent`, `_format_decimal`) so missing metrics surface as `n.d.` instead of breaking the narrative.
- Example consumption:
  ```python
  from src.report import nlg_patterns

  summary_lines = nlg_patterns.render_pattern_summary(
    cliente="00001234",
    tables={
      "client_pattern_scores": pattern_scores_df,
      "pattern_table": pattern_table_df,
    },
  )
  for line in summary_lines:
    print(line)
  ```
- Downstream services can reuse these sentences directly (they are the same ones embedded in the pattern card HTML) or inspect the helper functions in `nlg_patterns.py` for finer-grained evidence assembly.

#### 4.8.2 PROMPT E13 — Explainability snippets nel report
- Ogni report HTML ora include una sezione **"Explainability snippets"** che mostra per i 3 eventi più recenti del cliente:
  - Tabella comparativa a 4 colonne: **Fatture reali** vs **Top-k Policy dominante** vs **Top-k Modello A** (CatBoost) vs **Top-k Modello B** (AutoGluon).
  - Evidenziazione visiva con colori: celle verdi per i match (fattura suggerita coincide con quella reale), celle rosse per i mismatch.
  - Per le prime 2 fatture suggerite dal modello A, elenco delle **top-3 feature SHAP** con:
    - Nome della feature e valore corrente dell'invoice candidato.
    - Soglia estratta dal nome della feature (se disponibile).
    - Valore SHAP (contributo marginale alla probabilità di match).
- Le funzioni helper introdotte:
  - `build_explainability_cards(...)` orchestra la generazione delle card per ogni evento.
  - `build_highlighted_comparison_html(...)` crea la tabella HTML con evidenziazione match/mismatch.
  - `_format_shap_details(...)` produce l'elenco puntato delle feature SHAP con valore/soglia/contributo.
- **Allineamento con Streamlit**: la stessa logica di highlight verde/rosso è già presente nell'app interattiva; questa sezione garantisce coerenza tra report stampabili e GUI live.
- Output: i riquadri compaiono nel report HTML subito dopo la sezione "Pattern sintetico" e prima del "Confronto Regole vs Modelli".

### 4.9 Interactive application (`app/streamlit_app.py`)
- Four pages: Pattern Card (default), KPI Dashboard, Data Explorer, Clusters.
- **Pattern Card**: dashboard per cliente con metriche pattern, policy dominante, grafico recall@k e tabelle eventi/fatture.
- **KPI Dashboard**: interfaccia interattiva per metriche di ranking (precision@k, hit_rate@k, nDCG@k, MRR, coverage@k) e calibrazione (Brier score, ECE, MCE, reliability curves); filtri per cliente, bucket cardinalità, modello; download CSV (vedi sezione 4.8.6).
- **Data Explorer**: filtri multipli (cliente, intervallo date, tipo documento, chiave contabile), ricerca full-text, tabella interattiva con tutte le colonne e download CSV immediato della vista filtrata; tab dedicate per analisi cadence, policy, cardinalità con grafici Plotly.
- **Clusters**: visualizzazione UMAP/t-SNE dei cluster HDBSCAN con colori per cluster, sidebar per filtri, tabelle cluster statistics e membri.
- La sidebar include il toggle **Developer mode** che, se attivo, mostra un overlay diagnostico con metriche tecniche e log.
- Leverages caching (`st.cache_data` / `st.cache_resource`) to minimise reload times.
- Reads the same curated artefacts listed above and surfaces warnings for missing dependencies.
- Run `streamlit run app/streamlit_app.py` after pipeline refresh.

#### 4.8.3 PROMPT S1 — KPI estesi per ranking
- Il modulo `src/analytics/kpi_ranking.py` calcola metriche di ranking estese su hold-out test (split temporale 70/30):
  - **precision@k**: frazione di item top-k che sono rilevanti.
  - **hit_rate@k** (alias recall@k): frazione di item rilevanti trovati nei top-k.
  - **MRR (Mean Reciprocal Rank)**: reciproco del rank del primo item rilevante.
  - **nDCG@k**: normalized Discounted Cumulative Gain, che penalizza i match rilevanti in posizioni basse.
  - **coverage@k**: frazione di eventi che hanno almeno un hit nei top-k.
- K-values di riferimento: {1, 2, 3, 5}.
- Aggregazioni disponibili:
  - **Global**: medie su tutti gli eventi di test.
  - **Per-client**: medie per singolo cliente (utile per analisi cliente-specifiche).
  - **Per-cardinality-bucket**: k=1, k=2-3, k≥4 (raggruppa eventi per numero di fatture reali abbinate).
- Il modulo carica predizioni dal modello CatBoost con calibrazione (se disponibile), altrimenti usa un'euristica fallback.
- Outputs:
  - `data/curated/kpi_ranking.parquet` / `.csv`: tabella con colonne `aggregation_key`, `aggregation_value`, `n_events`, `precision@k`, `hit_rate@k`, `ndcg@k`, `coverage@k`, `mrr`.
  - Console summary: stampa tabella KPI globali, per-bucket e top 5 clienti per hit_rate@3.
- Usage standalone:
  ```powershell
  python -m src.analytics.kpi_ranking
  ```
- Integrazione in `run_all.py` con flag `--make-kpi`:
  ```powershell
  python -m src.pipeline.run_all --model catboost --make-kpi
  ```

#### 4.8.4 PROMPT S2 — Diagnostica di calibrazione probabilistica
- Il modulo `src/analytics/calibration.py` valuta la qualità delle probabilità predette dai modelli pointwise su hold-out test (split 70/30):
  - **Brier score**: errore quadratico medio delle predizioni probabilistiche (MSE tra p_i e label).
  - **ECE (Expected Calibration Error)**: media pesata dello scostamento tra confidence e accuracy su bin equi-popolati.
  - **MCE (Maximum Calibration Error)**: massimo scostamento tra confidence e accuracy (worst-case bin).
  - **Reliability table**: 20 bin equi-popolati con `pred_mean` (confidence media) vs `obs_rate` (accuracy osservata) per costruire curve di calibrazione.
- Metriche calcolate per:
  - **Global**: intero test set.
  - **Per-cardinality-bucket**: k=1, k=2-3, k≥4 (eventi raggruppati per numero di fatture reali).
  - **Per-client**: top 10 clienti per volume (minimum 10 sample/client).
- Il modulo supporta CatBoost e AutoGluon (se disponibili); carica modelli salvati con calibratori isotonic; se il modello non è disponibile o ha feature mismatch, usa predizioni euristiche fallback (age_days-based).
- Outputs:
  - `data/curated/calibration_summary.parquet`: tabella con colonne `model`, `segment_type`, `segment_value`, `n_samples`, `brier_score`, `ece`, `mce`.
  - `data/curated/calibration_bins.parquet`: curve di reliability con colonne `bin_id`, `pred_mean`, `obs_rate`, `n_samples`, `model`, `segment_type`, `segment_value`.
  - Console summary: stampa metriche globali, per bucket e top 5 clienti (ordinati per Brier score).
- Usage standalone:
  ```powershell
  python -m src.analytics.calibration
  ```
- Integrazione in `run_all.py` con flag `--make-calibration`:
  ```powershell
  python -m src.pipeline.run_all --model catboost --make-calibration
  ```

#### 4.8.5 PROMPT S3 — Metriche evento-level per il policy classifier
- Il modulo `src/analytics/event_policy_metrics.py` valuta le performance del classificatore di policy dominante per evento:
  - **Confusion matrix** (true vs predicted policy): matrice normalizzata per riga (percentuali), mostra quanto spesso ogni policy vera viene predetta correttamente o confusa con altre.
  - **Top-1 accuracy**: frazione di eventi in cui la policy predetta coincide esattamente con la policy vera.
  - **Top-2 hit rate**: frazione di eventi in cui la policy vera è tra le top-2 policy predette (per probabilità).
  - **Recall uplift**: confronto tra recall@k della policy predetta vs policy baseline (FIFO) per misurare il guadagno nel coverage delle fatture reali.
- Segmentazioni disponibili:
  - **Global**: medie su tutti gli eventi.
  - **Per-cardinality-bucket**: k=1, k=2-3, k≥4 (eventi raggruppati per cardinalità).
  - **Per-client**: top-10 clienti per volume eventi (minimum 2 eventi/client).
- Il modulo richiede:
  - `per_event_policy_scores.parquet`: recall@k per ogni policy per evento (colonne: cliente, event_key, fifo, lifo, same_period, largest, smallest, eomfirst, regmonth, quarter, seqblock, typeprio).
  - `event_policy_predictions.parquet`: policy predetta e probabilità per evento (colonne: event_key, policy_pred, policy_true, proba_*).
- Outputs:
  - `data/curated/event_policy_metrics.parquet`: tabella KPI con colonne `segment_type`, `segment_value`, `n_events`, `top1_accuracy`, `top2_hit_rate`, `baseline_recall`, `predicted_recall`, `uplift_absolute`, `uplift_relative`.
  - `data/curated/event_policy_confusion.parquet`: matrice di confusione normalizzata con colonne `policy_true` e una colonna per policy predetta.
  - Console summary: stampa accuracy, uplift % per segmento, dimensione matrice di confusione.
- Usage standalone:
  ```powershell
  python -m src.analytics.event_policy_metrics
  ```
- Integrazione in `run_all.py` con flag `--make-policy-metrics`:
  ```powershell
  python -m src.pipeline.run_all --make-policy-metrics
  ```

#### 4.8.6 PROMPT S4 — Streamlit: pagina KPI Dashboard
- La pagina **KPI Dashboard** in `app/pages/kpi_dashboard.py` fornisce un'interfaccia interattiva per esplorare le metriche di ranking e calibrazione con filtri dinamici, grafici e download CSV.
- **Architettura**:
  - Modulo standalone con funzione `render_kpi_dashboard_page()` integrata nel router di `app/streamlit_app.py`.
  - Caricamento dati cachato (`@st.cache_data`) per performance su 50-100k righe.
  - Layout a tab: **Ranking** e **Calibration**.
- **Filtri sidebar**:
  - **Cliente multiplo**: multiselect per filtrare metriche per clienti specifici.
  - **Bucket cardinalità**: selector per k=1, k=2-3, k≥4, all.
  - **Modello**: selector per modello CatBoost/AutoGluon (nella tab Calibration).
  - **Segmento**: selector per global/cardinality_bucket/cliente (nella tab Calibration).
- **Tab Ranking**:
  - **Global Metrics**: cards con precision@k, hit_rate@k, coverage@k per k∈{1,2,3,5} e MRR.
  - **Cardinality Bucket Metrics**: tabella con metriche per bucket di cardinalità.
  - **Per-Client Metrics**: tabella con ricerca full-text e ordinamento interattivo; mostra n_events, precision@3, hit_rate@3, coverage@3, nDCG@3, MRR per cliente.
  - **Download CSV**: button per esportare metriche filtrate.
- **Tab Calibration**:
  - **Metric Cards**: Brier score, ECE, MCE per modello selezionato (segmento global).
  - **Reliability Curves**: grafico Plotly a linee con pred_mean (x) vs obs_rate (y) per 20 bin equal-populated; include diagonale di calibrazione perfetta e tooltip con n_samples/bin.
  - **Filtri grafici**: multiselect modelli, selector segmento, selector valore segmento.
  - **Download CSV**: button per esportare summary e bins calibrazione.
- **Requisiti input**:
  - `data/curated/kpi_ranking.parquet` (output di `src/analytics/kpi_ranking.py`).
  - `data/curated/calibration_summary.parquet` (output di `src/analytics/calibration.py`).
  - `data/curated/calibration_bins.parquet` (output di `src/analytics/calibration.py`).
- **Navigazione**: selezionare "KPI Dashboard" dal menu radio in sidebar; la pagina si carica automaticamente con dati filtrabili e interattivi.
- **Performance**: cached data loading garantisce latenza <1s su reload; tabelle con pagination automatica per grandi dataset.
- **Acceptance test**:
  1. Avviare Streamlit: `streamlit run app/streamlit_app.py` (oppure `python -m streamlit run app/streamlit_app.py`).
  2. Navigare alla pagina "KPI Dashboard".
  3. Verificare caricamento tab Ranking con global metrics, bucket table, per-client table.
  4. Testare filtri cliente e bucket cardinalità; verificare aggiornamento tabelle.
  5. Testare ricerca full-text nella tabella per-client.
  6. Navigare alla tab Calibration; verificare metric cards e grafico reliability curve.
  7. Testare filtri modello/segmento; verificare aggiornamento grafico.
  8. Scaricare CSV da entrambe le tab; verificare formato corretto.

### 4.10 Unified pipeline runner (`src/pipeline/run_all.py`)
- Orchestrates the entire sequence with optional embeddings + clustering toggles.
- Execution order described in Section 3.
- Prima del training esegue un controllo di coerenza sui curated set (eventi, event_items, train_features): verifica cardinalità evento >0, importi non negativi, date documento coerenti (soglia 5%) e presenza delle colonne critiche del feature store; stampa una tabella riassuntiva e interrompe il run (exit code >0) se trova anomalie.
- Flags: `--input`, `--docmap`, `--model`, `--make-embeddings`, `--run-hdbscan`, `--neighbors`, `--umap`, `--min-cluster-size`, `--min-samples`, `--make-kpi`, `--make-calibration`, `--make-policy-metrics`, `--verbose`.
- Example full refresh:
  ```powershell
  python -m src.pipeline.run_all \
    --input data/raw/*.csv \
    --model catboost \
    --make-embeddings \
    --run-hdbscan \
    --umap \
    --make-kpi \
    --make-calibration \
    --make-policy-metrics \
    --docmap data/curated/doc_map.parquet
  ```

---

## 5. Feature catalogue (selected columns)

| Feature | Source | Type | Description |
| --- | --- | --- | --- |
| `age_days` | Feature stage | numeric | Days between `data_evento` and `data_documento`. |
| `lag_reg_vs_doc` | Feature stage | numeric | Days from invoice date to registration date. |
| `lag_acq_vs_doc` | Feature stage | numeric | Days from invoice date to acquisition date. |
| `is_eom_event` | Feature stage | integer | 1 when the clearing date is within 3 days of the month end. |
| `days_to_eom` | Feature stage | integer | Days remaining from the event date to the end of the month. |
| `is_quarter_end_event` | Feature stage | integer | 1 when the clearing falls on the last calendar day of the quarter. |
| `days_to_qe` | Feature stage | integer | Days between the event date and quarter end (0 when already at quarter close). |
| `dow_event` | Feature stage | integer | ISO weekday index of the clearing event (`0 = Monday`). |
| `dow_run_is_thu` | Feature stage | integer | 1 when the clearing event occurs on Thursday (used by cadence heuristics). |
| `weekly_cadence_client` | Feature stage | numeric | Share of events in the last 180 days occurring on the same weekday. |
| `eom_cadence_client` | Feature stage | numeric | Share of events in the last 180 days occurring within the month-end window. |
| `amt_rank_open` | Feature stage | integer | Rank of invoice amount within open invoices per event. |
| `amt_share_rank_event` | Feature stage | numeric | Normalised rank position (0 = largest candidate, 1 = smallest). |
| `event_gini_amounts_open` | Feature stage | numeric | Gini coefficient of absolute invoice amounts for the event. |
| `event_amount_scale` | Feature stage | numeric | Event total divided by the client's median event total. |
| `amt_decile_client` | Feature stage | integer | Decile placement within client historical amounts. |
| `amt_z_client` | Feature stage | numeric | Z-score of invoice relative to client mean/std. |
| `amt_ratio_B` | Feature stage | numeric | |invoice| / |event total|. |
| `dow_sin`, `dow_cos` | Feature stage | numeric | Cyclical encoding of day-of-week. |
| `dom_sin`, `dom_cos` | Feature stage | numeric | Cyclical encoding of day-of-month. |
| `mon_sin`, `mon_cos` | Feature stage | numeric | Cyclical encoding of month. |
| `same_period` | Feature stage | integer | 1 if `periodo_contabile` matches event month. |
| `same_fiscal_year` | Feature stage | integer | 1 if invoice fiscal year equals the event year. |
| `same_quarter_as_event` | Feature stage | integer | 1 if invoice quarter equals the event quarter in the same year. |
| `docnum_delta_prev` | Feature stage | integer | Difference vs previous numeric document inside the client-month. |
| `docnum_is_consecutive_block` | Feature stage | integer | 1 if the invoice belongs to a sequential document-number block (length ≥ 2). |
| `seq_block_len_client_month` | Feature stage | integer | Size of the sequential block within the client-month. |
| `close_to_recent_block` | Feature stage | integer | 1 when the sequential block's most recent document is the closest prior to the event date. |
| `p_same_period_hist` | Client profiles | numeric | Historical probability of same-period clearings. |
| `gap_med_events` | Client profiles | numeric | Median inter-event gap in days. |
| `dow_sin_event_mean` | Client profiles | numeric | Mean sine encoding of event weekday. |
| `share_tipo_documento_*` | Client profiles | numeric | Fraction of transactions per document type. |
| `policy_fifo_score` | Pattern scores | numeric | Stability-weighted recall for FIFO policy. |
| `amount_pref_slope` | Pattern scores | numeric | Preference slope for large amounts. |
| `stability_index` | Pattern scores | numeric | 1 / (1 + avg rolling std of policy recalls). |

---

## 6. Sample outputs and diagnostics

> **Note**: The following snippets are illustrative. Replace client identifiers and numbers with actual results from your dataset.

### 6.1 CatBoost evaluation sample
```text
Invoice-level metrics: {"log_loss": 0.3421, "auc_pr": 0.7115}
Event metrics (head):
 cliente   event_key                     recall_at_k  match_rate
 00001234  00001234|2024-03-15|90000123       1.0000       1.0
 00005678  00005678|2024-03-18|80000456       0.6667       0.0
 00005678  00005678|2024-04-02|80000488       1.0000       1.0
Client metrics sample:
 cliente   model_recall_at_k  model_match_rate  events_tested
 00001234            0.9500              0.80              10
 00005678            0.7200              0.60               5
```

### 6.2 AutoGluon leaderboard sample
```text
AutoGluon leaderboard (test data):
                model  score_val  score_test  fit_time  pred_time_val  pred_time_test
0     LightGBM_BAG_L1   -0.35811    -0.36745    45.321          0.123           0.118
1   CatBoost_BAG_L1     -0.36509    -0.37180    87.912          0.227           0.221
2  WeightedEnsemble_L2  -0.34455    -0.35124    95.174          0.245           0.238
Invoice metrics: {"log_loss": 0.3512, "auc_pr": 0.7024}
```

### 6.3 Embedding slice sample
```text
Embedding dimension: 64
Cumulative explained variance: 0.9421
Sample embeddings (top 3 rows):
 cliente  embedding_pc01  embedding_pc02  embedding_pc03  embedding_pc04  ...
 00001234        1.284517       -0.114872        0.092311       -0.004551  ...
 00005678       -0.428901        0.735444       -0.137820        0.018645  ...
 00007890       -0.855616       -0.620572        0.045509       -0.012883  ...
```
> The parquet contains the PCA-projected components, while `client_embeddings_meta.json` retains the pre-scaling feature list and the variance ratios per component.

### 6.4 HDBSCAN clustering summary sample
```text
Cluster summary (head):
 cluster_id  members  mean_match_rate  mean_fifo_score  dominant_pattern  notes
          0       54            0.78             0.81              FIFO  "High FIFO adherence, weekly cadence"
          1       37            0.64             0.35              MIXED "Irregular cadence, high entropy"
         -1       12            0.42             0.22            NOISE  "Sparse behaviour, monitor manually"
```

---

## 7. Observability and logging upgrades (PROMPT E7)

- **Centralised setup**: `src/utils/logging_setup.configure_logging()` now wires a dual-handler stack — a Rich console for colourful human-readable output plus a JSONL handler that writes one file per run under `logs/run_YYYYMMDD_HHMMSS.jsonl`.
- **Structured telemetry**: every JSON record carries the core fields `timestamp`, `level`, `logger`, and `message`, plus pipeline-specific metadata (`stage`, `cliente`, `event_key`, `duration_ms`, `rows`, `seed`, `rss_mb`). Additional keys passed via `logger.info(..., extra={...})` are preserved verbatim.
- **Timing decorator**: wrap long-running stages with `@log_timed("Stage name")` (already applied across `PipelineRunner`). It automatically emits start / completion / exception events, measuring execution time in milliseconds and sampling RSS memory when `psutil` is available.
- **Discoverability helpers**:
  ```python
  from src.utils import logging_setup

  log_path = logging_setup.configure_logging()
  print("JSON log file:", log_path)
  ```
  Use `logging_setup.get_current_log_file()` whenever you need to attach the run log to artefacts or diagnostics.
- **Operational quick wins**: tail the JSON file for machine-friendly monitoring (`Get-Content logs/run_*.jsonl -Wait`), while the Rich console keeps interactive runs legible.

## 7. Execution playbooks

### 7.1 Full refresh (daily/weekly)
```powershell
python -m src.pipeline.run_all \
  --input data/raw/*.csv \
  --model catboost \
  --make-embeddings \
  --run-hdbscan \
  --umap \
  --docmap data/curated/doc_map.parquet
```

### 7.2 Model re-training only
```powershell
python -m src.models.train --model catboost --seed 2025
python -m src.models.train --model autogluon --seed 2025
```

### 7.3 Embedding & clustering refresh without ingestion
```powershell
python -m src.analytics.make_client_embeddings
python -m src.analytics.embedding_viz --embeddings data/curated/client_embeddings.parquet --umap
python -m src.analytics.run_hdbscan --embeddings data/curated/client_embeddings.parquet --min-cluster-size 25 --min-samples 12
python -m src.analytics.cluster_profile
```

### 7.4 Report re-render only
```powershell
python -m src.report.make_report --output-dir reports
```
- Se il pre-check di PROMPT E10 fallisce (es. più del 5% di righe con `data_documento > data_pareggio`), il comando `run_all` esce con errore dopo aver mostrato la tabella riassuntiva delle anomalie, evitando il training.

### 7.5 Streamlit launch (local)
```powershell
streamlit run app/streamlit_app.py
```
- La pagina **Data Explorer** supporta l'ispezione rapida dei dataset con filtri avanzati, toggle per le sezioni analitiche e un pulsante "Download CSV" per esportare la vista corrente.

---

## 8. Logging, monitoring, and troubleshooting

- All modules use the standard `logging` module with INFO verbosity by default.
- To increase verbosity: pass `--verbose` to `run_all`, or set `LOGLEVEL=DEBUG` environment variable.
- Key warnings to watch:
  - **Missing input**: `FileNotFoundError` with explicit path.
  - **Degenerate validation**: CatBoost calibrator fallback (indicates poor class balance, revisit label generation).
  - **GPU not detected**: Informational message in AutoGluon trainer; training continues on CPU.
- Investigate the `logs/` directory (create one by configuring `logging.basicConfig(filename=...)` if needed).
- La dashboard Streamlit espone una modalità **Developer mode** (toggle nella sidebar) che visualizza: latenza per sezione UI (misurata via `log_timed`), footprint dei dataframe correnti, estratto di warning/error recenti dal JSON log più recente e parametri di split/seed del modello `models/champion.json`.

### 8.1 Common remediation steps
1. **Mismatch in schema**: rerun ingestion after verifying column headers; use `--dedupe-on` to customise keys.
2. **Empty label set**: check that events contain positives; ensure invoices have `data_pareggio` when cleared.
3. **Low match rate**: inspect `client_pattern_scores` and `client_model_metrics_*` for clients with stability issues.
4. **UMAP missing**: install `umap-learn`; otherwise pipeline defaults to PCA projection.
5. **HDBSCAN noise explosion**: adjust `--min-cluster-size` and `--min-samples` to fit dataset density.

---

## 9. Storage layout and retention

| Directory | Lifecycle recommendation |
| --- | --- |
| `data/raw/` | Refresh as new SAP exports arrive; keep source files for audit (immutable). |
| `data/curated/` | Overwritten per run; snapshot externally if versioning required. |
| `models/` | Keep latest champion and optionally archive previous versions with timestamped directories. |
| `reports/` | Regenerate per run; provide version in filename if needed. |
| `app/` | Source code only. |

---

## 10. Appendix A — CLI reference

### Ingestion
```powershell
python -m src.data.ingest_multi --help
```
Key options: `--input`, `--output`, `--docmap`, `--dedupe-on`, `--keep-first`.

### Event builder
```powershell
python -m src.features.build_events
```
No CLI options; uses project root automatically.

### Label maker
```powershell
python -m src.features.make_labels
```
Generates labels and prints summary; stops early if no positives.

### Feature engineering
```powershell
python -m src.features.make_features
```
Requires `train_pointwise`, `events`, and `event_items` in `data/curated`.

### Training
```powershell
python -m src.models.train --model catboost
python -m src.models.train --model autogluon
```

### Pattern detectors
```powershell
python -m src.analytics.pattern_detectors
```

### Rules backtest
```powershell
python -m src.analytics.rules_backtest
```

### Event policy classifier
```powershell
python -m src.analytics.event_policy_classifier
```

### Pattern table
```powershell
python -m src.analytics.make_pattern_table
```

### Embeddings
```powershell
python -m src.analytics.make_client_embeddings --max-samples 100
```

### Embedding projection
```powershell
python -m src.analytics.embedding_viz --embeddings data/curated/client_embeddings.parquet --neighbors 15 --umap
```

### Clustering
```powershell
python -m src.analytics.run_hdbscan --embeddings data/curated/client_embeddings.parquet --min-cluster-size 30 --min-samples 15
```

### Cluster profiling
```powershell
python -m src.analytics.cluster_profile
```

### Reporting
```powershell
python -m src.report.make_report --output-dir reports
```

### Unified pipeline
```powershell
python -m src.pipeline.run_all --model catboost --make-embeddings --run-hdbscan --umap
```

---

## 11. Appendix B — Data dictionary (curated outputs)

### 11.1 `line_items.parquet`
- `cliente` (string)
- `numero_documento` (string)
- `doc_pareggio` (string)
- `data_documento` (datetime64[ns])
- `data_pareggio` (datetime64[ns])
- `imp_div_int` (float)
- `tipo_documento` (string)
- `chiave_contabile` (string)
- `periodo_contabile` (string)
- `event_key` (string, derived)

### 11.2 `events.parquet`
- `cliente`
- `data_pareggio`
- `doc_pareggio`
- `event_key`
- `importo_evento`
- `n_righe_evento`
- `prev_event_date`
- `inter_event_gap_days`

### 11.3 `train_pointwise.parquet`
- `cliente`
- `numero_documento`
- `event_key`
- `label`
- Source columns from `event_items` appended for context.

### 11.4 `train_features.parquet`
- All engineered numeric and categorical columns enumerated in Section 5.

### 11.5 `client_profiles.parquet`
- One row per `cliente` with aggregated behaviour metrics.

### 11.6 `client_pattern_scores.parquet`
- Policy scores, periodicity metrics, amount preference slope, stability index.

### 11.7 `client_model_metrics_<model>.parquet`
- `cliente`, `model_recall_at_k`, `model_match_rate`, `events_tested`.

### 11.8 `client_embeddings.parquet`
- `cliente`
- `embedding_pcXX` columns (PCA projection of the scaled SHAP, policy, and profile feature stack; `XX` ranges from `01` up to the effective component count ≤ 64).
- Companion metadata file `client_embeddings_meta.json` includes the raw feature column list, PCA explained-variance ratios, cumulative variance, SHAP availability flag, and the champion model used.

### 11.9 `client_embedding_2d.parquet`
- `cliente`, `x`, `y`, `projection_method`.

### 11.10 `client_neighbors.parquet`
- `cliente`, `neighbor_cliente`, `distance`.

### 11.11 `client_clusters.parquet`
- `cliente`, `cluster_id`, `probability` (if available).

### 11.12 `cluster_summary.parquet`
- `cluster_id`, aggregated metrics summarising behaviour per cluster.

### 11.13 `pattern_table.parquet`
- Joined view for Streamlit/reporting (policy, model, rule metrics).

### 11.14 `per_event_policy_scores.parquet`
- `cliente`, `event_key`, `data_evento`
- Deterministic policy recalls (`fifo`, `lifo`, `same_period`, `largest`, `smallest`, `eomfirst`, `regmonth`, `quarter`, `seqblock`, `typeprio`)
- Aliased columns prefixed with `policy_` mirroring deterministic policy scores for convenience.

### 11.15 `event_policy_predictions.parquet`
- `event_key`, `policy_pred`, `policy_true`
- Global metrics columns repeated per row (`accuracy`, `mean_recall`)
- Probability columns `proba_<policy>` covering the deterministic policy set.

### 11.16 `doc_map.parquet`
- `cliente`, `numero_documento`, `record_count`, `importo_totale`
- `data_documento_min`, `data_documento_max`, `data_pareggio_min`, `data_pareggio_max`
- `divisa`, `tipo_documento`, `doc_pareggio_primo`

### 11.17 `client_rules_accuracy.parquet`
- `cliente`, `policy_dom`
- `rules_recall_at_k`, `rules_match_rate`, `events_tested`

---

## 12. Appendix C — Mathematical notes

1. **Periodicity kernel score**
   - For gap array `g`, target `t`, bandwidth `b=3`: \( score = \frac{1}{|g|} \sum_i \exp\left(-\frac{1}{2} \left(\frac{g_i - t}{b}\right)^2\right) \).
2. **Amount preference slope**
   - Point-biserial correlation: \( r = \frac{(\bar{x}_1 - \bar{x}_0)}{s_x} \sqrt{p_1 p_0} \).
3. **Stability index**
   - For rolling std set `s`: \( stability = \frac{1}{1 + mean(s)} \).
4. **Event recall@k**
   - For event of cardinality `k`: compute top-k invoices by predicted probability, compare with ground truth.
5. **Match rate**
   - Binary indicator: 1 if recall@k = 1, else 0; aggregated per event or client.

---

## 13. Appendix D — Environment notes

- Python: 3.11.x.
- Recommended packages (minimum versions):
  - pandas ≥ 2.1
  - numpy ≥ 1.24
  - scikit-learn ≥ 1.3
  - catboost ≥ 1.2
  - autogluon.tabular ≥ 1.0
  - shap ≥ 0.44
  - plotly ≥ 5.17
  - streamlit ≥ 1.25
  - umap-learn ≥ 0.5 (optional but recommended)
  - hdbscan ≥ 0.8
- Ensure `pyarrow` is installed for Parquet IO.

### GPU considerations
- AutoGluon automatically detects CUDA-ready GPUs through PyTorch.
- CatBoost can leverage GPU if compiled with CUDA; adjust parameters accordingly (not enabled by default in scripts).

---

## 14. Appendix E — Troubleshooting FAQ

1. **`data/raw` is empty**
   - Populate with SAP extracts before ingestion. Use `--input` flags pointing to absolute paths if necessary.
2. **`ValueError: File X is missing required columns`**
   - Inspect header row; ensure Italian column names match exactly.
3. **`No labelled samples generated`**
   - Likely due to missing `data_pareggio`. Validate ingestion results.
4. **`CatBoost calibrator not applicable`**
   - Check class distribution; consider increasing `MAX_NEGATIVES_PER_EVENT` or adjusting sampling logic.
5. **`HDBSCAN` returns all noise**
   - Lower `--min-cluster-size` or ensure embeddings carry signal (verify `shap_available`).
6. **Streamlit loads but pages are empty**
   - Confirm that `pattern_table.parquet` and model metrics exist.
7. **`shap` raises GPU errors**
   - Install CPU-only SHAP or set `CUDA_VISIBLE_DEVICES=""` before running `make_client_embeddings`.

---

## 15. Change management

- When modifying feature engineering, update the Feature catalogue (Section 5) and Data dictionary (Section 11).
- Any changes to model hyperparameters or pipeline order must be reflected in Stage walkthroughs and Execution playbooks.
- Keep sample outputs representative: regenerate after significant performance shifts.

---

_End of technical runbook._
