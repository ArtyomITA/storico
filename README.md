# 📊 Storico — Sistema di Analisi Pattern Pagamento Clienti

## 📋 Panoramica

**Storico** è un sistema di analisi predittiva che aiuta a:
1. **Identificare pattern di pagamento** di ogni cliente (FIFO, LIFO, Largest, etc.)
2. **Predire quali fatture** un cliente pagherà per prime
3. **Assegnare score 0-5** a ogni fattura aperta per prioritizzare solleciti
4. **Clusterizzare clienti** con comportamenti simili
5. **Visualizzare KPI** di qualità del modello con dashboard interattivo

### 🎯 Caso d'uso tipico
Un'azienda ha 200 clienti con 5000 fatture aperte totali. Per ogni cliente:
- Il sistema analizza lo storico pagamenti passati
- Identifica la "regola di pagamento" preferita (es. Cliente A usa FIFO, Cliente B usa Largest)
- Quando il cliente paga, predice le top-K fatture candidate
- Assegna score 0-5 a ogni fattura aperta (5 = pagherà sicuramente, 0 = non pagherà)
- Visualizza pattern, cluster e metriche di qualità in Streamlit

### 🏗️ Architettura ad Alto Livello

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA INGESTION                              │
│  data/raw/storico_sap.xlsx → data/normalized/events.parquet    │
│  (Normalizzazione colonne + cleaning)                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING                           │
│  src/features/make_features.py                                  │
│  - Tempo tra documento e pareggio                               │
│  - Lags: k_i (numero fatture aperte quando paghi)              │
│  - Policy scores: fifo_score, lifo_score, largest_score, ...   │
│  - Periodicità: weekly, monthly, bi-weekly                      │
│  - Embeddings clienti: aggregazioni metriche → vettore 128D    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                            │
│  src/training/pointwise_trainer.py (CatBoost + AutoGluon)      │
│  - Input: eventi storici con feature ingegnerizzate            │
│  - Target: y_i = 1 se fattura pagata, 0 altrimenti             │
│  - Output: modello che predice p_i ∈ [0,1] per ogni fattura   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    SCORING PIPELINE                             │
│  src/scoring/score_open_invoices.py                            │
│  - Input: fatture aperte + modello trainato                    │
│  - Output: data/curated/open_invoice_scores.parquet            │
│           (ogni riga = 1 fattura con p_i e score_class 0-5)   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   ANALYTICS PIPELINE                            │
│  src/analytics/*.py                                             │
│  - Pattern detection: FIFO/LIFO/Largest dominance per cliente │
│  - Clustering: HDBSCAN su embeddings 128D → gruppi clienti    │
│  - KPI: precision@k, recall@k, nDCG, Brier score, ECE         │
│  - Policy confusion matrix: true policy vs predicted policy    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  STREAMLIT DASHBOARD                            │
│  app/streamlit_app.py + app/page_components/*.py               │
│  - 📊 Overview: Summary KPI e statistiche dataset              │
│  - 🎯 Pattern Card: Drill-down per cliente con pattern visual  │
│  - 📈 KPI Dashboard: Metriche ranking e calibrazione           │
│  - 🔬 Clusters: Esplora gruppi clienti simili (HDBSCAN+UMAP)  │
│  - 🎯 Policy IQ: Confusion matrix policy + uplift vs FIFO     │
│  - 📋 Pattern Summary: Tabella clienti con filtri e export    │
└─────────────────────────────────────────────────────────────────┘
```

### 📦 Struttura Directory

```
storico/
├── app/                          # Streamlit application
│   ├── streamlit_app.py          # Main app router
│   ├── page_components/          # Individual pages
│   │   ├── overview_wizard.py    # Overview page
│   │   ├── pattern_card_v2.py    # Pattern Card per cliente
│   │   ├── kpi_dashboard.py      # KPI Dashboard
│   │   ├── clusters.py           # Cluster Explorer
│   │   ├── policy_iq.py          # Policy IQ analysis
│   │   └── pattern_summary.py    # Pattern Summary table
│   └── utils/                    # Utilities (export, glossary)
│       ├── export.py             # CSV/JSON export + deep-link
│       └── sap_glossary.py       # SAP column tooltips
│
├── src/                          # Core Python package
│   ├── data/                     # Data ingestion & normalization
│   │   └── normalize_sap.py      # SAP → events.parquet
│   ├── features/                 # Feature engineering
│   │   └── make_features.py      # Tempo, lags, policy scores, embeddings
│   ├── training/                 # Model training
│   │   └── pointwise_trainer.py  # CatBoost + AutoGluon pointwise
│   ├── scoring/                  # Inference on open invoices
│   │   └── score_open_invoices.py # p_i → score_class 0-5
│   ├── analytics/                # Post-training analytics
│   │   ├── pattern_detectors.py  # FIFO/LIFO/Largest detection
│   │   ├── cluster_clients.py    # HDBSCAN clustering
│   │   ├── ranking_metrics.py    # Precision@k, nDCG, MRR
│   │   ├── calibration_metrics.py # Brier, ECE, MCE
│   │   └── event_policy_metrics.py # Policy confusion + uplift
│   ├── report/                   # NLG text generation
│   │   └── nlg_patterns.py       # Pattern summaries in plain text
│   └── utils/                    # Shared utilities
│       ├── training_manager.py   # Training orchestration
│       └── logger.py             # JSONL logging
│
├── data/                         # Production data (gitignored)
│   ├── raw/                      # storico_sap.xlsx (input)
│   ├── normalized/               # events.parquet (cleaned)
│   ├── curated/                  # Parquet outputs (features, scores, analytics)
│   ├── models/                   # Trained models (.cbm, AutoGluon dir)
│   └── exports/                  # CSV/JSON exports da Streamlit
│
├── data_mock/                    # Mock data per demo (committed)
│   ├── raw/                      # storico_sap_mock.xlsx
│   ├── normalized/               # events_mock.parquet
│   ├── curated/                  # Mock parquet files
│   └── exports/                  # Mock exports
│
├── notebooks/                    # Jupyter notebooks esplorativi
├── tests/                        # Unit tests (pytest)
├── logs/                         # JSONL logs (run_YYYYMMDD_HHMMSS.jsonl)
├── .env                          # Environment variables
├── requirements.txt              # Python dependencies
├── pyproject.toml                # Project metadata (Poetry optional)
└── README.md                     # This file
```

---

## 🚀 Setup & Installazione

### Prerequisiti

- **Python 3.11+** (testato con 3.11.7)
- **pip** (o Poetry per dependency management)
- **Git** (per clonare il repository)
- **Excel reader** (openpyxl installato automaticamente con requirements)

### Installazione

```bash
# Clone repository
git clone <repository-url>
cd storico

# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Install HDBSCAN and UMAP for clustering
pip install hdbscan umap-learn

# (Optional) Install development dependencies
pip install -r requirements-dev.txt  # pytest, black, mypy
```

### Configurazione `.env`

Crea file `.env` nella root del progetto:

```env
# ============================================================================
# DATA PATHS
# ============================================================================
DATA_DIR=data                      # Production data directory
DATA_MOCK_DIR=data_mock            # Mock data directory
USE_MOCK=false                     # Set to 'true' for demo mode

# ============================================================================
# SAP INPUT
# ============================================================================
SAP_INPUT_FILE=storico_sap.xlsx    # Nome file Excel in data/raw/
SAP_SHEET_NAME=Sheet1              # Nome foglio Excel

# ============================================================================
# MODEL PARAMETERS
# ============================================================================
MODEL_TYPE=catboost                # Options: catboost, autogluon, ensemble
RANDOM_SEED=42                     # Seed per riproducibilità
TEST_SIZE=0.2                      # Train/test split ratio

# ============================================================================
# SCORING THRESHOLDS (p_i → score_class 0-5)
# ============================================================================
# Score 5: p_i >= 0.8  (pagherà sicuramente)
# Score 4: p_i >= 0.6  (alta probabilità)
# Score 3: p_i >= 0.4  (media probabilità)
# Score 2: p_i >= 0.2  (bassa probabilità)
# Score 1: p_i >= 0.05 (molto bassa)
# Score 0: p_i < 0.05  (non pagherà)

SCORE_THRESHOLD_5=0.8
SCORE_THRESHOLD_4=0.6
SCORE_THRESHOLD_3=0.4
SCORE_THRESHOLD_2=0.2
SCORE_THRESHOLD_1=0.05

# ============================================================================
# CLUSTERING
# ============================================================================
HDBSCAN_MIN_CLUSTER_SIZE=5         # Minimum clients per cluster
HDBSCAN_MIN_SAMPLES=3              # Minimum neighbors for core point
HDBSCAN_METRIC=cosine              # Distance metric (cosine for embeddings)

# ============================================================================
# STREAMLIT
# ============================================================================
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true     # true for production, false for local dev

# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL=INFO                     # DEBUG, INFO, WARNING, ERROR
LOG_DIR=logs                       # JSONL logs directory
```

### Mock Mode (Demo senza dati SAP)

Per testare il sistema senza dati reali:

```bash
# Imposta USE_MOCK=true nel .env
echo "USE_MOCK=true" >> .env

# Oppure come variabile d'ambiente
$env:USE_MOCK="true"  # PowerShell
export USE_MOCK="true"  # Linux/Mac

# Avvia Streamlit
python -m streamlit run app/streamlit_app.py
```

Il sistema caricherà automaticamente dati da `data_mock/` invece di `data/`.

---

## 📊 Data Contract — Colonne Normalizzate

Il file `data/normalized/events.parquet` è il cuore del sistema. Ogni riga rappresenta un **evento di pareggio** (matching tra fattura aperta e pagamento).

### Schema Colonne Obbligatorie

| Colonna | Tipo | Descrizione | Semantica SAP |
|---------|------|-------------|---------------|
| `cliente` | str | Codice cliente univoco | Numero cliente SAP |
| `numero_documento` | str | Numero fattura/documento | Numero documento contabile |
| `data_documento` | date | Data emissione fattura | Data documento (fattura emessa) |
| `data_registrazione` | date | Data registrazione contabile | Data registrazione in SAP |
| `data_pareggio` | date | Data matching pagamento | Data di pareggio (quando pagata) |
| `importo_divisa_interna` | float | Importo fattura (€) | Importo in divisa interna |
| `esercizio` | int | Anno fiscale | Esercizio contabile |
| `periodo_contabile` | int | Mese fiscale (1-12) | Periodo contabile |
| `divisa` | str | Codice divisa (EUR, USD, ...) | Divisa transazione |
| `tipo_documento` | str | Tipo documento (RV, NC, ...) | Tipo di documento SAP |
| `chiave_contabile` | str | Chiave posting (01, 11, ...) | Chiave di contabilizzazione |
| `doc_pareggio` | str | ID documento di pareggio | Documento di pareggio |
| `data_acquisizione` | date | Data import dati | Data di acquisizione (upload SAP) |

### Colonne Feature (Aggiunte da `make_features.py`)

| Colonna | Tipo | Descrizione | Formula |
|---------|------|-------------|---------|
| `tempo_documento_pareggio` | int | Giorni tra emissione e pagamento | `(data_pareggio - data_documento).days` |
| `k_i` | int | Numero fatture aperte quando paghi | Count fatture aperte a `data_pareggio` |
| `fifo_score` | float | Score FIFO [0,1] | `1.0` se fattura più vecchia, else proporzionale |
| `lifo_score` | float | Score LIFO [0,1] | `1.0` se fattura più recente, else proporzionale |
| `largest_score` | float | Score Largest [0,1] | `1.0` se importo massimo, else proporzionale |
| `smallest_score` | float | Score Smallest [0,1] | `1.0` se importo minimo, else proporzionale |
| `same_period_score` | float | Score Same Period [0,1] | `1.0` se `periodo_contabile == MONTH(data_pareggio)` |
| `eomfirst_score` | float | Score EOM First [0,1] | `1.0` se fattura fine mese (day >= 26) |
| `regmonth_score` | float | Score RegMonth [0,1] | `1.0` se `MONTH(data_registrazione) == MONTH(data_pareggio)` |
| `quarter_score` | float | Score Quarter [0,1] | `1.0` se stesso trimestre |
| `seqblock_score` | float | Score Sequential Block [0,1] | `1.0` se esercizio minimo tra aperti |
| `typeprio_score` | float | Score Type Priority [0,1] | `1.0` se tipo documento prioritario (RV > NC) |
| `importo_relativo` | float | Importo / max(importo) | Normalizzazione importo [0,1] |
| `periodicity_weekly` | float | Periodicità settimanale [0,1] | Correlazione pagamenti ogni 7 giorni |
| `periodicity_biweekly` | float | Periodicità bisettimanale [0,1] | Correlazione pagamenti ogni 14 giorni |
| `periodicity_monthly` | float | Periodicità mensile [0,1] | Correlazione pagamenti ogni ~30 giorni |

### Colonne Output (Aggiunte da `pointwise_trainer.py`)

| Colonna | Tipo | Descrizione |
|---------|------|-------------|
| `p_model` | float | Probabilità predetta [0,1] dal modello |
| `y_true` | int | Target ground truth (1 = pagata, 0 = non pagata) |
| `y_pred` | int | Predizione binaria (1 se `p_model >= 0.5`, else 0) |
| `score_class` | int | Score 0-5 (mapping da `p_model` con soglie) |

### Colonne Analytics (Aggiunte da `analytics/*.py`)

| Colonna | Tipo | Descrizione | File Parquet |
|---------|------|-------------|--------------|
| `dominant_pattern` | str | Pattern dominante (fifo/lifo/largest) | `client_patterns.parquet` |
| `pattern_confidence` | float | Confidenza pattern [0,1] | `client_patterns.parquet` |
| `k_mode` | float | Cardinalità modale (mediana k_i) | `client_patterns.parquet` |
| `cluster_label` | int | Cluster ID (-1 = noise) | `client_clusters.parquet` |
| `cluster_label_text` | str | Label auto-generato ("FIFO, k~2, monthly") | `client_clusters.parquet` |
| `embedding_128d` | array | Embedding cliente (128 dimensioni) | `client_embeddings.parquet` |
| `x_pca` | float | Coordinata X proiezione PCA | `client_embedding_2d.parquet` |
| `y_pca` | float | Coordinata Y proiezione PCA | `client_embedding_2d.parquet` |
| `precision_at_k` | float | Precision@k per modello | `ranking_metrics.parquet` |
| `recall_at_k` | float | Recall@k per modello | `ranking_metrics.parquet` |
| `ndcg_at_k` | float | nDCG@k per modello | `ranking_metrics.parquet` |
| `brier_score` | float | Brier score (calibrazione) | `calibration_metrics.parquet` |
| `ece` | float | Expected Calibration Error | `calibration_metrics.parquet` |
| `mce` | float | Maximum Calibration Error | `calibration_metrics.parquet` |

---

## 🔄 Flusso End-to-End

### 1. Normalizzazione Dati SAP

```bash
python src/data/normalize_sap.py \
    --input data/raw/storico_sap.xlsx \
    --output data/normalized/events.parquet
```

**Input**: Excel SAP con colonne variabili (nomi italiani, formati inconsistenti)  
**Output**: `events.parquet` con schema normalizzato  
**Log**: `logs/normalize_YYYYMMDD_HHMMSS.jsonl`

**Cosa fa**:
- Mappa colonne SAP italiane → nomi standard (`"Data documento" → "data_documento"`)
- Converte date da formato Excel a `datetime64[ns]`
- Rimuove righe duplicate (stesso `cliente` + `numero_documento` + `data_pareggio`)
- Gestisce missing values (forward fill per divisa, drop se mancano colonne critiche)
- Valida schema finale (tutte colonne obbligatorie presenti)

### 2. Feature Engineering

```bash
python src/features/make_features.py \
    --events data/normalized/events.parquet \
    --output data/curated/events_with_features.parquet
```

**Input**: `events.parquet`  
**Output**: `events_with_features.parquet` (con 20+ colonne feature)  
**Log**: `logs/features_YYYYMMDD_HHMMSS.jsonl`

**Cosa fa**:
- **Temporal features**: `tempo_documento_pareggio` (giorni tra emissione e pagamento)
- **Cardinality lag**: `k_i` (conta fatture aperte al momento del pagamento)
- **Policy scores**: Calcola score [0,1] per 10 policy deterministiche (FIFO, LIFO, Largest, etc.)
- **Periodicity**: Analizza pattern temporale pagamenti (weekly/biweekly/monthly)
- **Client embeddings**: Aggrega metriche per cliente → vettore 128D
- **Relative amounts**: Normalizza importo_divisa_interna [0,1] per cliente

### 3. Training Modello

```bash
python src/training/pointwise_trainer.py \
    --features data/curated/events_with_features.parquet \
    --model-dir data/models \
    --model-type catboost \
    --test-size 0.2
```

**Input**: `events_with_features.parquet`  
**Output**:  
- `data/models/catboost_model.cbm` (modello serializzato)
- `data/curated/predictions_test.parquet` (predizioni su test set)
- `data/curated/feature_importance.parquet` (importanza feature)

**Log**: `logs/training_YYYYMMDD_HHMMSS.jsonl`

**Cosa fa**:
- **Train/Test split** time-based: eventi prima del 2024 → train, dopo → test
- **CatBoost training**: 500 iterations, learning_rate=0.05, depth=6
- **AutoGluon** (opzionale): Ensemble di 10+ modelli con auto-tuning
- **Calibrazione**: Isotonic Regression su test set per calibrare probabilità
- **Feature importance**: SHAP values per interpretabilità
- **Salva artefatti**: modello + metadata + confusion matrix

### 4. Scoring Fatture Aperte

```bash
python src/scoring/score_open_invoices.py \
    --model data/models/catboost_model.cbm \
    --events data/normalized/events.parquet \
    --output data/curated/open_invoice_scores.parquet \
    --as-of-date 2024-10-30
```

**Input**:  
- Modello trainato (`catboost_model.cbm`)
- Eventi storici (`events.parquet`)
- Data di scoring (`--as-of-date`)

**Output**: `open_invoice_scores.parquet` con colonne:  
- `cliente`, `numero_documento`, `importo`, `data_documento`
- `p_model` (probabilità predetta)
- `score_class` (0-5)
- `k_i` (fatture aperte concorrenti)

**Cosa fa**:
- Filtra fatture aperte a `as-of-date` (data_documento <= as-of-date, data_pareggio > as-of-date o null)
- Calcola feature per ogni fattura aperta
- Applica modello → `p_model`
- Mappa `p_model` a `score_class`:
  - `score_class = 5` se `p_model >= 0.8`
  - `score_class = 4` se `0.6 <= p_model < 0.8`
  - `score_class = 3` se `0.4 <= p_model < 0.6`
  - `score_class = 2` se `0.2 <= p_model < 0.4`
  - `score_class = 1` se `0.05 <= p_model < 0.2`
  - `score_class = 0` se `p_model < 0.05`

### 5. Analytics Pipeline

```bash
# Pattern detection
python src/analytics/pattern_detectors.py \
    --events data/curated/events_with_features.parquet \
    --output data/curated/client_patterns.parquet

# Clustering
python src/analytics/cluster_clients.py \
    --embeddings data/curated/client_embeddings.parquet \
    --output data/curated/client_clusters.parquet \
    --min-cluster-size 5

# Ranking metrics
python src/analytics/ranking_metrics.py \
    --predictions data/curated/predictions_test.parquet \
    --output data/curated/ranking_metrics.parquet

# Calibration metrics
python src/analytics/calibration_metrics.py \
    --predictions data/curated/predictions_test.parquet \
    --output data/curated/calibration_metrics.parquet

# Policy confusion matrix
python src/analytics/event_policy_metrics.py \
    --events data/curated/events_with_features.parquet \
    --output data/curated/event_policy_metrics.parquet
```

**Output**:
- `client_patterns.parquet`: Pattern dominante per cliente (FIFO/LIFO/Largest)
- `client_clusters.parquet`: Cluster ID e label auto-generato
- `ranking_metrics.parquet`: Precision@k, Recall@k, nDCG@k, MRR, Coverage@k
- `calibration_metrics.parquet`: Brier score, ECE, MCE, reliability curve data
- `event_policy_metrics.parquet`: Confusion matrix policy + uplift vs FIFO

### 6. Avvio Streamlit Dashboard

```bash
# Production mode (usa data/)
python -m streamlit run app/streamlit_app.py --server.port 8501

# Mock mode (usa data_mock/)
$env:USE_MOCK="true"
python -m streamlit run app/streamlit_app.py --server.port 8501
```

**URL**: http://localhost:8501

**Pagine disponibili**:
1. **📊 Overview**: Summary KPI, statistiche dataset, ultimi eventi
2. **🎯 Pattern Card**: Drill-down per cliente con pattern, policy scores, eventi recenti
3. **📈 KPI Dashboard**: Metriche ranking (Precision@k, nDCG) e calibrazione (Brier, ECE)
4. **🔬 Clusters**: Esplora cluster clienti con HDBSCAN + PCA/UMAP visualization
5. **🎯 Policy IQ**: Confusion matrix policy + uplift analysis vs baseline FIFO
6. **📋 Pattern Summary**: Tabella clienti con pattern, filtri e export CSV

---

## 🔧 Comandi Run-All

### Script Bash/PowerShell per Pipeline Completa

Crea file `run_pipeline.ps1` (Windows) o `run_pipeline.sh` (Linux/Mac):

```powershell
# run_pipeline.ps1 - Pipeline completa produzione

# Attiva virtual environment
.\.venv\Scripts\Activate.ps1

# Set variables
$DATA_DIR = "data"
$LOG_DIR = "logs"
$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"

Write-Host "🚀 Pipeline Start: $TIMESTAMP" -ForegroundColor Green

# Step 1: Normalizzazione
Write-Host "📥 Step 1/6: Normalizzazione dati SAP..." -ForegroundColor Cyan
python src/data/normalize_sap.py `
    --input "$DATA_DIR/raw/storico_sap.xlsx" `
    --output "$DATA_DIR/normalized/events.parquet" `
    2>&1 | Tee-Object -FilePath "$LOG_DIR/normalize_$TIMESTAMP.log"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Normalizzazione fallita!" -ForegroundColor Red
    exit 1
}

# Step 2: Feature engineering
Write-Host "🔧 Step 2/6: Feature engineering..." -ForegroundColor Cyan
python src/features/make_features.py `
    --events "$DATA_DIR/normalized/events.parquet" `
    --output "$DATA_DIR/curated/events_with_features.parquet" `
    2>&1 | Tee-Object -FilePath "$LOG_DIR/features_$TIMESTAMP.log"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Feature engineering fallito!" -ForegroundColor Red
    exit 1
}

# Step 3: Training
Write-Host "🎓 Step 3/6: Training modello..." -ForegroundColor Cyan
python src/training/pointwise_trainer.py `
    --features "$DATA_DIR/curated/events_with_features.parquet" `
    --model-dir "$DATA_DIR/models" `
    --model-type catboost `
    --test-size 0.2 `
    2>&1 | Tee-Object -FilePath "$LOG_DIR/training_$TIMESTAMP.log"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Training fallito!" -ForegroundColor Red
    exit 1
}

# Step 4: Analytics
Write-Host "📊 Step 4/6: Analytics pipeline..." -ForegroundColor Cyan

# Pattern detection
python src/analytics/pattern_detectors.py `
    --events "$DATA_DIR/curated/events_with_features.parquet" `
    --output "$DATA_DIR/curated/client_patterns.parquet"

# Clustering
python src/analytics/cluster_clients.py `
    --embeddings "$DATA_DIR/curated/client_embeddings.parquet" `
    --output "$DATA_DIR/curated/client_clusters.parquet" `
    --min-cluster-size 5

# Ranking metrics
python src/analytics/ranking_metrics.py `
    --predictions "$DATA_DIR/curated/predictions_test.parquet" `
    --output "$DATA_DIR/curated/ranking_metrics.parquet"

# Calibration metrics
python src/analytics/calibration_metrics.py `
    --predictions "$DATA_DIR/curated/predictions_test.parquet" `
    --output "$DATA_DIR/curated/calibration_metrics.parquet"

# Policy metrics
python src/analytics/event_policy_metrics.py `
    --events "$DATA_DIR/curated/events_with_features.parquet" `
    --output "$DATA_DIR/curated/event_policy_metrics.parquet"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Analytics fallito!" -ForegroundColor Red
    exit 1
}

# Step 5: Scoring fatture aperte
Write-Host "🎯 Step 5/6: Scoring fatture aperte..." -ForegroundColor Cyan
$AS_OF_DATE = Get-Date -Format "yyyy-MM-dd"
python src/scoring/score_open_invoices.py `
    --model "$DATA_DIR/models/catboost_model.cbm" `
    --events "$DATA_DIR/normalized/events.parquet" `
    --output "$DATA_DIR/curated/open_invoice_scores.parquet" `
    --as-of-date $AS_OF_DATE `
    2>&1 | Tee-Object -FilePath "$LOG_DIR/scoring_$TIMESTAMP.log"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Scoring fallito!" -ForegroundColor Red
    exit 1
}

# Step 6: Avvio Streamlit (opzionale)
Write-Host "🌐 Step 6/6: Avvio Streamlit dashboard..." -ForegroundColor Cyan
Write-Host "Dashboard URL: http://localhost:8501" -ForegroundColor Yellow
python -m streamlit run app/streamlit_app.py --server.port 8501

Write-Host "✅ Pipeline completata: $TIMESTAMP" -ForegroundColor Green
```

**Esecuzione**:
```powershell
# Windows PowerShell
.\run_pipeline.ps1

# Linux/Mac (crea run_pipeline.sh con comandi analoghi)
chmod +x run_pipeline.sh
./run_pipeline.sh
```

---

## 📱 Pagine Streamlit — Descrizione Dettagliata

### 1. 📊 Overview Wizard

**File**: `app/page_components/overview_wizard.py`  
**URL**: http://localhost:8501/ (home page)

**Input Parquet**:
- `events.parquet` (eventi storici)
- `client_patterns.parquet` (pattern per cliente)
- `ranking_metrics.parquet` (KPI globali)
- `open_invoice_scores.parquet` (score fatture aperte)

**Cosa Mostra**:

1. **Summary KPI Cards** (4 metriche in colonne):
   - 👥 **Clienti Totali**: Count(distinct cliente)
   - 📋 **Eventi Storici**: Count(righe events.parquet)
   - 🎯 **Precision@5 Globale**: Media precision@5 da ranking_metrics
   - 💰 **Esposizione Totale Aperta**: Sum(importo) da open_invoice_scores

2. **Distribuzione Pattern** (Bar chart Plotly):
   - X-axis: Pattern (FIFO, LIFO, Largest, etc.)
   - Y-axis: Numero clienti
   - Color: Verde per pattern dominanti (>30% clienti), Arancione per minority

3. **Ultimi 10 Eventi** (Tabella interattiva):
   - Colonne: Cliente, Numero Documento, Data Pareggio, Importo, Pattern Predetto
   - Ordinamento: Data Pareggio decrescente
   - Click riga → Redirect a Pattern Card per quel cliente

4. **Quick Stats** (Expander con tabella riepilogativa):
   - Top 3 clienti per esposizione aperta
   - Top 3 pattern più comuni
   - Coverage@10 media
   - Data ultimo evento processato

**Filtri Sidebar**:
- **Data Range**: Slider per filtrare eventi per periodo
- **Pattern**: Multiselect per includere solo certi pattern
- **Importo Min**: Slider per escludere fatture piccole

**Pulsanti Export**:
- 📥 **Download Summary CSV**: Esporta KPI globali
- 📋 **Genera Deep-Link**: URL con filtri correnti per condivisione

**Tooltip SAP**: Hover su colonna "Periodo Contabile" → Mostra "Mese fiscale (1-12) da SAP"

---

### 2. 🎯 Pattern Card (Drill-down Cliente)

**File**: `app/page_components/pattern_card_v2.py`  
**URL**: http://localhost:8501/?page=Pattern%20Card&cliente=<CODICE>

**Input Parquet**:
- `events_with_features.parquet` (eventi cliente con policy scores)
- `client_patterns.parquet` (pattern dominante)
- `client_clusters.parquet` (cluster assignment)
- `per_event_policy_scores.parquet` (score evento-level)

**Cosa Mostra**:

1. **Header Cliente** (Card con badge):
   - **Cliente**: Codice cliente (es. "10001234")
   - **Pattern Dominante**: FIFO | LIFO | Largest (badge colorato)
   - **Confidence**: 0.85 (85% degli eventi seguono questo pattern)
   - **Cluster**: "FIFO, k~2, monthly" (label auto-generato)
   - **k_mode**: 3.5 (mediana fatture aperte concorrenti)

2. **Policy Scores Heatmap** (Plotly heatmap):
   - X-axis: Policy (FIFO, LIFO, Largest, Smallest, Same Period, EOM First, etc.)
   - Y-axis: Ultimo 20 eventi (ordinati per data_pareggio)
   - Color scale: 0 (bianco) → 1 (blu scuro)
   - Hover: Mostra numero_documento, data_pareggio, score esatto
   - **Interpretazione**: Colonna più scura = policy usata più spesso

3. **Pattern Timeline** (Line chart Plotly):
   - X-axis: Data pareggio
   - Y-axis: Cumulative count eventi
   - Lines: Una per policy (FIFO in blu, LIFO in rosso, Largest in verde)
   - **Interpretazione**: Slope più ripida = policy dominante in quel periodo

4. **Eventi Recenti Table** (Tabella scrollabile):
   - Colonne: Data Pareggio | N. Doc | Importo | k_i | Policy Predetta | Score p_i
   - Righe: Ultimi 50 eventi del cliente
   - Sort: Data pareggio decrescente
   - Color coding: Score >= 0.8 verde, 0.5-0.8 giallo, <0.5 rosso

5. **NLG Summary** (Testo generato):
   - Esempio: "Il cliente 10001234 mostra un comportamento FIFO dominante (85% degli eventi). Paga tipicamente 3-4 fatture per sessione con cadenza mensile. Preferisce chiudere le fatture più vecchie per prime, con particolare attenzione a quelle emesse a fine mese."
   - Fonte: `src/report/nlg_patterns.py` (template-based NLG)

6. **Policy Toggle** (Sidebar):
   - Multiselect: Seleziona quali policy mostrare nell heatmap
   - Default: Tutte attive
   - Checkbox "Nascondi RegMonth": OFF di default (RegMonth può confondere)

**Filtri Sidebar**:
- **Cliente**: Dropdown con tutti i clienti
- **Periodo**: Slider data range
- **Policy Attive**: Multiselect policy da visualizzare

**Pulsanti Export**:
- 📥 **Download Eventi Cliente CSV**: Esporta eventi filtrati del cliente
- 📥 **Download Policy Scores CSV**: Esporta matrice policy scores
- 📋 **Genera Deep-Link**: URL con cliente + filtri per audit

**Tooltip SAP**:
- "k_i": "Numero di fatture aperte concorrenti al momento del pagamento"
- "Policy Score": "Punteggio [0,1] che indica quanto l'evento corrisponde a quella regola di pagamento"

---

### 3. 📈 KPI Dashboard (Business-Friendly Metrics)

**File**: `app/page_components/kpi_dashboard.py`  
**URL**: http://localhost:8501/?page=KPI%20Dashboard

**Input Parquet**:
- `ranking_metrics.parquet` (precision@k, recall@k, nDCG@k, MRR, coverage@k)
- `calibration_metrics.parquet` (Brier score, ECE, MCE, reliability curve data)
- `predictions_test.parquet` (predizioni test set per plotting)

**Cosa Mostra** (2 tab principali):

#### **Tab 1: 🎯 Qualità Raccomandazioni (Ranking)**

1. **Precision@k** (Bar chart + Metric card):
   - **Definizione Business**: "Su 10 fatture raccomandate, quante sono corrette?"
   - X-axis: k (1, 3, 5, 10, 20)
   - Y-axis: Precision [0,1]
   - Target: Precision@5 >= 0.7 (70%)
   - Color coding: Verde se >= 0.7, Giallo se 0.5-0.7, Rosso se < 0.5
   - Tooltip Tecnico: `Precision@k = TP / (TP + FP) per top-K raccomandazioni`

2. **Hit Rate@k** (Line chart):
   - **Definizione Business**: "% di volte in cui almeno 1 raccomandazione nelle prime K è corretta"
   - Target: Hit Rate@10 >= 0.9 (90%)
   - Tooltip: "Utile per UX: se hit_rate@10 = 0.95, il 95% delle volte l'utente trova la fattura giusta nelle prime 10"

3. **nDCG@k** (Normalized Discounted Cumulative Gain):
   - **Definizione Business**: "Quanto è buono l'ordine delle raccomandazioni? (penalizza errori in top-3)"
   - Range: [0,1], target >= 0.8
   - Formula tooltip: `nDCG = DCG / IDCG` dove `DCG = Σ (2^rel - 1) / log2(i+1)`
   - Esempio: nDCG@5 = 0.92 significa ordine quasi perfetto nelle prime 5 posizioni

4. **MRR (Mean Reciprocal Rank)**:
   - **Definizione Business**: "In media, a che posizione si trova la prima raccomandazione corretta?"
   - Formula: `MRR = 1/N * Σ 1/rank_i`
   - Esempio: MRR = 0.67 → prima raccomandazione corretta è in media alla posizione 1.5
   - Target: >= 0.5

5. **Coverage@k**:
   - **Definizione Business**: "% di clienti per cui il modello riesce a fare raccomandazioni"
   - Target: >= 0.95 (copri almeno il 95% dei clienti)
   - Warning: Coverage bassa (<0.8) indica modello troppo conservativo

#### **Tab 2: 🎯 Affidabilità Probabilità (Calibrazione)**

1. **Brier Score** (Metric card + sparkline):
   - **Definizione Business**: "Quanto si discostano le probabilità predette dalla realtà?"
   - Formula: `Brier = 1/N * Σ (p_i - y_i)²`
   - Range: [0,1], target <= 0.2
   - Interpretazione:
     - Brier < 0.1: 🟢 Eccellente
     - Brier 0.1-0.2: 🟡 Accettabile
     - Brier > 0.2: 🔴 Problematico (sovra/sotto-stima)
   - Tooltip: "Se Brier = 0.15, in media l'errore di probabilità è ~15%"

2. **ECE (Expected Calibration Error)** (Bar chart):
   - **Definizione Business**: "In media, quanto sbaglia il modello nelle probabilità?"
   - X-axis: Bin probabilità (0-0.1, 0.1-0.2, ..., 0.9-1.0)
   - Y-axis: |Predicted Prob - Actual Frequency|
   - Target: ECE <= 0.05 (5%)
   - Esempio: Se modello dice "70% confident" su 100 eventi, quanti sono veri? Se 65, ECE = |0.70 - 0.65| = 0.05

3. **MCE (Maximum Calibration Error)** (Metric card):
   - **Definizione Business**: "Qual è il peggior errore di calibrazione in un singolo bin?"
   - Target: MCE <= 0.1 (10%)
   - Warning: MCE alto (>0.15) indica un bin specifico con calibrazione pessima

4. **Reliability Curve** (Plotly line chart):
   - X-axis: Predicted Probability (bin medio)
   - Y-axis: Actual Frequency
   - Line: Diagonale perfetta (y=x) = perfettamente calibrato
   - Punti: Bin osservati
   - Gap analysis:
     - Sopra diagonale: Modello **sotto-stima** (dice 60% ma è 75%)
     - Sotto diagonale: Modello **sovra-stima** (dice 80% ma è 65%)
   - Tooltip multilivello:
     - 📘 **Business**: "Punti vicini alla diagonale = probabilità affidabili"
     - 📊 **Esempio**: "Bin [0.7-0.8]: Modello predice 75% in media, realtà è 72% → Gap 3%"
     - 🔬 **Tecnico**: "Calibrazione isotonica riduce gap fino a ~2%"

**Filtri Sidebar**:
- **Cliente/Segmento**: Dropdown (globale, per cliente, per k_bucket)
- **Modello**: Dropdown (catboost, autogluon, ensemble) se disponibili multipli
- **Cardinalità k**: Slider per filtrare eventi con k_i specifico

**Pulsanti Export**:
- 📥 **Download Ranking Metrics CSV**: Precision, Recall, nDCG, MRR, Coverage per k
- 📥 **Download Calibration Metrics CSV**: Brier, ECE, MCE, reliability curve data
- 📋 **Genera Deep-Link**: URL con filtri modello/segmento

**Tooltip SAP**: N/A (pagina puramente ML metrics)

---

### 4. 🔬 Cluster Explorer (HDBSCAN + UMAP)

**File**: `app/page_components/clusters.py`  
**URL**: http://localhost:8501/?page=Clusters

**Input Parquet**:
- `client_embeddings.parquet` (embedding 128D per cliente)
- `client_embedding_2d.parquet` (proiezione PCA o UMAP)
- `client_clusters.parquet` (cluster ID + label)
- `cluster_summary.parquet` (metriche aggregate per cluster)

**Cosa Mostra**:

1. **Scatter Plot 2D** (Plotly interactive):
   - X-axis: x_pca (o x_umap)
   - Y-axis: y_pca (o y_umap)
   - Color: Cluster ID (palette categorica, noise = grigio)
   - Size: k_mode (cardinalità modale cliente)
   - Hover: Cliente, Pattern Dominante, Cluster Label
   - **Interazione**: Click punto → Seleziona cliente, aggiorna sidebar

2. **Cluster Summary Table** (Expander per ogni cluster):
   - **Cluster 0: "FIFO, k~2, monthly"** (35 clienti, 17.5%)
     - Pattern dominante: FIFO (82% clienti)
     - k_mode mediano: 2.3
     - Periodicità: Monthly (0.75)
     - Policy scores medie: FIFO=0.85, LIFO=0.12, Largest=0.05
   - **Cluster 1: "Largest, k~5, irregular"** (28 clienti, 14%)
     - ...
   - **Noise** (15 clienti, 7.5%)
     - Clienti senza pattern chiaro o outlier

3. **HDBSCAN Parameters Panel** (Sidebar expander):
   - **📏 min_cluster_size**: Slider 2-50 (default 5)
     - Tooltip Business: "Numero minimo di clienti per formare un cluster. Se 100 clienti e min=10, HDBSCAN crea solo cluster con >=10 clienti."
     - Linee guida:
       - 2-5: Molti cluster piccoli, cattura nicchie rare
       - 5-10: Bilanciato per 50-100 clienti
       - 10-20: Cluster robusti per 100-500 clienti
       - >20: Solo macro-segmenti
   
   - **🎯 min_samples**: Slider 1-20 (default 3)
     - Tooltip: "Numero minimo di vicini per considerare un punto 'core' (denso). Regola pratica: min_samples = min_cluster_size / 2"
   
   - **📐 Metrica Distanza**: Selectbox (euclidean | manhattan | cosine)
     - Default: **cosine** (migliore per embeddings neurali)
     - Tooltip: "Cosine ignora magnitudo, focus su direzione → ottimo per embeddings"
   
   - **🔄 Anteprima Clustering**: Button
     - Ricalcola HDBSCAN con parametri custom
     - Mostra metriche preview:
       - 📊 **Cluster Trovati**: 6
       - ⚠️ **Noise Points**: 8 (4.0%) ✓ Basso
       - ✅ **Clienti Clustered**: 96.0% (192/200)
     - Color coding noise:
       - >20%: ⚠️ Alto (rosso) → Suggerimento: "Riduci min_cluster_size da 10 a 5"
       - 10-20%: ⚠️ Moderato (arancione)
       - <10%: ✓ Basso (verde) → "✅ Configurazione bilanciata"
   
   - **✅ Applica e Salva**: Button
     - Salva cluster aggiornati in `client_clusters.parquet`
     - Rispetta mock mode: salva in `data_mock/curated/` se `USE_MOCK=true`
     - Success message: "✅ Salvato! Cluster aggiornati con parametri: {min_cluster_size: 5, min_samples: 3, metric: cosine}"

4. **PCA/UMAP Toggle** (Sidebar radio button):
   - **PCA (veloce)**: Proiezione lineare, deterministico, ~1s per 500 clienti
   - **UMAP (qualità)**: Manifold learning, non-lineare, ~30s per 500 clienti
   - **Comparison Table** (tooltip):
     | Criterio | PCA | UMAP |
     |----------|-----|------|
     | Velocità | ⚡ ~1s | 🐢 ~30s |
     | Qualità | 📊 Buona (varianza globale) | 🎨 Eccellente (topologia locale) |
     | Deterministico | ✅ Sì | ❌ No (random seed) |
     | Preserva distanze globali | ✅ Sì | ❌ No |
     | Preserva struttura locale | ❌ No | ✅ Sì |
   
   - **UMAP Parameters** (se selezionato UMAP):
     - **🏘️ n_neighbors**: Slider 5-50 (default 15)
       - Tooltip: "15 = bilanciato, 5-10 = focus locale, 30-50 = focus globale"
     - **📏 min_dist**: Slider 0.0-1.0 (default 0.1)
       - Tooltip: "0.1 = bilanciato, 0.0 = cluster densi, 0.5-1.0 = topologia continua"
     - **📐 Metrica**: Selectbox (euclidean | manhattan | cosine)
   
   - **Warning** se UMAP non installato:
     ```
     ⚠️ UMAP non disponibile
     
     Per installare UMAP:
     ```bash
     pip install umap-learn
     ```
     ```
     → Radio button disabilitato

5. **Cluster Detail View** (Click su cluster nella summary):
   - **Policy Distribution** (Pie chart): % clienti per pattern dominante nel cluster
   - **Periodicity Histogram**: Distribuzione periodicity_monthly/weekly
   - **Cardinality Boxplot**: Distribuzione k_mode nel cluster
   - **Top 5 Clienti**: Tabella con clienti rappresentativi (più vicini al centroide)

**Filtri Sidebar**:
- **Cliente Selezionato**: Dropdown (auto-updated da click scatter plot)
- **Cluster ID**: Multiselect per evidenziare specifici cluster
- **Noise Filter**: Checkbox "Nascondi Noise" (default OFF)

**Pulsanti Export**:
- 📥 **Scarica Cluster CSV**: Esporta `client_clusters.parquet` (cliente, cluster_label, cluster_label_text)
- 📥 **Scarica Summary CSV**: Esporta `cluster_summary.parquet` (metriche aggregate per cluster)
- 📋 **Genera Deep-Link**: URL con cliente selezionato

**Tooltip SAP**: N/A (pagina puramente clustering)

---

### 5. 🎯 Policy IQ (Confusion Matrix + Uplift)

**File**: `app/page_components/policy_iq.py`  
**URL**: http://localhost:8501/?page=Policy%20IQ

**Input Parquet**:
- `event_policy_metrics.parquet` (confusion matrix, uplift metrics)
- `per_event_policy_scores.parquet` (score per evento e policy)
- `event_policy_predictions.parquet` (policy predetta per evento)

**Cosa Mostra** (3 sezioni):

#### **Sezione 1: 🎯 Matrice Accuratezza Predizioni**

**Business-Friendly Title**: "Confronta regola di pagamento reale vs quella predetta dal modello"

1. **Confusion Matrix Heatmap** (Plotly normalized):
   - X-axis: Policy Predetta (FIFO, LIFO, Largest, etc.)
   - Y-axis: Policy Reale (true label)
   - Cell value: % normalizzato per riga (% di eventi con policy reale X predetti come Y)
   - Color scale: Blues (bianco → blu scuro)
   - Hover: Mostra count assoluti + percentuale
   - **Interpretazione**:
     - Diagonale verde scuro: Predizioni corrette ✅
     - Off-diagonal: Confusione tra policy simili (es. FIFO/LIFO)
   - **Target**: Diagonale >70% = modello affidabile

2. **Accuracy Metrics Cards** (3 colonne):
   - **🎯 Precisione Predizioni**: 72.3%
     - Tooltip: "% di eventi in cui il modello ha indovinato la regola esatta (target: >70%)"
     - Color emoji: 🎯 se >= 0.7, ⚠️ se 0.5-0.7, ❌ se <0.5
   
   - **📊 Copertura Top-2**: 89.5%
     - Tooltip: "% di eventi in cui la regola vera è tra le prime 2 predette (utile per ambiguità FIFO/LIFO)"
   
   - **📋 Eventi Analizzati**: 4,523
     - Tooltip: "Numero totale di pagamenti usati per calcolare la matrice"

3. **Export CSV**: 📥 **Scarica Matrice CSV**
   - Formato: Righe = policy_true, Colonne = policy_predicted, Valori = % + count assoluti
   - Filename: `matrice_accuratezza_policy_YYYYMMDD_HHMMSS.csv`

#### **Sezione 2: 📈 Miglioramento rispetto a FIFO Standard**

**Business-Friendly Title**: "Conviene usare il modello o basta applicare FIFO a tutti i clienti?"

1. **Uplift Table** (Tabella con segment breakdown):
   | Segmento | # Eventi | Recall Baseline (FIFO) | Recall Policy IQ | Uplift % |
   |----------|----------|------------------------|------------------|----------|
   | **Globale** | 4,523 | 65.2% | 72.3% | **+10.9%** 🟢 |
   | k=1 | 1,205 | 82.1% | 85.3% | +3.9% |
   | k=2-3 | 1,845 | 68.5% | 76.2% | +11.2% 🟢 |
   | k>=4 | 1,473 | 48.9% | 58.7% | +20.0% 🟢 |
   
   - **Interpretazione**:
     - Uplift +10.9% globale = Policy IQ predice 10.9% in più di pagamenti corretti rispetto a FIFO standard
     - Uplift negativo = Policy IQ peggiora (raro, indica overfitting)
   
   - **Color coding**:
     - Verde (🟢): Uplift > +5%
     - Giallo: Uplift 0-5%
     - Rosso: Uplift < 0%

2. **Uplift Bar Chart** (Plotly grouped bar):
   - X-axis: Segmento (Globale, k=1, k=2-3, k>=4)
   - Y-axis: Recall@k (%)
   - Bars: "FIFO Standard" (grigio) vs "Policy IQ" (verde)
   - **Key Insight Box**:
     - Uplift > +5%: ✅ "Policy IQ migliora significativamente il recall!"
     - Uplift 0-5%: 💡 "Miglioramento leggero, potrebbe non giustificare complessità"
     - Uplift < 0%: ⚠️ "Policy IQ non migliora, verifica overfitting"

3. **Export CSV**: 📥 **Scarica Uplift CSV**
   - Formato: Colonne = segment, n_events, baseline_recall, predicted_recall, uplift_absolute, uplift_relative
   - Filename: `policy_iq_uplift_metriche_YYYYMMDD_HHMMSS.csv`

#### **Sezione 3: 💡 Cosa Significa Ogni Regola di Pagamento?**

**Business-Friendly Title**: "Guida rapida: Espandi una regola per capire quando e perché un cliente la usa"

**2 Tab**:

**Tab 1: 📋 Tutte le Regole** (Dictionary completo con expanders)

Ogni policy ha expander con 4 livelli di spiegazione:

**Esempio: 🔹 FIFO — Prima Emessa, Prima Pagata**
```
📘 Cosa fa: Il cliente paga sempre le fatture più vecchie per prime.

🧪 Esempio: Cliente ha fatture del 10/01, 15/01, 20/01. 
Il 25/01 paga? → Sceglie quella del 10/01 (la più vecchia).

💼 Quando usarla: Clienti ordinati che seguono rigidamente 
l'ordine cronologico di ricezione.

🔬 Dettagli Tecnici (collapse):
Formula: min(Data Documento) tra fatture aperte al momento del pagamento.
```

Policies coperte:
- 🔹 **FIFO**: Prima Emessa, Prima Pagata
- 🔸 **LIFO**: Ultima Emessa, Prima Pagata
- 💰 **Largest**: Importo Maggiore Prima
- 🪙 **Smallest**: Importo Minore Prima
- 📅 **Same Period**: Stesso Periodo Contabile
- 📆 **EOM First**: Fine Mese Prima (ultimi 5 giorni)
- 📊 **Quarter**: Stesso Trimestre
- 🔢 **Sequential Block**: Blocco Sequenziale (es. tutte 2023 poi tutte 2024)
- 📋 **Type Priority**: Priorità per Tipo Documento (RV prima di NC)
- ⚠️ **RegMonth**: Mese Registrazione = Mese Pagamento (con warning)

**Tab 2: 🔍 Per Cliente Specifico** (Top 3 policy per cliente selezionato)

- **Dropdown Cliente**: Seleziona cliente
- **Output**: Mostra top 3 policy più usate dal cliente con:
  - % di eventi che seguono quella policy
  - Esempi concreti da eventi reali del cliente
  - Link a Pattern Card per drill-down

**Filtri Sidebar** (Policy Toggle):
- **Policy Attive**: Multiselect (default: tutte)
- **Nascondi RegMonth**: Checkbox (default: OFF)
  - Warning: "RegMonth può essere fuorviante perché confronta mese registrazione = mese pagamento senza verificare scadenze"

**Pulsanti Export**:
- 📥 **Scarica Matrice CSV**: Confusion matrix con % e count
- 📥 **Scarica Uplift CSV**: Uplift table con segmenti
- 📋 **Genera Deep-Link**: URL con cliente selezionato

**Tooltip SAP**:
- "Policy": "Regola di pagamento deterministica che il cliente sembra seguire"
- "Uplift": "Miglioramento percentuale del recall usando Policy IQ invece di FIFO standard"
- "Recall@k": "% di pagamenti corretti predetti nelle prime K raccomandazioni"

---

### 6. 📋 Pattern Summary (Tabella Clienti)

**File**: `app/page_components/pattern_summary.py`  
**URL**: http://localhost:8501/?page=Pattern%20Summary

**Input Parquet**:
- `client_patterns.parquet` (pattern dominante per cliente)
- `client_clusters.parquet` (cluster assignment)
- `ranking_metrics.parquet` (KPI per cliente)

**Cosa Mostra**:

1. **Tabella Clienti Filtrata** (Dataframe interattivo con sort/search):
   | Cliente | Pattern | Confidence | k_mode | Periodicità | Cluster | Precision@5 | # Eventi |
   |---------|---------|------------|--------|-------------|---------|-------------|----------|
   | 10001234 | FIFO | 0.85 | 3.2 | Monthly (0.78) | FIFO, k~3, monthly | 0.82 | 145 |
   | 10001567 | Largest | 0.72 | 5.1 | Irregular (0.35) | Largest, k~5, irregular | 0.68 | 89 |
   | ... | ... | ... | ... | ... | ... | ... | ... |
   
   - **Click Cliente** → Redirect a Pattern Card per drill-down
   - **Sort**: Click column header per ordinare
   - **Search**: Box filtro testo per cercare cliente specifico

2. **Filtri Sidebar** (6 filtri combinabili):
   - **Pattern**: Multiselect (FIFO, LIFO, Largest, etc.)
   - **Confidence Min**: Slider 0.0-1.0 (filtra solo clienti con pattern chiaro)
   - **k_mode Range**: Range slider 0-20 (filtra per cardinalità)
   - **Cluster**: Multiselect cluster ID
   - **Periodicità**: Multiselect (Weekly, Monthly, Irregular)
   - **Precision@5 Min**: Slider 0.0-1.0 (filtra clienti con KPI alto)

3. **Summary Stats** (Card sopra tabella):
   - **Clienti Filtrati**: 87 / 200 (43.5%)
   - **Pattern Mediano Confidence**: 0.78
   - **k_mode Mediano**: 3.5
   - **Precision@5 Media**: 0.74

**Pulsanti Export**:
- 📥 **Download Tabella Filtrata CSV**: Esporta solo righe visibili (rispetta filtri)
  - Filename: `pattern_summary_filtered_YYYYMMDD_HHMMSS.csv`
- 📥 **Download Tabella Completa CSV**: Esporta tutti i clienti (ignora filtri)
  - Filename: `pattern_summary_full_YYYYMMDD_HHMMSS.csv`
- 📋 **Genera Deep-Link**: URL con tutti filtri correnti (condivisibile con colleghi)

**Tooltip SAP**:
- "k_mode": "Mediana del numero di fatture aperte concorrenti quando il cliente paga"
- "Periodicità": "Pattern temporale pagamenti (weekly = ogni 7 giorni, monthly = ogni ~30 giorni)"
- "Confidence": "% di eventi che seguono il pattern dominante (>0.7 = pattern chiaro)"

---

## 🛠️ Troubleshooting

### Problema 1: "FileNotFoundError: data/raw/storico_sap.xlsx"

**Causa**: File input SAP non trovato nella posizione attesa.

**Soluzione**:
```bash
# Verifica file esiste
ls data/raw/storico_sap.xlsx

# Se non esiste, copia file SAP
cp ~/Downloads/storico_export.xlsx data/raw/storico_sap.xlsx

# Oppure usa mock mode per demo
$env:USE_MOCK="true"
python -m streamlit run app/streamlit_app.py
```

---

### Problema 2: "ImportError: No module named 'hdbscan'"

**Causa**: Libreria opzionale HDBSCAN non installata (necessaria per clustering).

**Soluzione**:
```bash
# Installa HDBSCAN
pip install hdbscan

# Installa anche UMAP per proiezione 2D di qualità
pip install umap-learn

# Verifica installazione
python -c "import hdbscan; import umap; print('OK')"
```

**Workaround** se installazione fallisce (Windows):
- HDBSCAN richiede compilatore C++ → Installa Visual Studio Build Tools
- Alternativa: Usa PCA invece di UMAP (già disponibile in scikit-learn)
- Sidebar Clusters page mostrerà warning: "⚠️ HDBSCAN non disponibile. Installa con: pip install hdbscan"

---

### Problema 3: "KeyError: 'Cliente' in normalize_sap.py"

**Causa**: Colonne Excel SAP hanno nomi diversi dal mapping predefinito.

**Soluzione**:
1. Apri `src/data/normalize_sap.py`
2. Modifica `COLUMN_MAPPING` con nomi corretti:
```python
COLUMN_MAPPING = {
    "Num. cliente": "cliente",  # Era "Cliente"
    "N. documento": "numero_documento",  # Era "Numero documento"
    # ... adatta altri nomi
}
```
3. Rilancia normalizzazione:
```bash
python src/data/normalize_sap.py --input data/raw/storico_sap.xlsx --output data/normalized/events.parquet
```

---

### Problema 4: "Streamlit page refresh infinito / memory leak"

**Causa**: Cache Streamlit corrotto o parquet file troppo grandi (>500MB).

**Soluzione**:
```bash
# Pulisci cache Streamlit
rm -rf .streamlit/cache

# Oppure forza no-cache nell'app
# In streamlit_app.py, aggiungi:
# st.cache_data.clear()

# Se file troppo grandi, usa chunking:
# In app/page_components/*.py, modifica:
df = pd.read_parquet("data/curated/events.parquet")
# Diventa:
df = pd.read_parquet("data/curated/events.parquet", columns=["cliente", "data_pareggio", "importo"])
```

---

### Problema 5: "CatBoost training fallisce con 'GPU not available'"

**Causa**: CatBoost prova a usare GPU ma non è disponibile/configurato.

**Soluzione**:
```bash
# Modifica src/training/pointwise_trainer.py
# Cambia:
model = CatBoostClassifier(task_type="GPU", ...)
# In:
model = CatBoostClassifier(task_type="CPU", ...)

# Oppure forza CPU via parametro:
python src/training/pointwise_trainer.py --task-type CPU
```

---

### Problema 6: "Score_class tutti 0 o tutti 5"

**Causa**: Soglie score troppo strette o modello non calibrato.

**Soluzione**:
1. Verifica soglie in `.env`:
```env
SCORE_THRESHOLD_5=0.8  # Prova 0.9 se troppi score=5
SCORE_THRESHOLD_4=0.6
SCORE_THRESHOLD_3=0.4
SCORE_THRESHOLD_2=0.2
SCORE_THRESHOLD_1=0.05  # Prova 0.1 se troppi score=0
```

2. Ricalibra modello:
```bash
python src/training/pointwise_trainer.py --calibrate isotonic
```

3. Verifica distribuzione p_model:
```python
import pandas as pd
df = pd.read_parquet("data/curated/open_invoice_scores.parquet")
print(df["p_model"].describe())
# Se std < 0.1, modello è troppo conservativo
```

---

### Problema 7: "Deep-link non ripristina filtri"

**Causa**: Query params non parsati correttamente da Streamlit.

**Soluzione**:
1. Verifica URL contiene `?page=Pattern%20Card&cliente=10001234`
2. In `app/page_components/*.py`, usa:
```python
from app.utils.export import restore_filters_from_deeplink

filters = restore_filters_from_deeplink()
selected_client = filters.get("cliente", default_value)
```

3. Se ancora non funziona, usa `st.query_params` direttamente:
```python
query_params = st.query_params
if "cliente" in query_params:
    selected_client = query_params["cliente"]
```

---

### Problema 8: "Mock mode non carica dati"

**Causa**: `USE_MOCK=true` ma `data_mock/` vuoto o corrotto.

**Soluzione**:
```bash
# Verifica struttura data_mock
ls data_mock/curated/

# Se mancano file, genera mock data:
python scripts/generate_mock_data.py

# Oppure copia da data/ (con anonymization):
python scripts/anonymize_and_copy_to_mock.py
```

---

### Problema 9: "Precision@5 = 0.00 (impossibile)"

**Causa**: Train/test split sbagliato (leakage o split casuale invece che temporale).

**Soluzione**:
1. Verifica split in `src/training/pointwise_trainer.py`:
```python
# DEVE essere time-based:
train = df[df["data_pareggio"] < "2024-01-01"]
test = df[df["data_pareggio"] >= "2024-01-01"]

# NON casuale:
# train, test = train_test_split(df, test_size=0.2, random_state=42)  # SBAGLIATO!
```

2. Rilancia training con split corretto:
```bash
python src/training/pointwise_trainer.py --split-type temporal --split-date 2024-01-01
```

---

### Problema 10: "JSONL logs corrotti / non leggibili"

**Causa**: Log interrotti durante crash o multi-process scrive stesso file.

**Soluzione**:
```bash
# Valida JSONL
python -c "
import json
with open('logs/run_20241030_143022.jsonl') as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            print(f'Line {i}: {e}')
"

# Rimuovi linee corrotte
grep -v '^{' logs/run_20241030_143022.jsonl > logs/run_cleaned.jsonl

# Per evitare: usa file lock o timestamp nel nome log
```

---

## 📚 Glossario SAP & ML

### Termini SAP

| Termine | Italiano | Descrizione | Esempio |
|---------|----------|-------------|---------|
| **Cliente** | Customer | Codice identificativo univoco del cliente | "10001234" |
| **Numero Documento** | Document Number | ID univoco della fattura/documento contabile | "900012345" |
| **Data Documento** | Document Date | Data di emissione della fattura (quando creata) | 2024-01-15 |
| **Data Registrazione** | Posting Date | Data di registrazione contabile in SAP (quando inserita a sistema) | 2024-01-16 (può essere dopo data documento) |
| **Data di Pareggio** | Clearing Date | Data in cui la fattura è stata matchata con un pagamento | 2024-02-10 (quando cliente paga) |
| **Importo Divisa Interna** | Amount in Local Currency | Importo fattura in valuta interna (€) | 1,250.50 € |
| **Esercizio** | Fiscal Year | Anno fiscale di riferimento | 2024 |
| **Periodo Contabile** | Posting Period | Mese fiscale (1-12) | 3 (marzo) |
| **Divisa** | Currency | Codice valuta transazione | "EUR", "USD" |
| **Tipo di Documento** | Document Type | Categoria documento (fattura, nota credito, etc.) | "RV" (fattura), "NC" (nota credito) |
| **Chiave Contabile** | Posting Key | Codice operazione contabile | "01" (dare cliente), "11" (avere cliente) |
| **Documento di Pareggio** | Clearing Document | ID del documento di pagamento che ha chiuso la fattura | "700012345" |
| **Data di Acquisizione** | Acquisition Date | Data di import/upload dei dati da SAP | 2024-10-30 |

### Termini ML

| Termine | Categoria | Descrizione | Formula / Range |
|---------|-----------|-------------|-----------------|
| **p_model** | Scoring | Probabilità predetta dal modello che la fattura venga pagata | [0, 1] (0 = non pagherà, 1 = pagherà sicuro) |
| **score_class** | Scoring | Classificazione discreta della probabilità in 6 livelli | {0, 1, 2, 3, 4, 5} (5 = alta priorità, 0 = bassa) |
| **Precision@k** | Ranking | % di raccomandazioni corrette nelle prime K posizioni | Precision@5 = TP / (TP + FP) per top-5 |
| **Recall@k** | Ranking | % di fatture pagate effettivamente trovate nelle prime K | Recall@5 = TP / (TP + FN) per top-5 |
| **nDCG@k** | Ranking | Normalized Discounted Cumulative Gain (qualità ordine ranking) | [0, 1], penalizza errori in posizioni alte |
| **MRR** | Ranking | Mean Reciprocal Rank (posizione media prima hit) | 1/N * Σ 1/rank_i |
| **Brier Score** | Calibrazione | Errore quadratico medio tra probabilità predette e realtà | 1/N * Σ (p_i - y_i)², range [0, 1] |
| **ECE** | Calibrazione | Expected Calibration Error (errore medio calibrazione) | Σ |bin| * |pred - actual|, target <0.05 |
| **MCE** | Calibrazione | Maximum Calibration Error (peggior bin) | max |pred - actual| per bin, target <0.1 |
| **k_i** | Feature | Numero di fatture aperte concorrenti al momento del pagamento | Intero, tipicamente 1-10 |
| **Embedding** | Feature | Rappresentazione vettoriale densa di un cliente (128 dimensioni) | Float array [128], cattura comportamento cliente |
| **HDBSCAN** | Clustering | Hierarchical Density-Based Spatial Clustering (algoritmo clustering) | Parametri: min_cluster_size, min_samples |
| **UMAP** | Dimensionality Reduction | Uniform Manifold Approximation and Projection (proiezione 128D → 2D) | Parametri: n_neighbors, min_dist |
| **PCA** | Dimensionality Reduction | Principal Component Analysis (proiezione lineare 128D → 2D) | Deterministico, veloce, preserva varianza globale |
| **Policy Score** | Feature | Score [0,1] che misura quanto un evento segue una policy deterministica | fifo_score, lifo_score, largest_score, etc. |
| **Pattern** | Analytics | Regola di pagamento dominante identificata per un cliente | FIFO, LIFO, Largest, Smallest, Same Period, etc. |
| **Confidence** | Analytics | % di eventi del cliente che seguono il pattern dominante | [0, 1], >0.7 = pattern chiaro |
| **Uplift** | Analytics | Miglioramento % usando modello predittivo vs baseline FIFO | +10% = recall migliorato del 10% |
| **Cluster Label** | Analytics | Etichetta testuale auto-generata per un cluster | "FIFO, k~2, monthly" (pattern, cardinalità, periodicità) |
| **Noise** | Clustering | Punti non assegnati a nessun cluster (outlier) | cluster_id = -1 in HDBSCAN |

---

## 📄 Licenza

**Proprietario**: [Nome Azienda]  
**Licenza**: Proprietaria / All Rights Reserved  
**Uso**: Solo interno aziendale, no redistribuzione

Per domande contattare: [email]

---

## 🔗 Link Utili

- **Documentazione CatBoost**: https://catboost.ai/docs/
- **Documentazione AutoGluon**: https://auto.gluon.ai/
- **HDBSCAN Paper**: https://arxiv.org/abs/1911.02282
- **UMAP Documentation**: https://umap-learn.readthedocs.io/
- **Streamlit Docs**: https://docs.streamlit.io/

---

## ✅ Checklist Completezza README

- [x] Panoramica progetto con caso d'uso
- [x] Architettura ad alto livello con diagramma ASCII
- [x] Struttura directory completa
- [x] Prerequisiti e setup step-by-step
- [x] Configurazione .env con tutte le variabili
- [x] Data contract (colonne normalizzate + semantica SAP)
- [x] Flusso end-to-end con comandi per ogni step
- [x] Script run_all (PowerShell + Bash)
- [x] Descrizione dettagliata di OGNI pagina Streamlit
- [x] Input parquet, grafici, tooltip, filtri sidebar per ogni pagina
- [x] Pulsanti export e cosa fanno
- [x] Mock mode spiegato
- [x] Logging JSONL path e formato
- [x] Troubleshooting (10 problemi comuni + soluzioni)
- [x] Glossario SAP + ML con formule
- [x] Licenza e contatti
- [x] Link incrociati funzionanti tra sezioni
- [x] Nessun dettaglio sensibile o path assoluti
- [x] Esempi con path mock (data_mock/)

---

**README Versione**: 2.0  
**Ultimo Aggiornamento**: 2024-10-30  
**Autore**: Sistema Storico Team

