# 📊 Storico — Sistema Intelligente Gestione Incassi SAP FI-AR

> **v2.0.0** | 11 Nov 2024 | Pipeline: Features → HDBSCAN → Learning-to-Rank

---

## 📋 Overview Tecnico

Sistema di ranking predittivo per fatture aperte basato su storico pagamenti SAP FI-AR.

**Pipeline**: (1) Normalizza SAP → schema contratto | (2) Feature engineering → profilo cliente | (3) HDBSCAN clustering → P(policy|cluster) | (4) CatBoost Ranker → score 0-5 | (5) Dashboard Streamlit

**Metriche validation**: AUC=0.87 | Precision@5=0.74 | Recall@5=0.71 | nDCG@10=0.82

---

## 🏗️ Architettura (4 Step: A→B→C→D)

### A. Ingest & Normalizzazione SAP → Schema Contratto Dati

**Cosa fa**: Legge i file Excel che arrivano da SAP (export contabilità clienti) e li trasforma in un formato pulito e standard che il sistema può usare.

**Input**: File `data/raw/storico_sap.xlsx` — export SAP FI-AR (moduli contabilità clienti). Contiene le tabelle BSID (partite ancora aperte) e BSAD (partite chiuse/pagate).

**Script**: `src/data/ingest.py`

**Output**: 
- `line_items.parquet` — ogni riga è una fattura (aperta o pagata)
- `events.parquet` — ogni riga è un pagamento (1 evento = quando il cliente paga, può saldare 1 o più fatture insieme)

**Trasformazioni applicate**:
1. **Mapping colonne SAP → schema contratto**: Prende i campi SAP (KUNNR, BELNR, BLDAT, AUGDT, DMBTR, BLART, MONAT, AUGBL) e li rinomina in modo semplice (cliente, numero_documento, data_documento, data_compensazione, importo, tipo_documento, periodo_fiscale, documento_clearing). Mantiene solo le colonne utili per l'analisi.

2. **Parsing date**: SAP può esportare date in formati diversi (YYYYMMDD tipo "20240115" oppure DD.MM.YYYY tipo "15.01.2024"). Lo script le riconosce e le converte tutte in formato datetime standard.

3. **Pulizia importi**: Rimuove caratteri strani (punti separatori migliaia, virgole decimali) e converte in numeri float puliti.

4. **Aggregazione line_items → events**: Raggruppa tutte le fatture che sono state pagate nello stesso momento (stesso `data_compensazione` + stesso `documento_clearing`) in un unico evento di pagamento. Esempio: Cliente paga fatture #001, #002, #003 il 15/01/2024 → 1 evento con 3 righe.

5. **Deduplicazione e validazione**: Rimuove righe duplicate, scarta record con dati mancanti critici (es. cliente vuoto o importo nullo).

**Risultato**: Dati SAP grezzi trasformati in tabelle parquet strutturate, pronte per l'analisi. Ogni evento di pagamento ha associato quante fatture ha saldato, quali importi, in che data, ecc.

---

### B. Feature Engineering → Profilo Comportamentale Cliente

**Cosa fa**: Per ogni cliente, calcola delle "metriche comportamentali" guardando tutto lo storico dei suoi pagamenti. Queste metriche descrivono il "carattere" del cliente: paga sempre le fatture più vecchie? Paga solo a fine mese? Paga sempre gli importi grandi per primi?

**Script**: `src/features/build_events.py`, `src/features/make_features.py`, `src/analytics/pattern_detectors.py`

**Output**: 
- `client_profiles.parquet` — 1 riga per cliente con le sue metriche aggregate
- `train_features.parquet` — feature calcolate per ogni evento di pagamento (usate per addestrare il modello)

**Metriche calcolate per ogni cliente**:

1. **Cardinalità** (quante fatture aperte ha di solito quando paga):
   - `k_mode`: numero più frequente di fatture aperte quando decide di pagare (es. di solito ha 8 fatture aperte)
   - `k_entropy`: quanto varia questo numero (alta entropia = imprevedibile, bassa = costante)

2. **Periodicità** (quando paga):
   - `is_weekly_payer`: TRUE se paga più o meno ogni settimana
   - `is_biweekly_payer`: TRUE se paga ogni 2 settimane
   - `is_eom_payer`: TRUE se paga quasi sempre negli ultimi 5 giorni del mese (End of Month)

3. **Preferenze importi** (sceglie fatture grandi o piccole?):
   - `largest_first_preference`: punteggio 0-1, alto se tende a pagare fatture con importi maggiori per prime
   - `smallest_first_preference`: punteggio 0-1, alto se tende a pagare fatture con importi minori per prime

4. **Policy scores aggregati** (quali strategie di pagamento segue):
   - `avg_fifo_score`: media dei punteggi FIFO (First In First Out - paga fatture vecchie per prime)
   - `avg_lifo_score`: media dei punteggi LIFO (Last In First Out - paga fatture recenti per prime)
   - `avg_largest_score`: media dei punteggi "Largest First"
   - `avg_same_period_score`: percentuale eventi in cui paga solo fatture dello stesso periodo fiscale (es. tutte fatture gennaio insieme)

5. **Temporal** (tempi medi):
   - `avg_doc_pareggio_days`: quanti giorni in media passano tra emissione fattura e pagamento (es. paga dopo 45 giorni in media)
   - `std_inter_event_gap`: quanto varia l'intervallo tra un pagamento e l'altro (es. paga ogni 30±5 giorni → bassa varianza, oppure ogni 10-90 giorni → alta varianza)

**Come funziona**: Per ogni evento storico del cliente, lo script calcola i "policy score" (quanto quel pagamento somigliava a FIFO, LIFO, Largest, ecc.) e poi aggrega questi score su tutti gli eventi per ottenere il profilo. Esempio: Cliente A ha 50 eventi storici, in 40 di questi ha pagato le fatture più vecchie per prime → `avg_fifo_score=0.80` → probabilmente segue strategia FIFO.

**Risultato**: Ogni cliente ha un "identikit comportamentale" numerico. Il sistema sa se è un pagatore regolare o irregolare, se segue schemi prevedibili o casuali, se preferisce certe strategie di pagamento.

---

### C. Clustering HDBSCAN → Gruppi Clienti Simili + Probabilità Policy per Cluster

**Cosa fa**: Raggruppa i clienti in "cluster" (gruppi di clienti con comportamenti simili) usando l'algoritmo HDBSCAN. Per ogni cluster, calcola le probabilità che un cliente di quel gruppo segua una determinata strategia di pagamento (es. probabilità che clienti Cluster 0 paghino FIFO = 78%, EOM = 12%, ecc.). Queste probabilità si chiamano "cluster priors" e vengono usate nel modello finale.

**Script**: `src/analytics/make_client_embeddings.py`, `src/analytics/run_hdbscan.py`

**Output**: 
- `client_embeddings.parquet` — vettori numerici che rappresentano ogni cliente (es. 32 numeri che riassumono profilo)
- `client_clusters.parquet` — per ogni cliente, il suo cluster_id (es. Cliente A → Cluster 0, Cliente B → Cluster 2, ecc.)
- **`cluster_priors.parquet`** ← **ARTEFATTO CHIAVE** — per ogni cluster, le probabilità P(policy|cluster) con smoothing Dirichlet

**Processo in 5 step**:

1. **Vettore cliente da feature Step B**: Prende le metriche calcolate in Step B (policy scores aggregati, periodicità, preferenze importi, cardinalità) e le mette in un vettore numerico. Esempio: Cliente A → [fifo=0.8, lifo=0.1, largest=0.6, eom=0.2, weekly=1, ...] → vettore 128 dimensioni.

2. **PCA opzionale (riduzione dimensionalità)**: Se il vettore ha troppe dimensioni (128), lo comprime a 32 dimensioni mantenendo l'informazione principale. Questo rende il clustering più veloce e robusto.

3. **HDBSCAN clustering**: Algoritmo che trova automaticamente gruppi di clienti simili. Parametri: `min_cluster_size=5` (cluster minimo 5 clienti), `metric=euclidean` (distanza euclidea tra vettori). HDBSCAN è robusto agli outlier: clienti con comportamenti anomali finiscono in cluster_id=-1 (rumore).

4. **Calcola P(policy|cluster) con smoothing Dirichlet**: Per ogni cluster, conta quante volte gli eventi dei clienti di quel cluster hanno seguito ciascuna policy (FIFO, LIFO, Largest, EOM, ecc.). Applica smoothing Dirichlet con alpha=0.1 per evitare probabilità 0 (se nessun evento ha seguito policy X nel cluster, non vuol dire P(X)=0, ma P(X)=piccolo). Esempio: Cluster 0, 100 eventi totali: 78 FIFO, 12 EOM, 5 Largest, 5 altri → dopo smoothing P(FIFO|cluster_0)=0.77, P(EOM|cluster_0)=0.12, P(Largest|cluster_0)=0.06, ecc.

5. **Salva cluster_id + priors**: Ogni cliente ha il suo cluster_id; ogni cluster ha la sua distribuzione P(policy|cluster). Questi priors vengono salvati e usati come feature nel modello finale (Step D).

**Perché è importante**: I cluster priors sono "informazione condivisa" tra clienti simili. Se il modello vede un nuovo cliente del Cluster 0 (dove il 78% degli eventi segue FIFO), sa che probabilmente anche questo cliente seguirà FIFO, anche se ha pochi eventi storici. È un modo per "condividere conoscenza" tra clienti simili e migliorare le predizioni.

**Nessun data leakage**: I priors sono calcolati solo su dati storici (eventi già avvenuti), mai su eventi futuri. Quando il modello predice, usa solo informazione disponibile "a quella data".

**Risultato**: Clienti raggruppati in cluster omogenei + distribuzione probabilità policy per cluster. Il sistema sa che "Cluster 0 = pagatori FIFO regolari", "Cluster 2 = pagatori EOM con alta varianza", ecc.

---

### D. Modello Learning-to-Rank → Ordina Fatture per Probabilità Pagamento

**Cosa fa**: Addestra un modello di machine learning che, dato un evento di pagamento (cliente X paga il giorno Y) e l'insieme di fatture aperte in quel momento, ordina le fatture dalla più probabile alla meno probabile di essere pagata. L'obiettivo è mettere le fatture che verranno effettivamente pagate il più in alto possibile nella classifica.

**Script**: `src/models/train_catboost.py`, `src/analytics/score_open_invoices.py`

**Output**: 
- `champion.json` — file di configurazione del modello migliore
- `catboost.cbm` — modello addestrato (CatBoost binario)
- `event_policy_predictions.parquet` — predizioni per ogni fattura in ogni evento (score 0-1)

**Algoritmi usati (Learning-to-Rank)**:
- **CatBoostRanker** con loss function YetiRank o QueryCrossEntropy (default consigliato)
- **LightGBM LGBMRanker** con algoritmo LambdaMART
- **XGBoost** con objective `rank:pairwise`

Tutti e 3 sono algoritmi specifici per il "ranking": non predicono semplicemente "pagata/non pagata", ma ottimizzano l'ordine relativo delle fatture (fattura A deve stare sopra fattura B se ha più probabilità di essere pagata).

**Feature per fattura** (in contesto di un evento):

1. **Base** (proprietà fattura):
   - `doc_pareggio_days`: quanti giorni sono passati dalla data emissione fattura al giorno del pagamento
   - `importo_aperto`: importo ancora dovuto (€)
   - `tipo_doc`: tipo documento (RV=fattura, NC=nota credito, ecc.)

2. **Policy scores** (quanto questa fattura "matcha" ciascuna strategia):
   - `fifo_score`: 1.0 se è la fattura più vecchia tra le aperte, 0.0 se è la più recente, valori intermedi per posizioni intermedie
   - `lifo_score`: 1.0 se è la fattura più recente, 0.0 se più vecchia
   - `largest_score`: 1.0 se ha l'importo maggiore, 0.0 se importo minore
   - `smallest_score`: inverso di largest_score
   - `same_period_score`: 1.0 se è dello stesso periodo fiscale (mese) del pagamento, 0.0 altrimenti

3. **Cluster priors** (dalla Step C) — **FEATURE CHIAVE**:
   - `P_eom_cluster`: probabilità che clienti del cluster paghino EOM (da cluster_priors.parquet)
   - `P_largest_cluster`: probabilità che clienti del cluster paghino Largest First
   - `P_fifo_cluster`: probabilità che clienti del cluster paghino FIFO
   - (tutte le altre policy priors per quel cluster)

4. **Context flags** (caratteristiche evento):
   - `same_period`: boolean, TRUE se fattura è dello stesso periodo fiscale del pagamento
   - `EOM_window`: boolean, TRUE se pagamento è negli ultimi 5 giorni del mese
   - `is_nota_credito`: boolean, TRUE se tipo_doc è NC (nota credito)

5. **Temporal** (tempi):
   - `inter_event_gap`: giorni passati dall'ultimo pagamento di questo cliente
   - `day_of_week`: giorno della settimana del pagamento (1=lunedì, 7=domenica)
   - `month`: mese fiscale (1-12)

**Target**: `y_i = 1` se la fattura è stata effettivamente pagata in quell'evento, `y_i = 0` altrimenti.

**Come funziona l'addestramento**: Il modello vede migliaia di eventi storici. Per ogni evento sa quali fatture erano aperte e quali sono state pagate. Impara a riconoscere i pattern: "quando cliente Cluster 0 paga e c'è alta P_fifo_cluster, la fattura con fifo_score=1.0 viene quasi sempre pagata", "quando EOM_window=TRUE e alta P_eom_cluster, vengono pagate fatture con same_period=TRUE", ecc.

**Calibrazione probabilità**: Dopo l'addestramento, applica isotonic regression sulle probabilità output. Questo garantisce che se il modello dice "p=0.70", nel 70% dei casi simili la fattura viene davvero pagata (probabilità "oneste", non distorte).

**Metriche validation** (su dati mai visti durante training):
- **AUC = 0.87**: capacità di distinguere fatture pagate da non pagate (0.5=random, 1.0=perfetto)
- **Precision@5 = 0.74**: delle 5 fatture che il modello mette in cima, il 74% sono effettivamente pagate
- **Recall@5 = 0.71**: delle fatture effettivamente pagate, il 71% finisce nelle top-5 del modello
- **nDCG@10 = 0.82**: qualità complessiva del ranking nelle top-10 posizioni (0.0=pessimo, 1.0=perfetto)

**Risultato**: Modello addestrato che prende un evento + fatture aperte e restituisce un ranking ordinato. Le fatture in cima alla lista sono quelle con maggiore probabilità di essere pagate. Gli utenti (tesoreria, credit management) usano questo ranking per prioritizzare solleciti, allocare risorse, pianificare flussi cassa.

---

## 📖 Glossario Termini SAP & Logica Sistema

### Termini SAP FI-AR (Accounts Receivable - Contabilità Clienti)
- **Cliente**: Anagrafica cliente SAP (campo `KUNNR` in tabelle BSID/BSAD)
- **Numero documento**: ID univoco fattura/nota credito (campo `BELNR`)
- **Data documento**: Data emissione fattura (campo `BLDAT`)
- **Data compensazione**: Data in cui fattura è stata pareggiata/pagata (campo `AUGDT` in BSAD). Se vuota in BSID → fattura ancora aperta
- **Importo div. int.**: Importo in valuta interna (EUR) della fattura (campo `DMBTR`)
- **Tipo documento**: RV=Fattura cliente, NC=Nota credito, DR=Nota debito, etc. (campo `BLART`)
- **Periodo contabile**: Mese fiscale (campo `MONAT`, 1-12)
- **Clearing document**: Documento di compensazione che chiude la partita aperta (campo `AUGBL`)

### Concetti Chiave Sistema

**Evento di compensazione**: Momento in cui cliente effettua un pagamento. Può saldare 1 o più fatture contemporaneamente. Esempio: 15/01/2024 Cliente A paga fatture #001, #002, #003 → 1 evento con `data_pareggio=15/01/2024`, `n_righe_evento=3`

**Fatture aperte (k_i)**: Numero di fatture che il cliente aveva aperte (non ancora pagate) nel momento in cui ha deciso di pagare. Esempio: Cliente ha 10 fatture aperte, paga 3 → k_i=10 per questo evento. Questa metrica indica la "cardinalità" del set di scelta.

**Policy di pagamento**: Strategia con cui cliente sceglie quali fatture pagare tra quelle aperte. Esempi:
- **FIFO** (First In First Out): Paga sempre fatture più vecchie per prime (es. emesse 3 mesi fa prima di quelle emesse 1 mese fa)
- **LIFO** (Last In Last Out): Paga fatture più recenti per prime
- **Largest First**: Paga fatture con importo maggiore per prime
- **Smallest First**: Paga fatture con importo minore per prime
- **Same Period**: Paga fatture dello stesso periodo fiscale (es. tutte le fatture di gennaio insieme)
- **EOM** (End of Month): Paga solo negli ultimi 5 giorni del mese
- Altre policy: Nearest Due Date, Random, etc.

**Policy score**: Punteggio 0-1 che indica quanto una fattura "matcha" una determinata policy in un dato contesto. Esempio: 5 fatture aperte con date emissione [01/01, 05/01, 10/01, 15/01, 20/01], valutando fattura 01/01 → `fifo_score=1.0` (più vecchia), `lifo_score=0.0` (meno recente)

**Pattern dominante cliente**: Policy seguita più frequentemente nello storico. Calcolato testando tutte le policy su tutti gli eventi storici del cliente e scegliendo quella con recall@5 massimo. Esempio: Cliente A, 50 eventi storici, policy FIFO predice correttamente 82% fatture pagate → `dominant_policy="fifo"`, `pattern_confidence=0.82`

**Cluster priors P(policy|cluster)**: Distribuzione probabilità policy per cluster. Conteggio policy seguite dagli eventi di tutti i clienti nel cluster + smoothing Dirichlet. Esempio: Cluster 0, 100 eventi totali: 78 FIFO, 12 EOM, 10 altri → P(FIFO|cluster_0)=0.78, P(EOM|cluster_0)=0.12

**Agreement score**: Coerenza tra ranking modello e policy dominante cliente. Overlap top-5 fatture modello vs top-5 policy dominante. Overlap >0.7 → modello coerente con comportamento storico → predizione affidabile

**Calibrazione**: Garantisce probabilità p_i "oneste" del modello. Se p_i=0.7, nel 70% casi simili la fattura viene pagata. Isotonic regression su validation set.

**Learning-to-Rank**: Algoritmi ML che ordinano fatture per probabilità pagamento (vs predire classe paid/not paid). Obiettivo: fatture che verranno pagate il più in alto possibile. Algoritmi: YetiRank (CatBoost), LambdaMART (LightGBM), rank:pairwise (XGBoost).

---

## 📦 Struttura Directory

```
storico/
├── app/                          # Streamlit Frontend
│   ├── streamlit_app.py          # Main router
│   ├── page_components/
│   │   ├── overview_wizard.py    # Wizard 4-step + FAQ
│   │   ├── pattern_card_v2.py    # Drill-down cliente + coerenza
│   │   ├── kpi_dashboard.py      # Metriche ranking + calibrazione
│   │   ├── clusters.py           # HDBSCAN explorer + UMAP
│   │   └── policy_iq.py          # Confusion + uplift + cluster breakdown
│   └── utils/export.py           # CSV export + deep-links
├── src/
│   ├── data/
│   │   ├── ingest.py             # A: Normalizza SAP
│   │   └── ingest_multi.py       # Multi-sheet
│   ├── features/
│   │   ├── build_events.py       # B: Eventi
│   │   ├── make_labels.py        # B: Labels
│   │   └── make_features.py      # B: Feature temporali
│   ├── analytics/
│   │   ├── make_client_embeddings.py  # C: Embeddings
│   │   ├── run_hdbscan.py             # C: Clustering + priors
│   │   ├── make_pattern_table.py      # Pattern dominante
│   │   ├── pattern_detectors.py       # Policy detection
│   │   ├── score_open_invoices.py     # D: Scoring
│   │   ├── kpi_ranking.py             # Precision@k, nDCG
│   │   ├── calibration.py             # Brier, ECE
│   │   └── event_policy_metrics.py    # Confusion + uplift
│   ├── models/
│   │   ├── train_catboost.py     # D: CatBoost ranker
│   │   └── train_autogluon.py    # Fallback
│   ├── pipeline/run_all.py       # A→B→C→D orchestrator
│   └── utils/{sap_glossary, logging_setup}.py
├── data/                         # Production (gitignored)
│   ├── raw/storico_sap.xlsx
│   ├── normalized/{line_items, events}.parquet
│   ├── curated/
│   │   ├── train_features.parquet
│   │   ├── client_embeddings.parquet
│   │   ├── client_clusters.parquet
│   │   ├── cluster_priors.parquet     # ← KEY! P(policy|cluster)
│   │   ├── client_patterns.parquet
│   │   ├── event_policy_predictions.parquet
│   │   └── kpi_ranking.parquet
│   └── models/{champion.json, catboost.cbm, calibrator.pkl}
├── data_mock/                    # Mock data per demo
├── .env + requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Demo Mode (Mock Data)
```powershell
$env:USE_MOCK="true"
python -m streamlit run app/streamlit_app.py
# http://localhost:8501
```

### Production Mode (Dati SAP Reali)
```powershell
# 1. Copia Excel SAP
cp ~/Downloads/storico.xlsx data/raw/storico_sap.xlsx

# 2. Pipeline completa (un comando)
python src/pipeline/run_all.py --verbose
# Output: ✓ A: Ingest (5.2s) | ✓ B: Features (40.5s) | ✓ C: HDBSCAN+priors (90.2s, 12 clusters) | ✓ D: CatBoost ranker (382.1s, AUC=0.87) | ✓ Scoring (25.3s, 2340 fatture) | ✓ Analytics (15.1s) | Completata 9m25s

# 3. Dashboard
python -m streamlit run app/streamlit_app.py
```

### Step-by-Step (Debug)
```powershell
python src/data/ingest.py                          # A
python src/features/{build_events,make_features}.py # B
python src/analytics/{make_client_embeddings,run_hdbscan}.py # C
python src/models/train_catboost.py --objective YetiRank     # D
python src/analytics/score_open_invoices.py        # Scoring
python src/analytics/{kpi_ranking,calibration}.py  # Analytics
python -m streamlit run app/streamlit_app.py       # Dashboard
```

---

## 🖥️ Dashboard Streamlit (5 Pagine)

### 1️⃣ Overview Wizard
Wizard 4-step (A→B→C→D) | FAQ ("Perché HDBSCAN prima del modello?", "Cosa sono cluster priors?") | Deep-links altre pagine | Progress bar pipeline | Summary stats (# clienti, eventi, cluster)

### 2️⃣ Pattern Card
Drill-down cliente specifico | Visualizzazioni: heatmap calendar, cadence histogram, ACF | **Coerenza panel**: Badge verde/giallo/rosso (agreement_score) confronto top-5 modello vs top-5 policy dominante | SHAP explainability top-3 feature | JSON explain download (cluster_label + dominant_policy) | Policy gating simulator (boost alpha)

### 3️⃣ KPI Dashboard
Metriche ranking: Precision@5, Recall@5, nDCG@10, MRR | Time series charts (precision@k nel tempo) | Breakdown per cluster | Calibrazione: Brier score, ECE, MCE | Reliability diagram (expected vs observed frequency) | Histogram calibrated probabilities | Export CSV

### 4️⃣ Clusters
UMAP projection 2D (scatter plot cluster-colored) | Configurazione HDBSCAN live (min_cluster_size slider 2-50, min_samples 1-20, metric selector) | Pulsante "Anteprima clustering" → live metrics preview | Cluster detail view: tabella clienti, policy distribution bar chart, periodicity/k_mode summary | Click su punto → redirect Pattern Card | Export CSV cluster assignments

### 5️⃣ Policy IQ
Confusion matrix heatmap (policy vera vs predetta, normalizzato) | Metric cards: Top-1 accuracy, Top-2 hit rate, # eventi | Uplift vs FIFO: tabella recall@k modello vs baseline per segmento | Bar chart FIFO standard vs Policy IQ grouped per k-bucket | **Cluster breakdown**: Cluster summary table (cluster_label, n_clienti, dominant_policy, k_mode) | Bar chart policy dominante per cluster | Metriche cards (# cluster, # clienti, dimensione media) | Policy explanations tabs: dizionario completo policy + per cliente top-3 policy frequenti | Export CSV

---

## 📊 Schema Dati Chiave

### `client_clusters.parquet`
| Colonna | Tipo | Descrizione |
|---------|------|-------------|
| cliente | int64 | ID cliente |
| cluster_label | int32 | Cluster ID (-1=noise) |
| membership_prob | float64 | Probabilità appartenenza |

### `cluster_priors.parquet` ← **KEY per Step D**
| Colonna | Tipo | Descrizione |
|---------|------|-------------|
| cluster_label | int32 | Cluster ID |
| policy_name | str | es. "fifo", "lifo", "largest", "EOM" |
| prior_prob | float64 | P(policy\|cluster) con smoothing Dirichlet |
| count_events | int32 | Eventi cluster con questa policy |
| total_events | int32 | Eventi totali cluster |

**Esempio**: `cluster_label=0, policy_name="fifo", prior_prob=0.78` → Cluster 0 preferisce FIFO (78%) | `cluster_label=3, policy_name="largest", prior_prob=0.85` → Cluster 3 preferisce Largest (85%)

### `event_policy_predictions.parquet`
| Colonna | Tipo | Descrizione |
|---------|------|-------------|
| numero_documento | str | ID fattura |
| cliente | int64 | ID cliente |
| p_i | float64 | Probabilità pagamento (0-1, calibrata) |
| rank | int32 | Ranking fattura (1=top) |
| score_class | int32 | Score 0-5 (binning p_i): 5=pagherà sicuro, 0=non pagherà |
| agreement_score | float64 | Overlap modello vs policy dominante (0-1) |
| alignment_status | str | aligned (>0.7) / partial (0.4-0.7) / divergent (<0.4) |
| dominant_policy | str | Policy dominante cliente (da client_patterns) |
| cluster_label | int32 | Cluster cliente |

---

## 📈 Metriche: Come Leggerle

### Precision@k
`Precision@5 = (# fatture pagate tra top-5 predette) / 5`
Esempio: Modello predice top-5=[A,B,C,D,E], cliente paga [A,C,E,F] → Precision@5=3/5=0.60
Interpretazione: >0.7=Eccellente | 0.5-0.7=Buono | <0.5=Scarso

### Recall@k
`Recall@5 = (# fatture pagate tra top-5) / (# fatture pagate totali)`
Esempio: Predette top-5=[A,B,C,D,E], pagate=[A,C,E,F] (4 totali) → Recall@5=3/4=0.75
Interpretazione: >0.7=Eccellente (cattura 70%+ pagamenti) | 0.5-0.7=Buono | <0.5=Scarso

### nDCG@k (Normalized Discounted Cumulative Gain)
Misura qualità ranking considerando posizione. Penalizza errori prime posizioni più che ultime. nDCG@10=0.82 → Ranking molto buono (ideale=1.0)
Interpretazione: >0.8=Eccellente | 0.6-0.8=Buono | <0.6=Da migliorare

### Agreement Score
`top_k_model = Set(top-5 fatture per p_i)` | `top_k_policy = Set(top-5 per policy dominante cliente)` | `Overlap = |top_k_model ∩ top_k_policy| / 5`
`agreement_score > 0.7 → aligned (verde)` | `0.4 ≤ agreement_score ≤ 0.7 → partial (giallo)` | `agreement_score < 0.4 → divergent (rosso)`
**Business value**: Alta coerenza (>0.7) = modello ha "capito" pattern cliente → predizioni affidabili | Media (0.4-0.7) = pattern misto/stagionalità → validazione manuale | Bassa (<0.4) = cliente ha cambiato strategia o modello non cattura pattern → flag revisione

### Brier Score
`Brier = avg((p_i - y_i)^2)` → Differenza quadratica probabilità predetta vs outcome reale
Brier=0.18 → Ben calibrato (ideale=0.0)
Interpretazione: <0.20=Eccellente | 0.20-0.30=Buona | >0.30=Scarsa (overconfident/underconfident)

### ECE (Expected Calibration Error)
Misura gap tra probabilità predette e frequenze osservate
ECE=0.05 → Molto ben calibrato (ideale=0.0)
Interpretazione: <0.05=Eccellente | 0.05-0.10=Buono | >0.10=Necessaria ricalibrazione (isotonic/platt)

---

## ️ Troubleshooting

### "FileNotFoundError: storico_sap.xlsx"
```powershell
cp ~/Downloads/export.xlsx data/raw/storico_sap.xlsx
# O usa mock mode: $env:USE_MOCK="true"
```

### "ImportError: No module named 'hdbscan'"
```powershell
pip install hdbscan umap-learn
# Windows: installa Visual Studio Build Tools (https://visualstudio.microsoft.com/downloads/)
```

### "KeyError: 'Cliente' in normalize"
Adatta `src/utils/sap_glossary.py`:
```python
COLUMN_MAPPING = {
    "Num. cliente": "cliente",  # Nome SAP custom
    "N. documento": "numero_documento",
    # ...
}
```

### "Score_class tutti 0 o 5"
Adatta soglie `.env`:
```env
SCORE_THRESHOLD_5=0.90  # Se troppi score=5, aumenta
SCORE_THRESHOLD_1=0.10  # Se troppi score=0, aumenta
```
Poi ritraina con calibrazione:
```powershell
python src/models/train_catboost.py --calibrate isotonic
```

### "Precision@5 = 0.00 (impossibile)"
Diagnosi:
```python
import pandas as pd
df = pd.read_parquet("data/curated/train_features.parquet")
print(df["label"].value_counts())
# Expected: mix 50/50 circa (0=not paid, 1=paid)
```
Soluzione: Se label tutte 0: Bug `make_labels.py` → Verifica data_compensazione not null | Se label tutte 1: Filtra solo eventi con almeno 1 fattura non pagata

---

## 📚 Configurazione `.env`

```env
# DATA
DATA_DIR=data
USE_MOCK=false

# SAP
SAP_INPUT_FILE=storico_sap.xlsx
SAP_SHEET_NAME=Sheet1

# MODEL
MODEL_TYPE=catboost                # catboost | lightgbm | xgboost
RANKING_OBJECTIVE=YetiRank         # YetiRank | QueryCrossEntropy | LambdaMART
RANDOM_SEED=42
TEST_SIZE=0.2

# SCORING THRESHOLDS (p_i → score_class 0-5)
SCORE_THRESHOLD_5=0.80             # score=5: pagherà sicuramente
SCORE_THRESHOLD_4=0.60             # score=4: molto probabile
SCORE_THRESHOLD_3=0.40             # score=3: probabile
SCORE_THRESHOLD_2=0.20             # score=2: poco probabile
SCORE_THRESHOLD_1=0.05             # score=1: improbabile
# p_i < 0.05 → score=0: non pagherà

# CLUSTERING (HDBSCAN)
HDBSCAN_MIN_CLUSTER_SIZE=5         # Minimo clienti per cluster
HDBSCAN_MIN_SAMPLES=1              # Densità (1=permissivo)
HDBSCAN_METRIC=euclidean           # euclidean | manhattan | cosine

# CLUSTER PRIORS (Dirichlet Smoothing)
DIRICHLET_ALPHA=0.1                # Smoothing parameter per P(policy | cluster)

# LOGGING
LOG_LEVEL=INFO
LOG_FORMAT=jsonl
LOG_DIR=logs
```

---

## 📝 Setup & Installazione

### Prerequisiti
Python 3.11+ | pip 23.0+ | Git | HDBSCAN (opzionale, richiede compilatore C++ su Windows: Visual Studio Build Tools) | UMAP (opzionale, proiezioni 2D migliori di PCA)

### Installazione Rapida
```powershell
git clone <repository-url>
cd storico
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install hdbscan umap-learn  # Opzionale ma raccomandato
python -c "import pandas, streamlit, catboost; print('✓ Core OK')"
python -c "import hdbscan, umap; print('✓ Clustering OK')"
```

---

**Fine README** • v2.0.0 • 11 Nov 2024 • Team Data Analytics
