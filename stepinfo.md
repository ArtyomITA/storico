
# 📄 Piano d'Azione Tecnico: Refactoring Pipeline & Validation System

**Obiettivo:** Validare matematicamente il flusso "Cluster → Modello", ripulire l'interfaccia Streamlit e creare un ambiente di debug deterministico con dati Mock.

***

## 1. Strategia di Validazione (Mock Data Factory)
Non useremo dati casuali. Creeremo un generatore di dati sintetici "truccati" per verificare se il modello impara le regole.

**Azione Richiesta:** Creare script `src/data/generate_mock_scenarios.py`.
**Logica del Dataset Mock:**
Deve generare 3 tipi di "Clienti Artificiali" (100 clienti totali, 2 anni di storico):
1.  **Il Metodico (FIFO):** Paga sempre e solo le fatture più vecchie.
    *   *Expected Result:* HDBSCAN deve metterli tutti nello stesso cluster. CatBoost deve dare peso massimo alla feature `is_oldest`.
2.  **Il "Big Spender" (Largest First):** Paga sempre le fatture con importo più alto, indipendentemente dalla data.
    *   *Expected Result:* Cluster separato. CatBoost deve dare peso massimo alla feature `amount_rank`.
3.  **Il Caotico (Random):** Paga a caso.
    *   *Expected Result:* Cluster "Noise" (-1) o cluster a bassa confidenza. Score predittivi bassi (vicini a 0.5).

**Verifica Successiva:**
Se il sistema funziona, lanciando la pipeline su questo mock, nella dashboard dobbiamo vedere:
*   Cluster 0 = Utenti FIFO (Probabilità FIFO > 90%).
*   Cluster 1 = Utenti Largest (Probabilità Largest > 90%).
*   CatBoost Accuracy > 95% (poiché le regole sono deterministiche).

***

## 2. Verifica Backend: Flusso "Cluster Priors" → "CatBoost"
Bisogna garantire che l'intelligenza generata da HDBSCAN arrivi effettivamente al modello.

**Analisi Critica (`src/models/train_catboost.py`):**
L'agente deve verificare la funzione `enrich_features`.
1.  **Input:** Deve ricevere `train_features` E `cluster_priors` (o `client_clusters` unito ai priors).
2.  **Join:** Deve esserci un *merge* esplicito basato su `cluster_id` del cliente.
3.  **Check Nulli:** Se un cliente è nuovo o "Noise" (-1), il sistema deve assegnare dei "Global Priors" (la media di tutti) invece di lasciare Null/Zero, altrimenti il modello crasha o sbaglia.
4.  **Feature Importance:** Dopo il training, forzare la stampa della "Feature Importance". Se le feature `cluster_prior_fifo` o `cluster_prior_largest` sono a zero, il passaggio dati è rotto.

***

## 3. Refactoring Streamlit (UI Cleanup & Pipeline Order)
Attualmente la dashboard è dispersiva e visivamente errata nel flusso.

**A. Correzione Pipeline Visuale**
*   **Target:** `app/page_components/overview_wizard.py` (o dove risiede il diagramma).
*   **Fix:** Modificare l'ordine di rendering dei blocchi. L'ordine logico visualizzato DEVE essere:
    1.  **Ingest & Features** (Data Prep)
    2.  **Unsupervised Learning** (HDBSCAN, Pattern Detection)
    3.  **Supervised Learning** (CatBoost Training)
    4.  **Scoring & Analytics** (Output finale)

**B. Aggregazione Pagine (Semplificazione Menu)**
Ridurre il menu laterale a sole 3 voci logiche:

1.  **🏠 Home & Monitor:**
    *   Stato Pipeline (barra avanzamento corretta).
    *   Wizards di configurazione.
    *   Loghi errori/warning globali.
2.  **🧠 Model Internals (Analisi Globale):**
    *   *Fusione di:* `Clusters`, `KPI Dashboard` e `Policy IQ`.
    *   Usa dei TAB interni alla pagina per navigare: [Tab 1: Cluster Map] | [Tab 2: Model Metrics] | [Tab 3: Policy Confusion].
3.  **🔎 Client Debugger (Nuova Pagina Richiesta):**
    *   Vedi sezione 4 sotto.

***

## 4. Nuova Pagina: "Client Debugger"
Questa è la pagina operativa per l'uomo (tu) per validare il singolo caso.

**Specifiche Funzionali:**
*   **Input:** Dropdown con ricerca per "Nome Cliente" o "ID Cliente".
*   **Sezione A: "Test Storico" (Il Passato)**
    *   Prende tutto lo storico *chiuso* di quel cliente.
    *   Divide 80% (usato per calcolare i suoi pattern) e 20% (test).
    *   **Output:** Un semaforo.
        *   🟢 "Prevedibile" (Accuracy > 80% sul test set).
        *   🔴 "Imprevedibile" (Il modello non capisce come paga questo cliente).
*   **Sezione B: "Prossima Riconciliazione" (Il Presente)**
    *   Carica le fatture *attualmente aperte* (BSID) per quel cliente.
    *   Applica il modello `catboost.cbm` caricato.
    *   **Tabella Output:**
        *   Colonna 1: ID Fattura.
        *   Colonna 2: Importo.
        *   Colonna 3: Data Scadenza.
        *   **Colonna 4 (Core):** `Score (0-5)` (Barra colorata visuale).
        *   **Colonna 5 (Debug):** `Why?` (Es. "Alta prob. FIFO + Cluster 2").

***

## 5. Ordine di Intervento per l'Agente

Per evitare conflitti, l'agente deve lavorare in questo ordine rigoroso:

1.  **Backend Core Fix:** Verificare la funzione di `enrich` in `train_catboost.py`. Senza questo, il resto è inutile.
2.  **Mock Data Gen:** Creare lo script dei dati finti.
3.  **Pipeline Test:** Lanciare `run_all.py` con i dati finti e assicurarsi che non esploda e che i cluster siano coerenti (FIFO con FIFO).
4.  **UI Refactoring:** Aggregare le pagine esistenti e fixare l'ordine visuale della pipeline.
5.  **Client Debugger:** Implementare l'ultima pagina (richiede che il backend sia stabile).