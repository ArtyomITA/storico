""" Homepage Overview & Wizard - flusso umano comprensibile per CEO/business users."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

LOGGER = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

def check_pipeline_initialization() -> Dict[str, Any]:
    """Check if pipeline has been initialized with data.
    
    Returns:
        Dict with keys:
        - initialized (bool): True if data files exist
        - source (str): "none" | "mock" | "real"
        - missing_files (List[str]): List of missing required files
        - has_raw (bool): True if raw data directory has files
    """
    if st.session_state.get("mock_mode"):
        data_root = PROJECT_ROOT / "data_mock"
        source_hint = "mock"
    else:
        data_root = PROJECT_ROOT / "data"
        source_hint = "real"

    curated_dir = data_root / "curated"
    raw_dir = data_root / "raw"

    required_files = ["line_items.parquet"]
    missing_files = []

    for filename in required_files:
        file_path = curated_dir / filename
        if not file_path.exists():
            missing_files.append(filename)

    has_raw = False
    if raw_dir.exists():
        raw_files = list(raw_dir.glob("*.xlsx")) + list(raw_dir.glob("*.csv"))
        raw_files += list((raw_dir / "uploads").glob("*.xlsx")) if (raw_dir / "uploads").exists() else []
        has_raw = len(raw_files) > 0

    initialized = len(missing_files) == 0
    source = source_hint if initialized else "none"

    return {
        "initialized": initialized,
        "source": source,
        "missing_files": missing_files,
        "has_raw": has_raw,
    }

def generate_mock_data_flow() -> None:
    """Generate synthetic data and run normalization automatically."""
    with st.spinner("🎭 Generazione dataset sintetico..."):
        try:
            result = subprocess.run(
                [sys.executable, "-m", "scripts.make_mock_data"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                st.error(f"❌ Errore generazione mock: {result.stderr}")
                LOGGER.error(f"Mock data generation failed: {result.stderr}")
                return

            mock_file = PROJECT_ROOT / "data_mock" / "raw" / "sap_mock.xlsx"
            if not mock_file.exists():
                st.error("❌ File mock non trovato dopo generazione")
                LOGGER.error(f"Mock file not found at {mock_file}")
                return

            st.success("✅ Dataset sintetico generato!")

        except subprocess.TimeoutExpired:
            st.error("❌ Timeout generazione mock (>120s)")
            return
        except Exception as exc:
            st.error(f"❌ Errore imprevisto: {exc}")
            LOGGER.exception("Error generating mock data")
            return

    with st.spinner("📥 Ingestion dataset sintetico..."):
        try:
            from src.data import ingest_multi

            output_path = PROJECT_ROOT / "data_mock" / "curated" / "line_items.parquet"
            docmap_path = PROJECT_ROOT / "data_mock" / "curated" / "docmap.parquet"

            cli_args = [
                "--input", str(mock_file),
                "--output", str(output_path),
                "--docmap", str(docmap_path),
            ]

            ingest_multi.run(cli_args)

            st.success("✅ Ingestion completata!")

        except Exception as exc:
            st.error(f"❌ Errore ingestion: {exc}")
            LOGGER.exception("Error ingesting mock data")
            return

    st.session_state["pipeline_initialized"] = True
    st.session_state["data_source"] = "mock"
    st.session_state["success_message"] = "✅ Dataset demo pronto! Attiva 'Mock mode' nella sidebar."

    LOGGER.info("Mock data generation completed successfully")
    st.balloons()
    st.rerun()

def ingest_real_data_flow(uploaded_files: List) -> None:
    """Ingest real SAP Excel files uploaded by user.
    
    Args:
        uploaded_files: List of UploadedFile objects from st.file_uploader
    """
    if not uploaded_files:
        st.warning("⚠️ Nessun file selezionato")
        return

    raw_dir = PROJECT_ROOT / "data" / "raw" / "uploads"
    raw_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for uploaded in uploaded_files:
        target = raw_dir / uploaded.name
        target.write_bytes(uploaded.getbuffer())
        saved_paths.append(target)
        LOGGER.info(f"Saved uploaded file: {target}")

    st.info(f"📁 Salvati {len(saved_paths)} file in {raw_dir}")

    with st.spinner(f"📥 Ingestion di {len(saved_paths)} file..."):
        try:
            from src.data import ingest_multi

            cli_args = ["--input"] + [str(p) for p in saved_paths]
            cli_args += [
                "--output", str(PROJECT_ROOT / "data" / "curated" / "line_items.parquet"),
                "--docmap", str(PROJECT_ROOT / "data" / "curated" / "docmap.parquet"),
            ]

            ingest_multi.run(cli_args)

            st.success(f"✅ Ingestion completata: {len(saved_paths)} file")

        except Exception as exc:
            for path in saved_paths:
                if path.exists():
                    path.unlink()
            st.error(f"❌ Errore ingestion: {exc}")
            st.warning(f"⚠️ File caricati rimossi a causa dell'errore ({len(saved_paths)} file)")
            LOGGER.exception("Error ingesting real data")
            return

    st.session_state["pipeline_initialized"] = True
    st.session_state["data_source"] = "real"
    st.session_state["success_message"] = f"✅ Ingestion completata: {len(saved_paths)} file processati"

    LOGGER.info(f"Ingestion completed successfully: {len(saved_paths)} files")
    st.balloons()
    st.rerun()

def render_welcome_screen() -> None:
    """Render welcome screen for cold start (no data loaded yet)."""
    st.title("🚀 Inizia la tua Analisi Intelligente dei Pagamenti")

    st.markdown(
        """
        Benvenuto nel sistema di analisi predittiva delle **partite aperte**! 
        
        Prima di procedere con l'analisi, dobbiamo caricare i dati SAP. 
        Scegli una delle opzioni qui sotto per iniziare.
        """
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎭 Dataset Demo (Mock)")

        st.markdown(
            """
            **Perfetto per:**
            - Test e demo rapide
            - Familiarizzare con il sistema
            - Training team
            
            **Cosa include:**
            - 50 clienti sintetici
            - 12-18 mesi di storico
            - Pattern realistici (FIFO, LIFO, EOM, etc.)
            - Dati completi per tutti gli step
            """
        )

        if st.button("📦 Genera Dataset Demo", use_container_width=True, type="primary"):
            generate_mock_data_flow()
            st.stop()

    with col2:
        st.markdown("### 📁 Carica Dati SAP Reali")

        st.markdown(
            """
            **Perfetto per:**
            - Analisi produzione
            - Report esecutivi
            - Decisioni di credito
            
            **Formati supportati:**
            - Excel (.xlsx)
            - CSV (.csv)
            - Tabelle SAP: BSID, BSAD
            """
        )

        uploaded_files = st.file_uploader(
            "Seleziona file SAP (.xlsx o .csv)",
            type=["xlsx", "csv"],
            accept_multiple_files=True,
            key="sap_uploader_welcome",
        )

        if st.button("📥 Avvia Ingestion", use_container_width=True, type="primary", disabled=not uploaded_files):
            ingest_real_data_flow(uploaded_files)
            st.stop()

    st.markdown("---")
    st.markdown("### 📊 Stato Pipeline")

    init_status = check_pipeline_initialization()

    if init_status["initialized"]:
        st.success("✅ Dati caricati e pronti")
        st.info(f"📂 Fonte dati: **{init_status['source']}**")
    elif init_status["has_raw"]:
        st.warning("⏳ File raw presenti ma non processati. Clicca 'Avvia Ingestion' per procedere.")
    else:
        st.info("⏳ In attesa di dati. Seleziona un'opzione qui sopra per iniziare.")

    with st.expander("❓ Serve aiuto?"):
        st.markdown(
            """
            **Q: Qual è la differenza tra Mock e Real?**  
            A: Mock genera dati sintetici per test/demo. Real usa i tuoi file SAP effettivi per analisi produzione.
            
            **Q: Quanto tempo richiede la generazione mock?**  
            A: Circa 30-60 secondi per generare + ingest automatico.
            
            **Q: Posso caricare più file SAP?**  
            A: Sì, seleziona tutti i file insieme (BSID, BSAD, etc.) e clicca "Avvia Ingestion". 
            Verranno consolidati automaticamente.
            
            **Q: I dati vengono inviati online?**  
            A: No, tutto rimane locale. Nessun dato lascia il tuo computer/server.
            
            **Q: Come verifico se i dati sono stati caricati?**  
            A: La sezione "Stato Pipeline" qui sotto mostra lo stato in tempo reale. 
            Vedrai ✅ verde quando i dati sono pronti.
            """
        )

SAP_GLOSSARY = {
    "partite_aperte": {
        "term": "Partite Aperte",
        "sap_term": "Open Items",
        "definition": "Fatture o documenti contabili che non sono ancora stati completamente pagati dal cliente.",
        "example": "Una fattura di €10.000 emessa il 15 gennaio è ancora 'aperta' se il cliente non ha ancora effettuato il pagamento.",
    },
    "pareggio": {
        "term": "Pareggio Fatture",
        "sap_term": "Clearing / Document Clearing",
        "definition": "Il processo di chiusura di una partita aperta quando il pagamento viene ricevuto. Il documento viene 'pareggiato' con l'incasso.",
        "example": "Il 30 gennaio arriva il bonifico del cliente per €10.000 → la fattura del 15 gennaio viene 'pareggiata' (chiusa).",
    },
    "doc_pareggio": {
        "term": "Documento di Pareggio",
        "sap_term": "Clearing Document",
        "definition": "Il documento contabile che registra il pagamento ricevuto e chiude la partita aperta.",
        "example": "Bonifico SEPA del 30 gennaio per €10.000 → genera doc. pareggio che chiude la fattura.",
    },
    "data_registrazione": {
        "term": "Data di Registrazione",
        "sap_term": "Posting Date",
        "definition": "La data in cui il documento viene inserito nel sistema contabile SAP.",
        "example": "Fattura creata il 15 gennaio → data di registrazione = 15 gennaio.",
    },
    "data_documento": {
        "term": "Data Documento",
        "sap_term": "Document Date",
        "definition": "La data riportata sul documento originale (es. data fattura).",
        "example": "Fattura con data 10 gennaio ma registrata in SAP il 15 gennaio → data documento = 10 gennaio.",
    },
    "evento_pagamento": {
        "term": "Evento di Pagamento",
        "sap_term": "Payment Event",
        "definition": "Un gruppo di fatture (partite aperte) che vengono pagate insieme dallo stesso cliente nello stesso momento.",
        "example": "Cliente paga 3 fatture (€5k, €3k, €2k) con un unico bonifico di €10k il 30 gennaio → 1 evento di pagamento con 3 fatture.",
    },
}

WIZARD_STEPS = [
    {
        "id": 1,
        "title": "📥 Ingest SAP Data",
        "subtitle": "Carica e normalizza dati grezzi SAP",
        "description": (
            "**Cosa succede:** Importiamo i file SAP (BSAD, BSID) e li normalizziamo in uno schema standard.\n\n"
            "**Cosa ottieni:** File `line_items.parquet` con tutte le righe documento pulite e validate."
        ),
        "business_value": "💡 Fondamento: dati SAP trasformati in formato analitico.",
        "sap_pills": ["partite_aperte"],
        "technical_details": "Gestito dall'upload nella welcome screen - eseguito automaticamente.",
        "pipeline_steps": [],
        "output_files": ["line_items.parquet", "docmap.parquet"],
        "estimated_time": "Completato",
        "link_page": "Data Explorer",
    },
    {
        "id": 2,
        "title": "🔄 Build Events",
        "subtitle": "Costruisci eventi di pagamento da line items",
        "description": (
            "**Cosa succede:** Dai line items creiamo eventi di pagamento (fattura→pareggio) con tutte le feature temporali.\n\n"
            "**Cosa ottieni:** File `events.parquet` e `event_items.parquet` con eventi pronti per training."
        ),
        "business_value": "💡 Trasformiamo righe SAP in eventi business (fattura pagata/non pagata).",
        "sap_pills": ["pareggio", "data_documento"],
        "technical_details": "CLI: `python -m src.features.build_events`",
        "pipeline_steps": ["events"],
        "output_files": ["events.parquet", "event_items.parquet"],
        "estimated_time": "2-5 min",
        "link_page": "Data Explorer",
    },
    {
        "id": 3,
        "title": "🏷️ Make Labels",
        "subtitle": "Crea etichette per training ML",
        "description": (
            "**Cosa succede:** Da eventi creiamo dataset pointwise con label (pagato=1, non_pagato=0).\n\n"
            "**Cosa ottieni:** File `train_pointwise.parquet` pronto per feature engineering."
        ),
        "business_value": "💡 Prepariamo i 'casi di studio' per insegnare al modello.",
        "sap_pills": [],
        "technical_details": "CLI: `python -m src.features.make_labels`",
        "pipeline_steps": ["labels"],
        "output_files": ["train_pointwise.parquet"],
        "estimated_time": "1-3 min",
        "link_page": None,
    },
    {
        "id": 4,
        "title": "⚙️ Make Features",
        "subtitle": "Ingegnerizza feature per ML",
        "description": (
            "**Cosa succede:** Calcoliamo feature avanzate (aging, importo relativo, recency, etc.) per ogni evento.\n\n"
            "**Cosa ottieni:** File `train_features.parquet` completo di tutte le variabili predittive."
        ),
        "business_value": "💡 Creiamo le 'variabili intelligenti' che il modello userà per predire.",
        "sap_pills": [],
        "technical_details": "CLI: `python -m src.features.make_features`",
        "pipeline_steps": ["features"],
        "output_files": ["train_features.parquet"],
        "estimated_time": "3-7 min",
        "link_page": None,
    },

    {
        "id": 5,
        "title": "🧬 Client Embeddings",
        "subtitle": "Crea rappresentazioni numeriche clienti",
        "description": (
            "**Cosa succede:** Trasformiamo ogni cliente in un vettore numerico (embedding) che cattura il suo comportamento.\n\n"
            "**Cosa ottieni:** File `client_embeddings.parquet` con feature aggregate per clustering."
        ),
        "business_value": "💡 Rappresentazione matematica del comportamento cliente.",
        "sap_pills": [],
        "technical_details": "CLI: `python -m src.analytics.make_client_embeddings`",
        "pipeline_steps": ["embeddings"],
        "output_files": ["client_embeddings.parquet"],
        "estimated_time": "2-5 min",
        "link_page": None,
    },
    {
        "id": 6,
        "title": "🗺️ Embedding Projection",
        "subtitle": "Proietta embeddings in 2D (UMAP/PCA)",
        "description": (
            "**Cosa succede:** Riduciamo dimensionalità embeddings per visualizzazione scatter plot 2D.\n\n"
            "**Cosa ottieni:** File `client_embedding_2d.parquet` con coordinate (x, y) per ogni cliente."
        ),
        "business_value": "💡 'Mappa' visuale dei clienti: chi è simile a chi.",
        "sap_pills": [],
        "technical_details": "CLI: `python -m src.analytics.embedding_viz --umap`",
        "pipeline_steps": ["projection"],
        "output_files": ["client_embedding_2d.parquet"],
        "estimated_time": "3-8 min",
        "link_page": None,
    },
    {
        "id": 7,
        "title": "🧩 HDBSCAN Clustering",
        "subtitle": "Segmenta clienti in cluster automatici",
        "description": (
            "**Cosa succede:** HDBSCAN trova gruppi di clienti con comportamento simile (density-based clustering).\n\n"
            "**Cosa ottieni:** File `client_clusters.parquet` con cluster_id per ogni cliente."
        ),
        "business_value": "💡 Segmentazione automatica: clienti con pattern simili nello stesso gruppo.",
        "sap_pills": [],
        "technical_details": "CLI: `python -m src.analytics.run_hdbscan`",
        "pipeline_steps": ["hdbscan"],
        "output_files": ["client_clusters.parquet"],
        "estimated_time": "2-5 min",
        "link_page": "Clusters",
    },
    {
        "id": 8,
        "title": "📊 Cluster Profiles",
        "subtitle": "Riassumi caratteristiche cluster",
        "description": (
            "**Cosa succede:** Per ogni cluster calcoliamo metriche aggregate (policy dominante, size, coverage media).\n\n"
            "**Cosa ottieni:** File `cluster_summary.parquet` con profilo di ogni cluster."
        ),
        "business_value": "💡 'Identikit' di ogni segmento: cosa li accomuna, quanto sono grandi.",
        "sap_pills": [],
        "technical_details": "CLI: `python -m src.analytics.cluster_profile`",
        "pipeline_steps": ["clusters"],
        "output_files": ["cluster_summary.parquet"],
        "estimated_time": "1-3 min",
        "link_page": "Clusters",
    },

    {
        "id": 9,
        "title": "✅ Pre-Training Validation",
        "subtitle": "Verifica qualità dati prima del training",
        "description": (
            "**Cosa succede:** Controlliamo che i dati siano pronti per training (no leak, balance OK, feature complete).\n\n"
            "**Cosa ottieni:** Report di validazione + conferma che si può procedere al training."
        ),
        "business_value": "💡 Quality gate: evitiamo training su dati problematici.",
        "sap_pills": [],
        "technical_details": "CLI: `python -m src.models.train --validate-only`",
        "pipeline_steps": ["validate"],
        "output_files": [],
        "estimated_time": "1-2 min",
        "link_page": None,
    },
    {
        "id": 10,
        "title": "🎯 Train Model",
        "subtitle": "Allena CatBoost/AutoGluon per scoring",
        "description": (
            "**Cosa succede:** Training del modello ML su storico pagamenti → predice p_i (probabilità pagamento fattura).\n\n"
            "**Cosa ottieni:** File `champion.json`, `catboost.cbm`, `calibrator.pkl` con modello pronto."
        ),
        "business_value": "💡 Il 'cervello' predittivo: impara pattern di pagamento da migliaia di esempi. USA cluster_id come feature!",
        "sap_pills": [],
        "technical_details": "CLI: `python -m src.models.train --model catboost`",
        "pipeline_steps": ["train"],
        "output_files": ["champion.json", "catboost.cbm", "calibrator.pkl"],
        "estimated_time": "10-30 min",
        "link_page": "KPI Dashboard",
    },

    {
        "id": 11,
        "title": "🔍 Pattern Detectors",
        "subtitle": "Identifica policy di pagamento dominanti",
        "description": (
            "**Cosa succede:** Per ogni cliente calcoliamo quale policy (FIFO, LIFO, largest, etc.) spiega meglio il suo storico.\n\n"
            "**Cosa ottieni:** File `client_pattern_scores.parquet` con dominant_policy per cliente."
        ),
        "business_value": "💡 Scopriamo 'come paga' ogni cliente (es. sempre le fatture più vecchie prima).",
        "sap_pills": ["evento_pagamento"],
        "technical_details": "CLI: `python -m src.analytics.pattern_detectors`",
        "pipeline_steps": ["patterns"],
        "output_files": ["client_pattern_scores.parquet"],
        "estimated_time": "2-5 min",
        "link_page": "Pattern Summary",
    },
    {
        "id": 12,
        "title": "📊 Rules Backtest",
        "subtitle": "Backtesta accuratezza policy rules",
        "description": (
            "**Cosa succede:** Verifichiamo quanti pagamenti storici sono predetti correttamente da ogni policy.\n\n"
            "**Cosa ottieni:** File `client_rules_accuracy.parquet` con accuracy per policy/cliente."
        ),
        "business_value": "💡 Misuriamo quanto sono 'predicibili' i clienti con regole semplici.",
        "sap_pills": [],
        "technical_details": "CLI: `python -m src.analytics.rules_backtest`",
        "pipeline_steps": ["rules"],
        "output_files": ["client_rules_accuracy.parquet"],
        "estimated_time": "2-4 min",
        "link_page": None,
    },
    {
        "id": 13,
        "title": "📋 Pattern Table",
        "subtitle": "Crea tabella riassuntiva pattern clienti",
        "description": (
            "**Cosa succede:** Assembliamo una tabella con dominant_policy, confidence, coverage mensile per ogni cliente.\n\n"
            "**Cosa ottieni:** File `pattern_table.parquet` - vista unificata pattern cliente."
        ),
        "business_value": "💡 'Carta d'identità' di ogni cliente: come paga, quanto è stabile, coverage.",
        "sap_pills": [],
        "technical_details": "CLI: `python -m src.analytics.make_pattern_table`",
        "pipeline_steps": ["table"],
        "output_files": ["pattern_table.parquet"],
        "estimated_time": "1-2 min",
        "link_page": "Pattern Summary",
    },
    {
        "id": 14,
        "title": "📸 Pattern Reports",
        "subtitle": "Genera report grafici pattern",
        "description": (
            "**Cosa succede:** Creiamo PNG visuali per ogni pattern cliente (timeline, policy chart, etc.).\n\n"
            "**Cosa ottieni:** Cartella `reports/` con immagini pattern card."
        ),
        "business_value": "💡 Report visivi pronti per condivisione con team business.",
        "sap_pills": [],
        "technical_details": "CLI: `python -m src.report.make_report`",
        "pipeline_steps": ["reports"],
        "output_files": ["reports/"],
        "estimated_time": "5-10 min",
        "link_page": None,
    },

    {
        "id": 15,
        "title": "📈 KPI Ranking",
        "subtitle": "Calcola ranking KPI fatture aperte",
        "description": (
            "**Cosa succede:** Per ogni cliente con fatture aperte, calcoliamo p_i (scoring ML) per ogni fattura e rankin KPI.\n\n"
            "**Cosa ottieni:** File `kpi_ranking.parquet` con ranking predittivo fatture."
        ),
        "business_value": "💡 Top-k fatture da sollecitare: modello ordina per probabilità pagamento.",
        "sap_pills": [],
        "technical_details": "CLI: `python -m src.analytics.kpi_ranking`",
        "pipeline_steps": ["kpi"],
        "output_files": ["kpi_ranking.parquet"],
        "estimated_time": "3-7 min",
        "link_page": "KPI Dashboard",
    },
    {
        "id": 16,
        "title": "🎚️ Calibration Diagnostics",
        "subtitle": "Valuta calibrazione probabilità modello",
        "description": (
            "**Cosa succede:** Verifichiamo se le probabilità p_i sono ben calibrate (p=0.8 → 80% pagato davvero).\n\n"
            "**Cosa ottieni:** File `calibration_summary.parquet` con metriche ECE, Brier, reliability curve."
        ),
        "business_value": "💡 Quality check: le probabilità ML sono affidabili per decisioni business.",
        "sap_pills": [],
        "technical_details": "CLI: `python -m src.analytics.calibration`",
        "pipeline_steps": ["calibration"],
        "output_files": ["calibration_summary.parquet"],
        "estimated_time": "2-5 min",
        "link_page": None,
    },
    {
        "id": 17,
        "title": "🎯 Policy Metrics",
        "subtitle": "Confronta ranking ML vs policy cliente",
        "description": (
            "**Cosa succede:** Per ogni cliente confrontiamo ranking modello con ranking policy dominante → agreement_score.\n\n"
            "**Cosa ottieni:** File `event_policy_metrics.parquet` con coerenza modello/pattern storico."
        ),
        "business_value": "💡 Coherence check: badge verde/rosso se modello è allineato al pattern cliente.",
        "sap_pills": [],
        "technical_details": "CLI: `python -m src.analytics.event_policy_metrics`",
        "pipeline_steps": ["policy"],
        "output_files": ["event_policy_metrics.parquet"],
        "estimated_time": "2-5 min",
        "link_page": "Pattern Card",
    },
]

def _get_data_root() -> Path:
    """Return data root directory: data_mock/ when mock_mode active, else data/."""
    project_root = Path(__file__).resolve().parents[2]
    if st.session_state.get("mock_mode"):
        return project_root / "data_mock"
    return project_root / Path(os.getenv("APP_DATA_ROOT", "data"))

def _get_curated_dir() -> Path:
    return _get_data_root() / "curated"

def _get_models_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "models"

def _cleanup_all_data() -> None:
    """Delete all curated files and models to start fresh."""
    import shutil

    data_root = _get_data_root()
    curated = data_root / "curated"
    models = _get_models_dir()

    try:
        if curated.exists():
            shutil.rmtree(curated)
            LOGGER.info(f"Deleted curated directory: {curated}")

        if models.exists():
            shutil.rmtree(models)
            LOGGER.info(f"Deleted models directory: {models}")

        st.session_state["pipeline_initialized"] = False
        st.session_state["success_message"] = "🗑️ Tutti i dati sono stati cancellati. Puoi ricominciare da zero."
        LOGGER.info("Cleanup completed successfully")

    except Exception as exc:
        error_msg = f"❌ Errore durante cleanup: {exc}"
        st.session_state["error_message"] = error_msg
        LOGGER.exception("Error during cleanup")

def _check_step_completion(step_id: int) -> Dict[str, Any]:
    """Check if a wizard step has been completed by verifying all output files exist."""
    wizard_step = WIZARD_STEPS[step_id - 1]
    curated = _get_curated_dir()
    models = _get_models_dir()

    output_files = wizard_step["output_files"]
    completed_count = 0
    missing_files = []

    for filename in output_files:
        if filename.endswith(".json") or filename.endswith(".cbm") or filename.endswith(".pkl"):
            file_path = models / filename
        else:
            file_path = curated / filename

        if file_path.exists():
            completed_count += 1
        else:
            missing_files.append(filename)

    return {
        "completed": completed_count == len(output_files),
        "progress": completed_count / len(output_files) if output_files else 0,
        "completed_count": completed_count,
        "total_count": len(output_files),
        "missing_files": missing_files,
    }

def _run_wizard_step(step_id: int) -> None:
    """Execute all pipeline steps for a wizard step."""
    wizard_step = WIZARD_STEPS[step_id - 1]
    pipeline_steps = wizard_step["pipeline_steps"]

    project_root = Path(__file__).resolve().parents[2]
    use_mock = "true" if st.session_state.get("mock_mode") else "false"

    progress_bar = st.progress(0, text=f"Preparazione step {step_id}...")
    status_text = st.empty()

    for i, step_name in enumerate(pipeline_steps):
        progress = (i + 1) / len(pipeline_steps)
        status_text.text(f"Esecuzione: {step_name} ({i+1}/{len(pipeline_steps)})")
        progress_bar.progress(progress)

        step_modules = {
            "events": "src.features.build_events",
            "labels": "src.features.make_labels",
            "features": "src.features.make_features",
            "validate": "src.models.train",
            "train": "src.models.train",
            "patterns": "src.analytics.pattern_detectors",
            "rules": "src.analytics.rules_backtest",
            "table": "src.analytics.make_pattern_table",
            "reports": "src.report.make_report",
            "embeddings": "src.analytics.make_client_embeddings",
            "projection": "src.analytics.embedding_viz",
            "hdbscan": "src.analytics.run_hdbscan",
            "clusters": "src.analytics.cluster_profile",
            "kpi": "src.analytics.kpi_ranking",
            "calibration": "src.analytics.calibration",
            "policy": "src.analytics.event_policy_metrics",
        }

        if step_name not in step_modules:
            st.error(f"❌ Step '{step_name}' non riconosciuto")
            return

        cmd = [sys.executable, "-m", step_modules[step_name]]

        data_root = _get_data_root()
        curated = data_root / "curated"

        if step_name == "events":
            cmd.extend(["--input", str(curated / "line_items.parquet")])
            cmd.extend(["--output", str(curated / "events.parquet")])
            cmd.extend(["--output-items", str(curated / "event_items.parquet")])
        elif step_name == "labels":
            cmd.extend(["--events", str(curated / "events.parquet")])
            cmd.extend(["--event-items", str(curated / "event_items.parquet")])
            cmd.extend(["--output", str(curated / "train_pointwise.parquet")])
        elif step_name == "features":
            cmd.extend(["--train-pointwise", str(curated / "train_pointwise.parquet")])
            cmd.extend(["--events", str(curated / "events.parquet")])
            cmd.extend(["--output", str(curated / "train_features.parquet")])
        elif step_name == "validate":
            cmd.append("--validate-only")

        env = os.environ.copy()
        env["USE_MOCK"] = "true" if st.session_state.get("mock_mode") else "false"

        try:
            result = subprocess.run(
                cmd,
                cwd=str(project_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.stdout:
                LOGGER.info(f"Step {step_name} stdout:\n{result.stdout}")
            if result.stderr:
                LOGGER.warning(f"Step {step_name} stderr:\n{result.stderr}")

            if result.returncode != 0:
                error_msg = f"❌ Errore durante esecuzione '{step_name}'"
                st.session_state["error_message"] = error_msg
                st.session_state["error_details"] = result.stderr or "Nessun dettaglio disponibile"
                LOGGER.error(f"{error_msg}\n{result.stderr}")
                st.error(error_msg)
                if result.stderr:
                    with st.expander("📋 Dettagli errore"):
                        st.code(result.stderr, language="text")
                progress_bar.empty()
                status_text.empty()
                return
        except subprocess.TimeoutExpired:
            error_msg = f"⏱️ Timeout durante esecuzione '{step_name}' (>600s)"
            st.session_state["error_message"] = error_msg
            LOGGER.error(error_msg)
            st.error(error_msg)
            progress_bar.empty()
            status_text.empty()
            return
        except Exception as e:
            error_msg = f"❌ Errore imprevisto: {e}"
            st.session_state["error_message"] = error_msg
            LOGGER.exception(f"Unexpected error in step {step_name}")
            st.error(error_msg)
            progress_bar.empty()
            status_text.empty()
            return

    progress_bar.progress(1.0, text="✅ Completato!")
    status_text.empty()

    st.session_state["success_message"] = f"✅ Step {step_id} completato con successo!"
    LOGGER.info(f"Wizard step {step_id} completed successfully")

    st.success(f"✅ Step {step_id} completato con successo!")
    st.balloons()

def _render_sap_pill(pill_id: str) -> None:
    """Render a SAP glossary term as an info box."""
    if pill_id not in SAP_GLOSSARY:
        return

    term_data = SAP_GLOSSARY[pill_id]
    with st.expander(f"📘 {term_data['term']} (SAP: {term_data['sap_term']})"):
        st.markdown(f"**Definizione:** {term_data['definition']}")
        st.markdown(f"**Esempio:** {term_data['example']}")

def _render_wizard_step_card(step: Dict[str, Any], step_completion: Dict[str, Any]) -> None:
    """Render a single wizard step as an expandable card."""
    step_id = step["id"]
    is_completed = step_completion["completed"]
    progress_pct = step_completion["progress"]

    col1, col2, col3 = st.columns([0.6, 0.3, 0.1])
    with col1:
        st.markdown(f"### {step['title']}")
        st.caption(step["subtitle"])
    with col2:
        if is_completed:
            st.success("✅ Completato")
        elif progress_pct > 0:
            st.warning(f"⏸️ Parziale ({int(progress_pct*100)}%)")
        else:
            st.info("⏳ Da fare")
    with col3:
        st.markdown(f"**{step_id}**", help=f"Step {step_id} di 4")

    if not is_completed and progress_pct > 0:
        st.progress(progress_pct)
        st.caption(
            f"{step_completion['completed_count']}/{step_completion['total_count']} file completati. "
            f"Mancanti: {', '.join(step_completion['missing_files'])}"
        )

    if step_completion["missing_files"]:
        with st.expander("❌ File Mancanti - Clicca per dettagli"):
            for missing_file in step_completion["missing_files"]:
                st.markdown(f"- ❌ `{missing_file}`")
            st.caption("💡 Esegui questo step per generare i file mancanti.")

    st.markdown(step["description"])

    st.info(step["business_value"])

    if step["sap_pills"]:
        st.markdown("**📚 Glossario SAP:**")
        cols = st.columns(len(step["sap_pills"]))
        for i, pill_id in enumerate(step["sap_pills"]):
            with cols[i]:
                _render_sap_pill(pill_id)

    with st.expander("🔧 Dettagli Tecnici (per esperti)"):
        st.markdown(step["technical_details"])
        st.markdown(f"**Pipeline steps:** `{', '.join(step['pipeline_steps'])}`")
        st.markdown(f"**Output files:** `{', '.join(step['output_files'])}`")
        st.markdown(f"**⏱️ Tempo stimato:** {step['estimated_time']}")

    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        if st.button(
            f"▶️ Esegui Step {step_id}" if not is_completed else f"🔄 Ri-esegui Step {step_id}",
            key=f"run_wizard_{step_id}",
            use_container_width=True,
            type="primary" if not is_completed else "secondary",
        ):
            _run_wizard_step(step_id)
            st.rerun()
    with col2:
        if step.get("link_page"):
            if st.button(
                f"📖 Scopri di più",
                key=f"learn_more_{step_id}",
                use_container_width=True,
                help=f"Vai alla pagina '{step['link_page']}' per esplorare i risultati",
            ):
                st.query_params["page"] = step["link_page"]
                st.rerun()

    st.divider()

def render_overview_wizard_page() -> None:
    """Render Prompt 9 Overview & Wizard page - HDBSCAN→Scoring flow in 4 business-friendly steps."""

    if "error_message" in st.session_state:
        st.error(st.session_state.pop("error_message"))
        if "error_details" in st.session_state:
            with st.expander("📋 Dettagli errore"):
                st.code(st.session_state.pop("error_details"), language="text")

    if "success_message" in st.session_state:
        st.success(st.session_state.pop("success_message"))

    init_status = check_pipeline_initialization()

    if not init_status["initialized"]:
        render_welcome_screen()
        return

    st.title("🏠 Benvenuto nell'Analisi Intelligente dei Pagamenti")
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 10px; color: white; margin-bottom: 2rem;'>
            <h2 style='margin: 0; color: white;'>Da Dati SAP a Decisioni Strategiche</h2>
            <p style='font-size: 1.1rem; margin: 0.5rem 0 0 0; color: #f0f0f0;'>
                🧩 Segmenta i clienti per comportamento → 🎯 Prevedi quali fatture verranno pagate → 📊 Valida coerenza con pattern storici
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        st.markdown("### 🎯 Flusso HDBSCAN → Scoring Predittivo")
        st.markdown(
            """
            Questo sistema usa un approccio **in due fasi** per massimizzare precisione e trasparenza:
            
            **1️⃣ HDBSCAN prima:** Identifica automaticamente **gruppi di clienti con comportamenti simili** 
            analizzando lo storico pagamenti (mensile, settimanale, FIFO, LIFO, irregolare).
            
            **2️⃣ Scoring dopo:** Allena un modello predittivo (CatBoost/AutoGluon) che stima **p_i = probabilità 
            che fattura i sia pagata**, usando sia segnali locali (importo, scadenza) che il **pattern del cluster**.
            
            **Risultato:** Ranking fatture aperte + badge coerenza (verde="in linea col pattern", rosso="anomalia").
            
            ---
            
            **Domande a cui rispondiamo:**
            
            - � **Quando** arriveranno i prossimi pagamenti?
            - 🧩 **Quali** cluster comportamentali esistono nella base clienti?
            - � **Quali** fatture hanno alta probabilità di pagamento?
            - ⚠️ **Chi** si comporta diversamente dal solito (divergenza dal pattern)?
            - 📊 **Come** posso validare le predizioni con lo storico?
            """
        )

    with col2:
        st.markdown("### 📊 Dati di Input")
        st.markdown(
            """
            **Cosa ti serve:**
            - Export SAP delle **partite aperte** (Open Items)
            - Export SAP dei **pagamenti ricevuti** (Clearing Documents)
            - Formato: Excel (.xlsx) o CSV
            
            **Colonne minime:**
            - `Cliente`, `Numero Documento`, `Importo`
            - `Data Documento`, `Data Pareggio`
            - `Documento di Pareggio`, `Divisa`
            
            **File opzionali:**
            - `Tipo Documento`, `Chiave Contabile`
            - `Periodo Contabile`, `Società`
            
            _(Vedi mapping completo in Data Explorer)_
            """
        )

    st.divider()

    st.markdown("## 🧙‍♂️ Wizard Guidato: Dal Caricamento Dati al Ranking Fatture")
    st.markdown(
        """
        **Flusso in 4 passi:**  
        1️⃣ **Carica dati SAP** → normalizza e prepara storico  
        2️⃣ **Segmenta clienti** → HDBSCAN identifica cluster comportamentali  
        3️⃣ **Allena modello** → CatBoost impara a predire p_i (probabilità pagamento)  
        4️⃣ **Rank fatture** → scoring + coerenza con pattern cliente
        
        **🎯 Perché questo ordine?**  
        HDBSCAN prima cattura **strategie stabili** (mensile, FIFO, k-mode). Il modello scoring poi usa 
        queste feature per predizioni più accurate. La coerenza finale valida se le predizioni sono **allineate al comportamento storico**.
        
        **📌 Ogni step richiede il completamento del precedente.** Clicca **"▶️ Esegui Step X"** per avviare.
        """
    )

    step_statuses = [_check_step_completion(i + 1) for i in range(len(WIZARD_STEPS))]

    col_progress, col_refresh = st.columns([0.85, 0.15])

    with col_progress:
        total_progress = sum(s["progress"] for s in step_statuses) / len(WIZARD_STEPS)
        st.progress(total_progress, text=f"Progresso complessivo: {int(total_progress * 100)}%")

    with col_refresh:
        if st.button("🔄 Verifica File", help="Aggiorna lo stato dei file di output", use_container_width=True):
            st.rerun()

    st.markdown("---")

    for i, step in enumerate(WIZARD_STEPS):
        with st.container():
            _render_wizard_step_card(step, step_statuses[i])

    st.markdown("## 🚀 Quick Start - Step Critici")

    cols = st.columns(min(6, len(WIZARD_STEPS)))
    for i in range(min(6, len(WIZARD_STEPS))):
        step_num = i + 1
        with cols[i]:
            is_disabled = i > 0 and not step_statuses[i - 1]["completed"]
            button_type = "primary" if i == 0 else "secondary"

            if st.button(
                f"▶️ Step {step_num}",
                use_container_width=True,
                type=button_type,
                disabled=is_disabled,
                help=WIZARD_STEPS[i]["title"]
            ):
                _run_wizard_step(step_num)
                st.rerun()

    st.markdown("---")
    col_reset, col_spacer = st.columns([0.2, 0.8])
    with col_reset:
        if st.button("🗑️ Reset Tutto", use_container_width=True, type="secondary", help="Cancella tutti i dati curated e ricomincia da zero"):
            _cleanup_all_data()
            st.rerun()

    with st.expander("❓ Domande Frequenti (FAQ)"):
        st.markdown(
            """
            **Q: Quanto tempo richiede l'analisi completa?**  
            A: Circa 25-65 minuti totali, a seconda del volume di dati. Step 1 (5-7min), Step 2 (5-12min), Step 3 (10-30min), Step 4 (5-15min). Progresso visibile in tempo reale.
            
            **Q: Perché HDBSCAN prima del modello predittivo?**  
            A: HDBSCAN cattura **pattern stabili** a livello cliente (mensile, FIFO, cardinalità tipica) che diventano **feature robuste** per il scoring fatture. Questo approccio combina interpretabilità (cluster visibili in UI) con performance predittiva.
            
            **Q: Cosa significa 'coerenza con pattern'?**  
            A: L'agreement_score confronta il ranking del modello con la policy dominante del cliente (es. FIFO, largest). Badge verde = allineato, rosso = divergente. Aiuta a identificare anomalie comportamentali.
            
            **Q: I miei dati sono al sicuro?**  
            A: Sì, tutti i dati rimangono sul tuo computer/server. Niente viene inviato online.
            
            **Q: Serve conoscere il machine learning?**  
            A: No, l'interfaccia è business-friendly. Dettagli tecnici sono opzionali (collapse "Dettagli Tecnici").
            
            **Q: Cosa succede se interrompo l'elaborazione?**  
            A: Ogni step è idempotente: puoi ri-eseguirlo senza perdere progressi. I file .parquet già generati non vengono sovrascritti inutilmente.
            
            **Q: Posso testare con dati di esempio?**  
            A: Sì, attiva **"Mock Mode"** nella sidebar. Genera dataset sintetico (50 clienti, 12-18 mesi di storico) con il pulsante "📦 Crea dataset demo".
            
            **Q: Come interpreto i risultati?**  
            A: Ogni pagina ha guide integrate:
            - **Clusters:** Esplora gruppi comportamentali, clicca su punti per drill-down
            - **Pattern Card:** Vedi ranking fatture + badge coerenza + SHAP top-3
            - **KPI Dashboard:** Metriche globali (precision@k, calibrazione, uplift vs FIFO)
            - **Policy IQ:** Confusion matrix policy classifier, uplift per cluster
            
            **Q: Quali modelli sono supportati?**  
            A: CatBoost (primario, SHAP incluso) e AutoGluon (opzionale, no SHAP). Selezionabili via flag `--model catboost|autogluon` in CLI o radio button in Pattern Card.
            
            **Q: Posso esportare i risultati?**  
            A: Sì, ogni pagina ha pulsante "Download CSV". Pattern Card ha anche "Scarica JSON Explain" con cluster/policy/SHAP.
            """
        )

    st.markdown("---")
    st.caption(
        "💡 **Quick Start:** Attiva 'Mock Mode' nella sidebar → clicca '📦 Crea dataset demo' → esegui Step 1-4 → "
        "esplora risultati in **Clusters** (segmenti) e **Pattern Card** (ranking + coerenza)."
    )

    st.markdown("### 🔗 Vai Direttamente a:")
    link_cols = st.columns(4)
    with link_cols[0]:
        if st.button("📊 Clusters", use_container_width=True):
            st.session_state["selected_page"] = "Clusters"
            st.rerun()
    with link_cols[1]:
        if st.button("📈 Pattern Card", use_container_width=True):
            st.session_state["selected_page"] = "Pattern Card"
            st.rerun()
    with link_cols[2]:
        if st.button("🎯 KPI Dashboard", use_container_width=True):
            st.session_state["selected_page"] = "KPI Dashboard"
            st.rerun()
    with link_cols[3]:
        if st.button("🧪 Data Explorer", use_container_width=True):
            st.session_state["selected_page"] = "Data Explorer"
            st.rerun()

