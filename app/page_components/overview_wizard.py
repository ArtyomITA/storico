""" Homepage Overview & Wizard - flusso umano comprensibile per CEO/business users."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

LOGGER = logging.getLogger(__name__)


# ================================================================================================
# SAP Business Glossary - Pillole per utenti business
# ================================================================================================

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


# ================================================================================================
# Wizard Step Definitions
# ================================================================================================

# Prompt 9: Updated wizard steps aligned with HDBSCAN→Scoring flow
WIZARD_STEPS = [
    {
        "id": 1,
        "title": "📥 Carica Dati SAP",
        "subtitle": "Importa e normalizza le partite aperte",
        "description": (
            "**Cosa succede:** Leggiamo i dati delle fatture dal tuo sistema SAP e li organizziamo "
            "in un formato pulito e pronto per l'analisi.\n\n"
            "**Cosa ottieni:** Una tabella con tutte le partite aperte, i pagamenti ricevuti, e le informazioni "
            "sui clienti (chi ha pagato cosa, quando, e quanto)."
        ),
        "business_value": (
            "💡 **Valore Business:** Trasformiamo i dati grezzi SAP (spesso dispersi in tabelle diverse) "
            "in una vista unificata e comprensibile. È come mettere ordine in un archivio cartaceo."
        ),
        "sap_pills": ["partite_aperte", "pareggio", "data_documento"],
        "technical_details": (
            "**Dettagli Tecnici:**\n"
            "- Normalizzazione SAP: mapping colonne (BSAD, BSID) → schema contrattuale\n"
            "- Validazioni: cliente non nullo, date coerenti, importi ≥0\n"
            "- Genera `line_items.parquet`, `events.parquet`, `event_items.parquet`\n"
            "- Engine: Pandas + Pyarrow per performance\n"
            "- CLI: `python -m src.data.normalize_sap` (Prompt 2)"
        ),
        "pipeline_steps": ["normalize", "ingest", "events", "labels", "features"],
        "output_files": ["line_items.parquet", "events.parquet", "event_items.parquet", "train_features.parquet", "client_profiles.parquet"],
        "estimated_time": "3-7 minuti",
        "link_page": "Data Explorer",
    },
    {
        "id": 2,
        "title": "🧩 Segmenta Clienti (HDBSCAN)",
        "subtitle": "Identifica gruppi di pagamento simili",
        "description": (
            "**Cosa succede:** Analizziamo lo storico di ogni cliente per identificare **pattern stabili**: "
            "chi paga sempre a fine mese, chi è settimanale, chi è irregolare. "
            "Poi raggruppiamo clienti con comportamenti simili in cluster automatici.\n\n"
            "**Cosa ottieni:** Una 'mappa dei comportamenti' dove ogni cliente appartiene a un gruppo "
            "con caratteristiche comuni (es. 'Pagatori FIFO mensili', 'Irregolari multifattura')."
        ),
        "business_value": (
            "💡 **Valore Business:** Clustering ti permette di **segmentare strategie**: "
            "solleciti personalizzati per gruppo, previsioni di cassa più accurate, identificazione rapida anomalie. "
            "È come avere una 'famiglia' di clienti con regole comuni."
        ),
        "sap_pills": ["evento_pagamento", "data_registrazione"],
        "technical_details": (
            "**Dettagli Tecnici:**\n"
            "- Feature engineering: periodicità (ACF), coverage mensile, policy score storici\n"
            "- Embeddings: PCA/UMAP su client-level features (storico aggregato)\n"
            "- Clustering: **HDBSCAN** (density-based, auto #cluster, metrica cosine)\n"
            "- Pattern detector (Prompt 5): dominant_policy, pattern_confidence, k_mode\n"
            "- Output: `client_clusters.parquet`, `client_patterns.parquet`, `cluster_summary.parquet`"
        ),
        "pipeline_steps": ["embeddings", "projection", "hdbscan", "clusters", "patterns"],
        "output_files": ["client_embeddings.parquet", "client_embedding_2d.parquet", "client_clusters.parquet", "client_patterns.parquet", "cluster_summary.parquet"],
        "estimated_time": "5-12 minuti",
        "link_page": "Clusters",
    },
    {
        "id": 3,
        "title": "🎯 Allena Modello Scoring",
        "subtitle": "Prevedi quali fatture verranno pagate (p_i)",
        "description": (
            "**Cosa succede:** Insegniamo al sistema a calcolare **p_i = probabilità che fattura i sia pagata** "
            "usando segnali locali (importo, scadenza, aging) + pattern del cliente (cluster, policy dominante). "
            "Il modello impara da migliaia di esempi storici 'fattura pagata vs non pagata'.\n\n"
            "**Cosa ottieni:** Un **ranking predittivo** per ogni cliente: data la lista fatture aperte, "
            "il sistema ordina per probabilità di pagamento e mostra coherenza con il pattern storico."
        ),
        "business_value": (
            "💡 **Valore Business:** Predizioni accurate ti permettono di **prioritizzare azioni**: "
            "quali fatture sollecitare, dove allocare team crediti, stimare liquidità settimanale. "
            "È come avere un esperto che conosce ogni cliente a memoria."
        ),
        "sap_pills": ["partite_aperte", "pareggio"],
        "technical_details": (
            "**Dettagli Tecnici:**\n"
            "- Algoritmi ML: **CatBoost** (primario) + AutoGluon (opzionale)\n"
            "- Feature merge: invoice-level + cluster/pattern del cliente (no leakage)\n"
            "- Training: split temporale, class imbalance handling, calibrazione isotonica\n"
            "- Validation: pre-training checks, SHAP explainability\n"
            "- Output: `champion.json`, `catboost.cbm`, `calibrator.pkl` (Prompt 6)"
        ),
        "pipeline_steps": ["validate", "train"],
        "output_files": ["champion.json", "catboost.cbm", "calibrator.pkl"],
        "estimated_time": "10-30 minuti",
        "link_page": "KPI Dashboard",
    },
    {
        "id": 4,
        "title": "📊 Rank Fatture Aperte",
        "subtitle": "Scoring + coerenza con pattern cliente",
        "description": (
            "**Cosa succede:** Per ogni cliente con fatture aperte, calcoliamo **p_i per ogni fattura** "
            "usando il modello allenato. Poi confrontiamo il ranking modello con la policy dominante del cliente "
            "per generare un **agreement_score** (indica se il modello è allineato al comportamento storico).\n\n"
            "**Cosa ottieni:** Tabella Top-k proposte con badge verde/giallo/rosso: "
            "'✅ In linea con pattern' oppure '⚠️ Divergente dal solito comportamento'."
        ),
        "business_value": (
            "💡 **Valore Business:** Scoring + coerenza ti aiutano a **prendere decisioni consapevoli**: "
            "se il modello propone fatture anomale rispetto al pattern, puoi indagare (es. cambio strategia cliente). "
            "È come avere sia il radar predittivo che il benchmark storico."
        ),
        "sap_pills": ["evento_pagamento"],
        "technical_details": (
            "**Dettagli Tecnici:**\n"
            "- Scoring: p_i con calibrazione isotonica per ogni fattura aperta\n"
            "- Coherence metric: confronto top-k modello vs top-k policy dominante\n"
            "- Agreement score: overlap Jaccard + ranking correlation\n"
            "- Policy ranking: FIFO, LIFO, largest, smallest, eomfirst (Prompt 7)\n"
            "- Output: `event_policy_predictions.parquet` con p_i, rank, agreement_score, alignment_status"
        ),
        "pipeline_steps": ["score", "kpi", "calibration", "policy"],
        "output_files": ["event_policy_predictions.parquet", "kpi_ranking.parquet", "calibration_summary.parquet", "event_policy_metrics.parquet"],
        "estimated_time": "5-15 minuti",
        "link_page": "Pattern Card",
    },
]


# ================================================================================================
# Helper Functions
# ================================================================================================


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


def _check_step_completion(step_id: int) -> Dict[str, Any]:
    """Check if a wizard step has been completed by verifying all output files exist."""
    wizard_step = WIZARD_STEPS[step_id - 1]
    curated = _get_curated_dir()
    models = _get_models_dir()

    output_files = wizard_step["output_files"]
    completed_count = 0
    missing_files = []

    for filename in output_files:
        # Check in both curated/ and models/ directories
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

        cmd = [sys.executable, "-m", "src.pipeline.run_all", "--step", step_name]
        env = os.environ.copy()
        env["USE_MOCK"] = use_mock

        try:
            result = subprocess.run(
                cmd,
                cwd=str(project_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                st.error(f"❌ Errore durante esecuzione '{step_name}'")
                if result.stderr:
                    with st.expander("📋 Dettagli errore"):
                        st.code(result.stderr, language="text")
                progress_bar.empty()
                status_text.empty()
                return
        except subprocess.TimeoutExpired:
            st.error(f"⏱️ Timeout durante esecuzione '{step_name}' (>600s)")
            progress_bar.empty()
            status_text.empty()
            return
        except Exception as e:
            st.error(f"❌ Errore imprevisto: {e}")
            progress_bar.empty()
            status_text.empty()
            return

    progress_bar.progress(1.0, text="✅ Completato!")
    status_text.empty()
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

    # Header with status badge
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

    # Progress bar
    if not is_completed and progress_pct > 0:
        st.progress(progress_pct)
        st.caption(
            f"{step_completion['completed_count']}/{step_completion['total_count']} file completati. "
            f"Mancanti: {', '.join(step_completion['missing_files'])}"
        )

    # Main description
    st.markdown(step["description"])

    # Business value highlight
    st.info(step["business_value"])

    # SAP Pills
    if step["sap_pills"]:
        st.markdown("**📚 Glossario SAP:**")
        cols = st.columns(len(step["sap_pills"]))
        for i, pill_id in enumerate(step["sap_pills"]):
            with cols[i]:
                _render_sap_pill(pill_id)

    # Technical details (collapsed by default)
    with st.expander("🔧 Dettagli Tecnici (per esperti)"):
        st.markdown(step["technical_details"])
        st.markdown(f"**Pipeline steps:** `{', '.join(step['pipeline_steps'])}`")
        st.markdown(f"**Output files:** `{', '.join(step['output_files'])}`")
        st.markdown(f"**⏱️ Tempo stimato:** {step['estimated_time']}")

    # Action buttons
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
                st.session_state["selected_page"] = step["link_page"]
                st.rerun()

    st.divider()


# ================================================================================================
# Main Rendering Function
# ================================================================================================


def render_overview_wizard_page() -> None:
    """Render Prompt 9 Overview & Wizard page - HDBSCAN→Scoring flow in 4 business-friendly steps."""

    # Hero section
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

    # What this app does
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

    # Wizard section - Prompt 9: Business-friendly flow explanation
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

    # Check completion status for all steps
    step_statuses = [_check_step_completion(i + 1) for i in range(4)]

    # Overall progress
    total_progress = sum(s["progress"] for s in step_statuses) / 4
    st.progress(total_progress, text=f"Progresso complessivo: {int(total_progress * 100)}%")

    st.markdown("---")

    # Render each wizard step
    for i, step in enumerate(WIZARD_STEPS):
        with st.container():
            _render_wizard_step_card(step, step_statuses[i])

    # Quick start section
    st.markdown("## 🚀 Quick Start")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🔹 Esegui Step 1", use_container_width=True, type="primary"):
            _run_wizard_step(1)
            st.rerun()

    with col2:
        if st.button("🔹 Esegui Step 2", use_container_width=True, disabled=not step_statuses[0]["completed"]):
            _run_wizard_step(2)
            st.rerun()

    with col3:
        if st.button("🔹 Esegui Step 3", use_container_width=True, disabled=not step_statuses[1]["completed"]):
            _run_wizard_step(3)
            st.rerun()

    with col4:
        if st.button("🔹 Esegui Step 4", use_container_width=True, disabled=not step_statuses[2]["completed"]):
            _run_wizard_step(4)
            st.rerun()

    # Help section - Prompt 9: Updated FAQ for HDBSCAN→Scoring flow
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

    # Footer - Prompt 9: Deep-link CTAs
    st.markdown("---")
    st.caption(
        "💡 **Quick Start:** Attiva 'Mock Mode' nella sidebar → clicca '📦 Crea dataset demo' → esegui Step 1-4 → "
        "esplora risultati in **Clusters** (segmenti) e **Pattern Card** (ranking + coerenza)."
    )
    
    # Deep-link shortcuts
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
