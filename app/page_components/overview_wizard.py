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

WIZARD_STEPS = [
    {
        "id": 1,
        "title": "📥 Carica & Organizza i Dati",
        "subtitle": "Importa i dati SAP e prepara il terreno",
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
            "- Legge file Excel/CSV con mapping colonne SAP (es. BSAD, BSID)\n"
            "- Normalizza formati data, importi, codici cliente\n"
            "- Genera `line_items.parquet` con 19 colonne standardizzate\n"
            "- Engine: Pandas + Pyarrow per performance"
        ),
        "pipeline_steps": ["ingest", "events"],
        "output_files": ["line_items.parquet", "events.parquet"],
        "estimated_time": "2-5 minuti",
        "link_page": "Data Explorer",
    },
    {
        "id": 2,
        "title": "🔍 Analizza i Pattern di Pagamento",
        "subtitle": "Scopri come e quando i tuoi clienti pagano",
        "description": (
            "**Cosa succede:** Studiamo lo storico dei pagamenti di ogni cliente per capire i loro comportamenti ricorrenti. "
            "Ad esempio: 'Il cliente X paga sempre a fine mese' oppure 'Il cliente Y paga settimanalmente'.\n\n"
            "**Cosa ottieni:** Una 'carta d'identità' di ogni cliente con le sue abitudini di pagamento, "
            "i ritardi tipici, la regolarità, e i pattern più frequenti."
        ),
        "business_value": (
            "💡 **Valore Business:** Conoscere i pattern ti permette di prevedere quando arriveranno i pagamenti, "
            "ottimizzare la gestione del cash flow, e identificare clienti a rischio. "
            "È come avere un radar per i pagamenti futuri."
        ),
        "sap_pills": ["evento_pagamento", "data_registrazione"],
        "technical_details": (
            "**Dettagli Tecnici:**\n"
            "- Feature engineering: età fattura, cadenza pagamenti, importo medio\n"
            "- Pattern detectors: settimanale, mensile, fine mese, pagamento totale\n"
            "- Genera `train_features.parquet`, `client_pattern_scores.parquet`\n"
            "- Algoritmi: time-based aggregations, percentile ranking"
        ),
        "pipeline_steps": ["labels", "features", "patterns", "rules", "table"],
        "output_files": ["train_features.parquet", "client_pattern_scores.parquet", "pattern_table.parquet"],
        "estimated_time": "5-10 minuti",
        "link_page": "Pattern Card",
    },
    {
        "id": 3,
        "title": "🎯 Costruisci il Modello Predittivo",
        "subtitle": "Allena il sistema a prevedere i pagamenti",
        "description": (
            "**Cosa succede:** Insegniamo al sistema a imparare dai comportamenti passati dei clienti. "
            "Il sistema studia migliaia di esempi di 'fattura pagata' vs 'fattura non pagata' e impara a riconoscere i segnali.\n\n"
            "**Cosa ottieni:** Un modello che, data una nuova fattura aperta, ti dice: "
            "'Questa verrà probabilmente pagata entro X giorni con probabilità Y%'."
        ),
        "business_value": (
            "💡 **Valore Business:** Predizioni accurate ti permettono di pianificare liquidità, "
            "decidere quali clienti sollecitare, e allocare risorse del team crediti in modo efficiente. "
            "È come avere un esperto che conosce a memoria lo storico di ogni cliente."
        ),
        "sap_pills": ["partite_aperte", "pareggio"],
        "technical_details": (
            "**Dettagli Tecnici:**\n"
            "- Algoritmi ML: CatBoost + AutoGluon (ensemble)\n"
            "- Training: split temporale, no data leakage\n"
            "- Validazione: pre-training checks, calibrazione isotonica\n"
            "- Output: `champion.json` con metriche MAE, accuracy, F1"
        ),
        "pipeline_steps": ["validate", "train"],
        "output_files": ["champion.json", "catboost.cbm", "calibrator.pkl"],
        "estimated_time": "10-30 minuti",
        "link_page": "KPI Dashboard",
    },
    {
        "id": 4,
        "title": "📊 Raggruppa & Valuta i Clienti",
        "subtitle": "Identifica gruppi simili e calcola KPI",
        "description": (
            "**Cosa succede:** Raggruppiamo i clienti in famiglie con comportamenti simili (es. 'pagatori puntuali', "
            "'pagatori a fine mese', 'ritardatari cronici'). Poi calcoliamo indicatori di performance per ogni cliente.\n\n"
            "**Cosa ottieni:** Una mappa visuale dei tuoi clienti organizzati per somiglianza, "
            "con KPI chiari (affidabilità, regolarità, esposizione rischio) per ogni gruppo."
        ),
        "business_value": (
            "💡 **Valore Business:** Clustering e KPI ti aiutano a segmentare la base clienti, "
            "personalizzare strategie di gestione crediti (es. solleciti automatici per gruppo X), "
            "e identificare opportunità di miglioramento. È come avere una dashboard executive per i pagamenti."
        ),
        "sap_pills": ["evento_pagamento"],
        "technical_details": (
            "**Dettagli Tecnici:**\n"
            "- Embeddings: rappresentazione vettoriale clienti (feature → vettore 64D)\n"
            "- Clustering: HDBSCAN (density-based, auto-detect numero cluster)\n"
            "- Dimensionality reduction: UMAP per visualizzazione 2D\n"
            "- KPI: ranking score, cluster profile, calibration diagnostics"
        ),
        "pipeline_steps": ["embeddings", "projection", "hdbscan", "clusters", "kpi", "calibration", "policy"],
        "output_files": [
            "client_embeddings.parquet",
            "client_clusters.parquet",
            "cluster_summary.parquet",
            "kpi_ranking.parquet",
        ],
        "estimated_time": "5-15 minuti",
        "link_page": "Clusters",
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
    """Render PROMPT 4 Overview & Wizard page with CEO-friendly language."""

    # Hero section
    st.title("🏠 Benvenuto nell'Analisi dei Pattern di Pagamento")
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 10px; color: white; margin-bottom: 2rem;'>
            <h2 style='margin: 0; color: white;'>Trasforma i tuoi dati SAP in insight azionabili</h2>
            <p style='font-size: 1.1rem; margin: 0.5rem 0 0 0; color: #f0f0f0;'>
                Capisci come e quando i tuoi clienti pagano. Prevedi i flussi di cassa. Ottimizza la gestione crediti.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # What this app does
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        st.markdown("### 🎯 Cosa Fa Questa Applicazione?")
        st.markdown(
            """
            Questa applicazione analizza lo **storico dei pagamenti** dei tuoi clienti estratto da SAP 
            per rispondere a domande cruciali:
            
            - 📅 **Quando** arriveranno i prossimi pagamenti?
            - 💰 **Quanto** posso aspettarmi di incassare questa settimana/mese?
            - ⚠️ **Chi** sono i clienti a rischio ritardo?
            - 📊 **Quali** pattern di pagamento hanno i miei top clienti?
            - 🎯 **Come** posso ottimizzare la gestione del credito?
            
            **Senza** bisogno di conoscenze tecniche o statistiche avanzate.
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
            
            **Colonne richieste:**
            - Cliente, Data documento, Importo
            - Data pareggio, Doc. pareggio
            - Divisa, Tipo documento
            
            _(Vedi mapping completo nella pagina Data Explorer)_
            """
        )

    st.divider()

    # Wizard section
    st.markdown("## 🧙‍♂️ Wizard: 4 Passi per l'Insight Completo")
    st.markdown(
        """
        Segui questi 4 passi nell'ordine per analizzare i tuoi dati. Ogni step richiede il completamento 
        del precedente. Clicca **"▶️ Esegui Step X"** per avviare l'elaborazione.
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

    # Help section
    with st.expander("❓ Domande Frequenti (FAQ)"):
        st.markdown(
            """
            **Q: Quanto tempo richiede l'analisi completa?**  
            A: Circa 25-60 minuti totali, a seconda del volume di dati. Puoi seguire il progresso in tempo reale.
            
            **Q: I miei dati sono al sicuro?**  
            A: Sì, tutti i dati rimangono sul tuo computer/server. Niente viene inviato online.
            
            **Q: Serve conoscere il machine learning?**  
            A: No, l'interfaccia è progettata per utenti business. I dettagli tecnici sono nascosti nei tooltip.
            
            **Q: Cosa succede se interrompo l'elaborazione?**  
            A: Ogni step è idempotente: puoi ri-eseguirlo senza perdere progressi precedenti.
            
            **Q: Posso testare con dati di esempio?**  
            A: Sì, attiva "Mock Mode" nella sidebar per usare un dataset di test pre-caricato.
            
            **Q: Come interpreto i risultati?**  
            A: Ogni pagina (Pattern Card, KPI Dashboard, Clusters) ha guide integrate e tooltip esplicativi.
            """
        )

    # Footer
    st.markdown("---")
    st.caption(
        "💡 **Suggerimento:** Inizia con 'Mock Mode' attivo per esplorare l'app con dati di esempio, "
        "poi carica i tuoi dati SAP reali."
    )
