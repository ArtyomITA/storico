""" Overview & Pipeline page with step buttons and log viewer."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List

import pandas as pd
import streamlit as st

LOGGER = logging.getLogger(__name__)

# Step definitions matching run_all.py step_registry
PIPELINE_STEPS = [
    {
        "id": "ingest",
        "label": "1. Ingest SAP Data",
        "description": "Load and normalize SAP raw data → line_items.parquet",
        "group": "data",
    },
    {
        "id": "events",
        "label": "2. Build Events",
        "description": "Construct payment events from line items → events.parquet",
        "group": "data",
    },
    {
        "id": "labels",
        "label": "3. Make Labels",
        "description": "Create training labels from events → train_pointwise.parquet",
        "group": "data",
    },
    {
        "id": "features",
        "label": "4. Make Features",
        "description": "Engineer features for training → train_features.parquet",
        "group": "data",
    },
    {
        "id": "validate",
        "label": "5. Pre-Training Validation",
        "description": "Check data quality before model training",
        "group": "training",
    },
    {
        "id": "train",
        "label": "6. Train Model",
        "description": "Train CatBoost/AutoGluon → champion.json",
        "group": "training",
    },
    {
        "id": "patterns",
        "label": "7. Pattern Detectors",
        "description": "Compute client payment patterns → client_pattern_scores.parquet",
        "group": "analytics",
    },
    {
        "id": "rules",
        "label": "8. Rules Backtest",
        "description": "Backtest policy rules → client_rules_accuracy.parquet",
        "group": "analytics",
    },
    {
        "id": "table",
        "label": "9. Pattern Table",
        "description": "Assemble pattern summary → pattern_table.parquet",
        "group": "analytics",
    },
    {
        "id": "reports",
        "label": "10. Pattern Reports",
        "description": "Generate pattern card PNGs → reports/",
        "group": "analytics",
    },
    {
        "id": "embeddings",
        "label": "11. Client Embeddings",
        "description": "Create client embeddings → client_embeddings.parquet",
        "group": "optional",
    },
    {
        "id": "projection",
        "label": "12. Embedding Projection",
        "description": "Compute 2D projection → client_embedding_2d.parquet",
        "group": "optional",
    },
    {
        "id": "hdbscan",
        "label": "13. HDBSCAN Clustering",
        "description": "Cluster clients → client_clusters.parquet",
        "group": "optional",
    },
    {
        "id": "clusters",
        "label": "14. Cluster Profiles",
        "description": "Summarize clusters → cluster_summary.parquet",
        "group": "optional",
    },
    {
        "id": "kpi",
        "label": "15. KPI Ranking",
        "description": "Compute KPI metrics → kpi_ranking.parquet",
        "group": "optional",
    },
    {
        "id": "calibration",
        "label": "16. Calibration Diagnostics",
        "description": "Compute calibration metrics → calibration_summary.parquet",
        "group": "optional",
    },
    {
        "id": "policy",
        "label": "17. Policy Metrics",
        "description": "Compute event policy metrics → event_policy_metrics.parquet",
        "group": "optional",
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


def _get_reports_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "reports"


def _check_step_status(step_id: str) -> Dict[str, Any]:
    """Check if a step has been completed by verifying output files exist."""
    curated = _get_curated_dir()
    models = _get_models_dir()
    reports = _get_reports_dir()

    output_map = {
        "ingest": curated / "line_items.parquet",
        "events": curated / "events.parquet",
        "labels": curated / "train_pointwise.parquet",
        "features": curated / "train_features.parquet",
        "validate": None,  # No output file
        "train": models / "champion.json",
        "patterns": curated / "client_pattern_scores.parquet",
        "rules": curated / "client_rules_accuracy.parquet",
        "table": curated / "pattern_table.parquet",
        "reports": reports,
        "embeddings": curated / "client_embeddings.parquet",
        "projection": curated / "client_embedding_2d.parquet",
        "hdbscan": curated / "client_clusters.parquet",
        "clusters": curated / "cluster_summary.parquet",
        "kpi": curated / "kpi_ranking.parquet",
        "calibration": curated / "calibration_summary.parquet",
        "policy": curated / "event_policy_metrics.parquet",
    }

    output_path = output_map.get(step_id)
    if output_path is None:
        return {"status": "unknown", "path": None}

    if output_path.is_dir():
        # Check if directory has files (e.g., reports/)
        if output_path.exists() and any(output_path.iterdir()):
            return {"status": "completed", "path": str(output_path)}
        return {"status": "not_started", "path": str(output_path)}

    if output_path.exists():
        return {"status": "completed", "path": str(output_path)}
    return {"status": "not_started", "path": str(output_path)}


def _load_last_log_lines(max_lines: int = 20) -> List[Dict[str, Any]]:
    """Load last N lines from JSON log file."""
    project_root = Path(__file__).resolve().parents[2]
    log_dir = project_root / "logs"
    if not log_dir.exists():
        return []

    log_files = sorted(log_dir.glob("run_*.jsonl"), reverse=True)
    if not log_files:
        return []

    log_path = log_files[0]
    lines: List[Dict[str, Any]] = []
    try:
        with log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return []

    return lines[-max_lines:]


def _run_step(step_id: str, model: str = "catboost") -> None:
    """Execute a single pipeline step by invoking src.pipeline.run_all."""
    import subprocess
    import sys

    project_root = Path(__file__).resolve().parents[2]
    use_mock = "true" if st.session_state.get("mock_mode") else "false"

    cmd = [
        sys.executable,
        "-m",
        "src.pipeline.run_all",
        "--step",
        step_id,
    ]
    if model:
        cmd.extend(["--model", model])

    env = os.environ.copy()
    env["USE_MOCK"] = use_mock

    with st.spinner(f"Executing step '{step_id}'..."):
        try:
            result = subprocess.run(
                cmd,
                cwd=str(project_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode == 0:
                st.success(f"Step '{step_id}' completed successfully")
            else:
                st.error(f"Step '{step_id}' failed with return code {result.returncode}")
                if result.stderr:
                    st.code(result.stderr, language="text")
        except subprocess.TimeoutExpired:
            st.error(f"Step '{step_id}' timed out after 600 seconds")
        except Exception as e:
            st.error(f"Error executing step '{step_id}': {e}")


def render_pipeline_overview_page() -> None:
    """Render PROMPT 3 Overview & Pipeline page with step buttons and log viewer."""
    st.title("🚀 Pipeline Overview & Control")
    st.markdown("**PROMPT 3**: Orchestratore unico con step idempotenti e UI pulsanti")

    # Pipeline status dashboard
    st.header("📊 Pipeline Status")
    
    # Group steps by category
    groups = {
        "data": "Data Preparation",
        "training": "Model Training",
        "analytics": "Analytics",
        "optional": "Optional Analytics",
    }

    for group_id, group_label in groups.items():
        with st.expander(f"**{group_label}**", expanded=(group_id in ["data", "training"])):
            group_steps = [s for s in PIPELINE_STEPS if s["group"] == group_id]
            cols = st.columns([3, 1, 1])
            cols[0].markdown("**Step**")
            cols[1].markdown("**Status**")
            cols[2].markdown("**Action**")

            for step in group_steps:
                status_info = _check_step_status(step["id"])
                cols = st.columns([3, 1, 1])
                
                # Step label + description
                cols[0].markdown(
                    f"**{step['label']}**  \n"
                    f"<small style='color: #666;'>{step['description']}</small>",
                    unsafe_allow_html=True,
                )
                
                # Status badge
                status = status_info["status"]
                if status == "completed":
                    cols[1].success("✅ Done")
                elif status == "not_started":
                    cols[1].warning("⏸️ Pending")
                else:
                    cols[1].info("❓ Unknown")
                
                # Run button
                if cols[2].button(f"▶ Run", key=f"run_{step['id']}"):
                    _run_step(step["id"])
                    st.rerun()

    st.divider()

    # Quick action buttons
    st.header("⚡ Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔹 Ingest → Features (1-4)", use_container_width=True):
            for step_id in ["ingest", "events", "labels", "features"]:
                _run_step(step_id)
            st.rerun()
    
    with col2:
        if st.button("🔹 Validate + Train (5-6)", use_container_width=True):
            for step_id in ["validate", "train"]:
                _run_step(step_id)
            st.rerun()
    
    with col3:
        if st.button("🔹 Embeddings → Clusters (11-14)", use_container_width=True):
            for step_id in ["embeddings", "projection", "hdbscan", "clusters"]:
                _run_step(step_id)
            st.rerun()

    st.divider()

    # Log viewer
    st.header("📜 Recent Log Entries")
    log_lines = _load_last_log_lines(max_lines=20)
    if log_lines:
        # Display as a table
        log_df = pd.DataFrame(log_lines)
        if "timestamp" in log_df.columns:
            log_df["timestamp"] = pd.to_datetime(log_df["timestamp"])
        display_cols = [c for c in ["timestamp", "level", "message", "module"] if c in log_df.columns]
        st.dataframe(
            log_df[display_cols] if display_cols else log_df,
            use_container_width=True,
            height=400,
        )
    else:
        st.info("No log entries found. Run a step to see logs.")

    # Developer info
    with st.expander("ℹ️ Developer Info"):
        st.markdown(
            """
            **CLI Equivalent:**
            ```bash
            # Run single step
            USE_MOCK=true python -m src.pipeline.run_all --step ingest
            
            # Run full pipeline
            USE_MOCK=true python -m src.pipeline.run_all --make-kpi --make-calibration
            ```
            
            **Acceptance Criteria (PROMPT 3):**
            - ✅ 17 idempotent steps in `src/pipeline/run_all.py`
            - ✅ CLI with `--step <name>` flag
            - ✅ UI buttons for each step
            - ⏳ 3-click workflow: Ingest → Train → Cluster
            """
        )
