"""Training Manager: gestisce stato ed esecuzione training CatBoost/AutoGluon/HDBSCAN.

Backend per UI Azioni con pulsanti Run & Stato per modelli e clustering.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

LOGGER = logging.getLogger(__name__)

# Status types
TrainingStatus = Literal["not_started", "running", "completed", "failed"]
ModelType = Literal["catboost", "autogluon", "pattern", "hdbscan"]


@dataclass
class TrainingRun:
    """Rappresenta un'esecuzione di training/processing."""
    
    model_type: ModelType
    status: TrainingStatus = "not_started"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    artifacts: List[Path] = field(default_factory=list)
    error_message: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON storage."""
        return {
            "model_type": self.model_type,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "artifacts": [str(p) for p in self.artifacts],
            "error_message": self.error_message,
            "parameters": self.parameters,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingRun":
        """Deserialize from dict."""
        return cls(
            model_type=data["model_type"],
            status=data["status"],
            started_at=datetime.fromisoformat(data["started_at"]) if data["started_at"] else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data["completed_at"] else None,
            duration_seconds=data.get("duration_seconds"),
            artifacts=[Path(p) for p in data.get("artifacts", [])],
            error_message=data.get("error_message"),
            parameters=data.get("parameters", {}),
        )


class TrainingManager:
    """Gestisce stato persistente e esecuzione di training/processing."""
    
    def __init__(self, state_file: Path = Path("data/training_state.json")):
        self.state_file = state_file
        self.runs: Dict[ModelType, TrainingRun] = {}
        self._load_state()
    
    def _load_state(self) -> None:
        """Carica stato salvato da JSON."""
        if not self.state_file.exists():
            LOGGER.info(f"Training state file not found, starting fresh: {self.state_file}")
            return
        
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for model_type, run_data in data.items():
                self.runs[model_type] = TrainingRun.from_dict(run_data)
            
            LOGGER.info(f"Loaded training state from {self.state_file}: {len(self.runs)} runs")
        except Exception as e:
            LOGGER.error(f"Failed to load training state: {e}", exc_info=True)
    
    def _save_state(self) -> None:
        """Salva stato corrente a JSON."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            data = {model_type: run.to_dict() for model_type, run in self.runs.items()}
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            LOGGER.debug(f"Saved training state to {self.state_file}")
        except Exception as e:
            LOGGER.error(f"Failed to save training state: {e}", exc_info=True)
    
    def get_run_status(self, model_type: ModelType) -> Optional[TrainingRun]:
        """Ottieni stato corrente di un training."""
        return self.runs.get(model_type)
    
    def start_training(
        self,
        model_type: ModelType,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> TrainingRun:
        """Inizia un nuovo training e salva stato."""
        run = TrainingRun(
            model_type=model_type,
            status="running",
            started_at=datetime.now(timezone.utc),
            parameters=parameters or {},
        )
        self.runs[model_type] = run
        self._save_state()
        LOGGER.info(f"Started training: {model_type} with params {parameters}")
        return run
    
    def complete_training(
        self,
        model_type: ModelType,
        artifacts: List[Path],
        error_message: Optional[str] = None,
    ) -> TrainingRun:
        """Completa un training con esito e artefatti."""
        if model_type not in self.runs:
            raise ValueError(f"No training run found for {model_type}")
        
        run = self.runs[model_type]
        run.completed_at = datetime.now(timezone.utc)
        
        if run.started_at:
            run.duration_seconds = (run.completed_at - run.started_at).total_seconds()
        
        if error_message:
            run.status = "failed"
            run.error_message = error_message
        else:
            run.status = "completed"
            run.artifacts = artifacts
        
        self._save_state()
        LOGGER.info(f"Completed training: {model_type} status={run.status} duration={run.duration_seconds}s")
        return run
    
    def check_artifacts_exist(self, model_type: ModelType) -> bool:
        """Verifica se esistono artefatti per un modello."""
        run = self.get_run_status(model_type)
        if not run or not run.artifacts:
            return False
        
        return all(p.exists() for p in run.artifacts)
    
    def execute_training_async(
        self,
        model_type: ModelType,
        command: List[str],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> subprocess.Popen:
        """Esegue training in subprocess asincrono.
        
        Args:
            model_type: Tipo di modello da trainare
            command: Comando da eseguire (es. ["python", "-m", "src.models.train", "--model", "catboost"])
            parameters: Parametri opzionali (es. {"use_gpu": true, "k_values": [1, 2, 3]})
        
        Returns:
            subprocess.Popen: Processo asincrono
        """
        self.start_training(model_type, parameters)
        
        try:
            # Start process asynchronously
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            LOGGER.info(f"Started async training process PID={process.pid}: {' '.join(command)}")
            return process
        
        except Exception as e:
            error_msg = f"Failed to start training process: {e}"
            self.complete_training(model_type, [], error_message=error_msg)
            raise
    
    def wait_for_completion(
        self,
        model_type: ModelType,
        process: subprocess.Popen,
        expected_artifacts: List[Path],
        timeout: Optional[int] = None,
    ) -> TrainingRun:
        """Attende completamento processo e aggiorna stato.
        
        Args:
            model_type: Tipo di modello
            process: Processo subprocess da attendere
            expected_artifacts: Path degli artefatti attesi
            timeout: Timeout in secondi (None = nessun timeout)
        
        Returns:
            TrainingRun: Stato finale del training
        """
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            returncode = process.returncode
            
            if returncode == 0:
                # Success: check artifacts
                existing_artifacts = [p for p in expected_artifacts if p.exists()]
                if existing_artifacts:
                    return self.complete_training(model_type, existing_artifacts)
                else:
                    error_msg = f"Training succeeded but no artifacts found: {expected_artifacts}"
                    return self.complete_training(model_type, [], error_message=error_msg)
            else:
                # Failure
                error_msg = f"Training failed with exit code {returncode}\n{stderr}"
                return self.complete_training(model_type, [], error_message=error_msg)
        
        except subprocess.TimeoutExpired:
            process.kill()
            error_msg = f"Training timeout after {timeout}s"
            return self.complete_training(model_type, [], error_message=error_msg)
        
        except Exception as e:
            error_msg = f"Training exception: {e}"
            return self.complete_training(model_type, [], error_message=error_msg)


# ============================================================================
# Helper functions for specific training types
# ============================================================================

def get_catboost_artifacts(data_dir: Path = Path("data")) -> List[Path]:
    """Ottieni path artefatti CatBoost attesi."""
    models_dir = Path("models")
    curated_dir = data_dir / "curated"
    
    return [
        models_dir / "catboost_champion.cbm",
        models_dir / "catboost_champion_metadata.json",
        curated_dir / "ranking_kpi.parquet",
        curated_dir / "calibration_kpi.parquet",
    ]


def get_autogluon_artifacts(data_dir: Path = Path("data")) -> List[Path]:
    """Ottieni path artefatti AutoGluon attesi."""
    models_dir = Path("models")
    curated_dir = data_dir / "curated"
    
    return [
        models_dir / "autogluon",  # Directory
        models_dir / "autogluon_champion_metadata.json",
        curated_dir / "ranking_kpi.parquet",
        curated_dir / "calibration_kpi.parquet",
    ]


def get_pattern_artifacts(data_dir: Path = Path("data")) -> List[Path]:
    """Ottieni path artefatti Pattern/Policy attesi."""
    curated_dir = data_dir / "curated"
    
    return [
        curated_dir / "per_client_patterns.parquet",
        curated_dir / "event_policy_confusion.parquet",
        curated_dir / "event_policy_metrics.parquet",
        curated_dir / "per_event_policy_scores.parquet",
    ]


def get_hdbscan_artifacts(data_dir: Path = Path("data")) -> List[Path]:
    """Ottieni path artefatti HDBSCAN attesi."""
    curated_dir = data_dir / "curated"
    
    return [
        curated_dir / "hdbscan_clusters.parquet",
        curated_dir / "cluster_profiles.parquet",
        curated_dir / "hdbscan_metadata.json",
    ]


def build_training_command(
    model_type: ModelType,
    use_gpu: bool = False,
    k_values: Optional[List[int]] = None,
    min_cluster_size: int = 30,
    min_samples: int = 15,
) -> List[str]:
    """Costruisce comando per eseguire training specifico.
    
    Args:
        model_type: Tipo di training da eseguire
        use_gpu: Usa GPU per training (CatBoost/AutoGluon)
        k_values: Valori k per metriche ranking (default: [1, 2, 3, 5])
        min_cluster_size: Parametro HDBSCAN min_cluster_size
        min_samples: Parametro HDBSCAN min_samples
    
    Returns:
        List[str]: Comando da eseguire
    """
    if model_type == "catboost":
        cmd = [sys.executable, "-m", "src.models.train", "--model", "catboost"]
        if use_gpu:
            cmd.append("--gpu")
        if k_values:
            for k in k_values:
                cmd.extend(["--k", str(k)])
        return cmd
    
    elif model_type == "autogluon":
        cmd = [sys.executable, "-m", "src.models.train", "--model", "autogluon"]
        if use_gpu:
            cmd.append("--gpu")
        if k_values:
            for k in k_values:
                cmd.extend(["--k", str(k)])
        return cmd
    
    elif model_type == "pattern":
        # Run pattern detection + policy analysis steps
        cmd = [
            sys.executable,
            "-m",
            "src.pipeline.run_all",
            "--step", "pattern_table",
            "--step", "pattern_detectors",
            "--step", "rules_backtest",
        ]
        return cmd
    
    elif model_type == "hdbscan":
        cmd = [
            sys.executable,
            "-m",
            "src.analytics.run_hdbscan",
            "--min-cluster-size", str(min_cluster_size),
            "--min-samples", str(min_samples),
        ]
        return cmd
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
