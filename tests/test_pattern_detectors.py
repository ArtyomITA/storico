"""Unit tests for pattern_detectors.py (Prompt 5).

Tests edge cases:
- Empty client_profiles
- Missing client_clusters
- No dominant policy (all scores near zero)
- Single-policy dominance (high confidence)
"""
import numpy as np
import pandas as pd
import pytest

from src.analytics.pattern_detectors import (
    build_client_patterns,
    compute_pattern_confidence,
    determine_dominant_policy,
    extract_periodicity_flags,
)


def test_determine_dominant_policy_clear_winner():
    """Test dominant policy detection with clear winner."""
    row = pd.Series({
        "fifo_score": 0.85,
        "lifo_score": 0.15,
        "same_period_score": 0.10,
        "largest_score": 0.05,
        "smallest_score": 0.02,
    })
    
    dominant = determine_dominant_policy(row)
    assert dominant == "fifo"


def test_determine_dominant_policy_no_clear_pattern():
    """Test dominant policy when all scores are low (irregolare fallback)."""
    row = pd.Series({
        "fifo_score": 0.05,
        "lifo_score": 0.04,
        "same_period_score": 0.03,
        "largest_score": 0.02,
        "smallest_score": 0.01,
    })
    
    dominant = determine_dominant_policy(row)
    assert dominant == "irregolare"


def test_determine_dominant_policy_missing_scores():
    """Test dominant policy with missing score columns."""
    row = pd.Series({"other_column": 1.0})
    
    dominant = determine_dominant_policy(row)
    assert dominant == "irregolare"


def test_compute_pattern_confidence_high():
    """Test pattern confidence with high dominant score and stability."""
    row = pd.Series({
        "fifo_score": 0.9,
        "lifo_score": 0.1,
        "stability_index": 0.8,
    })
    
    confidence = compute_pattern_confidence(row, "fifo")
    assert 0.5 <= confidence <= 1.0  # Should be high


def test_compute_pattern_confidence_low():
    """Test pattern confidence with low scores."""
    row = pd.Series({
        "fifo_score": 0.2,
        "lifo_score": 0.18,
        "stability_index": 0.3,
    })
    
    confidence = compute_pattern_confidence(row, "fifo")
    assert 0.0 <= confidence <= 0.5  # Should be low


def test_extract_periodicity_flags():
    """Test periodicity flag extraction with threshold."""
    row = pd.Series({
        "periodicity_weekly": 0.7,  # Above threshold
        "periodicity_biweekly": 0.5,  # Below threshold
        "periodicity_monthly": 0.65,  # Above threshold
    })
    
    flags = extract_periodicity_flags(row)
    assert flags["periodicity_weekly"] is True
    assert flags["periodicity_biweekly"] is False
    assert flags["periodicity_monthly"] is True


def test_extract_periodicity_flags_missing():
    """Test periodicity flags when columns are missing."""
    row = pd.Series({"other_column": 1.0})
    
    flags = extract_periodicity_flags(row)
    assert flags["periodicity_weekly"] is False
    assert flags["periodicity_biweekly"] is False
    assert flags["periodicity_monthly"] is False


def test_build_client_patterns_empty_profiles():
    """Test build_client_patterns with empty client_profiles."""
    pattern_scores = pd.DataFrame({
        "cliente": ["C001", "C002"],
        "fifo_score": [0.8, 0.3],
        "lifo_score": [0.2, 0.7],
        "periodicity_weekly": [0.1, 0.7],
        "periodicity_monthly": [0.8, 0.2],
        "k_mode": [2.0, 1.0],
        "stability_index": [0.9, 0.5],
    })
    
    clusters = pd.DataFrame({
        "cliente": ["C001", "C002"],
        "cluster_label": [0, 1],
        "membership_prob": [0.95, 0.88],
    })
    
    profiles = pd.DataFrame()  # Empty
    
    patterns = build_client_patterns(profiles, clusters, pattern_scores)
    
    assert len(patterns) == 2
    assert "dominant_policy" in patterns.columns
    assert "pattern_confidence" in patterns.columns
    assert "cluster_label" in patterns.columns
    assert patterns.loc[0, "dominant_policy"] in ["fifo", "irregolare"]
    assert 0.0 <= patterns.loc[0, "pattern_confidence"] <= 1.0


def test_build_client_patterns_no_clusters():
    """Test build_client_patterns without cluster assignments."""
    pattern_scores = pd.DataFrame({
        "cliente": ["C001"],
        "fifo_score": [0.9],
        "k_mode": [2.0],
        "periodicity_monthly": [0.8],
        "stability_index": [0.85],
    })
    
    profiles = pd.DataFrame({
        "cliente": ["C001"],
        "month_coverage_mean": [0.75],
    })
    
    clusters = pd.DataFrame()  # No clusters
    
    patterns = build_client_patterns(profiles, clusters, pattern_scores)
    
    assert len(patterns) == 1
    assert patterns.loc[0, "cluster_label"] == -1  # Default noise
    assert patterns.loc[0, "membership_prob"] == 0.0
    assert "month_coverage_mean" in patterns.columns


def test_build_client_patterns_full_integration():
    """Test build_client_patterns with all inputs."""
    pattern_scores = pd.DataFrame({
        "cliente": ["C001", "C002", "C003"],
        "fifo_score": [0.85, 0.15, 0.50],
        "lifo_score": [0.10, 0.80, 0.45],
        "periodicity_weekly": [0.1, 0.75, 0.3],
        "periodicity_monthly": [0.70, 0.2, 0.65],
        "k_mode": [2.0, 1.0, 3.0],
        "stability_index": [0.90, 0.75, 0.60],
    })
    
    clusters = pd.DataFrame({
        "cliente": ["C001", "C002", "C003"],
        "cluster_label": [0, 1, 0],
        "membership_prob": [0.95, 0.88, 0.72],
    })
    
    profiles = pd.DataFrame({
        "cliente": ["C001", "C002", "C003"],
        "month_coverage_mean": [0.80, 0.60, 0.70],
        "month_coverage_p95": [0.95, 0.85, 0.90],
        "acf_lag7": [0.5, 0.6, 0.4],
    })
    
    patterns = build_client_patterns(profiles, clusters, pattern_scores)
    
    # Validate structure
    assert len(patterns) == 3
    assert "dominant_policy" in patterns.columns
    assert "pattern_confidence" in patterns.columns
    assert "periodicity_weekly_flag" in patterns.columns
    assert "periodicity_monthly_flag" in patterns.columns
    assert "k_mode" in patterns.columns
    assert "month_coverage_mean" in patterns.columns
    assert "acf_lag7" in patterns.columns
    
    # Validate dominant policies
    assert patterns.loc[0, "dominant_policy"] == "fifo"  # C001: high fifo_score
    assert patterns.loc[1, "dominant_policy"] == "lifo"  # C002: high lifo_score
    
    # Validate confidence is in range
    assert all(0.0 <= conf <= 1.0 for conf in patterns["pattern_confidence"])
    
    # Validate periodicity flags
    assert patterns.loc[1, "periodicity_weekly_flag"] is True  # C002: 0.75 >= 0.6
    assert patterns.loc[0, "periodicity_monthly_flag"] is True  # C001: 0.70 >= 0.6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
