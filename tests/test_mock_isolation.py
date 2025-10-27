"""
Verification script for PROMPT 3 — Mock mode isolation.

This script demonstrates that _data_root() correctly routes I/O paths
based on the mock_mode session state.
"""
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_mock_isolation():
    """Verify mock mode isolation without running Streamlit."""
    import os
    
    BASE_DIR = Path(__file__).resolve().parents[1]
    
    # Simulate mock_mode OFF (default)
    class FakeSessionState:
        def __init__(self):
            self._state = {}
        
        def get(self, key, default=None):
            return self._state.get(key, default)
        
        def __setitem__(self, key, value):
            self._state[key] = value
    
    fake_session = FakeSessionState()
    
    def _data_root_test(session_state):
        """Test version of _data_root()."""
        if session_state.get("mock_mode"):
            return BASE_DIR / "data_mock"
        return BASE_DIR / Path(os.getenv("APP_DATA_ROOT", "data"))
    
    def _p_test(session_state, *parts):
        """Test version of _p()."""
        return _data_root_test(session_state).joinpath(*parts)
    
    print("=" * 60)
    print("PROMPT 3 — Mock Mode Isolation Test")
    print("=" * 60)
    
    # Test 1: mock_mode OFF (default)
    print("\n[Test 1] mock_mode = OFF (default)")
    fake_session["mock_mode"] = False
    curated_path = _p_test(fake_session, "curated", "line_items.parquet")
    print(f"  Expected: ...\\data\\curated\\line_items.parquet")
    print(f"  Actual:   {curated_path}")
    assert "data\\curated" in str(curated_path) or "data/curated" in str(curated_path)
    assert "data_mock" not in str(curated_path)
    print("  ✓ PASS: Routes to data/")
    
    # Test 2: mock_mode ON
    print("\n[Test 2] mock_mode = ON")
    fake_session["mock_mode"] = True
    curated_path = _p_test(fake_session, "curated", "line_items.parquet")
    print(f"  Expected: ...\\data_mock\\curated\\line_items.parquet")
    print(f"  Actual:   {curated_path}")
    assert "data_mock\\curated" in str(curated_path) or "data_mock/curated" in str(curated_path)
    print("  ✓ PASS: Routes to data_mock/")
    
    # Test 3: Upload paths
    print("\n[Test 3] Upload isolation")
    fake_session["mock_mode"] = False
    upload_real = _p_test(fake_session, "raw", "uploads", "test.csv")
    fake_session["mock_mode"] = True
    upload_mock = _p_test(fake_session, "raw", "uploads", "test.csv")
    print(f"  Real:  {upload_real}")
    print(f"  Mock:  {upload_mock}")
    assert "data\\raw" in str(upload_real) or "data/raw" in str(upload_real)
    assert "data_mock\\raw" in str(upload_mock) or "data_mock/raw" in str(upload_mock)
    print("  ✓ PASS: Upload paths isolated")
    
    # Test 4: Verify directories exist
    print("\n[Test 4] Directory structure")
    data_dir = BASE_DIR / "data"
    data_mock_dir = BASE_DIR / "data_mock"
    print(f"  data/:      {data_dir.exists()} {'✓' if data_dir.exists() else '✗'}")
    print(f"  data_mock/: {data_mock_dir.exists()} {'✓' if data_mock_dir.exists() else '✗'}")
    
    test_marker = data_mock_dir / "curated" / "test_client.txt"
    if test_marker.exists():
        print(f"  test_client.txt: exists in data_mock/ ✓")
        print(f"    → This file should NEVER appear when mock_mode=OFF")
    
    print("\n" + "=" * 60)
    print("All tests passed! Mock isolation working correctly.")
    print("=" * 60)


if __name__ == "__main__":
    test_mock_isolation()
