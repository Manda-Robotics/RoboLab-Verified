"""Dashboard event severity routes on whole name tokens, then the code range (P02)."""
import importlib.util
from pathlib import Path


def _load_severity():
    # dashboard.app pulls in h5py/fastapi; lift just the function out of the source.
    src = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
    code = src.read_text()
    start = code.index("def _event_severity(")
    end = code.index("\ndef ", start + 1)
    ns = {}
    exec(code[start:end], ns)
    return ns["_event_severity"]


def test_severity_tokens_and_code_range():
    s = _load_severity()
    assert s("WHITE_MUG_IN_BIN_SUCCESS", 120) == "success"
    assert s("WHITE_OBJECT_MOVED", None) == "neutral"        # 'HIT' inside WHITE is not a hit
    assert s("GRIPPER_HIT_TABLE", 255) == "failure"
    assert s("TARGET_OBJECT_DROPPED", None) == "failure"
    assert s("WRONG_OBJECT_DETACHED", 257) == "failure"
    assert s("OBJECT_BUMPED", 258) == "failure"              # by code range
    assert s("OK", 0) == "success"
