# Offline tests

Tests that run without Isaac Sim (`tests/conftest.py` boots Isaac for everything
under `tests/`, so anything that must stay GPU-free lives here).

    python -m pytest offline_tests/
