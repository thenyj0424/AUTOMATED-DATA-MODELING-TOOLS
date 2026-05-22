import os
import sys


def pytest_configure():
    # Ensure project root (one level up from tests/) is on sys.path so
    # imports like `from ai_agent.auto_engine import ...` work when running
    # `pytest` from the `AI Agent` folder.
    tests_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(tests_dir, os.pardir))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
