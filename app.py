"""Root entry point for Streamlit Community Cloud and local execution.

Allows running `streamlit run app.py` directly from the repository root.
"""

import os
import sys
from pathlib import Path

# Add stock-dashboard package path to sys.path
REPO_ROOT = Path(__file__).resolve().parent
DASHBOARD_DIR = REPO_ROOT / "stock-dashboard"

if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

# Change directory context so internal relative paths function identically
os.chdir(str(DASHBOARD_DIR))

# Import main application runner
from app import main

if __name__ == "__main__":
    main()

