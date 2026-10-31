"""
Ward Pulse: Frontend entrypoint proxy
Delegates execution to the civic dashboard main entrypoint.
"""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import main

if __name__ == "__main__":
    main()
