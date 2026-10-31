"""
Ward Pulse: Entrypoint proxy to frontend/app.py
"""
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from frontend.app import main

if __name__ == "__main__":
    main()
