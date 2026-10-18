"""Configuration for Stock_app web application.

Connects to the external ML_stock repository without modifying it, while keeping
all Stock_app state (predictions, screening runs, screen hits) in an isolated
local SQLite database.
"""

from __future__ import annotations

import os
from pathlib import Path

# Base directory of Stock_app
APP_DIR: Path = Path(__file__).resolve().parent

# Path to the external ML_stock repository
ML_STOCK_DIR: Path = Path(
    os.environ.get("ML_STOCK_DIR", "/home/amilap2/code/ML_stock")
).resolve()

# ML_stock resource paths (accessed via ML_stock modules / data_loader.py)
ML_STOCK_DATA_DIR: Path = Path(
    os.environ.get("ML_STOCK_DATA_DIR", str(ML_STOCK_DIR / "data" / "raw"))
).resolve()

ML_STOCK_CONFIG_DIR: Path = Path(
    os.environ.get(
        "ML_STOCK_CONFIG_DIR", str(ML_STOCK_DIR / "configs" / "universes")
    )
).resolve()

ML_STOCK_CHECKPOINT: Path = Path(
    os.environ.get(
        "ML_STOCK_CHECKPOINT",
        str(ML_STOCK_DIR / "checkpoints" / "patchtst_best.pt"),
    )
).resolve()

# Stock_app internal database path
DATA_DIR: Path = APP_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH: Path = Path(
    os.environ.get("STOCK_APP_DB_PATH", str(DATA_DIR / "stock_app.db"))
).resolve()

# Web server configuration
HOST: str = os.environ.get("STOCK_APP_HOST", "127.0.0.1")
PORT: int = int(os.environ.get("STOCK_APP_PORT", "8050"))
