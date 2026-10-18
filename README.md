# Stock_app — ML_stock Web Visualization & Prediction Terminal

`Stock_app` is a standalone web application for viewing and running `ML_stock` PatchTST quantile forecasts, market screening calls, probability cones, and realized price trajectories.

## Architecture & Separation of Concerns

- **External `ML_stock` Integration (`ml_bridge.py`)**:
  - Hooks directly into `/home/amilap2/code/ML_stock` without modifying any files in `ML_stock`.
  - Delegates **all** Parquet data reading, metadata inspection, and cache updating strictly to `ML_stock/src/data_loader.py` (`YahooParquetDataLoader`).
  - Uses `MarketScreener`, `PatchTST`, and `MarketQueryPatchTST` from `ML_stock` for model inference and screening.
- **Internal SQLite Database (`database.py`)**:
  - Stores all specific stock predictions, screening calls, and screen-hit probability cones in `/home/amilap2/code/Stock_app/data/stock_app.db`.
- **Zero-Extra-Dependency Web Server (`app.py`)**:
  - Uses Python's standard `ThreadingHTTPServer` so it runs directly inside the existing `ML_stock` Python environment without requiring external web framework installations.

## Running the Web App

Activate the same Python environment used for `ML_stock` and run:
```bash
source /home/amilap2/code/ML_stock/.venv/bin/activate
```

```bash
python /home/amilap2/code/Stock_app/app.py --host 127.0.0.1 --port 8050
```

Then open **http://127.0.0.1:8050** in your browser.

## Key Features

1. **Main Dashboard**:
   - **Stocks with Predictions & Screen Hits**: Consolidated table of all stocks currently tracked in `Stock_app`'s database with filters for Manual Predictions, Screen Hits, Bullish/Bearish/Volatility categories, and Global Indexes.
   - **Parquet Memory Explorer**: Search box and universe filter across all tickers cached in `ML_stock/data/raw` to inspect in detail or run predictions on.
   - **Screening Call History & Specific Prediction Call History**: Full audit log of every market screening call and specific stock prediction call.
2. **Selected Stock Detail View**:
   - **Market Chart & Prediction Band Overlay**: Candlesticks / Close Line + Volume chart. Clicking any prediction in the right-hand **Previous Predictions Side Menu** overlays its probability cone (`50% Core [25%–75%]`, `80% Standard [10%–90%]`, `90% Tail [5%–95%]`, and `50% Median`) directly onto the market chart.
   - **Point-in-Time Probability Cone vs. Realized Price Trajectory**: Focused graph comparing the model's probability cone at the exact anchor date the prediction was made against what the stock's price actually did afterward, complete with a milestone accuracy table (`Day +1`, `Day +2`, `Day +5`, `Day +20`).
   - **Live & Historical Point-in-Time Prediction Runner**: Run a new prediction as of the latest bar or pick any past date to backtest the probability cone against realized market prices.
   - **Cache Refresh**: One-click button that calls `ML_stock/src/data_loader.py` to update the cached Parquet file.
