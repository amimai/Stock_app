/**
 * Main Frontend Application Controller for Stock_app.
 */

const AppState = {
  view: "dashboard", // 'dashboard' | 'stock'
  overview: null,
  parquetTickers: [],
  activeScreenRun: null,
  activeScreenTab: "bullish",

  // Selected stock view state
  currentStock: null,
  selectedPredictionId: null,
  chartRange: 180,
  chartMode: "candle",
  showAllCones: false,
  coneUnitMode: "price", // 'price' | 'return'
};

function showToast(msg, isError = false) {
  const container = document.getElementById("toastContainer");
  if (!container) return;
  const el = document.createElement("div");
  el.className = "toast";
  if (isError) el.style.borderLeftColor = "#f43f5e";
  el.textContent = msg;
  container.appendChild(el);
  setTimeout(() => el.remove(), 4200);
}

async function apiFetch(url, options = {}) {
  const resp = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await resp.json();
  if (!resp.ok || !data.ok) {
    throw new Error(data.error || `Request failed (${resp.status})`);
  }
  return data;
}

function fmtRetBadge(logRet) {
  if (logRet === null || logRet === undefined || isNaN(logRet)) {
    return '<span class="text-muted">--</span>';
  }
  const pct = (Math.exp(Number(logRet)) - 1) * 100;
  const cls = pct >= 0 ? "text-bull" : "text-bear";
  const sign = pct >= 0 ? "+" : "";
  return `<span class="${cls}">${sign}${pct.toFixed(2)}%</span>`;
}

function renderCategoryBadges(source, screenCat) {
  const html = [];
  if (source === "manual") {
    html.push('<span class="badge badge-manual">Manual Prediction</span>');
  } else {
    const cats = (screenCat || "screen").split(",");
    cats.forEach((c) => {
      const clean = c.trim().toLowerCase();
      if (clean === "bullish") html.push('<span class="badge badge-bullish">Bullish Hit</span>');
      else if (clean === "bearish") html.push('<span class="badge badge-bearish">Bearish Hit</span>');
      else if (clean === "volatility") html.push('<span class="badge badge-volatility">Volatility Play</span>');
      else if (clean === "index") html.push('<span class="badge badge-index">Index</span>');
      else html.push(`<span class="badge badge-universe">${clean}</span>`);
    });
  }
  return html.join(" ");
}

/* ==================== DASHBOARD LOADING & RENDERING ==================== */

async function loadOverview() {
  try {
    const sourceFilter = document.getElementById("predHistorySourceSelect")?.value || "all";
    const data = await apiFetch(`/api/overview?source=${encodeURIComponent(sourceFilter)}`);
    AppState.overview = data;
    renderDashboardOverview();
  } catch (err) {
    showToast("Failed loading overview: " + err.message, true);
  }
}

function renderDashboardOverview() {
  const ov = AppState.overview;
  if (!ov) return;

  const st = ov.status || {};
  const counts = st.db_counts || {};
  const mod = st.model || {};

  document.getElementById("systemStatusLine").textContent =
    `ML_stock Connected • ${st.parquet_ticker_count || 0} Parquet Tickers • Model: ${(mod.model_type || "patchtst").toUpperCase()} (${(mod.device || "cpu").toUpperCase()})`;

  document.getElementById("kpiTrackedStocks").textContent = counts.tracked_stocks ?? 0;
  document.getElementById("kpiManualPreds").textContent = counts.manual_predictions ?? 0;
  document.getElementById("kpiTotalPredsSub").textContent =
    `${counts.total_predictions ?? 0} total stored cones (manual + screen)`;
  document.getElementById("kpiScreenRuns").textContent = counts.screening_runs ?? 0;
  document.getElementById("kpiScreenHitsSub").textContent =
    `${counts.screen_hit_predictions ?? 0} stored screen hit predictions`;
  document.getElementById("kpiParquetCount").textContent = st.parquet_ticker_count ?? 0;
  document.getElementById("kpiModelArchSub").textContent = mod.milestones
    ? `Milestones: ${mod.milestones.join(", ")}d | ${mod.quantiles?.length || 7} Quantiles`
    : "PatchTST Checkpoint Ready";

  renderTrackedStocksTable();
  renderScreeningHistory();
  renderPredictionHistory();
}

function renderTrackedStocksTable() {
  const tbody = document.getElementById("trackedStocksTbody");
  if (!tbody || !AppState.overview) return;

  const filterCat = document.getElementById("trackedFilterSelect")?.value || "all";
  const searchQ = (document.getElementById("trackedSearchInput")?.value || "").trim().toUpperCase();

  const list = (AppState.overview.tracked_stocks || []).filter((item) => {
    if (filterCat === "manual" && item.manual_count <= 0) return false;
    if (filterCat === "screen" && item.screen_hit_count <= 0) return false;
    if (
      ["bullish", "bearish", "volatility", "index"].includes(filterCat) &&
      !(item.screen_categories || []).some((c) => c.toLowerCase().includes(filterCat))
    ) {
      return false;
    }
    if (searchQ) {
      const match =
        item.ticker.toUpperCase().includes(searchQ) ||
        (item.name || "").toUpperCase().includes(searchQ) ||
        (item.sector || "").toUpperCase().includes(searchQ);
      if (!match) return false;
    }
    return true;
  });

  if (!list.length) {
    tbody.innerHTML = `
      <tr>
        <td colspan="8" style="text-align:center; padding: 24px;" class="text-muted">
          No tracked stocks match the current filter. Click <strong>Run Market Screen</strong> or <strong>+ Run Stock Prediction</strong> above, or pick any stock from the Parquet Memory Explorer below!
        </td>
      </tr>`;
    return;
  }

  tbody.innerHTML = list
    .map((item) => {
      const sum = item.latest_summary || {};
      const p10 = sum.price_2d_10 ? `$${Number(sum.price_2d_10).toFixed(2)}` : "--";
      const p90 = sum.price_2d_90 ? `$${Number(sum.price_2d_90).toFixed(2)}` : "--";
      const badges = [];
      if (item.manual_count > 0) {
        badges.push(`<span class="badge badge-manual">${item.manual_count} Manual</span>`);
      }
      if (item.screen_hit_count > 0) {
        badges.push(`<span class="badge badge-universe">${item.screen_hit_count} Screen Hits</span>`);
      }
      (item.screen_categories || []).forEach((cat) => {
        cat.split(",").forEach((c) => {
          const cl = c.trim();
          if (cl && !badges.some((b) => b.toLowerCase().includes(cl.toLowerCase()))) {
            badges.push(renderCategoryBadges("screen", cl));
          }
        });
      });

      return `
        <tr class="clickable-row" onclick="openStockDetail('${item.ticker}', ${item.latest_prediction_id})">
          <td><strong>${item.ticker}</strong> <span class="badge badge-universe">${item.universe}</span></td>
          <td>
            <div>${item.name}</div>
            <div class="text-muted" style="font-size:11px;">${item.sector}</div>
          </td>
          <td>${badges.join(" ")}</td>
          <td>${item.latest_anchor_date}</td>
          <td>$${Number(item.latest_anchor_close).toFixed(2)}</td>
          <td>${fmtRetBadge(sum.cum2d_ret_50)}</td>
          <td><span class="text-muted">${p10} – ${p90}</span></td>
          <td onclick="event.stopPropagation()">
            <button class="btn btn-xs active" onclick="openStockDetail('${item.ticker}', ${item.latest_prediction_id})">Chart &amp; Cones</button>
            <button class="btn btn-xs" onclick="quickRunPrediction('${item.ticker}')">Predict</button>
          </td>
        </tr>`;
    })
    .join("");
}

async function searchParquetMemory() {
  const q = document.getElementById("parquetSearchInput")?.value || "";
  const universe = document.getElementById("parquetUniverseSelect")?.value || "all";
  try {
    const data = await apiFetch(
      `/api/parquet/search?q=${encodeURIComponent(q)}&universe=${encodeURIComponent(universe)}&limit=120`
    );
    AppState.parquetTickers = data.tickers || [];
    renderParquetTable();
  } catch (err) {
    showToast("Parquet search error: " + err.message, true);
  }
}

function renderParquetTable() {
  const tbody = document.getElementById("parquetSearchTbody");
  if (!tbody) return;
  const rows = AppState.parquetTickers || [];
  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="6" style="text-align:center;padding:18px;" class="text-muted">No matching tickers found in Parquet memory.</td></tr>`;
    return;
  }

  tbody.innerHTML = rows
    .map(
      (r) => `
      <tr class="clickable-row" onclick="openStockDetail('${r.ticker}')">
        <td><strong>${r.ticker}</strong></td>
        <td>${r.name}</td>
        <td><span class="badge badge-sector">${r.sector}</span></td>
        <td><span class="badge badge-universe">${r.universe}</span></td>
        <td>
          ${
            r.total_predictions > 0
              ? `<span class="badge badge-manual">${r.total_predictions} stored</span>`
              : `<span class="text-muted">None yet</span>`
          }
        </td>
        <td onclick="event.stopPropagation()">
          <button class="btn btn-xs active" onclick="openStockDetail('${r.ticker}')">View Chart</button>
          <button class="btn btn-xs" onclick="quickRunPrediction('${r.ticker}')">Run Prediction</button>
        </td>
      </tr>`
    )
    .join("");
}

/* ==================== CALL HISTORIES (SCREENING & PREDICTIONS) ==================== */

function renderScreeningHistory() {
  const container = document.getElementById("screeningHistoryList");
  if (!container || !AppState.overview) return;
  const runs = AppState.overview.screening_history || [];
  if (!runs.length) {
    container.innerHTML = `<div class="text-muted" style="padding:18px;text-align:center;">No screening calls recorded yet. Click <strong>+ New Screen</strong> to run a universe screen.</div>`;
    return;
  }

  container.innerHTML = runs
    .map((run) => {
      const bullPrev = (run.top_bullish_preview || [])
        .map((x) => `${x.ticker} (${fmtRetBadge(x.cum2d_ret_50)})`)
        .join(", ");
      const bearPrev = (run.top_bearish_preview || [])
        .map((x) => `${x.ticker} (${fmtRetBadge(x.cum2d_ret_50)})`)
        .join(", ");
      const isAct = AppState.activeScreenRun && AppState.activeScreenRun.id === run.id;
      return `
        <div class="screen-history-card ${isAct ? "active" : ""}" onclick="inspectScreeningRun(${run.id})">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
            <div>
              <strong>Screen #${run.id}</strong>
              <span class="badge badge-universe">${run.universe.toUpperCase()}</span>
              <span class="text-muted" style="font-size:11.5px;">Top ${run.top_n} • Min $${Number(run.min_price).toFixed(2)} • ${run.total_screened} screened (${run.duration_sec}s)</span>
            </div>
            <span class="text-muted" style="font-size:11px;">${run.created_at.replace("T", " ").slice(0, 19)} UTC</span>
          </div>
          <div style="font-size:11.5px; margin-top:4px;">
            <span class="text-bull">Bullish:</span> ${bullPrev || "--"} &nbsp;|&nbsp;
            <span class="text-bear">Bearish:</span> ${bearPrev || "--"}
          </div>
        </div>`;
    })
    .join("");
}

async function inspectScreeningRun(runId) {
  try {
    const data = await apiFetch(`/api/screen/${runId}`);
    AppState.activeScreenRun = data.screening_run;
    renderScreeningHistory();
    renderScreenRunInspector();
  } catch (err) {
    showToast("Could not load screening run: " + err.message, true);
  }
}

function renderScreenRunInspector() {
  const panel = document.getElementById("screenRunInspectorPanel");
  const run = AppState.activeScreenRun;
  if (!panel || !run) {
    panel?.classList.add("hidden");
    return;
  }
  panel.classList.remove("hidden");
  document.getElementById("screenInspectorTitle").textContent =
    `Screening Call #${run.id} — Universe: ${run.universe.toUpperCase()}`;
  document.getElementById("screenInspectorMeta").textContent =
    `Executed ${run.created_at.replace("T", " ").slice(0, 19)} UTC • Screened ${run.total_screened}/${run.total_candidates} instruments in ${run.duration_sec}s`;

  const res = run.results || {};
  const tab = AppState.activeScreenTab || "bullish";
  const rows =
    tab === "bullish"
      ? res.top_bullish || []
      : tab === "bearish"
      ? res.top_bearish || []
      : tab === "volatility"
      ? res.top_volatility || []
      : res.indexes || [];

  const tbody = document.getElementById("screenInspectorTbody");
  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;padding:14px;" class="text-muted">No instruments in this category.</td></tr>`;
    return;
  }

  tbody.innerHTML = rows
    .map(
      (r) => `
      <tr class="clickable-row" onclick="openStockDetail('${r.ticker}')">
        <td><strong>${r.ticker}</strong></td>
        <td>${r.name || r.ticker}</td>
        <td>$${Number(r.last_close || 0).toFixed(2)}</td>
        <td>${fmtRetBadge(r.day1_ret_50)}</td>
        <td>${fmtRetBadge(r.cum2d_ret_50)}</td>
        <td class="text-muted">$${Number(r.price_10 || 0).toFixed(2)} – $${Number(r.price_90 || 0).toFixed(2)}</td>
        <td>${fmtRetBadge(r.cum20d_ret_50)}</td>
        <td onclick="event.stopPropagation()">
          <button class="btn btn-xs active" onclick="openStockDetail('${r.ticker}')">View Cone</button>
        </td>
      </tr>`
    )
    .join("");
}

function renderPredictionHistory() {
  const tbody = document.getElementById("predHistoryTbody");
  if (!tbody || !AppState.overview) return;
  const preds = AppState.overview.prediction_history || [];
  if (!preds.length) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;padding:18px;" class="text-muted">No predictions recorded yet.</td></tr>`;
    return;
  }

  tbody.innerHTML = preds
    .map((p) => {
      const sum = p.summary || {};
      return `
        <tr class="clickable-row" onclick="openStockDetail('${p.ticker}', ${p.id})">
          <td class="text-muted">${p.created_at.replace("T", " ").slice(0, 19)}</td>
          <td><strong>${p.ticker}</strong></td>
          <td>${renderCategoryBadges(p.source, p.screen_category)}</td>
          <td>${p.anchor_date}</td>
          <td>$${Number(p.anchor_close).toFixed(2)}</td>
          <td>${fmtRetBadge(sum.cum2d_ret_50)}</td>
          <td>${fmtRetBadge(sum.cum20d_ret_50)}</td>
          <td><button class="btn btn-xs active">Overlay</button></td>
        </tr>`;
    })
    .join("");
}

/* ==================== STOCK DETAIL VIEW & PREDICTION SIDE MENU ==================== */

async function openStockDetail(ticker, preferredPredId = null) {
  if (!ticker) return;
  const sym = ticker.trim().toUpperCase();
  try {
    const data = await apiFetch(`/api/stock/${encodeURIComponent(sym)}?bars=360&auto_fetch=1`);
    AppState.currentStock = data.stock;
    AppState.view = "stock";

    const preds = AppState.currentStock.predictions || [];
    if (preferredPredId && preds.some((p) => p.id === Number(preferredPredId))) {
      AppState.selectedPredictionId = Number(preferredPredId);
    } else if (preds.length > 0) {
      AppState.selectedPredictionId = preds[0].id;
    } else {
      AppState.selectedPredictionId = null;
    }

    document.getElementById("dashboardView").classList.add("hidden");
    document.getElementById("stockDetailView").classList.remove("hidden");
    renderStockDetailView();
  } catch (err) {
    showToast(`Failed loading stock ${sym}: ` + err.message, true);
  }
}

function getSelectedPrediction() {
  const st = AppState.currentStock;
  if (!st || !st.predictions) return null;
  return st.predictions.find((p) => p.id === AppState.selectedPredictionId) || null;
}

function renderStockDetailView() {
  const st = AppState.currentStock;
  if (!st) return;

  document.getElementById("detailTickerTitle").textContent = st.ticker;
  document.getElementById("detailUniverseBadge").textContent = st.universe || "custom";
  document.getElementById("detailSectorBadge").textContent = st.sector || "Equities";
  document.getElementById("detailCompanyName").textContent = st.name || st.ticker;

  if (st.latest_bar) {
    document.getElementById("detailLatestPrice").textContent =
      "$" + Number(st.latest_bar.close).toFixed(2);
    document.getElementById("detailLatestDate").textContent =
      `Last bar: ${st.latest_bar.date} (${st.bars.length} bars loaded)`;
    const chgEl = document.getElementById("detailLatestChange");
    if (st.day_change_pct !== null && st.day_change_pct !== undefined) {
      const pct = st.day_change_pct * 100;
      chgEl.className = "stock-latest-change " + (pct >= 0 ? "text-bull" : "text-bear");
      chgEl.textContent = (pct >= 0 ? "+" : "") + pct.toFixed(2) + "% (1d)";
    } else {
      chgEl.textContent = "";
    }
  }

  renderStockPredictionsSideMenu();
  renderStockChartsAndMilestones();
}

function renderStockPredictionsSideMenu() {
  const st = AppState.currentStock;
  const menu = document.getElementById("stockPredictionsMenu");
  const badge = document.getElementById("stockPredCountBadge");
  if (!st || !menu) return;

  const preds = st.predictions || [];
  badge.textContent = preds.length;

  if (!preds.length) {
    menu.innerHTML = `
      <div class="text-muted" style="padding:18px;text-align:center;">
        No stored predictions for <strong>${st.ticker}</strong> yet.<br/><br/>
        Click <strong>Run Prediction</strong> in the top bar (or pick a historical Point-in-Time Date) to generate and store a probability cone!
      </div>`;
    return;
  }

  menu.innerHTML = preds
    .map((p) => {
      const isSelected = p.id === AppState.selectedPredictionId;
      const sum = p.summary || {};
      const real = p.realized || {};
      const elapsedStr =
        real.elapsed_trading_days > 0
          ? `${real.elapsed_trading_days}d elapsed (${fmtRetBadge(real.latest_return_since_anchor)} actual)`
          : "Live / Latest Bar (0d elapsed)";

      return `
        <div class="pred-menu-item ${isSelected ? "active" : ""}" onclick="selectStockPrediction(${p.id})">
          <div class="pred-item-top">
            <span class="pred-item-anchor">Anchor: ${p.anchor_date}</span>
            ${renderCategoryBadges(p.source, p.screen_category)}
          </div>
          <div class="text-muted" style="font-size:11px;">
            Run #${p.id} • ${p.created_at.replace("T", " ").slice(0, 19)} UTC • Anchor Px: $${Number(p.anchor_close).toFixed(2)}
          </div>
          <div class="pred-item-metrics">
            <div>2d Med: ${fmtRetBadge(sum.cum2d_ret_50)}</div>
            <div>20d Med: ${fmtRetBadge(sum.cum20d_ret_50)}</div>
            <div style="grid-column: span 2; color: #94a3b8; margin-top: 2px;">
              Realized: ${elapsedStr}
            </div>
          </div>
        </div>`;
    })
    .join("");
}

function selectStockPrediction(predId) {
  AppState.selectedPredictionId = Number(predId);
  renderStockPredictionsSideMenu();
  renderStockChartsAndMilestones();
}

function renderStockChartsAndMilestones() {
  const st = AppState.currentStock;
  if (!st) return;
  const activePred = getSelectedPrediction();

  const mCanvas = document.getElementById("marketChartCanvas");
  window.StockAppCharts.renderMarketOverlayChart(mCanvas, {
    bars: st.bars || [],
    rangeLimit: AppState.chartRange,
    chartMode: AppState.chartMode,
    activePrediction: activePred,
    allPredictions: st.predictions || [],
    showAllCones: AppState.showAllCones,
  });

  const cCanvas = document.getElementById("coneChartCanvas");
  window.StockAppCharts.renderFocusedConeChart(cCanvas, activePred, AppState.coneUnitMode);

  // Update subtitles & milestone comparison table
  const mSub = document.getElementById("marketChartSubtitle");
  const cSub = document.getElementById("coneChartSubtitle");
  const tbody = document.getElementById("milestoneEvalTbody");

  if (!activePred) {
    mSub.textContent = "Showing historical market chart. Run or select a prediction to overlay probability bands.";
    cSub.textContent = "No prediction selected.";
    tbody.innerHTML = `<tr><td colspan="9" style="text-align:center;padding:16px;" class="text-muted">Run or select a prediction to inspect milestone quantile bands and realized accuracy.</td></tr>`;
    return;
  }

  mSub.textContent = `Overlaying Prediction #${activePred.id} (Anchor Date: ${activePred.anchor_date} @ $${Number(activePred.anchor_close).toFixed(2)}) onto ${st.ticker} market price chart.`;
  cSub.textContent = `Point-in-Time Probability Cone made on ${activePred.anchor_date} ($${Number(activePred.anchor_close).toFixed(2)}) vs. actual ${st.ticker} price performance (${activePred.realized?.elapsed_trading_days || 0} trading days elapsed).`;

  const milestones = (activePred.cone && activePred.cone.milestones) || [];
  tbody.innerHTML = milestones
    .map((m) => {
      const p = m.prices || {};
      const r = m.returns || {};
      let statusBadge = '<span class="badge badge-universe">Pending Future Horizon</span>';
      if (m.is_realized) {
        const b80 = m.in_80_band
          ? '<span class="badge badge-bullish">In 80% Band</span>'
          : '<span class="badge badge-bearish">Outside 80%</span>';
        const b50 = m.in_50_band ? '<span class="badge badge-manual">In 50% Core</span>' : "";
        const dir = m.direction_correct
          ? '<span class="badge badge-bullish">Dir ✓</span>'
          : '<span class="badge badge-volatility">Dir ✗</span>';
        statusBadge = `${b80} ${b50} ${dir}`;
      }
      return `
        <tr>
          <td><strong>Day +${m.horizon_days}</strong></td>
          <td>${m.target_date || "--"}</td>
          <td>$${Number(p.q50).toFixed(2)} (${fmtRetBadge(r.q50)})</td>
          <td>$${Number(p.q25).toFixed(2)} – $${Number(p.q75).toFixed(2)}</td>
          <td>$${Number(p.q10).toFixed(2)} – $${Number(p.q90).toFixed(2)}</td>
          <td>$${Number(p.q05).toFixed(2)} – $${Number(p.q95).toFixed(2)}</td>
          <td>${m.is_realized && m.actual_close !== null ? `<strong>$${Number(m.actual_close).toFixed(2)}</strong>` : '<span class="text-muted">Unrealized</span>'}</td>
          <td>${m.is_realized ? fmtRetBadge(m.actual_return) : '<span class="text-muted">--</span>'}</td>
          <td>${statusBadge}</td>
        </tr>`;
    })
    .join("");
}

/* ==================== PREDICTION, SCREENING & CACHE ACTIONS ==================== */

async function quickRunPrediction(ticker) {
  if (!ticker) return;
  showToast(`Running PatchTST prediction for ${ticker}...`);
  try {
    const res = await apiFetch("/api/predict", {
      method: "POST",
      body: JSON.stringify({ ticker, num_samples: 1 }),
    });
    showToast(`Saved prediction #${res.prediction.id} for ${ticker}`);
    await loadOverview();
    await openStockDetail(ticker, res.prediction.id);
  } catch (err) {
    showToast(`Prediction failed for ${ticker}: ` + err.message, true);
  }
}

async function runDetailStockPrediction() {
  const st = AppState.currentStock;
  if (!st) return;
  const btn = document.getElementById("btnRunStockPredictNow");
  const asOfDate = document.getElementById("detailAsOfDate")?.value || null;
  const numSamples = parseInt(document.getElementById("detailNumSamples")?.value || "1", 10);
  const mcDropout = Boolean(document.getElementById("detailMcDropout")?.checked);

  btn.disabled = true;
  btn.textContent = "Predicting...";
  try {
    const res = await apiFetch("/api/predict", {
      method: "POST",
      body: JSON.stringify({
        ticker: st.ticker,
        as_of_date: asOfDate,
        num_samples: numSamples,
        mc_dropout: mcDropout,
        market_resampling: true,
      }),
    });
    showToast(
      `Prediction #${res.prediction.id} stored for ${st.ticker} (Anchor: ${res.prediction.anchor_date})`
    );
    await loadOverview();
    await openStockDetail(st.ticker, res.prediction.id);
  } catch (err) {
    showToast("Prediction error: " + err.message, true);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Prediction";
  }
}

async function refreshCurrentStockCache() {
  const st = AppState.currentStock;
  if (!st) return;
  const btn = document.getElementById("btnRefreshStockCache");
  btn.disabled = true;
  btn.textContent = "Updating via data_loader.py...";
  try {
    const res = await apiFetch("/api/cache/refresh", {
      method: "POST",
      body: JSON.stringify({ tickers: [st.ticker], force_refresh: false }),
    });
    showToast(`Updated Parquet cache for ${st.ticker} (${res.result.rows || 0} rows)`);
    await openStockDetail(st.ticker, AppState.selectedPredictionId);
  } catch (err) {
    showToast("Cache refresh failed: " + err.message, true);
  } finally {
    btn.disabled = false;
    btn.textContent = "Update Cache (data_loader.py)";
  }
}

async function submitScreenModal() {
  const btn = document.getElementById("btnSubmitScreenModal");
  const universe = document.getElementById("modalScreenUniverse").value;
  const topN = parseInt(document.getElementById("modalScreenTopN").value || "10", 10);
  const minPrice = parseFloat(document.getElementById("modalScreenMinPrice").value || "5.0");
  const saveAll = Boolean(document.getElementById("modalScreenSaveAll").checked);
  const refreshData = Boolean(document.getElementById("modalScreenRefreshData").checked);

  btn.disabled = true;
  btn.textContent = "Screening Universe...";
  try {
    const res = await apiFetch("/api/screen", {
      method: "POST",
      body: JSON.stringify({
        universe,
        top_n: topN,
        min_price: minPrice,
        save_all_screened: saveAll,
        refresh_data: refreshData,
      }),
    });
    document.getElementById("screenModal").classList.add("hidden");
    showToast(
      `Screening Call #${res.screening_run.id} completed (${res.screening_run.total_screened} instruments in ${res.screening_run.duration_sec}s)`
    );
    AppState.activeScreenRun = res.screening_run;
    await loadOverview();
    renderScreenRunInspector();
  } catch (err) {
    showToast("Screening error: " + err.message, true);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Screening Call";
  }
}

async function submitPredictModal() {
  const btn = document.getElementById("btnSubmitPredictModal");
  const ticker = document.getElementById("modalPredictTicker").value.trim().toUpperCase();
  const asOfDate = document.getElementById("modalPredictAsOfDate").value || null;
  const numSamples = parseInt(document.getElementById("modalPredictSamples").value || "1", 10);
  const mcDropout = Boolean(document.getElementById("modalPredictMcDropout").checked);
  const refreshData = Boolean(document.getElementById("modalPredictRefreshData").checked);

  if (!ticker) {
    showToast("Please enter a ticker symbol.", true);
    return;
  }

  btn.disabled = true;
  btn.textContent = "Running Prediction...";
  try {
    const res = await apiFetch("/api/predict", {
      method: "POST",
      body: JSON.stringify({
        ticker,
        as_of_date: asOfDate,
        num_samples: numSamples,
        mc_dropout: mcDropout,
        refresh_data: refreshData,
      }),
    });
    document.getElementById("predictModal").classList.add("hidden");
    showToast(`Stored prediction #${res.prediction.id} for ${ticker}`);
    await loadOverview();
    await openStockDetail(ticker, res.prediction.id);
  } catch (err) {
    showToast("Prediction error: " + err.message, true);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run & Store Prediction";
  }
}

/* ==================== INITIALIZATION & EVENT LISTENERS ==================== */

function showDashboardView() {
  AppState.view = "dashboard";
  document.getElementById("stockDetailView").classList.add("hidden");
  document.getElementById("dashboardView").classList.remove("hidden");
  loadOverview();
}

window.addEventListener("DOMContentLoaded", () => {
  loadOverview();
  searchParquetMemory();

  // Navigation
  document.getElementById("navHomeBtn")?.addEventListener("click", showDashboardView);
  document.getElementById("btnNavDashboard")?.addEventListener("click", showDashboardView);
  document.getElementById("btnBackToDashboard")?.addEventListener("click", showDashboardView);

  // Tracked stocks filtering
  document.getElementById("trackedFilterSelect")?.addEventListener("change", renderTrackedStocksTable);
  document.getElementById("trackedSearchInput")?.addEventListener("input", renderTrackedStocksTable);

  // Parquet explorer filtering
  document.getElementById("btnParquetSearch")?.addEventListener("click", searchParquetMemory);
  document.getElementById("parquetUniverseSelect")?.addEventListener("change", searchParquetMemory);
  document.getElementById("parquetSearchInput")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") searchParquetMemory();
  });

  // Global search bar dropdown
  const globalInput = document.getElementById("globalSearchInput");
  const globalDrop = document.getElementById("globalSearchDropdown");
  let searchTimer = null;
  globalInput?.addEventListener("input", () => {
    clearTimeout(searchTimer);
    const q = globalInput.value.trim();
    if (!q) {
      globalDrop.classList.add("hidden");
      return;
    }
    searchTimer = setTimeout(async () => {
      try {
        const res = await apiFetch(`/api/parquet/search?q=${encodeURIComponent(q)}&limit=12`);
        const items = res.tickers || [];
        if (!items.length) {
          globalDrop.innerHTML = `<div class="search-item text-muted">No cached Parquet match for "${q}" — click "+ Run Stock Prediction" to fetch &amp; predict.</div>`;
        } else {
          globalDrop.innerHTML = items
            .map(
              (it) => `
              <div class="search-item" onclick="document.getElementById('globalSearchDropdown').classList.add('hidden'); openStockDetail('${it.ticker}')">
                <div><strong>${it.ticker}</strong> — ${it.name} <span class="badge badge-universe">${it.universe}</span></div>
                <div>${it.total_predictions > 0 ? `<span class="badge badge-manual">${it.total_predictions} preds</span>` : '<span class="text-muted">View Chart</span>'}</div>
              </div>`
            )
            .join("");
        }
        globalDrop.classList.remove("hidden");
      } catch (_) {}
    }, 180);
  });

  document.addEventListener("click", (e) => {
    if (!e.target.closest(".topbar-search")) {
      globalDrop?.classList.add("hidden");
    }
  });

  // History Tabs
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach((p) => p.classList.add("hidden"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.tab)?.classList.remove("hidden");
    });
  });

  document.getElementById("predHistorySourceSelect")?.addEventListener("change", loadOverview);

  // Screen Run Inspector Subtabs
  document.querySelectorAll(".subtab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".subtab-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      AppState.activeScreenTab = btn.dataset.screentab;
      renderScreenRunInspector();
    });
  });
  document.getElementById("btnCloseScreenInspector")?.addEventListener("click", () => {
    AppState.activeScreenRun = null;
    renderScreenRunInspector();
    renderScreeningHistory();
  });

  // Stock Detail Controls
  document.getElementById("btnRunStockPredictNow")?.addEventListener("click", runDetailStockPrediction);
  document.getElementById("btnRefreshStockCache")?.addEventListener("click", refreshCurrentStockCache);

  document.querySelectorAll("[data-range]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("[data-range]").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      AppState.chartRange = parseInt(btn.dataset.range, 10);
      renderStockChartsAndMilestones();
    });
  });

  document.getElementById("btnChartModeCandle")?.addEventListener("click", (e) => {
    AppState.chartMode = "candle";
    e.target.classList.add("active");
    document.getElementById("btnChartModeLine")?.classList.remove("active");
    renderStockChartsAndMilestones();
  });

  document.getElementById("btnChartModeLine")?.addEventListener("click", (e) => {
    AppState.chartMode = "line";
    e.target.classList.add("active");
    document.getElementById("btnChartModeCandle")?.classList.remove("active");
    renderStockChartsAndMilestones();
  });

  document.getElementById("chkShowAllCones")?.addEventListener("change", (e) => {
    AppState.showAllCones = Boolean(e.target.checked);
    renderStockChartsAndMilestones();
  });

  document.getElementById("btnConeUnitPrice")?.addEventListener("click", (e) => {
    AppState.coneUnitMode = "price";
    e.target.classList.add("active");
    document.getElementById("btnConeUnitReturn")?.classList.remove("active");
    renderStockChartsAndMilestones();
  });

  document.getElementById("btnConeUnitReturn")?.addEventListener("click", (e) => {
    AppState.coneUnitMode = "return";
    e.target.classList.add("active");
    document.getElementById("btnConeUnitPrice")?.classList.remove("active");
    renderStockChartsAndMilestones();
  });

  // Modals
  const screenModal = document.getElementById("screenModal");
  document.getElementById("btnOpenScreenModal")?.addEventListener("click", () => screenModal.classList.remove("hidden"));
  document.getElementById("btnQuickRunScreen")?.addEventListener("click", () => screenModal.classList.remove("hidden"));
  document.getElementById("btnCloseScreenModal")?.addEventListener("click", () => screenModal.classList.add("hidden"));
  document.getElementById("btnCancelScreenModal")?.addEventListener("click", () => screenModal.classList.add("hidden"));
  document.getElementById("btnSubmitScreenModal")?.addEventListener("click", submitScreenModal);

  const predictModal = document.getElementById("predictModal");
  document.getElementById("btnOpenPredictModal")?.addEventListener("click", () => {
    if (AppState.view === "stock" && AppState.currentStock) {
      document.getElementById("modalPredictTicker").value = AppState.currentStock.ticker;
    }
    predictModal.classList.remove("hidden");
  });
  document.getElementById("btnClosePredictModal")?.addEventListener("click", () => predictModal.classList.add("hidden"));
  document.getElementById("btnCancelPredictModal")?.addEventListener("click", () => predictModal.classList.add("hidden"));
  document.getElementById("btnSubmitPredictModal")?.addEventListener("click", submitPredictModal);

  window.addEventListener("resize", () => {
    if (AppState.view === "stock") {
      renderStockChartsAndMilestones();
    }
  });
});
