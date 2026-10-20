/**
 * Main Frontend Application Controller for Stock_app.
 */

const AppState = {
  view: "dashboard", // 'dashboard' | 'stock' | 'model_runs'
  overview: null,
  parquetTickers: [],
  activeScreenRun: null,
  activeScreenTab: "bullish",
  trackedViewMode: "directional", // 'directional' | 'volatility'
  activeHorizon: "2d", // '1d' | '2d' | '5d' | '20d'

  // Model switching & lineage tracking state
  availableModels: [],
  activeModelRunId: "default",
  selectedLeaderboardRunId: null,
  selectedModelDetail: null,
  activeModelSubTab: "subtabBenchmark",

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

function escapeHtml(str) {
  if (str === null || str === undefined) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function renderModelBadge(runId) {
  if (!runId || runId === "default" || runId === "active") {
    return '<span class="badge badge-model-default" title="Root Default Checkpoint (patchtst_best.pt)">Default</span>';
  }
  const s = String(runId).toLowerCase();
  const safeId = escapeHtml(runId);
  if (s.includes("3head")) {
    return `<span class="badge badge-model-3head" title="Model Run: ${safeId}">3-Head 4L</span>`;
  }
  if (s.includes("1head")) {
    if (s.includes("8l")) {
      return `<span class="badge badge-model-legacy" title="Model Run: ${safeId}">1-Head 8L</span>`;
    }
    return `<span class="badge badge-model-1head" title="Model Run: ${safeId}">1-Head 4L</span>`;
  }
  if (s.includes("8l")) {
    return `<span class="badge badge-model-legacy" title="Model Run: ${safeId}">8L Enc</span>`;
  }
  const shortId = runId.length > 18 ? runId.slice(0, 10) + "..." : runId;
  return `<span class="badge badge-model-1head" title="Model Run: ${safeId}">${escapeHtml(shortId)}</span>`;
}

function getHorizonMetrics(item, horizon = "2d") {
  const sum = item?.latest_summary || {};
  const horizons = sum.horizons || {};
  if (horizons[horizon]) {
    const h = horizons[horizon];
    const logRet = h.ret_50;
    const medPct = logRet !== null && logRet !== undefined && !isNaN(logRet)
      ? Math.round((Math.exp(Number(logRet)) - 1) * 1000000) / 10000
      : null;
    const sp = h.spread_80;
    const spreadPct = sp !== null && sp !== undefined && !isNaN(sp)
      ? Math.round(Number(sp) * 1000000) / 10000
      : null;
    return {
      ret_50: h.ret_50,
      ret_10: h.ret_10,
      ret_90: h.ret_90,
      price_10: h.price_10,
      price_50: h.price_50,
      price_90: h.price_90,
      medPct,
      spreadPct,
      skew: h.skew,
    };
  }

  // Fallback mappings if horizons dict wasn't populated on legacy prediction
  if (horizon === "1d") {
    const logRet = sum.day1_ret_50;
    const medPct = logRet !== null && logRet !== undefined && !isNaN(logRet)
      ? Math.round((Math.exp(Number(logRet)) - 1) * 1000000) / 10000
      : null;
    return {
      ret_50: sum.day1_ret_50,
      ret_10: null,
      ret_90: null,
      price_10: null,
      price_50: sum.day1_price_50,
      price_90: null,
      medPct,
      spreadPct: null,
      skew: null,
    };
  } else if (horizon === "5d") {
    const logRet = sum.cum5d_ret_50;
    const medPct = logRet !== null && logRet !== undefined && !isNaN(logRet)
      ? Math.round((Math.exp(Number(logRet)) - 1) * 1000000) / 10000
      : null;
    return {
      ret_50: sum.cum5d_ret_50,
      ret_10: null,
      ret_90: null,
      price_10: null,
      price_50: sum.price_5d_50,
      price_90: null,
      medPct,
      spreadPct: null,
      skew: null,
    };
  } else if (horizon === "20d") {
    const logRet = sum.cum20d_ret_50;
    const medPct = logRet !== null && logRet !== undefined && !isNaN(logRet)
      ? Math.round((Math.exp(Number(logRet)) - 1) * 1000000) / 10000
      : null;
    return {
      ret_50: sum.cum20d_ret_50,
      ret_10: null,
      ret_90: null,
      price_10: null,
      price_50: sum.price_20d_50,
      price_90: null,
      medPct,
      spreadPct: null,
      skew: null,
    };
  }

  // Default to 2d:
  const logRet = sum.cum2d_ret_50;
  const medPct = logRet !== null && logRet !== undefined && !isNaN(logRet)
    ? Math.round((Math.exp(Number(logRet)) - 1) * 1000000) / 10000
    : null;
  const spVal = get2dSpreadPct(item);
  return {
    ret_50: sum.cum2d_ret_50,
    ret_10: sum.cum2d_ret_10,
    ret_90: sum.cum2d_ret_90,
    price_10: sum.price_2d_10,
    price_50: sum.price_2d_50,
    price_90: sum.price_2d_90,
    medPct,
    spreadPct: spVal,
    skew: sum.skew_2d,
  };
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

function fmtPct(logRet, signed = true) {
  if (logRet === null || logRet === undefined || isNaN(logRet)) {
    return "--";
  }
  const pct = (Math.exp(Number(logRet)) - 1) * 100;
  const sign = pct >= 0 ? "+" : "";
  return signed ? `${sign}${pct.toFixed(2)}%` : `${pct.toFixed(2)}%`;
}

function get2dSpreadPct(item) {
  const sum = item?.latest_summary || {};
  let sp = sum.volatility_spread_2d;
  if (sp === null || sp === undefined || isNaN(sp)) {
    if (sum.cum2d_ret_90 !== undefined && sum.cum2d_ret_10 !== undefined) {
      sp = Number(sum.cum2d_ret_90) - Number(sum.cum2d_ret_10);
    } else {
      return null;
    }
  }
  return Math.round(Number(sp) * 100 * 10000) / 10000;
}

function formatDate(val) {
  if (!val) return "--";
  const s = String(val).trim();
  if (s.length >= 10 && /^\d{4}-\d{2}-\d{2}/.test(s)) {
    return s.slice(0, 10);
  }
  const d = new Date(val);
  if (!isNaN(d.getTime())) {
    const yyyy = d.getFullYear();
    const mm = String(d.getMonth() + 1).padStart(2, "0");
    const dd = String(d.getDate()).padStart(2, "0");
    return `${yyyy}-${mm}-${dd}`;
  }
  return s.slice(0, 10);
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
      else html.push(`<span class="badge badge-universe">${escapeHtml(clean)}</span>`);
    });
  }
  return html.join(" ");
}

/* ==================== DASHBOARD LOADING & RENDERING ==================== */

async function loadOverview() {
  try {
    const sourceFilter = document.getElementById("predHistorySourceSelect")?.value || "all";
    const modelFilter = document.getElementById("trackedModelFilterSelect")?.value || "all";
    const data = await apiFetch(`/api/overview?source=${encodeURIComponent(sourceFilter)}&model_run=${encodeURIComponent(modelFilter)}`);
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
  const counts = ov.counts || st.db_counts || {};
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

  populateTrackedFilterDropdowns();
  renderTrackedStocksTable();
  renderScreeningHistory();
  renderPredictionHistory();
}

function get2dMedianPct(item) {
  const sum = item?.latest_summary || {};
  const logRet = sum.cum2d_ret_50;
  if (logRet === null || logRet === undefined || isNaN(logRet)) {
    return null;
  }
  const pct = (Math.exp(Number(logRet)) - 1) * 100;
  return Math.round(pct * 10000) / 10000;
}

function populateTrackedFilterDropdowns() {
  const stocks = AppState.overview?.tracked_stocks || [];

  // Sector dropdown
  const sectorSelect = document.getElementById("filterTrackedSector");
  if (sectorSelect) {
    const currentVal = sectorSelect.value || "all";
    const sectorSet = new Set();
    stocks.forEach((item) => {
      if (item.sector) {
        sectorSet.add(item.sector);
      }
    });

    const sortedSectors = Array.from(sectorSet).sort();
    let optsHtml = '<option value="all">All Sectors</option>';
    sortedSectors.forEach((sec) => {
      optsHtml += `<option value="${sec}">${sec}</option>`;
    });

    sectorSelect.innerHTML = optsHtml;
    if (Array.from(sectorSelect.options).some((o) => o.value === currentVal)) {
      sectorSelect.value = currentVal;
    } else {
      sectorSelect.value = "all";
    }
  }

  // Predictions / Hits dropdown
  const predsSelect = document.getElementById("filterTrackedPredsHits");
  if (predsSelect) {
    const currentVal = predsSelect.value || "all";
    let hasManual = false;
    let hasScreen = false;
    const catSet = new Set();

    stocks.forEach((item) => {
      if (item.manual_count > 0) hasManual = true;
      if (item.screen_hit_count > 0) hasScreen = true;
      (item.screen_categories || []).forEach((cat) => {
        cat.split(",").forEach((c) => {
          const cl = c.trim().toLowerCase();
          if (cl) catSet.add(cl);
        });
      });
    });

    let optsHtml = '<option value="all">All Values</option>';
    if (hasManual) {
      optsHtml += '<option value="manual">Manual Preds</option>';
    }
    if (hasScreen) {
      optsHtml += '<option value="screen">Screen Hits</option>';
    }
    Array.from(catSet)
      .sort()
      .forEach((cat) => {
        const label = cat.charAt(0).toUpperCase() + cat.slice(1);
        optsHtml += `<option value="cat:${cat}">${label} Hits</option>`;
      });

    predsSelect.innerHTML = optsHtml;
    if (Array.from(predsSelect.options).some((o) => o.value === currentVal)) {
      predsSelect.value = currentVal;
    } else {
      predsSelect.value = "all";
    }
  }

  // Latest Anchor Date dropdown
  const anchorSelect = document.getElementById("filterTrackedAnchorDate");
  if (anchorSelect) {
    const currentVal = anchorSelect.value || "all";
    const dateSet = new Set();
    stocks.forEach((item) => {
      if (item.latest_anchor_date) {
        dateSet.add(formatDate(item.latest_anchor_date));
      }
    });

    const sortedDates = Array.from(dateSet).sort().reverse();
    let optsHtml = '<option value="all">All Dates</option>';
    sortedDates.forEach((d) => {
      optsHtml += `<option value="${d}">${d}</option>`;
    });

    anchorSelect.innerHTML = optsHtml;
    if (Array.from(anchorSelect.options).some((o) => o.value === currentVal)) {
      anchorSelect.value = currentVal;
    } else {
      anchorSelect.value = "all";
    }
  }
}

function setTrackedHorizon(horizon) {
  AppState.activeHorizon = horizon || "2d";
  document.querySelectorAll("#trackedHorizonToggle button").forEach((b) => {
    b.classList.toggle("active", b.dataset.horizon === AppState.activeHorizon);
  });
  const thMetric = document.getElementById("thTrackedMetricTitle");
  if (thMetric) {
    thMetric.textContent = AppState.trackedViewMode === "volatility"
      ? `${AppState.activeHorizon} Volatility Spread`
      : `${AppState.activeHorizon} Median (50%)`;
  }
  renderTrackedStocksTable();
}

function setTrackedViewMode(mode) {
  AppState.trackedViewMode = mode;
  const btnDir = document.getElementById("btnModeDirectional");
  const btnVol = document.getElementById("btnModeVolatility");
  const thMetric = document.getElementById("thTrackedMetricTitle");
  const thRange = document.getElementById("thTrackedRangeTitle");
  const colFilterSelect = document.getElementById("filterTracked2dRange");
  const sortSelect = document.getElementById("filterTracked2dSort");
  const minInput = document.getElementById("filterTracked2dMin");
  const maxInput = document.getElementById("filterTracked2dMax");
  const customRow = document.getElementById("filterTracked2dCustomRow");
  const curHorizon = AppState.activeHorizon || "2d";

  if (mode === "volatility") {
    btnDir?.classList.remove("active");
    btnVol?.classList.add("active");
    if (thMetric) thMetric.textContent = `${curHorizon} Volatility Spread`;
    if (thRange) thRange.innerHTML = "Directional &amp; Combined % Band";

    if (colFilterSelect) {
      colFilterSelect.title = `Filter by ${curHorizon} volatility spread %`;
      colFilterSelect.innerHTML = `
        <option value="all">All Spreads</option>
        <option value="vol_gt_6">&gt; 6.0% (High Vol)</option>
        <option value="vol_3_6">3.0% to 6.0% (Mid Vol)</option>
        <option value="vol_lt_3">&lt; 3.0% (Low Vol / Pin)</option>
        <option value="vol_gt_8">&gt; 8.0% (Extreme Move)</option>
        <option value="vol_4_8">4.0% to 8.0%</option>
        <option value="custom">Custom Spread...</option>
      `;
    }
    if (sortSelect) {
      sortSelect.title = `Sort by ${curHorizon} volatility spread`;
      const cur = sortSelect.value;
      sortSelect.innerHTML = `
        <option value="none">Sort: Default</option>
        <option value="desc">Sort: Widest Spread (High → Low)</option>
        <option value="asc">Sort: Narrowest Spread (Low → High)</option>
      `;
      sortSelect.value = cur;
    }
    if (minInput) minInput.placeholder = "Min Spread %";
    if (maxInput) maxInput.placeholder = "Max Spread %";
  } else {
    btnDir?.classList.add("active");
    btnVol?.classList.remove("active");
    if (thMetric) thMetric.textContent = `${curHorizon} Median (50%)`;
    if (thRange) thRange.textContent = "80% Range [10%–90%]";

    if (colFilterSelect) {
      colFilterSelect.title = `Filter by ${curHorizon} return range`;
      colFilterSelect.innerHTML = `
        <option value="all">All Ranges</option>
        <option value="0.5_1.0">0.50% to 1.00%</option>
        <option value="pos">Positive (&gt; 0%)</option>
        <option value="neg">Negative (&lt; 0%)</option>
        <option value="gt_2">&gt; +2.00%</option>
        <option value="1.0_2.0">+1.00% to +2.00%</option>
        <option value="0.0_0.5">0.00% to +0.50%</option>
        <option value="m0.5_0.0">-0.50% to 0.00%</option>
        <option value="m1.0_m0.5">-1.00% to -0.50%</option>
        <option value="lt_m1">&lt; -1.00%</option>
        <option value="custom">Custom Range...</option>
      `;
    }
    if (sortSelect) {
      sortSelect.title = `Sort by ${curHorizon} Median return`;
      const cur = sortSelect.value;
      sortSelect.innerHTML = `
        <option value="none">Sort: Default</option>
        <option value="desc">Sort: High → Low (Desc)</option>
        <option value="asc">Sort: Low → High (Asc)</option>
      `;
      sortSelect.value = cur;
    }
    if (minInput) minInput.placeholder = "Min %";
    if (maxInput) maxInput.placeholder = "Max %";
  }

  if (colFilterSelect) colFilterSelect.value = "all";
  if (minInput) minInput.value = "";
  if (maxInput) maxInput.value = "";
  if (customRow) customRow.style.display = "none";

  renderTrackedStocksTable();
}

function toggleSort2d() {
  const sortSelect = document.getElementById("filterTracked2dSort");
  const cur = sortSelect?.value || "none";
  let next = "desc";
  if (cur === "desc") next = "asc";
  else if (cur === "asc") next = "none";
  else next = "desc";

  if (sortSelect) sortSelect.value = next;
  renderTrackedStocksTable();
}

function clearTrackedTableFilters() {
  const t = document.getElementById("filterTrackedTicker");
  const n = document.getElementById("filterTrackedName");
  const sec = document.getElementById("filterTrackedSector");
  const m = document.getElementById("filterTrackedModel");
  const tm = document.getElementById("trackedModelFilterSelect");
  const p = document.getElementById("filterTrackedPredsHits");
  const a = document.getElementById("filterTrackedAnchorDate");
  const r = document.getElementById("filterTracked2dRange");
  const s = document.getElementById("filterTracked2dSort");
  const minEl = document.getElementById("filterTracked2dMin");
  const maxEl = document.getElementById("filterTracked2dMax");
  const customRow = document.getElementById("filterTracked2dCustomRow");
  const topF = document.getElementById("trackedFilterSelect");
  const topS = document.getElementById("trackedSearchInput");

  if (t) t.value = "";
  if (n) n.value = "";
  if (sec) sec.value = "all";
  if (m) m.value = "all";
  if (tm) tm.value = "all";
  if (p) p.value = "all";
  if (a) a.value = "all";
  if (r) r.value = "all";
  if (s) s.value = "none";
  if (minEl) minEl.value = "";
  if (maxEl) maxEl.value = "";
  if (customRow) customRow.style.display = "none";
  if (topF) topF.value = "all";
  if (topS) topS.value = "";

  renderTrackedStocksTable();
}

function renderTrackedStocksTable() {
  const tbody = document.getElementById("trackedStocksTbody");
  if (!tbody || !AppState.overview) return;

  const allStocks = AppState.overview.tracked_stocks || [];
  const isVolMode = AppState.trackedViewMode === "volatility";
  const curHorizon = AppState.activeHorizon || "2d";

  const thMetric = document.getElementById("thTrackedMetricTitle");
  if (thMetric) {
    thMetric.textContent = isVolMode ? `${curHorizon} Volatility Spread` : `${curHorizon} Median (50%)`;
  }

  // Column filter inputs
  const tickerQ = (document.getElementById("filterTrackedTicker")?.value || "").trim().toUpperCase();
  const nameQ = (document.getElementById("filterTrackedName")?.value || "").trim().toUpperCase();
  const sectorFilter = document.getElementById("filterTrackedSector")?.value || "all";
  const modelFilter = document.getElementById("filterTrackedModel")?.value || document.getElementById("trackedModelFilterSelect")?.value || "all";
  const predsFilter = document.getElementById("filterTrackedPredsHits")?.value || "all";
  const anchorFilter = document.getElementById("filterTrackedAnchorDate")?.value || "all";

  // Range & Custom inputs
  const range2d = document.getElementById("filterTracked2dRange")?.value || "all";
  const minInputVal = document.getElementById("filterTracked2dMin")?.value?.trim();
  const maxInputVal = document.getElementById("filterTracked2dMax")?.value?.trim();

  let minBound = null;
  let maxBound = null;

  if (isVolMode) {
    if (range2d === "vol_gt_6") {
      minBound = 6.0;
    } else if (range2d === "vol_3_6") {
      minBound = 3.0;
      maxBound = 6.0;
    } else if (range2d === "vol_lt_3") {
      maxBound = 3.0;
    } else if (range2d === "vol_gt_8") {
      minBound = 8.0;
    } else if (range2d === "vol_4_8") {
      minBound = 4.0;
      maxBound = 8.0;
    } else if (range2d === "custom") {
      if (minInputVal !== "" && minInputVal !== undefined && !isNaN(Number(minInputVal))) {
        minBound = Number(minInputVal);
      }
      if (maxInputVal !== "" && maxInputVal !== undefined && !isNaN(Number(maxInputVal))) {
        maxBound = Number(maxInputVal);
      }
    }
  } else {
    if (range2d === "0.5_1.0") {
      minBound = 0.50;
      maxBound = 1.00;
    } else if (range2d === "pos") {
      minBound = 0.00;
    } else if (range2d === "neg") {
      maxBound = 0.00;
    } else if (range2d === "gt_2") {
      minBound = 2.00;
    } else if (range2d === "1.0_2.0") {
      minBound = 1.00;
      maxBound = 2.00;
    } else if (range2d === "0.0_0.5") {
      minBound = 0.00;
      maxBound = 0.50;
    } else if (range2d === "m0.5_0.0") {
      minBound = -0.50;
      maxBound = 0.00;
    } else if (range2d === "m1.0_m0.5") {
      minBound = -1.00;
      maxBound = -0.50;
    } else if (range2d === "lt_m1") {
      maxBound = -1.00;
    } else if (range2d === "custom") {
      if (minInputVal !== "" && minInputVal !== undefined && !isNaN(Number(minInputVal))) {
        minBound = Number(minInputVal);
      }
      if (maxInputVal !== "" && maxInputVal !== undefined && !isNaN(Number(maxInputVal))) {
        maxBound = Number(maxInputVal);
      }
    }
  }

  // Top header filter controls (legacy / complementary)
  const topFilterCat = document.getElementById("trackedFilterSelect")?.value || "all";
  const topSearchQ = (document.getElementById("trackedSearchInput")?.value || "").trim().toUpperCase();

  const list = allStocks.filter((item) => {
    // Model filter
    if (modelFilter !== "all") {
      const rId = item.latest_run_id || "default";
      if (rId !== modelFilter) return false;
    }

    // 1. Ticker column search box
    if (tickerQ) {
      const match =
        (item.ticker || "").toUpperCase().includes(tickerQ) ||
        (item.universe || "").toUpperCase().includes(tickerQ);
      if (!match) return false;
    }

    // 2. Name column search box
    if (nameQ) {
      const match = (item.name || "").toUpperCase().includes(nameQ);
      if (!match) return false;
    }

    // 3. Sector column dropdown
    if (sectorFilter !== "all") {
      if (item.sector !== sectorFilter) return false;
    }

    // 4. Predictions / Hits column dropdown
    if (predsFilter !== "all") {
      if (predsFilter === "manual" && item.manual_count <= 0) return false;
      if (predsFilter === "screen" && item.screen_hit_count <= 0) return false;
      if (predsFilter.startsWith("cat:")) {
        const catKey = predsFilter.slice(4).toLowerCase();
        const hasCat = (item.screen_categories || []).some((c) =>
          c.toLowerCase().includes(catKey)
        );
        if (!hasCat) return false;
      }
    }

    // Top filter category
    if (topFilterCat !== "all") {
      if (topFilterCat === "manual" && item.manual_count <= 0) return false;
      if (topFilterCat === "screen" && item.screen_hit_count <= 0) return false;
      if (
        ["bullish", "bearish", "volatility", "index"].includes(topFilterCat) &&
        !(item.screen_categories || []).some((c) => c.toLowerCase().includes(topFilterCat))
      ) {
        return false;
      }
    }

    // 5. Latest Anchor date dropdown
    if (anchorFilter !== "all") {
      if (formatDate(item.latest_anchor_date) !== anchorFilter) return false;
    }

    // Top search input
    if (topSearchQ) {
      const match =
        (item.ticker || "").toUpperCase().includes(topSearchQ) ||
        (item.name || "").toUpperCase().includes(topSearchQ) ||
        (item.sector || "").toUpperCase().includes(topSearchQ);
      if (!match) return false;
    }

    // 6. Metric Range filter for currently selected horizon
    if (minBound !== null || maxBound !== null) {
      const hMetrics = getHorizonMetrics(item, curHorizon);
      const val = isVolMode ? hMetrics.spreadPct : hMetrics.medPct;
      if (val === null) return false;
      if (minBound !== null && val < minBound) return false;
      if (maxBound !== null && val > maxBound) return false;
    }

    return true;
  });

  // Sort by Metric (Median in directional mode, Spread in volatility mode)
  const sort2d = document.getElementById("filterTracked2dSort")?.value || "none";
  if (sort2d === "desc") {
    list.sort((a, b) => {
      const hA = getHorizonMetrics(a, curHorizon);
      const hB = getHorizonMetrics(b, curHorizon);
      const vA = isVolMode ? hA.spreadPct : hA.medPct;
      const vB = isVolMode ? hB.spreadPct : hB.medPct;
      if (vA === null && vB === null) return 0;
      if (vA === null) return 1;
      if (vB === null) return -1;
      return vB - vA;
    });
  } else if (sort2d === "asc") {
    list.sort((a, b) => {
      const hA = getHorizonMetrics(a, curHorizon);
      const hB = getHorizonMetrics(b, curHorizon);
      const vA = isVolMode ? hA.spreadPct : hA.medPct;
      const vB = isVolMode ? hB.spreadPct : hB.medPct;
      if (vA === null && vB === null) return 0;
      if (vA === null) return 1;
      if (vB === null) return -1;
      return vA - vB;
    });
  }

  // Update sort indicator icon in header
  const sortIcon = document.getElementById("sortIndicator2d");
  if (sortIcon) {
    if (sort2d === "desc") {
      sortIcon.textContent = "▼";
      sortIcon.title = isVolMode ? `Sorted: Widest ${curHorizon} Spread First` : `Sorted: Highest ${curHorizon} Return First`;
    } else if (sort2d === "asc") {
      sortIcon.textContent = "▲";
      sortIcon.title = isVolMode ? `Sorted: Narrowest ${curHorizon} Spread First` : `Sorted: Lowest ${curHorizon} Return First`;
    } else {
      sortIcon.textContent = "⇅";
      sortIcon.title = "Click to toggle sort";
    }
  }

  // Update filter count badge
  const badgeEl = document.getElementById("trackedFilterCountBadge");
  if (badgeEl) {
    badgeEl.style.display = "inline-block";
    if (list.length < allStocks.length) {
      badgeEl.textContent = `Showing ${list.length} of ${allStocks.length}`;
      badgeEl.className = "badge badge-manual tracked-count-badge";
    } else {
      badgeEl.textContent = `${allStocks.length} tracked`;
      badgeEl.className = "badge badge-universe tracked-count-badge";
    }
  }

  if (!list.length) {
    tbody.innerHTML = `
      <tr>
        <td colspan="10" style="text-align:center; padding: 24px;" class="text-muted">
          No tracked stocks match the current filters.
          <button type="button" class="btn btn-xs btn-secondary" style="margin-left:8px;" onclick="clearTrackedTableFilters()">Reset Filters</button>
        </td>
      </tr>`;
    return;
  }

  tbody.innerHTML = list
    .map((item) => {
      const hMetrics = getHorizonMetrics(item, curHorizon);
      const p10 = hMetrics.price_10 ? `$${Number(hMetrics.price_10).toFixed(2)}` : "--";
      const p90 = hMetrics.price_90 ? `$${Number(hMetrics.price_90).toFixed(2)}` : "--";
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

      const safeTicker = escapeHtml(item.ticker);
      const safeName = escapeHtml(item.name || "--");
      const safeSector = escapeHtml(item.sector || "--");
      const safeUniverse = escapeHtml(item.universe || "");
      const predId = item.latest_prediction_id ? Number(item.latest_prediction_id) : "null";

      let metricCell = "";
      let rangeCell = "";

      if (isVolMode) {
        const spreadVal = hMetrics.spreadPct;
        const spreadStr = spreadVal !== null ? `${spreadVal.toFixed(2)}% spread` : "--";
        const halfMoveStr = spreadVal !== null ? `±${(spreadVal / 2).toFixed(2)}%` : "";
        const r10 = fmtPct(hMetrics.ret_10);
        const r90 = fmtPct(hMetrics.ret_90);

        metricCell = `
          <div class="vol-metric-cell">
            <span class="badge badge-volatility font-mono">${spreadStr}</span>
            ${halfMoveStr ? `<span class="text-muted range-span-sm font-mono" title="Implied ± Expected Move">${halfMoveStr}</span>` : ""}
          </div>`;

        rangeCell = `
          <div class="vol-band-cell">
            <div class="vol-band-ret font-mono">
              <span class="ret-down" title="Downside 10% Bound">${r10}</span>
              <span class="text-muted" style="font-weight:400;margin:0 2px;">to</span>
              <span class="ret-up" title="Upside 90% Bound">${r90}</span>
            </div>
            <div class="range-span-sm font-mono text-muted">${p10} – ${p90}</div>
          </div>`;
      } else {
        const skewStr = hMetrics.skew ? `<span class="skew-indicator" title="Upside/Downside Skew Ratio">${Number(hMetrics.skew).toFixed(2)}x</span>` : "";
        metricCell = `<div class="col-nowrap">${fmtRetBadge(hMetrics.ret_50)}${skewStr}</div>`;
        rangeCell = `<span class="text-muted range-span font-mono">${p10} – ${p90}</span>`;
      }

      return `
        <tr class="clickable-row" onclick="openStockDetail('${safeTicker}', ${predId})">
          <td>
            <div class="ticker-cell">
              <span class="ticker-symbol">${safeTicker}</span>
              <span class="badge badge-universe ticker-sub-badge">${safeUniverse}</span>
            </div>
          </td>
          <td class="col-truncate" title="${safeName}">${safeName}</td>
          <td class="col-truncate" title="${safeSector}"><span class="badge badge-sector">${safeSector}</span></td>
          <td class="col-nowrap">${renderModelBadge(item.latest_run_id)}</td>
          <td>${badges.join(" ")}</td>
          <td class="col-nowrap">${formatDate(item.latest_anchor_date)}</td>
          <td class="col-nowrap">$${Number(item.latest_anchor_close).toFixed(2)}</td>
          <td class="col-nowrap">${metricCell}</td>
          <td class="col-nowrap">${rangeCell}</td>
          <td onclick="event.stopPropagation()">
            <div class="action-cell">
              <button class="btn btn-xs active" onclick="openStockDetail('${safeTicker}', ${predId})">Chart &amp; Cones</button>
              <button class="btn btn-xs" onclick="quickRunPrediction('${safeTicker}')">Predict</button>
            </div>
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
      (r) => {
        const safeTicker = escapeHtml(r.ticker);
        const safeName = escapeHtml(r.name || "--");
        const safeSector = escapeHtml(r.sector || "--");
        const safeUniverse = escapeHtml(r.universe || "--");
        return `
      <tr class="clickable-row" onclick="openStockDetail('${safeTicker}')">
        <td class="col-nowrap"><strong>${safeTicker}</strong></td>
        <td class="col-truncate" title="${safeName}">${safeName}</td>
        <td class="col-truncate"><span class="badge badge-sector">${safeSector}</span></td>
        <td class="col-nowrap"><span class="badge badge-universe">${safeUniverse}</span></td>
        <td class="col-nowrap">
          ${
            r.total_predictions > 0
              ? `<span class="badge badge-manual">${Number(r.total_predictions)} stored</span>`
              : `<span class="text-muted">None yet</span>`
          }
        </td>
        <td class="col-nowrap" onclick="event.stopPropagation()">
          <div class="row-actions">
            <button class="btn btn-xs active" onclick="openStockDetail('${safeTicker}')">View Chart</button>
            <button class="btn btn-xs" onclick="quickRunPrediction('${safeTicker}')">Run Prediction</button>
          </div>
        </td>
      </tr>`;
      }
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
              ${renderModelBadge(run.run_id)}
              <span class="text-muted" style="font-size:11.5px;">Top ${run.top_n} • Min $${Number(run.min_price).toFixed(2)} • ${run.total_screened} screened (${run.duration_sec}s)</span>
            </div>
            <span class="text-muted" style="font-size:11px;">${formatDate(run.created_at)}</span>
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
    `Executed ${formatDate(run.created_at)} • Screened ${run.total_screened}/${run.total_candidates} instruments in ${run.duration_sec}s`;

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

  const theadRow = document.getElementById("screenInspectorTheadRow");
  const tbody = document.getElementById("screenInspectorTbody");

  if (tab === "volatility") {
    if (theadRow) {
      theadRow.innerHTML = `
        <th>Ticker</th>
        <th>Name</th>
        <th>Last Px</th>
        <th>2-Day Spread</th>
        <th>Downside (10%)</th>
        <th>Median (50%)</th>
        <th>Upside (90%)</th>
        <th>Action</th>
      `;
    }
    if (!rows.length) {
      tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;padding:14px;" class="text-muted">No instruments in this category.</td></tr>`;
      return;
    }
    tbody.innerHTML = rows
      .map((r) => {
        const spreadVal =
          r.volatility_spread !== undefined && r.volatility_spread !== null
            ? (Number(r.volatility_spread) * 100).toFixed(2)
            : "--";
        const p10Str = r.price_10
          ? `$${Number(r.price_10).toFixed(2)} (${fmtRetBadge(r.cum2d_ret_10)})`
          : "--";
        const p50Str = r.price_50
          ? `$${Number(r.price_50).toFixed(2)} (${fmtRetBadge(r.cum2d_ret_50)})`
          : "--";
        const p90Str = r.price_90
          ? `$${Number(r.price_90).toFixed(2)} (${fmtRetBadge(r.cum2d_ret_90)})`
          : "--";
        const safeTicker = escapeHtml(r.ticker);
        const safeName = escapeHtml(r.name || r.ticker || "--");
        return `
        <tr class="clickable-row" onclick="openStockDetail('${safeTicker}')">
          <td class="col-nowrap"><strong>${safeTicker}</strong></td>
          <td class="col-truncate" title="${safeName}">${safeName}</td>
          <td class="col-nowrap">$${Number(r.last_close || 0).toFixed(2)}</td>
          <td class="col-nowrap"><span class="badge badge-volatility font-mono">${spreadVal}% spread</span></td>
          <td class="col-nowrap font-mono">${p10Str}</td>
          <td class="col-nowrap font-mono">${p50Str}</td>
          <td class="col-nowrap font-mono">${p90Str}</td>
          <td class="col-nowrap" onclick="event.stopPropagation()">
            <div class="row-actions">
              <button class="btn btn-xs active" onclick="openStockDetail('${safeTicker}')">View Cone</button>
            </div>
          </td>
        </tr>`;
      })
      .join("");
  } else {
    if (theadRow) {
      theadRow.innerHTML = `
        <th>Ticker</th>
        <th>Name</th>
        <th>Last Px</th>
        <th>1d (50%)</th>
        <th>2d (50%)</th>
        <th>80% Range [10%–90%]</th>
        <th>20d (50%)</th>
        <th>Action</th>
      `;
    }
    if (!rows.length) {
      tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;padding:14px;" class="text-muted">No instruments in this category.</td></tr>`;
      return;
    }
    tbody.innerHTML = rows
      .map(
        (r) => {
          const safeTicker = escapeHtml(r.ticker);
          const safeName = escapeHtml(r.name || r.ticker || "--");
          return `
        <tr class="clickable-row" onclick="openStockDetail('${safeTicker}')">
          <td class="col-nowrap"><strong>${safeTicker}</strong></td>
          <td class="col-truncate" title="${safeName}">${safeName}</td>
          <td class="col-nowrap">$${Number(r.last_close || 0).toFixed(2)}</td>
          <td class="col-nowrap">${fmtRetBadge(r.day1_ret_50)}</td>
          <td class="col-nowrap">${fmtRetBadge(r.cum2d_ret_50)}</td>
          <td class="text-muted col-nowrap range-span">$${Number(r.price_10 || 0).toFixed(2)} – $${Number(r.price_90 || 0).toFixed(2)}</td>
          <td class="col-nowrap">${fmtRetBadge(r.cum20d_ret_50)}</td>
          <td class="col-nowrap" onclick="event.stopPropagation()">
            <div class="row-actions">
              <button class="btn btn-xs active" onclick="openStockDetail('${safeTicker}')">View Cone</button>
            </div>
          </td>
        </tr>`;
        }
      )
      .join("");
  }
}

function renderPredictionHistory() {
  const tbody = document.getElementById("predHistoryTbody");
  if (!tbody || !AppState.overview) return;
  const preds = AppState.overview.prediction_history || [];
  const modelFilter = document.getElementById("predHistoryModelSelect")?.value || "all";

  const filtered = preds.filter((p) => {
    if (modelFilter !== "all" && (p.run_id || "default") !== modelFilter) return false;
    return true;
  });

  if (!filtered.length) {
    tbody.innerHTML = `<tr><td colspan="9" style="text-align:center;padding:18px;" class="text-muted">No predictions match current filters.</td></tr>`;
    return;
  }

  tbody.innerHTML = filtered
    .map((p) => {
      const sum = p.summary || {};
      const safeTicker = escapeHtml(p.ticker);
      const predId = Number(p.id);
      return `
        <tr class="clickable-row" onclick="openStockDetail('${safeTicker}', ${predId})">
          <td class="text-muted col-nowrap">${formatDate(p.created_at)}</td>
          <td class="col-nowrap"><strong>${safeTicker}</strong></td>
          <td class="col-nowrap">${renderModelBadge(p.run_id)}</td>
          <td>${renderCategoryBadges(p.source, p.screen_category)}</td>
          <td class="col-nowrap">${formatDate(p.anchor_date)}</td>
          <td class="col-nowrap">$${Number(p.anchor_close).toFixed(2)}</td>
          <td class="col-nowrap">${fmtRetBadge(sum.cum2d_ret_50)}</td>
          <td class="col-nowrap">${fmtRetBadge(sum.cum20d_ret_50)}</td>
          <td class="col-nowrap" onclick="event.stopPropagation()">
            <div class="row-actions">
              <button class="btn btn-xs active" onclick="openStockDetail('${safeTicker}', ${predId})">Overlay</button>
            </div>
          </td>
        </tr>`;
    })
    .join("");
}

/* ==================== STOCK DETAIL VIEW & PREDICTION SIDE MENU ==================== */

let stockDetailRequestSeq = 0;

async function openStockDetail(ticker, preferredPredId = null) {
  if (!ticker) return;
  const sym = ticker.trim().toUpperCase();
  const seq = ++stockDetailRequestSeq;
  try {
    const data = await apiFetch(`/api/stock/${encodeURIComponent(sym)}?bars=360&auto_fetch=1`);
    if (seq !== stockDetailRequestSeq) return;
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
    if (seq !== stockDetailRequestSeq) return;
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
      `Last bar: ${formatDate(st.latest_bar?.date)} (${st.bars.length} bars loaded)`;
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
            <span class="pred-item-anchor">Anchor: ${formatDate(p.anchor_date)}</span>
            <div style="display:flex;align-items:center;gap:4px;">
              ${renderModelBadge(p.run_id)}
              ${renderCategoryBadges(p.source, p.screen_category)}
            </div>
          </div>
          <div class="text-muted" style="font-size:11px;">
            Run #${p.id} • ${formatDate(p.created_at)} • Anchor Px: $${Number(p.anchor_close).toFixed(2)}
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

  mSub.textContent = `Overlaying Prediction #${activePred.id} (Anchor Date: ${formatDate(activePred.anchor_date)} @ $${Number(activePred.anchor_close).toFixed(2)}) onto ${st.ticker} market price chart.`;
  cSub.textContent = `Point-in-Time Probability Cone made on ${formatDate(activePred.anchor_date)} ($${Number(activePred.anchor_close).toFixed(2)}) vs. actual ${st.ticker} price performance (${activePred.realized?.elapsed_trading_days || 0} trading days elapsed).`;

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

      // 50% Core [25%-75%]
      const c25Price = p.q25 !== undefined ? `$${Number(p.q25).toFixed(2)}` : "--";
      const c75Price = p.q75 !== undefined ? `$${Number(p.q75).toFixed(2)}` : "--";
      const c25Ret = fmtPct(r.q25);
      const c75Ret = fmtPct(r.q75);
      const coreSpread = (r.q25 !== undefined && r.q75 !== undefined)
        ? (Math.exp(Number(r.q75)) - Math.exp(Number(r.q25))) * 100
        : (m.spread_50 ? Number(m.spread_50) * 100 : null);
      const coreSpreadStr = coreSpread !== null ? `${coreSpread.toFixed(2)}% spd` : "";

      // 80% Std [10%-90%]
      const s10Price = p.q10 !== undefined ? `$${Number(p.q10).toFixed(2)}` : "--";
      const s90Price = p.q90 !== undefined ? `$${Number(p.q90).toFixed(2)}` : "--";
      const s10Ret = fmtPct(r.q10);
      const s90Ret = fmtPct(r.q90);
      const stdSpread = (r.q10 !== undefined && r.q90 !== undefined)
        ? (Math.exp(Number(r.q90)) - Math.exp(Number(r.q10))) * 100
        : (m.spread_80 ? Number(m.spread_80) * 100 : null);
      const stdSpreadStr = stdSpread !== null ? `${stdSpread.toFixed(2)}% spd` : "";

      // 90% Tail [5%-95%]
      const t05Price = p.q05 !== undefined ? `$${Number(p.q05).toFixed(2)}` : "--";
      const t95Price = p.q95 !== undefined ? `$${Number(p.q95).toFixed(2)}` : "--";
      const t05Ret = fmtPct(r.q05);
      const t95Ret = fmtPct(r.q95);

      return `
        <tr>
          <td class="col-nowrap"><strong>Day +${m.horizon_days}</strong></td>
          <td class="col-nowrap">${formatDate(m.target_date)}</td>
          <td class="col-nowrap">$${Number(p.q50).toFixed(2)} (${fmtRetBadge(r.q50)})</td>
          <td class="col-nowrap milestone-band-cell">
            <div class="band-px font-mono"><strong>${c25Price} – ${c75Price}</strong></div>
            <div class="band-sub">
              <span class="sub-range font-mono">(${c25Ret} to ${c75Ret})</span>
              ${coreSpreadStr ? `<span class="badge badge-sub-spread font-mono">${coreSpreadStr}</span>` : ""}
            </div>
          </td>
          <td class="col-nowrap milestone-band-cell">
            <div class="band-px font-mono">${s10Price} – ${s90Price}</div>
            <div class="band-sub">
              <span class="sub-range font-mono">(${s10Ret} to ${s90Ret})</span>
              ${stdSpreadStr ? `<span class="badge badge-sub-std font-mono">${stdSpreadStr}</span>` : ""}
            </div>
          </td>
          <td class="col-nowrap milestone-band-cell">
            <div class="band-px font-mono text-muted">${t05Price} – ${t95Price}</div>
            <div class="band-sub">
              <span class="sub-range font-mono" style="opacity:0.85;">(${t05Ret} to ${t95Ret})</span>
            </div>
          </td>
          <td class="col-nowrap">${m.is_realized && m.actual_close !== null ? `<strong>$${Number(m.actual_close).toFixed(2)}</strong>` : '<span class="text-muted">Unrealized</span>'}</td>
          <td class="col-nowrap">${m.is_realized ? fmtRetBadge(m.actual_return) : '<span class="text-muted">--</span>'}</td>
          <td class="col-nowrap">${statusBadge}</td>
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
  const rawDate = document.getElementById("detailAsOfDate")?.value?.trim() || "";
  let asOfDate = null;
  if (rawDate) {
    if (!/^\d{4}-\d{2}-\d{2}$/.test(rawDate)) {
      showToast("Point-in-Time Date must be in YYYY-MM-DD format (e.g. 2026-02-11)", true);
      return;
    }
    asOfDate = rawDate;
  }
  const numSamples = parseInt(document.getElementById("detailNumSamples")?.value || "1", 10);
  const mcDropout = Boolean(document.getElementById("detailMcDropout")?.checked);
  const modelRun = document.getElementById("detailModelSelect")?.value || null;

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
        model_run: modelRun,
      }),
    });
    showToast(
      `Prediction #${res.prediction.id} stored for ${st.ticker} (Anchor: ${formatDate(res.prediction.anchor_date)})`
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
  const modelRun = document.getElementById("modalScreenModel")?.value || null;

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
        model_run: modelRun,
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
  const rawDate = document.getElementById("modalPredictAsOfDate")?.value?.trim() || "";
  let asOfDate = null;
  if (rawDate) {
    if (!/^\d{4}-\d{2}-\d{2}$/.test(rawDate)) {
      showToast("Point-in-Time Date must be in YYYY-MM-DD format (e.g. 2026-02-11)", true);
      return;
    }
    asOfDate = rawDate;
  }
  const numSamples = parseInt(document.getElementById("modalPredictSamples").value || "1", 10);
  const mcDropout = Boolean(document.getElementById("modalPredictMcDropout").checked);
  const refreshData = Boolean(document.getElementById("modalPredictRefreshData").checked);
  const modelRun = document.getElementById("modalPredictModel")?.value || null;

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
        model_run: modelRun,
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

function setupDateInputWrappers() {
  document.querySelectorAll(".date-input-wrap").forEach((wrap) => {
    const textInput = wrap.querySelector(".date-picker-text");
    const nativeInput = wrap.querySelector(".date-picker-native-overlay");
    const btn = wrap.querySelector(".btn-calendar-trigger");
    if (!textInput) return;

    if (nativeInput) {
      // Sync from native picker to text input
      const syncFromNative = () => {
        if (nativeInput.value) {
          textInput.value = nativeInput.value;
          textInput.dispatchEvent(new Event("input", { bubbles: true }));
          textInput.dispatchEvent(new Event("change", { bubbles: true }));
        }
      };
      nativeInput.addEventListener("change", syncFromNative);
      nativeInput.addEventListener("input", syncFromNative);

      // Sync from text input to native picker
      textInput.addEventListener("input", () => {
        let v = textInput.value.trim();
        // Auto-convert slashes YYYY/MM/DD -> YYYY-MM-DD
        if (/^\d{4}\/\d{2}\/\d{2}$/.test(v)) {
          v = v.replace(/\//g, "-");
          textInput.value = v;
        }
        // Auto-convert 8 digits YYYYMMDD -> YYYY-MM-DD
        else if (/^\d{8}$/.test(v)) {
          v = `${v.slice(0, 4)}-${v.slice(4, 6)}-${v.slice(6, 8)}`;
          textInput.value = v;
        }

        if (/^\d{4}-\d{2}-\d{2}$/.test(v)) {
          nativeInput.value = v;
        } else if (!v) {
          nativeInput.value = "";
        }
      });

      // Trigger button fallback
      if (btn) {
        btn.addEventListener("click", () => {
          if (typeof nativeInput.showPicker === "function") {
            try {
              nativeInput.showPicker();
            } catch (_) {
              nativeInput.focus();
            }
          } else {
            nativeInput.focus();
          }
        });
      }
    }
  });
}

/* ==================== VIEW SWITCHING & MODEL RUNS EXPLORER ==================== */

function showView(viewName) {
  AppState.view = viewName;
  const dashView = document.getElementById("dashboardView");
  const runsView = document.getElementById("modelRunsView");
  const stockView = document.getElementById("stockDetailView");
  const btnDash = document.getElementById("btnNavDashboard");
  const btnRuns = document.getElementById("btnNavModelRuns");

  dashView?.classList.add("hidden");
  runsView?.classList.add("hidden");
  stockView?.classList.add("hidden");
  btnDash?.classList.remove("active");
  btnRuns?.classList.remove("active");

  if (viewName === "dashboard") {
    dashView?.classList.remove("hidden");
    btnDash?.classList.add("active");
    loadOverview();
  } else if (viewName === "model_runs") {
    runsView?.classList.remove("hidden");
    btnRuns?.classList.add("active");
    renderModelRunsView();
  } else if (viewName === "stock") {
    stockView?.classList.remove("hidden");
  }
}

async function loadModels() {
  try {
    const res = await apiFetch("/api/models");
    AppState.availableModels = res.models || [];
    AppState.activeModelRunId = res.active_run_id || "default";

    // Populate #globalModelSelect
    const globalSel = document.getElementById("globalModelSelect");
    if (globalSel) {
      globalSel.innerHTML = AppState.availableModels
        .map((m) => {
          const actMark = m.run_id === AppState.activeModelRunId ? " (Active)" : "";
          const scoreStr = m.test_score ? ` [Score: ${m.test_score}]` : "";
          return `<option value="${m.run_id}" ${m.run_id === AppState.activeModelRunId ? "selected" : ""}>
            ${m.badge_label || m.run_id}: ${m.run_tag || m.run_id}${scoreStr}${actMark}
          </option>`;
        })
        .join("");
    }

    // Populate modal & detail dropdowns
    const screenSel = document.getElementById("modalScreenModel");
    const predSel = document.getElementById("modalPredictModel");
    const detailSel = document.getElementById("detailModelSelect");

    const modalOpts = AppState.availableModels
      .map((m) => {
        const actMark = m.run_id === AppState.activeModelRunId ? " (Active)" : "";
        return `<option value="${m.run_id}" ${m.run_id === AppState.activeModelRunId ? "selected" : ""}>
          ${m.badge_label || m.run_id} — ${m.run_tag || m.run_id}${actMark}
        </option>`;
      })
      .join("");

    if (screenSel) screenSel.innerHTML = modalOpts;
    if (predSel) predSel.innerHTML = modalOpts;
    if (detailSel) detailSel.innerHTML = modalOpts;

    // Populate table filter selects
    populateModelFilterDropdowns();

    if (AppState.view === "model_runs") {
      renderModelRunsView();
    }
  } catch (err) {
    console.error("Failed loading models:", err);
  }
}

function populateModelFilterDropdowns() {
  const models = AppState.availableModels || [];
  const trackedSel = document.getElementById("trackedModelFilterSelect");
  const filterModelSel = document.getElementById("filterTrackedModel");
  const predModelSel = document.getElementById("predHistoryModelSelect");

  let opts = '<option value="all">All Models</option>';
  models.forEach((m) => {
    opts += `<option value="${m.run_id}">${m.badge_label || m.run_id} (${m.run_tag || m.run_id})</option>`;
  });

  if (trackedSel) {
    const cur = trackedSel.value || "all";
    trackedSel.innerHTML = opts;
    if (Array.from(trackedSel.options).some((o) => o.value === cur)) trackedSel.value = cur;
  }
  if (filterModelSel) {
    const cur = filterModelSel.value || "all";
    filterModelSel.innerHTML = opts;
    if (Array.from(filterModelSel.options).some((o) => o.value === cur)) filterModelSel.value = cur;
  }
  if (predModelSel) {
    const cur = predModelSel.value || "all";
    predModelSel.innerHTML = opts;
    if (Array.from(predModelSel.options).some((o) => o.value === cur)) predModelSel.value = cur;
  }
}

async function selectActiveModel(runId, promoteToRoot = false) {
  if (!runId) return;
  try {
    showToast(`Switching active model to ${runId}...`);
    const res = await apiFetch("/api/models/select", {
      method: "POST",
      body: JSON.stringify({ run_id: runId, promote_to_root: Boolean(promoteToRoot) }),
    });
    AppState.activeModelRunId = res.active_run_id;
    showToast(`Active model updated: ${res.active_run_id}${res.promoted_to_root ? " (Promoted to root)" : ""}`);
    await loadModels();
    await loadOverview();
    if (AppState.view === "model_runs") {
      renderModelRunsView();
    }
  } catch (err) {
    showToast(`Model selection failed: ${err.message}`, true);
  }
}

function renderModelRunsView() {
  const tbody = document.getElementById("modelRunsLeaderboardTbody");
  if (!tbody) return;
  const models = AppState.availableModels || [];
  if (!models.length) {
    tbody.innerHTML = '<tr><td colspan="14" style="text-align:center;padding:24px;" class="text-muted">No model runs found in ML_stock/checkpoints.</td></tr>';
    return;
  }

  // Find lowest score - ignore runs where test_score <= 0.0 or score is missing/null (Issue 13)
  let bestScore = Infinity;
  models.forEach((m) => {
    if (m.test_score !== null && m.test_score !== undefined && !isNaN(m.test_score) && Number(m.test_score) > 0.0 && Number(m.test_score) < bestScore) {
      bestScore = Number(m.test_score);
    }
  });

  if (!AppState.selectedLeaderboardRunId) {
    AppState.selectedLeaderboardRunId = AppState.activeModelRunId || models[0].run_id;
  }

  tbody.innerHTML = models
    .map((m) => {
      const isSelected = m.run_id === AppState.selectedLeaderboardRunId;
      const isActive = m.run_id === AppState.activeModelRunId;
      const isBest = m.test_score !== null && m.test_score !== undefined && !isNaN(m.test_score) && Number(m.test_score) > 0.0 && Number(m.test_score) <= bestScore;

      const safeRunId = escapeHtml(m.run_id);
      const safeTag = escapeHtml(m.run_tag || m.run_id);
      const safeModelType = escapeHtml(m.model_type || "market_query");

      const statusHtml = isActive
        ? '<span class="active-run-indicator">● Active</span>'
        : '<span class="text-muted font-mono" style="font-size:11px;">Standby</span>';

      const headsStr = m.run_id.includes("3head") || (m.run_tag || "").includes("3head")
        ? '<span class="badge badge-model-3head">3-Head (1-1-1)</span>'
        : '<span class="badge badge-model-1head">1-Head</span>';

      const layersStr = `${m.num_layers || 4}L enc / ${m.market_layers || 4}L mkt`;
      const bestEpStr = m.best_epoch ? `Ep ${m.best_epoch}` : "--";
      const scoreStr = m.test_score !== null && m.test_score !== undefined
        ? (isBest ? `<span class="metric-pill-best">${m.test_score} ★</span>` : `<span class="metric-pill-val">${m.test_score}</span>`)
        : "--";
      const maeStr = m.test_mae !== null && m.test_mae !== undefined ? `${(Number(m.test_mae) * 100).toFixed(2)}%` : "--";
      const hitStr = m.test_dir_acc !== null && m.test_dir_acc !== undefined ? `${(Number(m.test_dir_acc) * 100).toFixed(1)}%` : "--";
      const icStr = m.test_ic !== null && m.test_ic !== undefined ? `${Number(m.test_ic) >= 0 ? "+" : ""}${Number(m.test_ic).toFixed(3)}` : "--";
      const covStr = m.test_cov80 !== null && m.test_cov80 !== undefined ? `${(Number(m.test_cov80) * 100).toFixed(1)}%` : "--";
      const calibMaeStr = m.test_calib_mae !== null && m.test_calib_mae !== undefined ? `${(Number(m.test_calib_mae) * 100).toFixed(2)}%` : "--";
      const durStr = m.duration_sec !== null && m.duration_sec !== undefined ? `${Number(m.duration_sec).toFixed(0)}s` : "--";

      return `
        <tr class="${isSelected ? "selected-run-row" : ""}" onclick="inspectModelRun('${safeRunId}')">
          <td class="col-nowrap">${statusHtml}</td>
          <td class="col-nowrap">
            <strong>${safeTag}</strong>
            <div style="margin-top:2px;">${renderModelBadge(m.run_id)}</div>
          </td>
          <td class="col-nowrap font-mono text-muted" style="font-size:11.5px;">${safeModelType}</td>
          <td class="col-nowrap font-mono" style="font-size:12px;">${layersStr}</td>
          <td class="col-nowrap">${headsStr}</td>
          <td class="col-nowrap font-mono">${bestEpStr}</td>
          <td class="col-nowrap font-mono">${scoreStr}</td>
          <td class="col-nowrap font-mono">${maeStr}</td>
          <td class="col-nowrap font-mono">${hitStr}</td>
          <td class="col-nowrap font-mono">${icStr}</td>
          <td class="col-nowrap font-mono">${covStr}</td>
          <td class="col-nowrap font-mono">${calibMaeStr}</td>
          <td class="col-nowrap font-mono text-muted" style="font-size:11.5px;">${durStr}</td>
          <td class="col-nowrap" onclick="event.stopPropagation()">
            <button class="btn btn-xs ${isActive ? "btn-secondary" : "btn-primary"}" onclick="selectActiveModel('${safeRunId}')">
              ${isActive ? "Active" : "Activate"}
            </button>
          </td>
        </tr>`;
    })
    .join("");

  if (AppState.selectedLeaderboardRunId) {
    inspectModelRun(AppState.selectedLeaderboardRunId, false);
  }
}

let inspectModelRequestSeq = 0;

async function inspectModelRun(runId, reRenderTable = true) {
  if (!runId) return;
  const seq = ++inspectModelRequestSeq;
  AppState.selectedLeaderboardRunId = runId;

  if (reRenderTable) {
    const rows = document.querySelectorAll(".runs-leaderboard-table tbody tr");
    rows.forEach((r) => {
      r.classList.toggle("selected-run-row", r.textContent.includes(runId));
    });
  }

  // Update activation bar
  const activeBarId = document.getElementById("activeBarRunId");
  const activeBarBadge = document.getElementById("activeBarBadge");
  const activeBarPath = document.getElementById("activeBarPath");
  if (activeBarId) activeBarId.textContent = runId;
  if (activeBarBadge) activeBarBadge.innerHTML = renderModelBadge(runId);

  try {
    const data = await apiFetch(`/api/models/${encodeURIComponent(runId)}`);
    if (seq !== inspectModelRequestSeq) return;
    AppState.selectedModelDetail = data;

    if (activeBarPath) activeBarPath.textContent = data.checkpoint_path || "";
    const headerTag = document.getElementById("modelInspectionHeaderTag");
    if (headerTag) headerTag.textContent = `Inspecting: ${runId} • Checkpoint: ${data.exists ? "Available" : "Missing"}`;

    renderRunSubtabs(data);
  } catch (err) {
    if (seq !== inspectModelRequestSeq) return;
    console.error("Failed fetching model detail for", runId, err);
  }
}

function renderRunSubtabs(data) {
  const sum = data.summary || {};
  const metrics = sum.test_evaluation || sum.test_metrics || {};

  // 1. Per-Horizon Test Benchmark
  const bTbody = document.getElementById("runBenchmarkTbody");
  if (bTbody) {
    const horizons = ["1d", "2d", "5d", "20d", "Aggregate"];
    const rows = [];
    horizons.forEach((hKey) => {
      const prefix = hKey === "Aggregate" ? "test" : `test_${hKey}`;
      const isAgg = hKey === "Aggregate";

      const mae = isAgg
        ? (metrics.test_mae ?? metrics.mae_median ?? metrics.mae)
        : (metrics[`${prefix}_mae`] ?? metrics[`${hKey}_mae`]);
      const rmse = isAgg
        ? (metrics.test_rmse ?? metrics.rmse)
        : (metrics[`${prefix}_rmse`] ?? metrics[`${hKey}_rmse`]);
      const pinball = isAgg
        ? (metrics.test_pinball ?? metrics.pinball_loss ?? metrics.test_loss ?? metrics.combined_score)
        : (metrics[`${prefix}_pinball`] ?? metrics[`${hKey}_pinball`]);
      const dirAcc = isAgg
        ? (metrics.test_dir_acc ?? metrics.directional_accuracy ?? metrics.dir_acc)
        : (metrics[`${prefix}_dir_acc`] ?? metrics[`${hKey}_dir_acc`]);
      const ic = isAgg
        ? (metrics.test_ic ?? metrics.ic_median ?? metrics.ic)
        : (metrics[`${prefix}_ic`] ?? metrics[`${hKey}_ic`]);
      const cov80 = isAgg
        ? (metrics.test_cov80 ?? metrics.coverage_80pct_band ?? metrics.cov80)
        : (metrics[`${prefix}_cov80`] ?? metrics[`${hKey}_cov80`]);
      const width80 = isAgg
        ? (metrics.test_width80 ?? metrics.mean_interval_width ?? metrics.mean_interval_width_80pct ?? metrics.width80)
        : (metrics[`${prefix}_width80`] ?? metrics[`${hKey}_width80`]);

      if (mae !== undefined || dirAcc !== undefined || ic !== undefined || cov80 !== undefined || pinball !== undefined) {
        rows.push(`
          <tr>
            <td class="col-nowrap"><strong>${hKey}</strong></td>
            <td class="col-nowrap font-mono">${mae !== undefined && mae !== null ? (Number(mae) * 100).toFixed(2) + "%" : "--"}</td>
            <td class="col-nowrap font-mono">${rmse !== undefined && rmse !== null ? (Number(rmse) * 100).toFixed(2) + "%" : "--"}</td>
            <td class="col-nowrap font-mono">${pinball !== undefined && pinball !== null ? Number(pinball).toFixed(5) : "--"}</td>
            <td class="col-nowrap font-mono">${dirAcc !== undefined && dirAcc !== null ? (Number(dirAcc) * 100).toFixed(1) + "%" : "--"}</td>
            <td class="col-nowrap font-mono">${ic !== undefined && ic !== null ? (Number(ic) >= 0 ? "+" : "") + Number(ic).toFixed(3) : "--"}</td>
            <td class="col-nowrap font-mono">${cov80 !== undefined && cov80 !== null ? (Number(cov80) * 100).toFixed(1) + "%" : "--"}</td>
            <td class="col-nowrap font-mono">${width80 !== undefined && width80 !== null ? (Number(width80) * 100).toFixed(2) + "%" : "--"}</td>
          </tr>
        `);
      }
    });

    if (rows.length) {
      bTbody.innerHTML = rows.join("");
    } else {
      bTbody.innerHTML = '<tr><td colspan="8" style="text-align:center;padding:16px;" class="text-muted">No test benchmark metrics found in this model checkpoint.</td></tr>';
    }
  }

  // 2. Probability Cone Calibration
  const cTbody = document.getElementById("runCalibrationTbody");
  if (cTbody) {
    const tiers = [
      {
        name: "50% Core Band",
        target: 0.50,
        covKeys: ["coverage_50pct_band", "test_cov50", "cov50"],
        widthKeys: ["mean_interval_width_50pct", "test_width50", "width50"]
      },
      {
        name: "80% Standard Band",
        target: 0.80,
        covKeys: ["coverage_80pct_band", "test_cov80", "cov80"],
        widthKeys: ["mean_interval_width", "mean_interval_width_80pct", "test_width80", "width80"]
      },
      {
        name: "90% Tail Band",
        target: 0.90,
        covKeys: ["coverage_90pct_band", "test_cov90", "cov90"],
        widthKeys: ["mean_interval_width_90pct", "test_width90", "width90"]
      },
    ];
    let hasCalib = false;
    const cRows = tiers.map((t) => {
      let realized;
      for (const k of t.covKeys) {
        if (metrics[k] !== undefined && metrics[k] !== null) {
          realized = metrics[k];
          break;
        }
      }
      let width;
      for (const k of t.widthKeys) {
        if (metrics[k] !== undefined && metrics[k] !== null) {
          width = metrics[k];
          break;
        }
      }
      if (realized !== undefined && realized !== null) {
        hasCalib = true;
        const gap = Number(realized) - t.target;
        const gapSign = gap >= 0 ? "+" : "";
        const gapCls = Math.abs(gap) <= 0.04 ? "text-bull" : "text-vol";
        return `
          <tr>
            <td class="col-nowrap"><strong>${t.name}</strong></td>
            <td class="col-nowrap font-mono">${(t.target * 100).toFixed(0)}%</td>
            <td class="col-nowrap font-mono"><strong>${(Number(realized) * 100).toFixed(1)}%</strong></td>
            <td class="col-nowrap font-mono ${gapCls}">${gapSign}${(gap * 100).toFixed(2)}%</td>
            <td class="col-nowrap font-mono">${width !== undefined && width !== null ? (Number(width) * 100).toFixed(2) + "%" : "--"}</td>
            <td class="col-nowrap"><span class="badge ${Math.abs(gap) <= 0.04 ? "badge-bullish" : "badge-volatility"}">${Math.abs(gap) <= 0.04 ? "Calibrated ✓" : "Slight Bias"}</span></td>
          </tr>
        `;
      }
      return "";
    });

    if (hasCalib) {
      cTbody.innerHTML = cRows.join("");
    } else {
      cTbody.innerHTML = '<tr><td colspan="6" style="text-align:center;padding:16px;" class="text-muted">No calibration tiers recorded for this run.</td></tr>';
    }
  }

  // 3. Epoch Training History
  const hTbody = document.getElementById("runHistoryTbody");
  if (hTbody) {
    const hist = data.history_rows || [];
    if (!hist.length) {
      hTbody.innerHTML = '<tr><td colspan="8" style="text-align:center;padding:16px;" class="text-muted">No history.csv available for this run.</td></tr>';
    } else {
      const bestEp = Number(sum.best_epoch || 0);
      hTbody.innerHTML = hist
        .map((r) => {
          const ep = Number(r.epoch || 0);
          const isBest = ep === bestEp;
          return `
            <tr class="${isBest ? "selected-run-row" : ""}">
              <td class="col-nowrap font-mono"><strong>${isBest ? '<span class="epoch-best-star">★</span>' : ""}Epoch ${ep}</strong></td>
              <td class="col-nowrap font-mono">${r.train_loss ? Number(r.train_loss).toFixed(5) : "--"}</td>
              <td class="col-nowrap font-mono">${r.val_loss ? Number(r.val_loss).toFixed(5) : "--"}</td>
              <td class="col-nowrap font-mono"><strong>${r.val_score ? Number(r.val_score).toFixed(5) : "--"}</strong></td>
              <td class="col-nowrap font-mono">${r.val_dir_acc ? (Number(r.val_dir_acc) * 100).toFixed(1) + "%" : "--"}</td>
              <td class="col-nowrap font-mono">${r.val_ic ? (Number(r.val_ic) >= 0 ? "+" : "") + Number(r.val_ic).toFixed(3) : "--"}</td>
              <td class="col-nowrap font-mono text-muted">${r.lr ? Number(r.lr).toExponential(2) : "--"}</td>
              <td class="col-nowrap font-mono text-muted">${r.duration_sec ? Number(r.duration_sec).toFixed(1) + "s" : "--"}</td>
            </tr>
          `;
        })
        .join("");
    }
  }

  // 4. Architecture & Hyperparameters
  const hpGrid = document.getElementById("runHyperparamsGrid");
  if (hpGrid) {
    const arch = sum.architecture || {};
    const hp = sum.training_hyperparameters || {};
    const seqLen = arch.seq_len ?? hp.seq_len ?? 60;
    const rawMilestones = arch.milestones || hp.milestones || [1, 2, 5, 20];
    const milestonesStr = Array.isArray(rawMilestones) ? rawMilestones.join(", ") : String(rawMilestones);
    const patchLen = arch.patch_len ?? hp.patch_len ?? 16;
    const stride = arch.stride ?? hp.stride ?? 8;
    const dModel = arch.d_model ?? hp.d_model ?? 128;
    const numLayers = arch.num_layers ?? hp.num_layers ?? 4;
    const nhead = arch.nhead ?? arch.n_heads ?? hp.nhead ?? hp.n_heads ?? 8;
    const marketLayers = arch.market_layers ?? hp.market_layers ?? 4;
    const marketContext = arch.market_context_size ?? hp.market_context_size ?? 64;
    const modelType = arch.model_type || hp.model_type || "market_query";

    const items = [
      { label: "Model Architecture", val: modelType },
      { label: "Encoder Layers (Target)", val: numLayers },
      { label: "Market Layers (Context)", val: marketLayers },
      { label: "Head Configuration", val: arch.head_architecture || hp.head_architecture || (sum.run_id?.includes("3head") ? "3-Head (1-1-1)" : "1-Head") },
      { label: "Embedding Dimension (d_model)", val: dModel },
      { label: "Attention Heads (nhead)", val: nhead },
      { label: "Sequence Lookback Length", val: `${seqLen} trading days` },
      { label: "Forecast Milestones", val: `${milestonesStr} days` },
      { label: "Patch Length & Stride", val: `Patch: ${patchLen}, Stride: ${stride}` },
      { label: "Market Context Size", val: `${marketContext} instruments` },
      { label: "Loss Function", val: arch.loss_type || hp.loss_type || "pinball + directional" },
      { label: "Loss Multi-Head Weights", val: arch.loss_weights ? JSON.stringify(arch.loss_weights) : (sum.run_id?.includes("3head") ? "1.0 / 1.0 / 1.0" : "1.0") },
      { label: "Optimizer", val: "AdamW (weight_decay=1e-2)" },
      { label: "Checkpoint Location", val: data.checkpoint_path || "--" },
    ];
    hpGrid.innerHTML = items
      .map(
        (it) => `
        <div class="model-meta-item">
          <div class="meta-label">${it.label}</div>
          <div class="meta-val font-mono">${it.val}</div>
        </div>
      `
      )
      .join("");
  }

  // 5. Terminal Report
  const rPre = document.getElementById("runReportPre");
  if (rPre) {
    rPre.textContent = data.report_txt || "No report.txt found for this run.";
  }
}

window.addEventListener("DOMContentLoaded", async () => {
  setupDateInputWrappers();
  await loadModels();
  await loadOverview();
  searchParquetMemory();

  // Navigation & View switching
  document.getElementById("navHomeBtn")?.addEventListener("click", () => showView("dashboard"));
  document.getElementById("btnNavDashboard")?.addEventListener("click", () => showView("dashboard"));
  document.getElementById("btnNavModelRuns")?.addEventListener("click", () => showView("model_runs"));
  document.getElementById("btnBackToDashboard")?.addEventListener("click", () => showView("dashboard"));

  // Topbar model selector
  document.getElementById("globalModelSelect")?.addEventListener("change", (e) => {
    selectActiveModel(e.target.value);
  });

  // Model Runs Explorer Controls
  document.getElementById("btnRefreshModelsList")?.addEventListener("click", loadModels);
  document.getElementById("btnActivateSelectedModel")?.addEventListener("click", () => {
    const promote = Boolean(document.getElementById("chkPromoteToRoot")?.checked);
    selectActiveModel(AppState.selectedLeaderboardRunId, promote);
  });

  // Model detail subtabs
  document.querySelectorAll("#modelDetailPanel .tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#modelDetailPanel .tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll("#modelDetailPanel .tab-pane").forEach((p) => p.classList.add("hidden"));
      btn.classList.add("active");
      const targetId = btn.dataset.modeltab;
      if (targetId) {
        document.getElementById(targetId)?.classList.remove("hidden");
      }
    });
  });

  // Horizon Switcher Pills on Tracked Stocks Table
  document.querySelectorAll("#trackedHorizonToggle button").forEach((btn) => {
    btn.addEventListener("click", () => {
      setTrackedHorizon(btn.dataset.horizon);
    });
  });

  // Analysis Mode Toggles (Directional vs Volatility)
  document.getElementById("btnModeDirectional")?.addEventListener("click", () => setTrackedViewMode("directional"));
  document.getElementById("btnModeVolatility")?.addEventListener("click", () => setTrackedViewMode("volatility"));

  // Tracked stocks column filtering
  document.getElementById("filterTrackedTicker")?.addEventListener("input", renderTrackedStocksTable);
  document.getElementById("filterTrackedName")?.addEventListener("input", renderTrackedStocksTable);
  document.getElementById("filterTrackedSector")?.addEventListener("change", renderTrackedStocksTable);
  document.getElementById("filterTrackedPredsHits")?.addEventListener("change", renderTrackedStocksTable);
  document.getElementById("filterTrackedAnchorDate")?.addEventListener("change", renderTrackedStocksTable);

  // Model filter sync
  const trackedModelSel = document.getElementById("trackedModelFilterSelect");
  const filterModelSel = document.getElementById("filterTrackedModel");

  trackedModelSel?.addEventListener("change", (e) => {
    if (filterModelSel) filterModelSel.value = e.target.value;
    loadOverview();
  });
  filterModelSel?.addEventListener("change", (e) => {
    if (trackedModelSel) trackedModelSel.value = e.target.value;
    loadOverview();
  });

  document.getElementById("filterTracked2dRange")?.addEventListener("change", (e) => {
    const customRow = document.getElementById("filterTracked2dCustomRow");
    if (e.target.value === "custom") {
      if (customRow) customRow.style.display = "flex";
      document.getElementById("filterTracked2dMin")?.focus();
    } else {
      if (customRow) customRow.style.display = "none";
    }
    renderTrackedStocksTable();
  });

  document.getElementById("filterTracked2dMin")?.addEventListener("input", () => {
    const rangeSelect = document.getElementById("filterTracked2dRange");
    if (rangeSelect && rangeSelect.value !== "custom") {
      rangeSelect.value = "custom";
      const customRow = document.getElementById("filterTracked2dCustomRow");
      if (customRow) customRow.style.display = "flex";
    }
    renderTrackedStocksTable();
  });

  document.getElementById("filterTracked2dMax")?.addEventListener("input", () => {
    const rangeSelect = document.getElementById("filterTracked2dRange");
    if (rangeSelect && rangeSelect.value !== "custom") {
      rangeSelect.value = "custom";
      const customRow = document.getElementById("filterTracked2dCustomRow");
      if (customRow) customRow.style.display = "flex";
    }
    renderTrackedStocksTable();
  });

  document.getElementById("filterTracked2dSort")?.addEventListener("change", renderTrackedStocksTable);
  document.getElementById("thTracked2dMedian")?.addEventListener("click", toggleSort2d);
  document.getElementById("btnClearTrackedFilters")?.addEventListener("click", clearTrackedTableFilters);

  // Tracked stocks legacy/topbar controls
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
  let globalSearchRequestSeq = 0;
  globalInput?.addEventListener("input", () => {
    clearTimeout(searchTimer);
    const q = globalInput.value.trim();
    if (!q) {
      globalDrop.classList.add("hidden");
      return;
    }
    searchTimer = setTimeout(async () => {
      const seq = ++globalSearchRequestSeq;
      try {
        const res = await apiFetch(`/api/parquet/search?q=${encodeURIComponent(q)}&limit=12`);
        if (seq !== globalSearchRequestSeq) return;
        const items = res.tickers || [];
        if (!items.length) {
          globalDrop.innerHTML = `<div class="search-item text-muted">No cached Parquet match for "${escapeHtml(q)}" — click "+ Run Stock Prediction" to fetch &amp; predict.</div>`;
        } else {
          globalDrop.innerHTML = items
            .map(
              (it) => {
                const safeTicker = escapeHtml(it.ticker);
                const safeName = escapeHtml(it.name || "");
                const safeUniverse = escapeHtml(it.universe || "");
                return `
              <div class="search-item" onclick="document.getElementById('globalSearchDropdown').classList.add('hidden'); openStockDetail('${safeTicker}')">
                <div><strong>${safeTicker}</strong> — ${safeName} <span class="badge badge-universe">${safeUniverse}</span></div>
                <div>${it.total_predictions > 0 ? `<span class="badge badge-manual">${Number(it.total_predictions)} preds</span>` : '<span class="text-muted">View Chart</span>'}</div>
              </div>`;
              }
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

  // History Tabs in right panel
  document.querySelectorAll(".panel-header .tabs-bar .tab-btn").forEach((btn) => {
    if (btn.dataset.tab) {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".panel-header .tabs-bar .tab-btn").forEach((b) => {
          if (b.dataset.tab) b.classList.remove("active");
        });
        document.querySelectorAll(".tab-pane").forEach((p) => {
          if (p.id === "tabScreeningHistory" || p.id === "tabPredictionHistory") {
            p.classList.add("hidden");
          }
        });
        btn.classList.add("active");
        document.getElementById(btn.dataset.tab)?.classList.remove("hidden");
      });
    }
  });

  document.getElementById("predHistorySourceSelect")?.addEventListener("change", loadOverview);
  document.getElementById("predHistoryModelSelect")?.addEventListener("change", renderPredictionHistory);

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

