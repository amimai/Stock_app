/**
 * Interactive Financial & Probability Cone Canvas Chart Engine for Stock_app.
 * Renders:
 *  1) Market OHLCV Chart with Overlaid Historical / Live Prediction Cones
 *  2) Point-in-Time Probability Cone vs. Realized Subsequent Price Trajectory
 */

(function () {
  function setupHiDPICanvas(canvas, logicalHeight) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const width = Math.max(320, rect.width - 24);
    const height = logicalHeight || parseInt(canvas.getAttribute("height") || "380", 10);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.width = width + "px";
    canvas.style.height = height + "px";
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, width, height };
  }

  function fmtPrice(v) {
    if (v === null || v === undefined || isNaN(v)) return "--";
    return "$" + Number(v).toFixed(2);
  }

  function fmtPctFromLog(r) {
    if (r === null || r === undefined || isNaN(r)) return "--";
    const pct = (Math.exp(Number(r)) - 1) * 100;
    return (pct >= 0 ? "+" : "") + pct.toFixed(2) + "%";
  }

  function drawBandPolygon(ctx, points, xFn, yLowFn, yHighFn, fillStyle, strokeStyle) {
    if (!points || points.length < 2) return;
    ctx.save();
    ctx.beginPath();
    for (let i = 0; i < points.length; i++) {
      const x = xFn(points[i], i);
      const y = yHighFn(points[i], i);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    for (let i = points.length - 1; i >= 0; i--) {
      const x = xFn(points[i], i);
      const y = yLowFn(points[i], i);
      ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fillStyle = fillStyle;
    ctx.fill();
    if (strokeStyle) {
      ctx.strokeStyle = strokeStyle;
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    ctx.restore();
  }

  /**
   * 1) Market Chart with Overlaid Prediction Probability Bands
   */
  function renderMarketOverlayChart(canvas, state) {
    if (!canvas) return;
    const { ctx, width, height } = setupHiDPICanvas(canvas, 400);
    const allBars = state.bars || [];
    const rangeLimit = state.rangeLimit || 180;
    const mode = state.chartMode || "candle";
    const activePred = state.activePrediction || null;
    const allPreds = state.allPredictions || [];
    const showAllCones = Boolean(state.showAllCones);

    ctx.clearRect(0, 0, width, height);

    if (!allBars.length) {
      ctx.fillStyle = "#94a3b8";
      ctx.font = "13px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("No historical market bars loaded for this ticker.", width / 2, height / 2);
      return;
    }

    // Slice visible bars, ensuring active prediction's anchor date is included if reasonable
    let startIdx = Math.max(0, allBars.length - rangeLimit);
    if (activePred && activePred.anchor_date) {
      const aIdx = allBars.findIndex((b) => b.date >= activePred.anchor_date);
      if (aIdx !== -1 && aIdx < startIdx) {
        startIdx = Math.max(0, aIdx - 25);
      }
    }
    const visBars = allBars.slice(startIdx);

    // Build unified timeline including any future projected dates from cones
    const conesToDraw = showAllCones
      ? allPreds.slice(0, 12)
      : activePred
      ? [activePred]
      : [];

    const timelineDates = visBars.map((b) => b.date);
    const dateToIndex = new Map();
    timelineDates.forEach((d, idx) => dateToIndex.set(d, idx));

    conesToDraw.forEach((pred) => {
      const dCone = (pred.cone && pred.cone.daily_cone) || [];
      dCone.forEach((pt) => {
        if (pt.date > timelineDates[timelineDates.length - 1] && !dateToIndex.has(pt.date)) {
          dateToIndex.set(pt.date, timelineDates.length);
          timelineDates.push(pt.date);
        }
      });
    });

    const totalSlots = Math.max(1, timelineDates.length);
    const pad = { top: 20, right: 68, bottom: 32, left: 14 };
    const plotW = width - pad.left - pad.right;
    const priceH = Math.round((height - pad.top - pad.bottom) * 0.8);
    const volTop = pad.top + priceH + 8;
    const volH = height - pad.top - pad.bottom - priceH - 8;

    // Determine Y price min/max across visible bars and visible cones
    let minP = Infinity;
    let maxP = -Infinity;
    let maxVol = 1;

    visBars.forEach((b) => {
      if (b.low < minP) minP = b.low;
      if (b.high > maxP) maxP = b.high;
      if ((b.volume || 0) > maxVol) maxVol = b.volume;
    });

    conesToDraw.forEach((pred) => {
      const dCone = (pred.cone && pred.cone.daily_cone) || [];
      dCone.forEach((pt) => {
        if (!dateToIndex.has(pt.date)) return;
        const p = pt.prices || {};
        const lo = p.q05 || p.q10 || p.q50;
        const hi = p.q95 || p.q90 || p.q50;
        if (lo && lo < minP) minP = lo;
        if (hi && hi > maxP) maxP = hi;
      });
    });

    if (!isFinite(minP) || !isFinite(maxP)) {
      minP = 0;
      maxP = 100;
    }
    const pSpan = Math.max(1e-4, maxP - minP);
    minP -= pSpan * 0.06;
    maxP += pSpan * 0.06;

    const xForSlot = (slot) =>
      pad.left + ((slot + 0.5) / totalSlots) * plotW;
    const yForPrice = (p) =>
      pad.top + (1 - (p - minP) / (maxP - minP)) * priceH;

    // Draw horizontal price gridlines
    ctx.strokeStyle = "rgba(38, 51, 77, 0.45)";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#94a3b8";
    ctx.font = "11px sans-serif";
    ctx.textAlign = "left";
    const gridSteps = 6;
    for (let i = 0; i <= gridSteps; i++) {
      const pVal = minP + (i / gridSteps) * (maxP - minP);
      const y = yForPrice(pVal);
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();
      ctx.fillText(fmtPrice(pVal), pad.left + plotW + 6, y + 4);
    }

    // Draw X-axis date labels
    ctx.textAlign = "center";
    const labelCount = Math.min(8, totalSlots);
    const stepSlots = Math.max(1, Math.floor(totalSlots / labelCount));
    for (let s = 0; s < totalSlots; s += stepSlots) {
      const x = xForSlot(s);
      ctx.fillStyle = "#64748b";
      ctx.fillText(timelineDates[s].slice(5), x, height - 10);
    }

    // Draw probability cones (background layer before/around candles)
    conesToDraw.forEach((pred) => {
      const isSelected = activePred && pred.id === activePred.id;
      const dCone = ((pred.cone && pred.cone.daily_cone) || []).filter((pt) =>
        dateToIndex.has(pt.date)
      );
      if (dCone.length < 2) return;

      const xFn = (pt) => xForSlot(dateToIndex.get(pt.date));
      // 90% Tail Band [5%-95%]
      drawBandPolygon(
        ctx,
        dCone,
        xFn,
        (pt) => yForPrice(pt.prices.q05),
        (pt) => yForPrice(pt.prices.q95),
        isSelected ? "rgba(168, 85, 247, 0.14)" : "rgba(168, 85, 247, 0.06)",
        null
      );
      // 80% Standard Band [10%-90%]
      drawBandPolygon(
        ctx,
        dCone,
        xFn,
        (pt) => yForPrice(pt.prices.q10),
        (pt) => yForPrice(pt.prices.q90),
        isSelected ? "rgba(56, 189, 248, 0.20)" : "rgba(56, 189, 248, 0.09)",
        isSelected ? "rgba(56, 189, 248, 0.45)" : null
      );
      // 50% Core Band [25%-75%]
      drawBandPolygon(
        ctx,
        dCone,
        xFn,
        (pt) => yForPrice(pt.prices.q25),
        (pt) => yForPrice(pt.prices.q75),
        isSelected ? "rgba(56, 189, 248, 0.36)" : "rgba(56, 189, 248, 0.15)",
        null
      );

      // Anchor vertical marker
      const anchorSlot = dateToIndex.get(dCone[0].date);
      if (anchorSlot !== undefined) {
        const ax = xForSlot(anchorSlot);
        ctx.save();
        ctx.strokeStyle = isSelected ? "#f59e0b" : "rgba(245, 158, 11, 0.45)";
        ctx.setLineDash([4, 4]);
        ctx.lineWidth = isSelected ? 1.5 : 1;
        ctx.beginPath();
        ctx.moveTo(ax, pad.top);
        ctx.lineTo(ax, pad.top + priceH);
        ctx.stroke();
        ctx.restore();
      }

      // Median 50% trajectory line
      ctx.save();
      ctx.strokeStyle = isSelected ? "#38bdf8" : "rgba(56, 189, 248, 0.55)";
      ctx.lineWidth = isSelected ? 2.2 : 1.3;
      ctx.beginPath();
      dCone.forEach((pt, idx) => {
        const x = xFn(pt);
        const y = yForPrice(pt.prices.q50);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      // Milestone dots on selected cone
      if (isSelected) {
        dCone.forEach((pt) => {
          if (pt.is_milestone || pt.step === 0) {
            const x = xFn(pt);
            const y = yForPrice(pt.prices.q50);
            ctx.fillStyle = pt.step === 0 ? "#f59e0b" : "#38bdf8";
            ctx.beginPath();
            ctx.arc(x, y, 3.5, 0, Math.PI * 2);
            ctx.fill();
          }
        });
      }
      ctx.restore();
    });

    // Draw Market Bars (Candlesticks or Line) + Volume
    const candleW = Math.max(1.5, Math.min(9, (plotW / totalSlots) * 0.68));
    if (mode === "line") {
      ctx.save();
      ctx.strokeStyle = "#e2e8f0";
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      visBars.forEach((b, i) => {
        const slot = dateToIndex.get(b.date);
        const x = xForSlot(slot);
        const y = yForPrice(b.close);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.restore();
    } else {
      visBars.forEach((b) => {
        const slot = dateToIndex.get(b.date);
        const x = xForSlot(slot);
        const isUp = b.close >= b.open;
        const color = isUp ? "#10b981" : "#f43f5e";

        // High-low wick
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, yForPrice(b.high));
        ctx.lineTo(x, yForPrice(b.low));
        ctx.stroke();

        // Open-close body
        const yOpen = yForPrice(b.open);
        const yClose = yForPrice(b.close);
        const topY = Math.min(yOpen, yClose);
        const bodyH = Math.max(1.5, Math.abs(yClose - yOpen));
        ctx.fillStyle = color;
        ctx.fillRect(x - candleW / 2, topY, candleW, bodyH);
      });
    }

    // Volume subplot
    visBars.forEach((b) => {
      const slot = dateToIndex.get(b.date);
      const x = xForSlot(slot);
      const isUp = b.close >= b.open;
      const vh = ((b.volume || 0) / maxVol) * volH;
      ctx.fillStyle = isUp ? "rgba(16, 185, 129, 0.32)" : "rgba(244, 63, 94, 0.32)";
      ctx.fillRect(x - candleW / 2, volTop + volH - vh, candleW, vh);
    });

    // Save layout metadata for interactive hover crosshair
    canvas._chartMeta = {
      timelineDates,
      visBarsByDate: new Map(visBars.map((b) => [b.date, b])),
      activeConeByDate: new Map(
        ((activePred && activePred.cone && activePred.cone.daily_cone) || []).map((pt) => [
          pt.date,
          pt,
        ])
      ),
      pad,
      plotW,
      priceH,
      totalSlots,
    };
  }

  /**
   * 2) Point-in-Time Probability Cone vs. Realized Price Trajectory Chart
   */
  function renderFocusedConeChart(canvas, pred, unitMode) {
    if (!canvas) return;
    const { ctx, width, height } = setupHiDPICanvas(canvas, 340);
    ctx.clearRect(0, 0, width, height);

    if (!pred || !pred.cone || !pred.cone.daily_cone || !pred.cone.daily_cone.length) {
      ctx.fillStyle = "#94a3b8";
      ctx.font = "13px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(
        "Select a prediction from the right-hand side menu (or run a new prediction) to view its probability cone.",
        width / 2,
        height / 2
      );
      return;
    }

    const useReturn = unitMode === "return";
    const anchorClose = Number(pred.anchor_close || 1);
    const dailyCone = pred.cone.daily_cone;
    const ctxBars = pred.context_bars || [];

    // Take up to 20 trading days before anchor (rel_step < 0) + all cone steps (0..max_horizon)
    const preBars = ctxBars.filter((b) => b.rel_step < 0).slice(-20);
    const slots = [];

    preBars.forEach((b) => {
      const logRet = anchorClose > 0 && b.close > 0 ? Math.log(b.close / anchorClose) : 0;
      slots.push({
        step: b.rel_step,
        date: b.date,
        isCone: false,
        actualVal: useReturn ? (Math.exp(logRet) - 1) * 100 : b.close,
        actualHigh: useReturn ? (b.high / anchorClose - 1) * 100 : b.high,
        actualLow: useReturn ? (b.low / anchorClose - 1) * 100 : b.low,
      });
    });

    dailyCone.forEach((pt) => {
      const conv = (qKey) => {
        if (useReturn) {
          return (Math.exp(Number(pt.returns[qKey] || 0)) - 1) * 100;
        }
        return Number(pt.prices[qKey] || anchorClose);
      };
      const hasAct = pt.actual_close !== null && pt.actual_close !== undefined;
      slots.push({
        step: pt.step,
        date: pt.date,
        isCone: true,
        isMilestone: Boolean(pt.is_milestone),
        q05: conv("q05"),
        q10: conv("q10"),
        q25: conv("q25"),
        q50: conv("q50"),
        q75: conv("q75"),
        q90: conv("q90"),
        q95: conv("q95"),
        actualVal: hasAct
          ? useReturn
            ? (pt.actual_close / anchorClose - 1) * 100
            : pt.actual_close
          : null,
        actualHigh:
          hasAct && pt.actual_high
            ? useReturn
              ? (pt.actual_high / anchorClose - 1) * 100
              : pt.actual_high
            : null,
        actualLow:
          hasAct && pt.actual_low
            ? useReturn
              ? (pt.actual_low / anchorClose - 1) * 100
              : pt.actual_low
            : null,
      });
    });

    let minY = Infinity;
    let maxY = -Infinity;
    slots.forEach((s) => {
      if (s.actualVal !== null && s.actualVal !== undefined) {
        minY = Math.min(minY, s.actualLow !== null ? s.actualLow : s.actualVal);
        maxY = Math.max(maxY, s.actualHigh !== null ? s.actualHigh : s.actualVal);
      }
      if (s.isCone) {
        minY = Math.min(minY, s.q05, s.q10);
        maxY = Math.max(maxY, s.q95, s.q90);
      }
    });

    if (!isFinite(minY) || !isFinite(maxY)) {
      minY = 0;
      maxY = 100;
    }
    const span = Math.max(1e-4, maxY - minY);
    minY -= span * 0.08;
    maxY += span * 0.08;

    const pad = { top: 24, right: 74, bottom: 34, left: 16 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;
    const totalSlots = Math.max(1, slots.length);

    const xAt = (idx) => pad.left + ((idx + 0.5) / totalSlots) * plotW;
    const yAt = (val) => pad.top + (1 - (val - minY) / (maxY - minY)) * plotH;

    // Gridlines
    ctx.strokeStyle = "rgba(38, 51, 77, 0.45)";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#94a3b8";
    ctx.font = "11px sans-serif";
    ctx.textAlign = "left";
    for (let i = 0; i <= 5; i++) {
      const v = minY + (i / 5) * (maxY - minY);
      const y = yAt(v);
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();
      const lbl = useReturn ? (v >= 0 ? "+" : "") + v.toFixed(2) + "%" : fmtPrice(v);
      ctx.fillText(lbl, pad.left + plotW + 6, y + 4);
    }

    //Cone points with index in `slots`
    const coneIdxs = [];
    slots.forEach((s, i) => {
      if (s.isCone) coneIdxs.push({ s, i });
    });

    // Draw 90% Tail Band [5%-95%], 80% Std Band [10%-90%], 50% Core Band [25%-75%]
    drawBandPolygon(
      ctx,
      coneIdxs,
      (item) => xAt(item.i),
      (item) => yAt(item.s.q05),
      (item) => yAt(item.s.q95),
      "rgba(168, 85, 247, 0.16)",
      "rgba(168, 85, 247, 0.35)"
    );
    drawBandPolygon(
      ctx,
      coneIdxs,
      (item) => xAt(item.i),
      (item) => yAt(item.s.q10),
      (item) => yAt(item.s.q90),
      "rgba(56, 189, 248, 0.22)",
      "rgba(56, 189, 248, 0.50)"
    );
    drawBandPolygon(
      ctx,
      coneIdxs,
      (item) => xAt(item.i),
      (item) => yAt(item.s.q25),
      (item) => yAt(item.s.q75),
      "rgba(56, 189, 248, 0.38)",
      "rgba(56, 189, 248, 0.65)"
    );

    // Anchor vertical marker (step == 0)
    const anchorIdx = slots.findIndex((s) => s.step === 0);
    if (anchorIdx !== -1) {
      const ax = xAt(anchorIdx);
      ctx.save();
      ctx.strokeStyle = "#f59e0b";
      ctx.setLineDash([5, 4]);
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(ax, pad.top);
      ctx.lineTo(ax, pad.top + plotH);
      ctx.stroke();
      ctx.fillStyle = "#f59e0b";
      ctx.font = "bold 11px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Prediction Anchor (" + slots[anchorIdx].date + ")", ax, pad.top - 7);
      ctx.restore();
    }

    // Draw Median Forecast (50%) trajectory
    ctx.save();
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 2.4;
    ctx.setLineDash([6, 3]);
    ctx.beginPath();
    coneIdxs.forEach((item, k) => {
      const x = xAt(item.i);
      const y = yAt(item.s.q50);
      if (k === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.restore();

    // Draw Actual Price trajectory (prior to prediction + realized after prediction)
    ctx.save();
    ctx.strokeStyle = "#10b981";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    let started = false;
    slots.forEach((s, idx) => {
      if (s.actualVal === null || s.actualVal === undefined) return;
      const x = xAt(idx);
      const y = yAt(s.actualVal);
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    // Draw High-Low whiskers & dots on realized post-prediction bars
    slots.forEach((s, idx) => {
      const x = xAt(idx);
      if (s.isCone && s.isMilestone) {
        // Milestone marker on Median forecast
        ctx.fillStyle = "#38bdf8";
        ctx.beginPath();
        ctx.arc(x, yAt(s.q50), 4, 0, Math.PI * 2);
        ctx.fill();
      }
      if (s.actualVal !== null && s.actualVal !== undefined && s.step >= 0) {
        if (s.actualHigh !== null && s.actualLow !== null) {
          ctx.strokeStyle = "rgba(16, 185, 129, 0.45)";
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          ctx.moveTo(x, yAt(s.actualHigh));
          ctx.lineTo(x, yAt(s.actualLow));
          ctx.stroke();
        }
        ctx.fillStyle = s.step === 0 ? "#f59e0b" : "#10b981";
        ctx.beginPath();
        ctx.arc(x, yAt(s.actualVal), s.isMilestone ? 4.2 : 2.8, 0, Math.PI * 2);
        ctx.fill();
      }
    });
    ctx.restore();

    // X-axis step / milestone labels
    ctx.fillStyle = "#94a3b8";
    ctx.font = "11px sans-serif";
    ctx.textAlign = "center";
    slots.forEach((s, idx) => {
      if (s.step === 0 || s.isMilestone || s.step === -15 || s.step === -5) {
        const lbl = s.step === 0 ? "T=0" : (s.step > 0 ? "T+" + s.step + "d" : "T" + s.step + "d");
        ctx.fillText(lbl, xAt(idx), height - 10);
      }
    });
    canvas._coneMeta = {
      slots,
      pad,
      plotW,
      plotH,
      totalSlots,
      useReturn,
    };
  }

  function ensureTooltipEl(canvas) {
    let tip = canvas.parentElement.querySelector(".chart-hover-tooltip");
    if (!tip) {
      tip = document.createElement("div");
      tip.className = "chart-hover-tooltip hidden";
      tip.style.cssText =
        "position:absolute;top:14px;left:22px;background:rgba(15,23,42,0.92);border:1px solid #26334d;padding:6px 10px;border-radius:6px;font-size:11.5px;color:#f1f5f9;pointer-events:none;z-index:10;";
      canvas.parentElement.appendChild(tip);
    }
    return tip;
  }

  function bindHoverTooltips() {
    const mCanvas = document.getElementById("marketChartCanvas");
    if (mCanvas && !mCanvas._hoverBound) {
      mCanvas._hoverBound = true;
      mCanvas.addEventListener("mousemove", (e) => {
        const meta = mCanvas._chartMeta;
        const tip = ensureTooltipEl(mCanvas);
        if (!meta || !meta.timelineDates.length) return;
        const rect = mCanvas.getBoundingClientRect();
        const relX = e.clientX - rect.left - meta.pad.left;
        const idx = Math.floor((relX / meta.plotW) * meta.totalSlots);
        if (idx < 0 || idx >= meta.totalSlots) {
          tip.classList.add("hidden");
          return;
        }
        const d = meta.timelineDates[idx];
        const b = meta.visBarsByDate.get(d);
        const c = meta.activeConeByDate.get(d);
        const parts = [`<strong>${d}</strong>`];
        if (b) {
          parts.push(
            `O: ${fmtPrice(b.open)} H: ${fmtPrice(b.high)} L: ${fmtPrice(b.low)} <strong>C: ${fmtPrice(b.close)}</strong>`
          );
        }
        if (c && c.prices) {
          parts.push(
            `<span style="color:#38bdf8">Cone T+${c.step}d Med: ${fmtPrice(c.prices.q50)}</span> | 50%: [${fmtPrice(c.prices.q25)}–${fmtPrice(c.prices.q75)}] | 80%: [${fmtPrice(c.prices.q10)}–${fmtPrice(c.prices.q90)}]`
          );
        }
        tip.innerHTML = parts.join(" &nbsp;•&nbsp; ");
        tip.classList.remove("hidden");
      });
      mCanvas.addEventListener("mouseleave", () => {
        ensureTooltipEl(mCanvas).classList.add("hidden");
      });
    }

    const cCanvas = document.getElementById("coneChartCanvas");
    if (cCanvas && !cCanvas._hoverBound) {
      cCanvas._hoverBound = true;
      cCanvas.addEventListener("mousemove", (e) => {
        const meta = cCanvas._coneMeta;
        const tip = ensureTooltipEl(cCanvas);
        if (!meta || !meta.slots.length) return;
        const rect = cCanvas.getBoundingClientRect();
        const relX = e.clientX - rect.left - meta.pad.left;
        const idx = Math.floor((relX / meta.plotW) * meta.totalSlots);
        if (idx < 0 || idx >= meta.totalSlots) {
          tip.classList.add("hidden");
          return;
        }
        const s = meta.slots[idx];
        const fmtV = (v) =>
          v === null || v === undefined
            ? "--"
            : meta.useReturn
            ? (v >= 0 ? "+" : "") + Number(v).toFixed(2) + "%"
            : fmtPrice(v);
        const stepLbl = s.step === 0 ? "T=0 (Anchor)" : s.step > 0 ? `T+${s.step}d` : `T${s.step}d`;
        const parts = [`<strong>${s.date} (${stepLbl})</strong>`];
        if (s.actualVal !== null && s.actualVal !== undefined) {
          parts.push(`<span style="color:#10b981">Actual: ${fmtV(s.actualVal)}</span>`);
        }
        if (s.isCone) {
          parts.push(
            `<span style="color:#38bdf8">Pred Med (50%): ${fmtV(s.q50)}</span> | 50% Core: [${fmtV(s.q25)}–${fmtV(s.q75)}] | 80% Band: [${fmtV(s.q10)}–${fmtV(s.q90)}]`
          );
        }
        tip.innerHTML = parts.join(" &nbsp;•&nbsp; ");
        tip.classList.remove("hidden");
      });
      cCanvas.addEventListener("mouseleave", () => {
        ensureTooltipEl(cCanvas).classList.add("hidden");
      });
    }
  }

  window.StockAppCharts = {
    renderMarketOverlayChart: function (canvas, state) {
      renderMarketOverlayChart(canvas, state);
      bindHoverTooltips();
    },
    renderFocusedConeChart: function (canvas, pred, unitMode) {
      renderFocusedConeChart(canvas, pred, unitMode);
      bindHoverTooltips();
    },
    fmtPrice,
    fmtPctFromLog,
  };
})();
