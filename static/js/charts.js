/* ==================================================================
   Heart Disease Prediction - charts
   Uses Chart.js (loaded from a CDN in the templates that need it) to draw
   every chart in the browser, so the server never renders images.

   Every chart reads its data from a <script type="application/json">
   block rendered by Jinja, so no data is ever built by string
   concatenation into JavaScript.
   ================================================================== */

(function () {
  'use strict';

  var PALETTE = {
    red:    '#ef4444',
    green:  '#10b981',
    amber:  '#f59e0b',
    blue:   '#3b82f6',
    violet: '#a855f7',
    grey:   '#9ca3af'
  };

  var SERIES = [PALETTE.red, PALETTE.blue, PALETTE.amber, PALETTE.green, PALETTE.violet];

  /* Read a JSON payload embedded in the page. Returns null if missing. */
  function readData(elementId) {
    var el = document.getElementById(elementId);
    if (!el) return null;
    try {
      return JSON.parse(el.textContent);
    } catch (e) {
      return null;
    }
  }

  /* Shared look for every chart */
  function baseOptions(extra) {
    var options = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { font: { family: 'Inter', size: 11 }, color: '#374151' }
        },
        tooltip: {
          backgroundColor: '#1f2937',
          titleFont: { family: 'Inter', size: 12 },
          bodyFont: { family: 'Inter', size: 12 },
          padding: 10,
          cornerRadius: 8
        }
      },
      scales: {
        x: {
          ticks: { color: '#6b7280', font: { family: 'Inter', size: 10 } },
          grid: { color: 'rgba(0,0,0,.05)' }
        },
        y: {
          ticks: { color: '#6b7280', font: { family: 'Inter', size: 10 } },
          grid: { color: 'rgba(0,0,0,.05)' },
          beginAtZero: true
        }
      }
    };
    if (extra) {
      Object.keys(extra).forEach(function (k) { options[k] = extra[k]; });
    }
    return options;
  }

  function makeChart(canvasId, config) {
    var canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === 'undefined') return null;
    return new Chart(canvas.getContext('2d'), config);
  }


  /* ---------------------------------------------------------------
     RISK TREND - a line of risk scores over time
     --------------------------------------------------------------- */
  function riskTrend(canvasId, dataId) {
    var data = readData(dataId);
    if (!data || !data.length) return;

    makeChart(canvasId, {
      type: 'line',
      data: {
        labels: data.map(function (d) { return d.label; }),
        datasets: [{
          label: 'Risk probability (%)',
          data: data.map(function (d) { return d.value; }),
          borderColor: PALETTE.red,
          backgroundColor: 'rgba(239,68,68,.12)',
          borderWidth: 2,
          pointRadius: 3,
          pointBackgroundColor: PALETTE.red,
          fill: true,
          tension: 0.3
        }]
      },
      options: baseOptions({
        scales: {
          x: { ticks: { color: '#6b7280', font: { size: 9 }, maxRotation: 45 },
               grid: { display: false } },
          y: { min: 0, max: 100,
               ticks: { color: '#6b7280', font: { size: 10 },
                        callback: function (v) { return v + '%'; } },
               grid: { color: 'rgba(0,0,0,.05)' } }
        }
      })
    });
  }


  /* ---------------------------------------------------------------
     RISK SPLIT - doughnut of high vs low risk
     --------------------------------------------------------------- */
  function riskSplit(canvasId, dataId) {
    var data = readData(dataId);
    if (!data) return;

    makeChart(canvasId, {
      type: 'doughnut',
      data: {
        labels: ['High Risk', 'Low Risk'],
        datasets: [{
          data: [data.high, data.low],
          backgroundColor: [PALETTE.red, PALETTE.green],
          borderColor: '#fff',
          borderWidth: 3
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '58%',
        plugins: {
          legend: { position: 'bottom',
                    labels: { font: { family: 'Inter', size: 11 }, color: '#374151',
                              padding: 14 } }
        }
      }
    });
  }


  /* ---------------------------------------------------------------
     BAR CHART - generic label/value pairs
     --------------------------------------------------------------- */
  function barChart(canvasId, dataId, label, horizontal) {
    var data = readData(dataId);
    if (!data || !data.length) return;

    makeChart(canvasId, {
      type: 'bar',
      data: {
        labels: data.map(function (d) { return d.label; }),
        datasets: [{
          label: label || 'Count',
          data: data.map(function (d) { return d.value; }),
          backgroundColor: data.map(function (d, i) {
            return d.colour || SERIES[i % SERIES.length];
          }),
          borderRadius: 6,
          borderSkipped: false
        }]
      },
      options: baseOptions({
        indexAxis: horizontal ? 'y' : 'x',
        plugins: { legend: { display: false },
                   tooltip: { backgroundColor: '#1f2937', padding: 10, cornerRadius: 8 } }
      })
    });
  }


  /* ---------------------------------------------------------------
     PREDICTIONS OVER TIME - bars split into high / low risk
     --------------------------------------------------------------- */
  function stackedOverTime(canvasId, dataId) {
    var data = readData(dataId);
    if (!data || !data.length) return;

    makeChart(canvasId, {
      type: 'bar',
      data: {
        labels: data.map(function (d) { return d.label; }),
        datasets: [
          { label: 'High Risk', data: data.map(function (d) { return d.high; }),
            backgroundColor: PALETTE.red, borderRadius: 4 },
          { label: 'Low Risk', data: data.map(function (d) { return d.low; }),
            backgroundColor: PALETTE.green, borderRadius: 4 }
        ]
      },
      options: baseOptions({
        scales: {
          x: { stacked: true, ticks: { color: '#6b7280', font: { size: 9 }, maxRotation: 45 },
               grid: { display: false } },
          y: { stacked: true, beginAtZero: true,
               ticks: { color: '#6b7280', font: { size: 10 }, precision: 0 },
               grid: { color: 'rgba(0,0,0,.05)' } }
        }
      })
    });
  }


  /* ---------------------------------------------------------------
     MODEL COMPARISON - grouped bars, one group per metric
     --------------------------------------------------------------- */
  function modelComparison(canvasId, dataId) {
    var data = readData(dataId);
    if (!data || !data.models || !data.models.length) return;

    var metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc'];
    var labels  = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC'];

    makeChart(canvasId, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: data.models.map(function (m, i) {
          return {
            label: m.short || m.name,
            data: metrics.map(function (k) { return m[k]; }),
            backgroundColor: SERIES[i % SERIES.length],
            borderRadius: 4
          };
        })
      },
      options: baseOptions({
        scales: {
          x: { grid: { display: false }, ticks: { color: '#6b7280', font: { size: 10 } } },
          y: { min: 0, max: 1,
               ticks: { color: '#6b7280', font: { size: 10 } },
               grid: { color: 'rgba(0,0,0,.05)' } }
        }
      })
    });
  }


  /* ── Calibration (reliability) curves ───────────────────────────
     One line per model: mean predicted probability (x) vs the observed
     disease rate in that bin (y), plus the dotted y = x diagonal that a
     perfectly calibrated model would follow. Data comes from
     results.json, computed once during training. */
  function calibrationCurves(canvasId, dataId) {
    var models = readData(dataId);
    if (!models || !models.length) return;

    var datasets = models.map(function (m, i) {
      return {
        label: m.short || m.name,
        data: m.mean_predicted.map(function (x, k) {
          return { x: x, y: m.fraction_positive[k] };
        }),
        borderColor: SERIES[i % SERIES.length],
        backgroundColor: SERIES[i % SERIES.length],
        showLine: true,
        fill: false,
        tension: 0.2,
        pointRadius: 3,
        borderWidth: 2
      };
    });

    // The reference diagonal: perfect calibration
    datasets.push({
      label: 'Perfect calibration',
      data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
      borderColor: '#9ca3af',
      borderDash: [6, 5],
      showLine: true,
      fill: false,
      pointRadius: 0,
      borderWidth: 1.5
    });

    makeChart(canvasId, {
      type: 'scatter',
      data: { datasets: datasets },
      options: baseOptions({
        scales: {
          x: { min: 0, max: 1,
               title: { display: true, text: 'Mean predicted probability',
                        color: '#6b7280', font: { size: 11 } },
               ticks: { color: '#6b7280', font: { size: 10 } },
               grid: { color: 'rgba(0,0,0,.05)' } },
          y: { min: 0, max: 1,
               title: { display: true, text: 'Observed disease rate',
                        color: '#6b7280', font: { size: 11 } },
               ticks: { color: '#6b7280', font: { size: 10 } },
               grid: { color: 'rgba(0,0,0,.05)' } }
        }
      })
    });
  }


  /* Expose the helpers so templates can call exactly what they need */
  window.HDPCharts = {
    riskTrend: riskTrend,
    riskSplit: riskSplit,
    barChart: barChart,
    stackedOverTime: stackedOverTime,
    modelComparison: modelComparison,
    calibrationCurves: calibrationCurves,
    palette: PALETTE
  };
})();
