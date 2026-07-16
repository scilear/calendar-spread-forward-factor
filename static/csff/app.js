(function () {
  'use strict';

  const API_BASE = '/csff/api';
  let activeDate = null;
  let pollTimer = null;

  // ── DOM refs ──────────────────────────────────────────────────────
  const $ = (s) => document.querySelector(s);
  const reportList = document.getElementById('report-list');
  const contentArea = document.getElementById('content-area');
  const reportIndex = document.getElementById('report-index');
  const reportDetail = document.getElementById('report-detail');
  const tickerInput = document.getElementById('ticker-input');
  const statusEl = document.getElementById('nav-status');
  const modalOverlay = document.getElementById('modal-overlay');
  const modalBody = document.getElementById('modal-body');
  const modalTitle = document.getElementById('modal-title');
  const progressFill = document.getElementById('progress-fill');

  // ── API helpers ────────────────────────────────────────────────────
  async function api(method, path, body) {
    const opts = { method, headers: {} };
    if (body) {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    }
    const resp = await fetch(API_BASE + path, opts);
    const ct = resp.headers.get('content-type') || '';
    if (ct.includes('application/json')) {
      return { ok: resp.ok, status: resp.status, data: await resp.json() };
    }
    return { ok: resp.ok, status: resp.status, text: await resp.text() };
  }

  // ── Sidebar: report list ───────────────────────────────────────────
  async function loadReportList() {
    const res = await api('GET', '/reports?max=30');
    if (!res.ok) {
      reportList.innerHTML = '<div class="loading">Failed to load reports.</div>';
      return;
    }
    const dates = res.data.dates || [];
    if (!dates.length) {
      reportList.innerHTML = '<div class="loading">No reports yet. Run a scan.</div>';
      return;
    }
    reportList.innerHTML = dates.map((d) =>
      `<div class="report-item${d.date === activeDate ? ' active' : ''}"
            onclick="window.CSFF.selectDate('${d.date}')">
         <span class="date-label">${d.date}</span>
         <span class="count">${d.count}</span>
       </div>`
    ).join('');
    if (!activeDate && dates.length) {
      selectDate(dates[0].date);
    }
  }

  // ── Date / ticker selection ────────────────────────────────────────
  window.CSFF = { selectDate, showTicker };

  function selectDate(dateStr) {
    activeDate = dateStr;
    reportDetail.style.display = 'none';
    reportIndex.style.display = 'block';
    loadReportIndex(dateStr);
    // update sidebar active
    document.querySelectorAll('.report-item').forEach((el) => {
      el.classList.toggle('active', el.textContent.trim().startsWith(dateStr));
    });
  }

  async function loadReportIndex(dateStr) {
    reportIndex.innerHTML = '<div class="loading">Loading...</div>';
    const res = await api('GET', `/report?date=${dateStr}`);
    if (!res.ok || !res.data) {
      reportIndex.innerHTML = '<div class="loading">No data for this date.</div>';
      return;
    }
    const tickers = res.data.tickers || [];
    if (!tickers.length) {
      reportIndex.innerHTML = '<div class="loading">No ticker reports for this date.</div>';
      return;
    }
    // We render a clickable list; for the full HTML we embed the index.html
    // But better: render a simple table matching the existing index style
    let html = `<h2 style="color:#fff;margin-bottom:12px">${dateStr}</h2>
                <table>
                  <thead><tr>
                    <th>Ticker</th><th>Action</th>
                  </tr></thead>
                  <tbody>`;
    for (const t of tickers) {
      html += `<tr>
        <td><a class="ticker-link" onclick="window.CSFF.showTicker('${dateStr}','${t}')">${t}</a></td>
        <td><button class="btn btn-sm" onclick="window.CSFF.refreshTicker('${t}')">Refresh</button></td>
      </tr>`;
    }
    html += '</tbody></table>';
    reportIndex.innerHTML = html;
  }

  async function showTicker(dateStr, ticker) {
    reportDetail.style.display = 'block';
    reportIndex.style.display = 'none';
    reportDetail.innerHTML = '<div class="loading">Loading report...</div>';
    const res = await api('GET', `/report?date=${dateStr}&ticker=${ticker}`);
    if (!res.ok || !res.data) {
      reportDetail.innerHTML = '<div class="loading">Report not found.</div>';
      return;
    }
    // Embed the report HTML in an iframe (clean isolation)
    reportDetail.innerHTML = `<iframe srcdoc="${escapeHtml(res.data.html)}"></iframe>`;
  }

  // ── Ticker management ──────────────────────────────────────────────
  async function loadTickers() {
    const res = await api('GET', '/tickers');
    if (res.ok && res.data) {
      tickerInput.value = (res.data.tickers || []).join(', ');
    }
  }

  async function saveTickers() {
    const val = tickerInput.value;
    const res = await api('POST', '/tickers', { tickers: val });
    if (res.ok) {
      tickerInput.value = (res.data.tickers || []).join(', ');
      setStatus('Tickers saved');
    } else {
      setStatus('Failed to save tickers', true);
    }
  }

  // ── Scan / Refresh ─────────────────────────────────────────────────
  async function startScan(type) {
    const tickers = tickerInput.value.trim();
    const body = tickers ? { tickers, type } : { type };
    const res = await api('POST', '/scan', body);
    if (!res.ok) {
      setStatus(res.data?.error || 'Scan failed', true);
      return;
    }
    const jobId = res.data.job_id;
    if (jobId) {
      showModal(`${type} scan in progress...`);
      pollJob(jobId);
    }
  }

  async function refreshTicker(ticker) {
    setStatus(`Refreshing ${ticker}...`);
    const res = await api('POST', `/refresh?ticker=${ticker}`);
    if (res.ok) {
      setStatus(`${ticker} refreshed`);
      if (activeDate) loadReportIndex(activeDate);
    } else {
      setStatus(`${ticker} refresh failed: ${res.data?.error || ''}`, true);
    }
  }

  // ── Job polling ────────────────────────────────────────────────────
  function pollJob(jobId) {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(async () => {
      const res = await api('GET', `/status?job_id=${jobId}`);
      if (!res.ok || !res.data) {
        clearInterval(pollTimer);
        pollTimer = null;
        updateModalLog('Lost connection to job.');
        return;
      }
      const job = res.data;
      if (job.status === 'done') {
        clearInterval(pollTimer);
        pollTimer = null;
        progressFill.style.width = '100%';
        updateModalLog('Scan complete. Reloading reports...');
        setTimeout(() => {
          closeModal();
          loadReportList();
          setStatus('Scan complete');
        }, 500);
      } else if (job.status === 'failed') {
        clearInterval(pollTimer);
        pollTimer = null;
        updateModalLog(`FAILED: ${job.error || 'Unknown error'}`);
        progressFill.style.width = '0%';
        progressFill.style.background = '#f87171';
      } else {
        progressFill.style.width = '50%';
        updateModalLog(job.progress || 'Running...');
      }
    }, 2000);
  }

  // ── Modal ──────────────────────────────────────────────────────────
  function showModal(title) {
    modalTitle.textContent = title;
    modalBody.innerHTML = '<div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div><pre class="modal-log" id="modal-log"></pre>';
    modalOverlay.style.display = 'flex';
  }

  function closeModal() {
    modalOverlay.style.display = 'none';
  }

  function updateModalLog(msg) {
    const log = document.getElementById('modal-log');
    if (log) {
      log.textContent += msg + '\n';
      log.scrollTop = log.scrollHeight;
    }
  }

  // ── Status bar ─────────────────────────────────────────────────────
  function setStatus(msg, isError) {
    statusEl.textContent = msg;
    statusEl.style.color = isError ? '#f87171' : '#4ade80';
    setTimeout(() => { statusEl.textContent = ''; }, 5000);
  }

  // ── Helpers ────────────────────────────────────────────────────────
  function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/"/g, '&quot;')
              .replace(/'/g, '&#39;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  // ── Bind events ────────────────────────────────────────────────────
  document.getElementById('btn-save-tickers').addEventListener('click', saveTickers);
  document.getElementById('btn-scan-universe').addEventListener('click', () => startScan('universe'));
  document.getElementById('btn-scan-intraday').addEventListener('click', () => startScan('intraday'));
  document.getElementById('modal-close').addEventListener('click', closeModal);
  modalOverlay.addEventListener('click', (e) => { if (e.target === modalOverlay) closeModal(); });

  // ── Init ────────────────────────────────────────────────────────────
  loadReportList();
  loadTickers();

})();
