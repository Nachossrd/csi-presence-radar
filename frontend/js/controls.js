/**
 * Dashboard Controls — Wi-Fi Radar · Digital Twin
 * Wires up all UI buttons, tabs, and events to the scene, floor plan, and WebSocket.
 */

export class DashboardControls {
  /**
   * @param {import('./websocket-client.js').WebSocketClient} wsClient
   * @param {import('./three-scene.js').HouseScene3D} scene3d
   * @param {import('./floor-plan-2d.js').FloorPlan2D} floorPlan2d
   * @param {import('./floor-plan-editor.js').FloorPlanEditor} editor
   */
  constructor(wsClient, scene3d, floorPlan2d, editor) {
    this.ws = wsClient;
    this.scene3d = scene3d;
    this.floorPlan2d = floorPlan2d;
    this.editor = editor;

    // State
    this.isTracking = false;
    this.isSimulation = false;
    this.isCollecting = false;
    this.fingerprints = [];
    this._currentFloorPlan = null;
  }

  /** Initialize all controls and event bindings */
  init() {
    this._initTabs();
    this._initCameraButtons();
    this._initViewToggles();
    this._initWifiScan();
    this._initTracking();
    this._initSimulation();
    this._initFingerprints();
    this._initEditor();
    this._initFloorPlanIO();
    this._initSettings();
    this._initWSListeners();
  }

  /* ─── Tabs ───────────────────────────────────── */

  _initTabs() {
    const tabs = document.querySelectorAll('.tab');
    const contents = document.querySelectorAll('.tab-content');

    tabs.forEach((tab) => {
      tab.addEventListener('click', () => {
        const targetId = `tab-${tab.dataset.tab}`;

        tabs.forEach((t) => t.classList.remove('active'));
        contents.forEach((c) => c.classList.remove('active'));

        tab.classList.add('active');
        document.getElementById(targetId)?.classList.add('active');

        // Update mode badge
        const badge = document.getElementById('mode-badge');
        const labels = { monitor: 'MONITOR', fingerprint: 'RECOLECCIÓN', editor: 'EDITOR', settings: 'CONFIG' };
        if (badge) badge.textContent = labels[tab.dataset.tab] || 'MONITOR';

        // Activate/deactivate editor
        if (tab.dataset.tab === 'editor') {
          this.editor.activate();
        } else {
          this.editor.deactivate();
        }
      });
    });
  }

  /* ─── Camera Presets ─────────────────────────── */

  _initCameraButtons() {
    const btnTop = document.getElementById('btn-camera-top');
    const btnIso = document.getElementById('btn-camera-iso');
    const btnFollow = document.getElementById('btn-camera-follow');
    const buttons = [btnTop, btnIso, btnFollow];

    const setActive = (btn) => {
      buttons.forEach((b) => b?.classList.remove('active'));
      btn?.classList.add('active');
    };

    btnTop?.addEventListener('click', () => {
      this.scene3d.setCameraPreset('top');
      setActive(btnTop);
    });

    btnIso?.addEventListener('click', () => {
      this.scene3d.setCameraPreset('isometric');
      setActive(btnIso);
    });

    btnFollow?.addEventListener('click', () => {
      this.scene3d.setCameraPreset('follow');
      setActive(btnFollow);
    });
  }

  /* ─── View Toggles ──────────────────────────── */

  _initViewToggles() {
    const btnGrid = document.getElementById('btn-toggle-grid');
    const btnLabels = document.getElementById('btn-toggle-labels');
    const btnFP = document.getElementById('btn-toggle-fingerprints');

    btnGrid?.addEventListener('click', () => {
      const on = this.floorPlan2d.toggleGrid();
      btnGrid.classList.toggle('active', on);
    });

    btnLabels?.addEventListener('click', () => {
      const on = this.floorPlan2d.toggleLabels();
      btnLabels.classList.toggle('active', on);
    });

    btnFP?.addEventListener('click', () => {
      const on = this.floorPlan2d.toggleFingerprints();
      btnFP.classList.toggle('active', on);
    });
  }

  /* ─── Wi-Fi Scan ─────────────────────────────── */

  _initWifiScan() {
    const btn = document.getElementById('btn-scan-wifi');

    btn?.addEventListener('click', async () => {
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner"></span> Escaneando…';

      try {
        const res = await fetch('/api/wifi/scan');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const networks = this._extractNetworks(data);
        this._renderAPList(networks, data);
        this._updateStat('stat-aps', networks.length);
        // 0 APs is usually Windows rate-limiting (4s cooldown), not a real error.
        // Just log it quietly. The list shows the "no data" placeholder anyway.
        if (!networks.length && data.scan_status?.error) {
          console.info('[Controls] Wi-Fi scan returned 0 APs:',
                       data.scan_status.error.message);
        }
      } catch (err) {
        // Only show a toast on real network/HTTP errors. Suppress repeats.
        console.warn('[Controls] Wi-Fi scan failed:', err);
        if (!this._wifiToastShownAt || Date.now() - this._wifiToastShownAt > 30000) {
          this._showToast('Wi-Fi: el backend no respondió. Reintentando en silencio.', 'warning');
          this._wifiToastShownAt = Date.now();
        }
        this._renderAPList([], { message: 'Sin respuesta del backend.' });
      } finally {
        btn.disabled = false;
        btn.textContent = 'Escanear';
      }
    });
  }

  _extractNetworks(data) {
    return data?.networks || data?.access_points || data?.aps || [];
  }

  _renderAPList(networks, meta = {}) {
    const container = document.getElementById('ap-list');
    if (!container) return;

    if (!networks.length) {
      const error = meta.scan_status?.error;
      if (error) {
        container.innerHTML = `
          <div class="wifi-diagnostic">
            <strong>${this._escapeHtml(error.message)}</strong>
            <span>${this._escapeHtml(error.action || 'Revisa permisos de Windows y vuelve a intentar.')}</span>
          </div>
        `;
      } else {
        container.innerHTML = `<p class="placeholder">${this._escapeHtml(meta.message || 'Sin redes detectadas')}</p>`;
      }
      return;
    }

    // Sort by signal strength (strongest first)
    const sorted = [...networks].sort((a, b) => this._getRssi(b) - this._getRssi(a));

    container.innerHTML = sorted.map((ap) => {
      const ssid = ap.ssid || ap.SSID || 'Oculta';
      const bssid = ap.bssid || ap.BSSID || '--';
      const rssi = this._getRssi(ap);
      const bars = this._signalToBars(rssi);
      const strength = rssi > -50 ? 'strong' : rssi > -70 ? 'medium' : 'weak';

      return `
        <div class="ap-item">
          <div class="ap-info">
            <span class="ap-ssid">${this._escapeHtml(ssid)}</span>
            <span class="ap-bssid">${this._escapeHtml(bssid)}</span>
          </div>
          <div class="ap-signal">
            <div class="signal-bars">
              ${[1, 2, 3, 4, 5].map((n) =>
                `<div class="signal-bar ${n <= bars ? `active ${strength}` : ''}"></div>`
              ).join('')}
            </div>
            <span class="signal-dbm">${rssi} dBm</span>
          </div>
        </div>
      `;
    }).join('');
  }

  _getRssi(ap) {
    const value = ap.signal_dbm ?? ap.rssi ?? ap.signal;
    const numeric = Number(value);
    if (Number.isFinite(numeric)) return numeric;
    const percent = Number(ap.signal_percent);
    return Number.isFinite(percent) ? (percent / 2) - 100 : -100;
  }

  _signalToBars(rssi) {
    if (rssi >= -40) return 5;
    if (rssi >= -55) return 4;
    if (rssi >= -67) return 3;
    if (rssi >= -78) return 2;
    return 1;
  }

  /* ─── Tracking ───────────────────────────────── */

  _initTracking() {
    const btnTrain = document.getElementById('btn-train');
    const btnStart = document.getElementById('btn-start-tracking');
    const btnStop = document.getElementById('btn-stop-tracking');

    btnTrain?.addEventListener('click', async () => {
      btnTrain.disabled = true;
      btnTrain.innerHTML = '<span class="spinner"></span> Entrenando…';

      try {
        const res = await fetch('/api/localizer/train', { method: 'POST' });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const accuracy = data.accuracy || data.score || 0;
        this._updateStat('stat-accuracy', `${(accuracy * 100).toFixed(0)}%`);
        this._showToast(`Modelo entrenado — Precisión: ${(accuracy * 100).toFixed(1)}%`, 'success');
      } catch (err) {
        console.error('[Controls] Train error:', err);
        this._showToast('Error al entrenar modelo', 'error');
      } finally {
        btnTrain.disabled = false;
        btnTrain.textContent = 'Entrenar Modelo';
      }
    });

    btnStart?.addEventListener('click', async () => {
      try {
        const res = await fetch('/api/localizer/start', { method: 'POST' });
        const data = await res.json().catch(() => ({}));
        if (!res.ok || data.success === false) {
          throw new Error(data.action || data.message || `HTTP ${res.status}`);
        }
        this.isTracking = true;
        btnStart.classList.add('hidden');
        btnStop?.classList.remove('hidden');
        this._showToast('Tracking iniciado', 'success');
      } catch (err) {
        console.error('[Controls] Start tracking error:', err);
        this._showToast(err.message || 'Error al iniciar tracking', 'error');
      }
    });

    btnStop?.addEventListener('click', async () => {
      try {
        const res = await fetch('/api/localizer/stop', { method: 'POST' });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        this.isTracking = false;
        btnStop.classList.add('hidden');
        btnStart?.classList.remove('hidden');
        this._showToast('Tracking detenido', 'info');
      } catch (err) {
        console.error('[Controls] Stop tracking error:', err);
        this._showToast('Error al detener tracking', 'error');
      }
    });
  }

  /* ─── Simulation ─────────────────────────────── */

  _initSimulation() {
    const btn = document.getElementById('btn-simulation');

    btn?.addEventListener('click', async () => {
      try {
        const res = await fetch('/api/simulation/toggle', { method: 'POST' });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        this.isSimulation = Boolean(data.simulation);
        btn.classList.toggle('active', this.isSimulation);
        btn.textContent = this.isSimulation ? 'ON' : 'OFF';
        this._showToast(
          this.isSimulation ? 'Simulación activada' : 'Simulación desactivada',
          'info',
        );
      } catch (err) {
        console.error('[Controls] Simulation toggle error:', err);
        // Toggle locally anyway for UI
        this.isSimulation = !this.isSimulation;
        btn.classList.toggle('active', this.isSimulation);
        btn.textContent = this.isSimulation ? 'ON' : 'OFF';
      }
    });
  }

  /* ─── Fingerprints ───────────────────────────── */

  _initFingerprints() {
    const btn = document.getElementById('btn-collect-mode');

    btn?.addEventListener('click', () => {
      if (this.isCollecting) {
        // Deactivate collection
        this.isCollecting = false;
        this.floorPlan2d.disableCollectionMode();
        btn.textContent = 'Activar Modo Recolección';
        btn.classList.remove('btn-danger');
        btn.classList.add('btn-primary');
      } else {
        // Activate collection
        this.isCollecting = true;
        btn.textContent = '⏹ Desactivar Recolección';
        btn.classList.remove('btn-primary');
        btn.classList.add('btn-danger');

        this.floorPlan2d.enableCollectionMode(async (x, y) => {
          await this._collectFingerprint(x, y);
        });
      }
    });
  }

  async _collectFingerprint(x, y) {
    try {
      const res = await fetch('/api/fingerprint', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ position: [x, y] }),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const data = await res.json();
      this.fingerprints.push({
        id: data.id || Date.now(),
        x,
        y,
        networks: data.networks || [],
      });

      this._renderFingerprintList();
      this.floorPlan2d.setFingerprints(this.fingerprints);
      this._updateStat('stat-fingerprints', this.fingerprints.length);
      this._showToast(`Fingerprint #${this.fingerprints.length} registrado en (${x.toFixed(1)}, ${y.toFixed(1)})`, 'success');
    } catch (err) {
      console.error('[Controls] Fingerprint collect error:', err);
      // Still add locally even if server fails
      this.fingerprints.push({ id: Date.now(), x, y, networks: [] });
      this._renderFingerprintList();
      this.floorPlan2d.setFingerprints(this.fingerprints);
      this._updateStat('stat-fingerprints', this.fingerprints.length);
      this._showToast(`Fingerprint guardado localmente en (${x.toFixed(1)}, ${y.toFixed(1)})`, 'warning');
    }
  }

  _renderFingerprintList() {
    const container = document.getElementById('fingerprint-list');
    if (!container) return;

    if (!this.fingerprints.length) {
      container.innerHTML = '<p class="placeholder">Sin fingerprints</p>';
      return;
    }

    container.innerHTML = this.fingerprints.map((fp, i) => `
      <div class="fp-item" data-index="${i}">
        <div class="fp-info">
          <span class="fp-number">${i + 1}</span>
          <span class="fp-coords">(${fp.x.toFixed(1)}, ${fp.y.toFixed(1)})</span>
        </div>
        <button class="fp-delete" data-index="${i}" title="Eliminar">✕</button>
      </div>
    `).join('');

    // Wire delete buttons
    container.querySelectorAll('.fp-delete').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        const idx = parseInt(e.currentTarget.dataset.index, 10);
        this.fingerprints.splice(idx, 1);
        this._renderFingerprintList();
        this.floorPlan2d.setFingerprints(this.fingerprints);
        this._updateStat('stat-fingerprints', this.fingerprints.length);
      });
    });
  }

  /* ─── Editor ─────────────────────────────────── */

  _initEditor() {
    // Tool buttons
    const toolBtns = document.querySelectorAll('.tool-btn[data-tool]');
    toolBtns.forEach((btn) => {
      btn.addEventListener('click', () => {
        toolBtns.forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        this.editor.setTool(btn.dataset.tool);
      });
    });

    // Dimension inputs
    const widthInput = document.getElementById('house-width');
    const lengthInput = document.getElementById('house-length');
    const snapInput = document.getElementById('snap-grid');

    widthInput?.addEventListener('change', () => {
      this.editor.houseWidth = parseFloat(widthInput.value) || 10;
      this.editor.resize();
    });

    lengthInput?.addEventListener('change', () => {
      this.editor.houseLength = parseFloat(lengthInput.value) || 11;
      this.editor.resize();
    });

    snapInput?.addEventListener('change', () => {
      this.editor.snapToGrid = snapInput.checked;
    });

    // Clear
    document.getElementById('btn-clear-plan')?.addEventListener('click', () => {
      if (confirm('¿Limpiar todo el plano?')) {
        this.editor.clear();
      }
    });
  }

  /* ─── Floor Plan I/O ─────────────────────────── */

  _initFloorPlanIO() {
    // Save
    document.getElementById('btn-save-plan')?.addEventListener('click', async () => {
      const plan = this.editor.getFloorPlan();

      try {
        const res = await fetch('/api/floor-plan', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(plan),
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        this._showToast('Plano guardado exitosamente', 'success');

        // Apply to views
        this._applyFloorPlan(plan);
      } catch (err) {
        console.error('[Controls] Save floor plan error:', err);
        // Apply locally anyway
        this._applyFloorPlan(plan);
        this._showToast('Plano aplicado localmente (servidor no disponible)', 'warning');
      }
    });

    // Load
    document.getElementById('btn-load-plan')?.addEventListener('click', async () => {
      await this._loadFloorPlan();
    });

    // Load Default
    document.getElementById('btn-load-default')?.addEventListener('click', () => {
      const demoPlan = this._getDemoFloorPlan();
      this._applyFloorPlan(demoPlan);
      this.editor.loadFloorPlan(demoPlan);
      this._showToast('Plano demo cargado', 'info');
    });
  }

  async _loadFloorPlan() {
    try {
      const res = await fetch('/api/floor-plan');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      this._applyFloorPlan(data);
      this.editor.loadFloorPlan(data);
      this._showToast('Plano cargado desde servidor', 'success');
    } catch (err) {
      console.error('[Controls] Load floor plan error:', err);
      // Load demo as fallback
      const demo = this._getDemoFloorPlan();
      this._applyFloorPlan(demo);
      this.editor.loadFloorPlan(demo);
      this._showToast('Cargando plano demo (servidor no disponible)', 'warning');
    }
  }

  _applyFloorPlan(plan) {
    this._currentFloorPlan = plan;
    this.scene3d.loadFloorPlan(plan);
    this.floorPlan2d.loadFloorPlan(plan);
  }

  /**
   * Load the initial floor plan (try server, fallback to demo).
   */
  async loadInitialFloorPlan() {
    try {
      const res = await fetch('/api/floor-plan');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      this._applyFloorPlan(data);
      this.editor.loadFloorPlan(data);
    } catch {
      // Fallback to demo plan
      const demo = this._getDemoFloorPlan();
      this._applyFloorPlan(demo);
      this.editor.loadFloorPlan(demo);
    }
  }

  /* ─── Settings ───────────────────────────────── */

  _initSettings() {
    const trailInput = document.getElementById('trail-length');
    trailInput?.addEventListener('change', () => {
      const val = parseInt(trailInput.value, 10) || 50;
      this.scene3d.maxTrail = val;
      this.floorPlan2d.maxTrail = val;
    });
  }

  /* ─── WebSocket Listeners ────────────────────── */

  _initWSListeners() {
    this.ws.addEventListener('position-update', (e) => {
      const payload = e.detail;
      if (Array.isArray(payload.persons)) {
        if (typeof this.scene3d.updatePersons === 'function') {
          this.scene3d.updatePersons(payload.persons);
        }
        if (typeof this.floorPlan2d.updatePersons === 'function') {
          this.floorPlan2d.updatePersons(payload.persons);
        }
      }

      const data = payload.persons?.[0] || payload;
      const x = data.x ?? data.position?.x ?? 0;
      const y = data.y ?? data.position?.y ?? 0;
      const z = data.z ?? data.position?.z ?? 0.25;
      const confidence = data.confidence ?? data.score ?? 0;
      const zone = data.zone || data.room || null;

      if (!Array.isArray(payload.persons)) {
        this.scene3d.updatePersonPosition(x, y, z, confidence);
        this.floorPlan2d.updatePersonPosition(x, y, confidence);
      }

      // Update UI
      this._updatePosition(x, y, confidence, zone);
    });

    this.ws.addEventListener('wifi-update', (e) => {
      const data = e.detail;
      const networks = this._extractNetworks(data);
      this._renderAPList(networks, data);
      this._updateStat('stat-aps', networks.length);
    });

    this.ws.addEventListener('zone-update', (e) => {
      const data = e.detail;
      const zone = data.zone || data.room || '--';
      const zoneEl = document.getElementById('current-zone');
      if (zoneEl) zoneEl.textContent = zone;
    });

    this.ws.addEventListener('status-update', (e) => {
      const data = e.detail;
      if (data.fingerprints !== undefined) {
        this._updateStat('stat-fingerprints', data.fingerprints);
      }
      if (data.accuracy !== undefined) {
        this._updateStat('stat-accuracy', `${(data.accuracy * 100).toFixed(0)}%`);
      }
    });
  }

  /* ─── UI Helpers ─────────────────────────────── */

  _updatePosition(x, y, confidence, zone) {
    const posX = document.getElementById('pos-x');
    const posY = document.getElementById('pos-y');
    const confFill = document.getElementById('confidence-fill');
    const confValue = document.getElementById('confidence-value');
    const zoneEl = document.getElementById('current-zone');

    if (posX) posX.textContent = x.toFixed(2);
    if (posY) posY.textContent = y.toFixed(2);

    const confPct = Math.round(confidence * 100);
    if (confFill) confFill.style.width = confPct + '%';
    if (confValue) confValue.textContent = confPct + '%';

    if (zone && zoneEl) {
      zoneEl.textContent = zone;
    }
  }

  _updateStat(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  }

  _showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      toast.style.transition = 'all 0.3s ease-out';
      setTimeout(() => toast.remove(), 300);
    }, 3500);
  }

  _escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  /* ─── Demo Floor Plan ────────────────────────── */

  _getDemoFloorPlan() {
    return {
      width: 10,
      length: 11,
      rooms: [
        { name: 'Sala', x: 0, y: 0, width: 5, height: 4 },
        { name: 'Cocina', x: 5, y: 0, width: 5, height: 4 },
        { name: 'Comedor', x: 0, y: 4, width: 5, height: 3 },
        { name: 'Baño', x: 5, y: 4, width: 2.5, height: 3 },
        { name: 'Pasillo', x: 7.5, y: 4, width: 2.5, height: 3 },
        { name: 'Habitación Principal', x: 0, y: 7, width: 5, height: 4 },
        { name: 'Habitación 2', x: 5, y: 7, width: 5, height: 4 },
      ],
      walls: [
        // Outer walls
        { x1: 0, y1: 0, x2: 10, y2: 0 },
        { x1: 10, y1: 0, x2: 10, y2: 11 },
        { x1: 10, y1: 11, x2: 0, y2: 11 },
        { x1: 0, y1: 11, x2: 0, y2: 0 },
        // Inner walls
        { x1: 5, y1: 0, x2: 5, y2: 4 },
        { x1: 0, y1: 4, x2: 10, y2: 4 },
        { x1: 5, y1: 4, x2: 5, y2: 7 },
        { x1: 7.5, y1: 4, x2: 7.5, y2: 7 },
        { x1: 0, y1: 7, x2: 10, y2: 7 },
        { x1: 5, y1: 7, x2: 5, y2: 11 },
      ],
      doors: [
        { x: 2.5, y: 4 },
        { x: 6, y: 4 },
        { x: 9, y: 4 },
        { x: 2.5, y: 7 },
        { x: 7.5, y: 7 },
        { x: 5, y: 2 },
      ],
      router: { x: 5, y: 2 },
    };
  }
}
