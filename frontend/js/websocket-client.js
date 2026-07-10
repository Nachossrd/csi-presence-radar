/**
 * WebSocket Client — Wi-Fi Radar · Digital Twin
 * EventTarget-based WebSocket client with auto-reconnect and exponential backoff.
 */

export class WebSocketClient extends EventTarget {
  /** @param {string} url  WebSocket server URL (ws:// or wss://) */
  constructor(url) {
    super();
    this.url = url;
    /** @type {WebSocket|null} */
    this._ws = null;
    this._reconnectDelay = 1000;
    this._maxReconnectDelay = 30000;
    this._shouldReconnect = true;
    this._reconnectTimer = null;
    this._connected = false;
  }

  /** Whether the WebSocket connection is currently open */
  get isConnected() {
    return this._connected;
  }

  /** Establish the WebSocket connection */
  connect() {
    this._shouldReconnect = true;
    this._createConnection();
  }

  /** Gracefully disconnect and stop reconnection attempts */
  disconnect() {
    this._shouldReconnect = false;
    clearTimeout(this._reconnectTimer);
    if (this._ws) {
      this._ws.close(1000, 'Client disconnect');
      this._ws = null;
    }
    this._setConnected(false);
  }

  /**
   * Send JSON data through the WebSocket.
   * @param {object} data - Data to serialize and send
   */
  send(data) {
    if (!this._ws || this._ws.readyState !== WebSocket.OPEN) {
      console.warn('[WS] Cannot send — not connected');
      return;
    }
    try {
      this._ws.send(JSON.stringify(data));
    } catch (err) {
      console.error('[WS] Send error:', err);
    }
  }

  /* -------- Internal -------- */

  _createConnection() {
    // Clean up previous socket
    if (this._ws) {
      this._ws.onopen = null;
      this._ws.onclose = null;
      this._ws.onerror = null;
      this._ws.onmessage = null;
      if (this._ws.readyState === WebSocket.OPEN || this._ws.readyState === WebSocket.CONNECTING) {
        this._ws.close();
      }
    }

    try {
      this._ws = new WebSocket(this.url);
    } catch (err) {
      console.error('[WS] Connection creation failed:', err);
      this._scheduleReconnect();
      return;
    }

    this._ws.onopen = () => {
      console.log('[WS] Connected to', this.url);
      this._reconnectDelay = 1000; // reset backoff
      this._setConnected(true);
      this.dispatchEvent(new Event('open'));
    };

    this._ws.onclose = (event) => {
      console.log(`[WS] Closed (code=${event.code}, reason="${event.reason}")`);
      this._setConnected(false);
      this.dispatchEvent(new CustomEvent('close', { detail: { code: event.code, reason: event.reason } }));
      this._scheduleReconnect();
    };

    this._ws.onerror = (event) => {
      console.error('[WS] Error:', event);
      this.dispatchEvent(new CustomEvent('error', { detail: event }));
    };

    this._ws.onmessage = (event) => {
      this._handleMessage(event.data);
    };
  }

  _handleMessage(raw) {
    let data;
    try {
      data = JSON.parse(raw);
    } catch {
      console.warn('[WS] Non-JSON message:', raw);
      return;
    }

    // Dispatch typed events based on message type
    const type = data.type || data.event;
    switch (type) {
      case 'position':
      case 'position_update':
      case 'position-update':
        this.dispatchEvent(new CustomEvent('position-update', { detail: data }));
        break;

      case 'wifi':
      case 'wifi_scan':
      case 'wifi-update':
      case 'scan_result':
        this.dispatchEvent(new CustomEvent('wifi-update', { detail: data }));
        break;

      case 'zone':
      case 'zone_update':
        this.dispatchEvent(new CustomEvent('zone-update', { detail: data }));
        break;

      case 'status':
      case 'tracking_status':
        this.dispatchEvent(new CustomEvent('status-update', { detail: data }));
        break;

      case 'radar_event':
      case 'radar-event':
        this.dispatchEvent(new CustomEvent('radar-event', { detail: data }));
        break;

      default:
        if (Array.isArray(data.persons)) {
          this.dispatchEvent(new CustomEvent('position-update', { detail: data }));
          break;
        }
        // Generic message event for unknown types
        this.dispatchEvent(new CustomEvent('message', { detail: data }));
        break;
    }
  }

  _scheduleReconnect() {
    if (!this._shouldReconnect) return;

    clearTimeout(this._reconnectTimer);
    console.log(`[WS] Reconnecting in ${this._reconnectDelay}ms…`);

    this._reconnectTimer = setTimeout(() => {
      this._createConnection();
    }, this._reconnectDelay);

    // Exponential backoff with jitter
    this._reconnectDelay = Math.min(
      this._reconnectDelay * 1.5 + Math.random() * 500,
      this._maxReconnectDelay
    );
  }

  _setConnected(connected) {
    this._connected = connected;
    this._updateStatusUI(connected);
  }

  _updateStatusUI(connected) {
    const indicator = document.getElementById('ws-status');
    if (!indicator) return;

    const dot = indicator.querySelector('.status-dot');
    const text = indicator.querySelector('.status-text');

    if (connected) {
      dot?.classList.add('connected');
      if (text) text.textContent = 'Conectado';
    } else {
      dot?.classList.remove('connected');
      if (text) text.textContent = 'Desconectado';
    }
  }
}
