/**
 * 2D Floor Plan Renderer — Wi-Fi Radar · Digital Twin
 * Canvas-based 2D visualization of the house floor plan with person tracking.
 */

export class FloorPlan2D {
  /**
   * @param {HTMLCanvasElement} canvas
   */
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');

    // Data
    this.floorPlan = null;
    this.personPos = null;       // { x, y, confidence }
    this.persons = new Map();
    this.trailPositions = [];
    this.personTrails = new Map();
    this.maxTrail = 80;
    this.fingerprints = [];

    // Display toggles
    this.showGrid = true;
    this.showLabels = true;
    this.showFingerprints = false;

    // Collection mode
    this._collectionMode = false;
    this._collectionCallback = null;
    this._clickHandler = null;

    // Animation
    this._animFrame = null;
    this._pulsePhase = 0;

    // Transform (world → canvas)
    this._scale = 1;
    this._offsetX = 0;
    this._offsetY = 0;
    this._padding = 40;

    // Room colors
    this._roomColors = {
      Sala: '#00d4ff',
      Cocina: '#ff9500',
      'Habitación': '#bf5af2',
      'Habitación Principal': '#bf5af2',
      'Habitación 2': '#5e5ce6',
      Comedor: '#30d158',
      'Baño': '#ff375f',
      'Baño 2': '#ff6482',
      Pasillo: '#636366',
      Garaje: '#8e8e93',
      Entrada: '#ffd60a',
      Oficina: '#64d2ff',
      Lavandería: '#ac8e68',
      default: '#48484a',
    };

    // ResizeObserver
    this._resizeObserver = new ResizeObserver(() => this.resize());
    this._resizeObserver.observe(this.canvas.parentElement);

    this.resize();
    this._startAnimation();
  }

  /** Get a color for a room name */
  _getRoomColor(name) {
    for (const [key, color] of Object.entries(this._roomColors)) {
      if (name && name.toLowerCase().includes(key.toLowerCase())) return color;
    }
    return this._roomColors.default;
  }

  /** Convert world coordinates (meters) to canvas pixels */
  _worldToCanvas(wx, wy) {
    return {
      x: this._offsetX + wx * this._scale,
      y: this._offsetY + wy * this._scale,
    };
  }

  /** Convert canvas pixels to world coordinates (meters) */
  _canvasToWorld(cx, cy) {
    return {
      x: (cx - this._offsetX) / this._scale,
      y: (cy - this._offsetY) / this._scale,
    };
  }

  /**
   * Load floor plan data.
   * @param {object} data - Floor plan JSON
   */
  loadFloorPlan(data) {
    this.floorPlan = data;
    this._calculateTransform();
    this.render();
  }

  /** Recalculate scale and offset to fit the plan in the canvas */
  _calculateTransform() {
    if (!this.floorPlan) return;

    const w = this.canvas.width;
    const h = this.canvas.height;
    const fw = this.floorPlan.width || 10;
    const fh = this.floorPlan.length || this.floorPlan.height || 11;
    const pad = this._padding;

    const scaleX = (w - pad * 2) / fw;
    const scaleY = (h - pad * 2) / fh;
    this._scale = Math.min(scaleX, scaleY);

    // Center
    this._offsetX = (w - fw * this._scale) / 2;
    this._offsetY = (h - fh * this._scale) / 2;
  }

  /** Handle canvas resize */
  resize() {
    const container = this.canvas.parentElement;
    const headerH = container.querySelector('.view-header')?.offsetHeight || 0;
    const w = container.clientWidth;
    const h = container.clientHeight - headerH;

    if (w <= 0 || h <= 0) return;

    const dpr = Math.min(window.devicePixelRatio, 2);
    this.canvas.width = w * dpr;
    this.canvas.height = h * dpr;
    this.canvas.style.width = w + 'px';
    this.canvas.style.height = h + 'px';
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    this._calculateTransform();
    this.render();
  }

  /** Start render animation loop */
  _startAnimation() {
    const loop = () => {
      this._animFrame = requestAnimationFrame(loop);
      this._pulsePhase += 0.04;
      this.render();
    };
    loop();
  }

  /** Main render method */
  render() {
    const ctx = this.ctx;
    const w = this.canvas.width / (Math.min(window.devicePixelRatio, 2));
    const h = this.canvas.height / (Math.min(window.devicePixelRatio, 2));

    // Clear
    ctx.clearRect(0, 0, w, h);

    // Background
    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, w, h);

    if (!this.floorPlan) {
      ctx.fillStyle = '#55556a';
      ctx.font = '14px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Cargue un plano para visualizar', w / 2, h / 2);
      return;
    }

    // Grid
    if (this.showGrid) {
      this._drawGrid(ctx);
    }

    // Rooms
    if (this.floorPlan.rooms) {
      this.floorPlan.rooms.forEach((room) => this._drawRoom(ctx, room));
    }

    // Walls
    if (this.floorPlan.walls) {
      this.floorPlan.walls.forEach((wall) => this._drawWall(ctx, wall));
    }

    // Doors
    if (this.floorPlan.doors) {
      this.floorPlan.doors.forEach((door) => this._drawDoor(ctx, door));
    }

    // Router
    if (this.floorPlan.router) {
      this._drawRouter(ctx, this.floorPlan.router);
    }
    if (this.floorPlan.routers) {
      this.floorPlan.routers.forEach((r) => this._drawRouter(ctx, r));
    }

    // Room labels
    if (this.showLabels && this.floorPlan.rooms) {
      this.floorPlan.rooms.forEach((room) => this._drawRoomLabel(ctx, room));
    }

    // Fingerprints
    if (this.showFingerprints && this.fingerprints.length) {
      this.fingerprints.forEach((fp, i) => this._drawFingerprint(ctx, fp, i));
    }

    if (this.persons.size) {
      this.persons.forEach((person) => this._drawPersonTrail(ctx, person.id));
      this.persons.forEach((person) => this._drawPerson(ctx, person));
    } else {
      // Trail
      this._drawTrail(ctx);

      // Person
      if (this.personPos) {
        this._drawPerson(ctx, this.personPos);
      }
    }

    // Collection mode indicator
    if (this._collectionMode) {
      ctx.save();
      ctx.fillStyle = 'rgba(0, 212, 255, 0.08)';
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = '#00d4ff';
      ctx.font = '12px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('📍 Modo Recolección — Haz clic para registrar posición', w / 2, 20);
      ctx.restore();
    }
  }

  _drawGrid(ctx) {
    if (!this.floorPlan) return;

    const fw = this.floorPlan.width || 10;
    const fh = this.floorPlan.length || this.floorPlan.height || 11;

    ctx.save();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 4]);

    // Vertical lines
    for (let x = 0; x <= fw; x++) {
      const p = this._worldToCanvas(x, 0);
      const p2 = this._worldToCanvas(x, fh);
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }

    // Horizontal lines
    for (let y = 0; y <= fh; y++) {
      const p = this._worldToCanvas(0, y);
      const p2 = this._worldToCanvas(fw, y);
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }

    // Labels
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.font = '9px JetBrains Mono, monospace';
    ctx.textAlign = 'center';

    for (let x = 0; x <= fw; x += 2) {
      const p = this._worldToCanvas(x, 0);
      ctx.fillText(`${x}m`, p.x, p.y - 6);
    }

    ctx.textAlign = 'right';
    for (let y = 0; y <= fh; y += 2) {
      const p = this._worldToCanvas(0, y);
      ctx.fillText(`${y}m`, p.x - 6, p.y + 3);
    }

    ctx.restore();
  }

  _drawRoom(ctx, room) {
    const x = room.x || 0;
    const y = room.y || 0;
    const w = room.width || 2;
    const h = room.height || room.length || 2;
    const color = this._getRoomColor(room.name);

    const p = this._worldToCanvas(x, y);
    const sw = w * this._scale;
    const sh = h * this._scale;

    ctx.save();
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.08;
    ctx.fillRect(p.x, p.y, sw, sh);

    // Room border
    ctx.globalAlpha = 0.25;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.strokeRect(p.x, p.y, sw, sh);
    ctx.restore();
  }

  _drawRoomLabel(ctx, room) {
    const x = room.x || 0;
    const y = room.y || 0;
    const w = room.width || 2;
    const h = room.height || room.length || 2;
    const color = this._getRoomColor(room.name);

    const center = this._worldToCanvas(x + w / 2, y + h / 2);

    ctx.save();
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.7;
    ctx.font = '600 11px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(room.name || '', center.x, center.y);
    ctx.restore();
  }

  _drawWall(ctx, wall) {
    const x1 = wall.x1 ?? wall.start?.[0] ?? 0;
    const y1 = wall.y1 ?? wall.start?.[1] ?? 0;
    const x2 = wall.x2 ?? wall.end?.[0] ?? 0;
    const y2 = wall.y2 ?? wall.end?.[1] ?? 0;

    const p1 = this._worldToCanvas(x1, y1);
    const p2 = this._worldToCanvas(x2, y2);

    ctx.save();
    ctx.strokeStyle = 'rgba(200, 200, 220, 0.9)';
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
    ctx.restore();
  }

  _drawDoor(ctx, door) {
    const x = door.x ?? 0;
    const y = door.y ?? 0;
    const p = this._worldToCanvas(x, y);

    ctx.save();
    ctx.fillStyle = '#00d4ff';
    ctx.globalAlpha = 0.6;
    const size = 6;
    ctx.fillRect(p.x - size, p.y - size, size * 2, size * 2);
    ctx.strokeStyle = '#00d4ff';
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.9;
    ctx.strokeRect(p.x - size, p.y - size, size * 2, size * 2);
    ctx.restore();
  }

  _drawRouter(ctx, router) {
    const x = router.x ?? 0;
    const y = router.y ?? 0;
    const p = this._worldToCanvas(x, y);

    ctx.save();
    // Diamond shape
    const s = 8;
    ctx.fillStyle = '#00ff88';
    ctx.globalAlpha = 0.8;
    ctx.beginPath();
    ctx.moveTo(p.x, p.y - s);
    ctx.lineTo(p.x + s, p.y);
    ctx.lineTo(p.x, p.y + s);
    ctx.lineTo(p.x - s, p.y);
    ctx.closePath();
    ctx.fill();

    // Glow
    ctx.globalAlpha = 0.15;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 16, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();
  }

  _drawFingerprint(ctx, fp, index) {
    const p = this._worldToCanvas(fp.x, fp.y);

    ctx.save();
    ctx.fillStyle = '#00ff88';
    ctx.globalAlpha = 0.6;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
    ctx.fill();

    // Number
    ctx.globalAlpha = 1;
    ctx.fillStyle = '#000';
    ctx.font = '700 7px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(index + 1), p.x, p.y);

    ctx.restore();
  }

  _drawTrail(ctx) {
    if (this.trailPositions.length < 2) return;

    ctx.save();
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    const len = this.trailPositions.length;
    for (let i = 1; i < len; i++) {
      const alpha = (i / len) * 0.5;
      ctx.strokeStyle = `rgba(255, 45, 85, ${alpha})`;
      ctx.beginPath();
      const p1 = this._worldToCanvas(this.trailPositions[i - 1].x, this.trailPositions[i - 1].y);
      const p2 = this._worldToCanvas(this.trailPositions[i].x, this.trailPositions[i].y);
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }

    ctx.restore();
  }

  _drawPerson(ctx, person = this.personPos) {
    if (!person) return;

    const p = this._worldToCanvas(person.x, person.y);
    const conf = person.confidence || 0;
    const color = this._personColor(person.id);

    ctx.save();

    // Pulsing rings
    const numRings = 3;
    for (let i = 0; i < numRings; i++) {
      const phase = (this._pulsePhase + i * 0.7) % (Math.PI * 2);
      const ringRadius = 12 + Math.sin(phase) * 8 + i * 10;
      const ringAlpha = Math.max(0, 0.25 - i * 0.08) * (0.5 + Math.sin(phase) * 0.5);

      ctx.beginPath();
      ctx.arc(p.x, p.y, ringRadius, 0, Math.PI * 2);
      ctx.strokeStyle = this._rgba(color, ringAlpha);
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    // Confidence halo
    const haloRadius = 10 + (1 - conf) * 20;
    const gradient = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, haloRadius);
    gradient.addColorStop(0, this._rgba(color, 0.2));
    gradient.addColorStop(1, this._rgba(color, 0));
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(p.x, p.y, haloRadius, 0, Math.PI * 2);
    ctx.fill();

    // Main dot
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 7, 0, Math.PI * 2);
    ctx.fill();

    // White center
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.arc(p.x, p.y, 2.5, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#fff';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(person.id ?? ''), p.x, p.y - 16);

    ctx.restore();
  }

  /**
   * Update tracked person position.
   * @param {number} x - X in meters
   * @param {number} y - Y in meters
   * @param {number} [confidence=0] - 0-1 confidence
   */
  updatePersonPosition(x, y, confidence = 0) {
    this.personPos = { x, y, confidence };

    this.trailPositions.push({ x, y });
    if (this.trailPositions.length > this.maxTrail) {
      this.trailPositions.shift();
    }
  }

  updatePersons(persons) {
    this.persons.clear();

    persons.forEach((person, index) => {
      const id = person.id ?? index + 1;
      const x = person.x ?? person.position?.x ?? 0;
      const y = person.y ?? person.position?.y ?? 0;
      const confidence = person.confidence ?? 0;
      const normalized = { id, x, y, confidence };
      this.persons.set(String(id), normalized);

      const trail = this.personTrails.get(String(id)) || [];
      trail.push({ x, y });
      if (trail.length > this.maxTrail) {
        trail.shift();
      }
      this.personTrails.set(String(id), trail);
    });
  }

  _drawPersonTrail(ctx, id) {
    const trail = this.personTrails.get(String(id));
    if (!trail || trail.length < 2) return;

    ctx.save();
    ctx.strokeStyle = this._rgba(this._personColor(id), 0.35);
    ctx.lineWidth = 2;
    ctx.beginPath();
    trail.forEach((pos, index) => {
      const p = this._worldToCanvas(pos.x, pos.y);
      if (index === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    });
    ctx.stroke();
    ctx.restore();
  }

  _personColor(id) {
    const colors = {
      1: '#ff2d55',
      2: '#00d4ff',
      3: '#00ff88',
    };
    return colors[id] || '#ffb000';
  }

  _rgba(hex, alpha) {
    const clean = hex.replace('#', '');
    const value = parseInt(clean, 16);
    const r = (value >> 16) & 255;
    const g = (value >> 8) & 255;
    const b = value & 255;
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  /**
   * Set fingerprint data.
   * @param {Array<{x: number, y: number}>} fingerprints
   */
  setFingerprints(fingerprints) {
    this.fingerprints = fingerprints || [];
  }

  /**
   * Enable collection mode — clicks on canvas call the callback with (x, y) in meters.
   * @param {function(number, number)} callback
   */
  enableCollectionMode(callback) {
    this._collectionMode = true;
    this._collectionCallback = callback;
    this.canvas.style.cursor = 'crosshair';

    this._clickHandler = (e) => {
      if (!this._collectionMode || !this._collectionCallback) return;

      const rect = this.canvas.getBoundingClientRect();
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;
      const world = this._canvasToWorld(cx, cy);

      // Clamp to floor plan bounds
      const fw = this.floorPlan?.width || 10;
      const fh = this.floorPlan?.length || this.floorPlan?.height || 11;
      const wx = Math.max(0, Math.min(fw, Math.round(world.x * 2) / 2));
      const wy = Math.max(0, Math.min(fh, Math.round(world.y * 2) / 2));

      this._collectionCallback(wx, wy);
    };

    this.canvas.addEventListener('click', this._clickHandler);
  }

  /** Disable collection mode */
  disableCollectionMode() {
    this._collectionMode = false;
    this._collectionCallback = null;
    this.canvas.style.cursor = 'default';

    if (this._clickHandler) {
      this.canvas.removeEventListener('click', this._clickHandler);
      this._clickHandler = null;
    }
  }

  toggleGrid() {
    this.showGrid = !this.showGrid;
    return this.showGrid;
  }

  toggleLabels() {
    this.showLabels = !this.showLabels;
    return this.showLabels;
  }

  toggleFingerprints() {
    this.showFingerprints = !this.showFingerprints;
    return this.showFingerprints;
  }

  /** Dispose and clean up */
  dispose() {
    cancelAnimationFrame(this._animFrame);
    this._resizeObserver?.disconnect();
    this.disableCollectionMode();
  }
}
