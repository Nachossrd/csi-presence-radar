/**
 * Floor Plan Editor — Wi-Fi Radar · Digital Twin
 * Interactive drawing editor for creating house floor plans.
 */

export class FloorPlanEditor {
  /**
   * @param {HTMLCanvasElement} canvas
   */
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');

    // State
    this.active = false;
    this.currentTool = 'wall';

    // Data
    this.walls = [];
    this.rooms = [];
    this.router = null;
    this.doors = [];

    // Drawing state
    this._drawing = false;
    this._startPoint = null;
    this._currentPoint = null;
    this._roomPoints = [];
    this._hoveredElement = null;

    // History for undo
    this._history = [];
    this._maxHistory = 50;

    // Dimensions
    this.houseWidth = 10;
    this.houseLength = 11;
    this.snapToGrid = true;

    // Transform
    this._scale = 1;
    this._offsetX = 0;
    this._offsetY = 0;
    this._padding = 40;

    // Event handlers
    this._onMouseDown = this._handleMouseDown.bind(this);
    this._onMouseMove = this._handleMouseMove.bind(this);
    this._onMouseUp = this._handleMouseUp.bind(this);
    this._onKeyDown = this._handleKeyDown.bind(this);
    this._onDblClick = this._handleDblClick.bind(this);

    // Animation
    this._animFrame = null;

    // ResizeObserver
    this._resizeObserver = new ResizeObserver(() => this.resize());
    this._resizeObserver.observe(this.canvas.parentElement);
  }

  /* ─── Activation ─────────────────────────────── */

  activate() {
    this.active = true;
    this.canvas.style.cursor = 'crosshair';
    this.canvas.addEventListener('mousedown', this._onMouseDown);
    this.canvas.addEventListener('mousemove', this._onMouseMove);
    this.canvas.addEventListener('mouseup', this._onMouseUp);
    this.canvas.addEventListener('dblclick', this._onDblClick);
    window.addEventListener('keydown', this._onKeyDown);
    this.resize();
    this._startAnimation();
  }

  deactivate() {
    this.active = false;
    this.canvas.style.cursor = 'default';
    this.canvas.removeEventListener('mousedown', this._onMouseDown);
    this.canvas.removeEventListener('mousemove', this._onMouseMove);
    this.canvas.removeEventListener('mouseup', this._onMouseUp);
    this.canvas.removeEventListener('dblclick', this._onDblClick);
    window.removeEventListener('keydown', this._onKeyDown);
    cancelAnimationFrame(this._animFrame);
  }

  /* ─── Tool selection ─────────────────────────── */

  setTool(tool) {
    this.currentTool = tool;
    this._drawing = false;
    this._startPoint = null;
    this._roomPoints = [];

    if (tool === 'erase') {
      this.canvas.style.cursor = 'not-allowed';
    } else {
      this.canvas.style.cursor = 'crosshair';
    }
  }

  /* ─── Coordinate Helpers ─────────────────────── */

  _worldToCanvas(wx, wy) {
    return {
      x: this._offsetX + wx * this._scale,
      y: this._offsetY + wy * this._scale,
    };
  }

  _canvasToWorld(cx, cy) {
    let x = (cx - this._offsetX) / this._scale;
    let y = (cy - this._offsetY) / this._scale;

    if (this.snapToGrid) {
      x = Math.round(x * 2) / 2;
      y = Math.round(y * 2) / 2;
    }

    return { x, y };
  }

  _calculateTransform() {
    const container = this.canvas.parentElement;
    const headerH = container.querySelector('.view-header')?.offsetHeight || 0;
    const w = container.clientWidth;
    const h = container.clientHeight - headerH;
    const pad = this._padding;

    const scaleX = (w - pad * 2) / this.houseWidth;
    const scaleY = (h - pad * 2) / this.houseLength;
    this._scale = Math.min(scaleX, scaleY);

    this._offsetX = (w - this.houseWidth * this._scale) / 2;
    this._offsetY = (h - this.houseLength * this._scale) / 2;
  }

  /* ─── Event Handlers ─────────────────────────── */

  _getMousePos(e) {
    const rect = this.canvas.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  _handleMouseDown(e) {
    const canvasPos = this._getMousePos(e);
    const worldPos = this._canvasToWorld(canvasPos.x, canvasPos.y);

    switch (this.currentTool) {
      case 'wall':
        if (!this._drawing) {
          this._drawing = true;
          this._startPoint = worldPos;
        } else {
          this._saveState();
          this.walls.push({
            x1: this._startPoint.x,
            y1: this._startPoint.y,
            x2: worldPos.x,
            y2: worldPos.y,
          });
          this._drawing = false;
          this._startPoint = null;
        }
        break;

      case 'room':
        this._roomPoints.push(worldPos);
        break;

      case 'router':
        this._saveState();
        this.router = { x: worldPos.x, y: worldPos.y };
        break;

      case 'door':
        this._saveState();
        this.doors.push({ x: worldPos.x, y: worldPos.y });
        break;

      case 'erase':
        this._eraseAt(worldPos);
        break;
    }
  }

  _handleMouseMove(e) {
    const canvasPos = this._getMousePos(e);
    this._currentPoint = this._canvasToWorld(canvasPos.x, canvasPos.y);

    // Find hovered element for erase mode highlight
    if (this.currentTool === 'erase') {
      this._hoveredElement = this._findNearestElement(this._currentPoint);
    }
  }

  _handleMouseUp(_e) {
    // Not needed for current click-based tools
  }

  _handleDblClick(_e) {
    // Finalize room polygon on double-click
    if (this.currentTool === 'room' && this._roomPoints.length >= 3) {
      const name = prompt('Nombre de la habitación:', 'Sala') || 'Habitación';
      this._saveState();

      // Calculate bounding box from points
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      this._roomPoints.forEach((p) => {
        minX = Math.min(minX, p.x);
        minY = Math.min(minY, p.y);
        maxX = Math.max(maxX, p.x);
        maxY = Math.max(maxY, p.y);
      });

      this.rooms.push({
        name,
        x: minX,
        y: minY,
        width: maxX - minX,
        height: maxY - minY,
        points: [...this._roomPoints],
      });

      this._roomPoints = [];
    }
  }

  _handleKeyDown(e) {
    // Ctrl+Z undo
    if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
      e.preventDefault();
      this._undo();
    }

    // Escape to cancel current drawing
    if (e.key === 'Escape') {
      this._drawing = false;
      this._startPoint = null;
      this._roomPoints = [];
    }
  }

  /* ─── Erase Logic ────────────────────────────── */

  _findNearestElement(worldPos) {
    let nearest = null;
    let minDist = 0.5; // threshold in meters

    // Check walls
    this.walls.forEach((wall, i) => {
      const dist = this._pointToSegmentDist(worldPos, wall);
      if (dist < minDist) {
        minDist = dist;
        nearest = { type: 'wall', index: i };
      }
    });

    // Check rooms
    this.rooms.forEach((room, i) => {
      const cx = room.x + room.width / 2;
      const cy = room.y + room.height / 2;
      const dist = Math.hypot(worldPos.x - cx, worldPos.y - cy);
      if (dist < minDist) {
        minDist = dist;
        nearest = { type: 'room', index: i };
      }
    });

    // Check doors
    this.doors.forEach((door, i) => {
      const dist = Math.hypot(worldPos.x - door.x, worldPos.y - door.y);
      if (dist < minDist) {
        minDist = dist;
        nearest = { type: 'door', index: i };
      }
    });

    // Check router
    if (this.router) {
      const dist = Math.hypot(worldPos.x - this.router.x, worldPos.y - this.router.y);
      if (dist < minDist) {
        nearest = { type: 'router', index: 0 };
      }
    }

    return nearest;
  }

  _pointToSegmentDist(point, wall) {
    const dx = wall.x2 - wall.x1;
    const dy = wall.y2 - wall.y1;
    const lenSq = dx * dx + dy * dy;

    if (lenSq === 0) return Math.hypot(point.x - wall.x1, point.y - wall.y1);

    let t = ((point.x - wall.x1) * dx + (point.y - wall.y1) * dy) / lenSq;
    t = Math.max(0, Math.min(1, t));

    const nearX = wall.x1 + t * dx;
    const nearY = wall.y1 + t * dy;

    return Math.hypot(point.x - nearX, point.y - nearY);
  }

  _eraseAt(worldPos) {
    const elem = this._findNearestElement(worldPos);
    if (!elem) return;

    this._saveState();

    switch (elem.type) {
      case 'wall':
        this.walls.splice(elem.index, 1);
        break;
      case 'room':
        this.rooms.splice(elem.index, 1);
        break;
      case 'door':
        this.doors.splice(elem.index, 1);
        break;
      case 'router':
        this.router = null;
        break;
    }

    this._hoveredElement = null;
  }

  /* ─── History ────────────────────────────────── */

  _saveState() {
    this._history.push({
      walls: JSON.parse(JSON.stringify(this.walls)),
      rooms: JSON.parse(JSON.stringify(this.rooms)),
      doors: JSON.parse(JSON.stringify(this.doors)),
      router: this.router ? { ...this.router } : null,
    });

    if (this._history.length > this._maxHistory) {
      this._history.shift();
    }
  }

  _undo() {
    const state = this._history.pop();
    if (!state) return;

    this.walls = state.walls;
    this.rooms = state.rooms;
    this.doors = state.doors;
    this.router = state.router;
  }

  /* ─── Data ───────────────────────────────────── */

  clear() {
    this._saveState();
    this.walls = [];
    this.rooms = [];
    this.doors = [];
    this.router = null;
    this._drawing = false;
    this._startPoint = null;
    this._roomPoints = [];
  }

  /**
   * Export floor plan as JSON.
   * @returns {object}
   */
  getFloorPlan() {
    const plan = {
      width: this.houseWidth,
      length: this.houseLength,
      walls: this.walls.map((w) => ({
        x1: w.x1,
        y1: w.y1,
        x2: w.x2,
        y2: w.y2,
      })),
      rooms: this.rooms.map((r) => ({
        name: r.name,
        x: r.x,
        y: r.y,
        width: r.width,
        height: r.height,
      })),
      doors: this.doors.map((d) => ({
        x: d.x,
        y: d.y,
      })),
    };

    if (this.router) {
      plan.router = { x: this.router.x, y: this.router.y };
    }

    return plan;
  }

  /**
   * Load floor plan data into the editor.
   * @param {object} data
   */
  loadFloorPlan(data) {
    if (!data) return;

    this.houseWidth = data.width || 10;
    this.houseLength = data.length || data.height || 11;
    this.walls = (data.walls || []).map((w) => ({ ...w }));
    this.rooms = (data.rooms || []).map((r) => ({ ...r }));
    this.doors = (data.doors || []).map((d) => ({ ...d }));
    this.router = data.router ? { ...data.router } : null;

    // Update dimension inputs
    const widthInput = document.getElementById('house-width');
    const lengthInput = document.getElementById('house-length');
    if (widthInput) widthInput.value = this.houseWidth;
    if (lengthInput) lengthInput.value = this.houseLength;

    this._calculateTransform();
  }

  /* ─── Rendering ──────────────────────────────── */

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
  }

  _startAnimation() {
    const loop = () => {
      if (!this.active) return;
      this._animFrame = requestAnimationFrame(loop);
      this.render();
    };
    loop();
  }

  render() {
    const ctx = this.ctx;
    const container = this.canvas.parentElement;
    const headerH = container.querySelector('.view-header')?.offsetHeight || 0;
    const w = container.clientWidth;
    const h = container.clientHeight - headerH;

    // Clear
    ctx.clearRect(0, 0, w, h);

    // Background
    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, w, h);

    // Grid
    this._drawGrid(ctx);

    // Rooms
    this.rooms.forEach((room, i) => {
      const highlight = this._hoveredElement?.type === 'room' && this._hoveredElement.index === i;
      this._drawRoom(ctx, room, highlight);
    });

    // Walls
    this.walls.forEach((wall, i) => {
      const highlight = this._hoveredElement?.type === 'wall' && this._hoveredElement.index === i;
      this._drawWall(ctx, wall, highlight);
    });

    // Doors
    this.doors.forEach((door, i) => {
      const highlight = this._hoveredElement?.type === 'door' && this._hoveredElement.index === i;
      this._drawDoor(ctx, door, highlight);
    });

    // Router
    if (this.router) {
      const highlight = this._hoveredElement?.type === 'router';
      this._drawRouter(ctx, this.router, highlight);
    }

    // Current wall being drawn
    if (this._drawing && this._startPoint && this._currentPoint && this.currentTool === 'wall') {
      const p1 = this._worldToCanvas(this._startPoint.x, this._startPoint.y);
      const p2 = this._worldToCanvas(this._currentPoint.x, this._currentPoint.y);

      ctx.save();
      ctx.strokeStyle = '#00d4ff';
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 3]);
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();

      // Start dot
      ctx.fillStyle = '#00d4ff';
      ctx.beginPath();
      ctx.arc(p1.x, p1.y, 4, 0, Math.PI * 2);
      ctx.fill();

      ctx.restore();
    }

    // Room points being collected
    if (this.currentTool === 'room' && this._roomPoints.length > 0) {
      ctx.save();
      ctx.strokeStyle = '#bf5af2';
      ctx.fillStyle = 'rgba(191, 90, 242, 0.1)';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);

      ctx.beginPath();
      const p0 = this._worldToCanvas(this._roomPoints[0].x, this._roomPoints[0].y);
      ctx.moveTo(p0.x, p0.y);

      this._roomPoints.forEach((pt) => {
        const p = this._worldToCanvas(pt.x, pt.y);
        ctx.lineTo(p.x, p.y);
      });

      if (this._currentPoint) {
        const pc = this._worldToCanvas(this._currentPoint.x, this._currentPoint.y);
        ctx.lineTo(pc.x, pc.y);
      }

      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      // Draw points
      this._roomPoints.forEach((pt) => {
        const p = this._worldToCanvas(pt.x, pt.y);
        ctx.fillStyle = '#bf5af2';
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
        ctx.fill();
      });

      ctx.restore();
    }

    // Cursor crosshair at current position
    if (this._currentPoint && this.currentTool !== 'erase') {
      const p = this._worldToCanvas(this._currentPoint.x, this._currentPoint.y);
      ctx.save();
      ctx.strokeStyle = 'rgba(255,255,255,0.2)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);

      ctx.beginPath();
      ctx.moveTo(p.x - 12, p.y);
      ctx.lineTo(p.x + 12, p.y);
      ctx.moveTo(p.x, p.y - 12);
      ctx.lineTo(p.x, p.y + 12);
      ctx.stroke();

      // Coordinate label
      ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(255,255,255,0.5)';
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.fillText(
        `(${this._currentPoint.x.toFixed(1)}, ${this._currentPoint.y.toFixed(1)})`,
        p.x + 14,
        p.y - 6,
      );

      ctx.restore();
    }

    // Tool indicator
    ctx.save();
    ctx.fillStyle = '#00d4ff';
    ctx.font = '600 11px Inter, sans-serif';
    ctx.textAlign = 'left';
    const toolNames = { wall: '🧱 Pared', room: '🏠 Habitación', router: '📡 Router', door: '🚪 Puerta', erase: '🗑️ Borrar' };
    ctx.fillText(`Herramienta: ${toolNames[this.currentTool] || this.currentTool}`, 10, h - 10);
    ctx.restore();
  }

  _drawGrid(ctx) {
    ctx.save();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 4]);

    for (let x = 0; x <= this.houseWidth; x += 0.5) {
      const p = this._worldToCanvas(x, 0);
      const p2 = this._worldToCanvas(x, this.houseLength);
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }

    for (let y = 0; y <= this.houseLength; y += 0.5) {
      const p = this._worldToCanvas(0, y);
      const p2 = this._worldToCanvas(this.houseWidth, y);
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }

    // Major grid lines (whole meters)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.setLineDash([]);

    for (let x = 0; x <= this.houseWidth; x++) {
      const p = this._worldToCanvas(x, 0);
      const p2 = this._worldToCanvas(x, this.houseLength);
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }

    for (let y = 0; y <= this.houseLength; y++) {
      const p = this._worldToCanvas(0, y);
      const p2 = this._worldToCanvas(this.houseWidth, y);
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }

    // Labels
    ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.font = '9px JetBrains Mono, monospace';
    ctx.textAlign = 'center';

    for (let x = 0; x <= this.houseWidth; x++) {
      const p = this._worldToCanvas(x, 0);
      ctx.fillText(`${x}`, p.x, p.y - 6);
    }

    ctx.textAlign = 'right';
    for (let y = 0; y <= this.houseLength; y++) {
      const p = this._worldToCanvas(0, y);
      ctx.fillText(`${y}`, p.x - 6, p.y + 3);
    }

    ctx.restore();
  }

  _drawRoom(ctx, room, highlight = false) {
    const p = this._worldToCanvas(room.x, room.y);
    const sw = room.width * this._scale;
    const sh = room.height * this._scale;

    ctx.save();

    // Fill
    ctx.fillStyle = highlight ? 'rgba(255, 45, 85, 0.15)' : 'rgba(191, 90, 242, 0.08)';
    ctx.fillRect(p.x, p.y, sw, sh);

    // Border
    ctx.strokeStyle = highlight ? '#ff2d55' : 'rgba(191, 90, 242, 0.3)';
    ctx.lineWidth = highlight ? 2 : 1;
    ctx.strokeRect(p.x, p.y, sw, sh);

    // Label
    ctx.fillStyle = highlight ? '#ff2d55' : 'rgba(191, 90, 242, 0.7)';
    ctx.font = '600 11px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(room.name || '', p.x + sw / 2, p.y + sh / 2);

    ctx.restore();
  }

  _drawWall(ctx, wall, highlight = false) {
    const p1 = this._worldToCanvas(wall.x1, wall.y1);
    const p2 = this._worldToCanvas(wall.x2, wall.y2);

    ctx.save();
    ctx.strokeStyle = highlight ? '#ff2d55' : 'rgba(200, 200, 220, 0.9)';
    ctx.lineWidth = highlight ? 4 : 3;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
    ctx.restore();
  }

  _drawDoor(ctx, door, highlight = false) {
    const p = this._worldToCanvas(door.x, door.y);
    const size = 6;

    ctx.save();
    ctx.fillStyle = highlight ? '#ff2d55' : '#00d4ff';
    ctx.globalAlpha = highlight ? 0.9 : 0.6;
    ctx.fillRect(p.x - size, p.y - size, size * 2, size * 2);
    ctx.strokeStyle = highlight ? '#ff2d55' : '#00d4ff';
    ctx.lineWidth = 1;
    ctx.globalAlpha = 1;
    ctx.strokeRect(p.x - size, p.y - size, size * 2, size * 2);
    ctx.restore();
  }

  _drawRouter(ctx, router, highlight = false) {
    const p = this._worldToCanvas(router.x, router.y);
    const s = 8;

    ctx.save();
    ctx.fillStyle = highlight ? '#ff2d55' : '#00ff88';
    ctx.globalAlpha = 0.8;
    ctx.beginPath();
    ctx.moveTo(p.x, p.y - s);
    ctx.lineTo(p.x + s, p.y);
    ctx.lineTo(p.x, p.y + s);
    ctx.lineTo(p.x - s, p.y);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  /** Dispose and clean up */
  dispose() {
    this.deactivate();
    this._resizeObserver?.disconnect();
  }
}
