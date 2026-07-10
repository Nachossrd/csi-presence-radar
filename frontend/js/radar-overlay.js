/**
 * Radar Overlay — adds the Wi-Fi / BLE / Tag visualization layer on top of
 * the existing 3D house scene.
 *
 * Concept:
 *   - Anchors live at fixed (x, y, z) positions: routers, BLE devices, the laptop.
 *   - For each anchor we draw a colored "ray" — a line from the anchor to the laptop.
 *     This is the EM line-of-sight that gets disturbed when someone crosses it.
 *   - When the backend broadcasts a radar_event of state="motion"/"moving" for an
 *     anchor, we pulse its ray, ramp emissive intensity on its marker, and drop
 *     a fading heatmap sphere at the ray midpoint.
 *   - When a tag fires (kind="tag_fired"), we flash a larger banner sphere.
 *
 * Data source: GET /api/anchors  +  WebSocket `radar_event` messages.
 */

import * as THREE from 'three';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

const RAY_FADE_MS = 2200;       // how long a motion pulse remains visible
const HEAT_FADE_MS = 2800;      // heatmap sphere fade
const TAG_FLASH_MS = 4000;      // tag-fired banner duration

// Path-loss model: distance from RSSI in dBm.
//   RSSI = TxPower(@1m) - 10 * n * log10(d)
//   d = 10 ^ ((TxPower - RSSI) / (10n))
// n is the path-loss exponent (2.0 free space, 3.0 typical indoor, 4.0 dense walls).
const TX_POWER_BY_TYPE = {
  wifi_ap:    -40,   // dBm at 1m for a typical home router
  ble:        -55,   // dBm at 1m for a BLE peripheral
  ble_mobile: -55,
};
const PATH_LOSS_N = 3.0;

function rssiToDistance(rssiDbm, type, txOverride) {
  const tx = (txOverride != null) ? txOverride : (TX_POWER_BY_TYPE[type] ?? -50);
  const d = Math.pow(10, (tx - rssiDbm) / (10 * PATH_LOSS_N));
  return Math.max(0.3, Math.min(30, d));   // clamp to plausible indoor range
}

function signalPctToRssi(pct) {
  // Windows reports % only; rough but standard mapping to dBm
  return Math.max(-100, Math.min(-30, pct / 2 - 100));
}

// EWMA smoothing factor per anchor type. The lower alpha is, the more
// samples it takes to follow a real movement — but the better outliers
// (multipath fading, body shadowing) get rejected. Static objects need
// heavy smoothing because they should NOT move; mobile ones less.
const SMOOTHING_ALPHA = {
  wifi_ap:    0.04,   // routers, hotspots placed on a table
  ble:        0.04,   // TVs, plugs, anything not on a person
  ble_mobile: 0.25,   // smartwatch, phone on the user's body
  wifi_mobile: 0.25,
};
const SMOOTHING_ALPHA_DEFAULT = 0.10;

// Deadband: minimum distance delta (in meters) for the anchor's visual
// position to update. Sub-deadband micro-oscillations are invisible, which
// makes static anchors look STATIC even when RSSI still wobbles a bit.
const DEADBAND_M = {
  wifi_ap:    0.5,
  ble:        0.5,
  ble_mobile: 0.0,    // mobile devices: always reflect every change
  wifi_mobile: 0.0,
};
const DEADBAND_DEFAULT = 0.3;

function _hexToRgb(hex) {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex || '#ffffff');
  return m ? [parseInt(m[1], 16) / 255, parseInt(m[2], 16) / 255, parseInt(m[3], 16) / 255]
           : [1, 1, 1];
}

function _toThreeColor(hex) {
  const [r, g, b] = _hexToRgb(hex);
  return new THREE.Color(r, g, b);
}

export class RadarOverlay {
  /**
   * @param {THREE.Scene} scene
   * @param {THREE.Group} labelGroup - existing CSS2D label group from HouseScene3D
   */
  constructor(scene, labelGroup) {
    this.scene = scene;
    this.labelGroup = labelGroup;

    this.root = new THREE.Group();
    this.root.name = 'RadarOverlay';
    this.scene.add(this.root);

    /** @type {Map<string, object>} anchor_id -> { mesh, ray, light, color, label } */
    this.anchors = new Map();

    /** {x,y,z} of the laptop, fixed endpoint for all rays */
    this.laptopPos = new THREE.Vector3(1, 1.2, 1);
    this.laptopMesh = null;

    /** active heatmap spheres (animated and culled) */
    this.heats = [];
    /** active tag-fired banners */
    this.tagFlashes = [];

    /** People detected by the camera (label -> { mesh, light, labelObj, target, current, lastSeen }) */
    this.people = new Map();
    /** Aggregated presence panel (DOM element) — created lazily on first snapshot */
    this._presencePanel = null;
    /** "People in house" master panel (DOM element, aggregates all sources) */
    this._peopleInHousePanel = null;
    /** How long after no update we drop a person marker */
    this._personStaleMs = 3500;
    /** Tracking: per-source recent activity for the in-house aggregator */
    this._cameraVisible = [];                    // [{label, x_m, y_m, score, identified, lastSeen}]
    this._bleLabeled = new Map();                // mac -> {label, state, rssi, lastSeen}
    this._anchorMotion = new Map();              // anchor_id -> {name, source, lastSeen}
    this._motionStaleMs = 6000;
    this._bleStaleMs = 12000;
    /** Rooms loaded from floor_plan.json — for "Yo en el Living" labels */
    this._rooms = [];
    this._roomsLoaded = false;
    /** System status panel (top-left) */
    this._statusPanel = null;
    /** Latest status snapshot (polled every 2s) */
    this._lastSystemStatus = null;

    this._tmp = new THREE.Vector3();
    this._lastUpdate = performance.now();
  }

  /** Apply config returned from /api/anchors. */
  setConfig(cfg) {
    if (!cfg) return;
    this._cfg = cfg; // keep reference for drag-save
    if (cfg.laptop) {
      this.laptopPos.set(cfg.laptop.x ?? 1, cfg.laptop.y ?? 1.2, cfg.laptop.z ?? 1);
      this._createLaptopMarker(cfg.laptop);
    }
    // Clear existing anchors
    for (const { mesh, ray, light, labelObj } of this.anchors.values()) {
      if (mesh) this.root.remove(mesh);
      if (ray) this.root.remove(ray);
      if (light) this.root.remove(light);
      if (labelObj) {
        labelObj.element?.remove();
        this.labelGroup?.remove(labelObj);
      }
    }
    this.anchors.clear();
    // Add anchors
    for (const anchor of cfg.anchors || []) {
      this.addAnchor(anchor);
    }
    // Rebuild rays for current laptop position
    this._rebuildAllRays();
  }

  addAnchor(anchor) {
    const id = anchor.id || `anchor_${this.anchors.size}`;
    const color = _toThreeColor(anchor.color || '#00d4ff');
    const pos = new THREE.Vector3(anchor.x ?? 0, anchor.y ?? 1.5, anchor.z ?? 0);

    // marker sphere
    const geo = new THREE.SphereGeometry(0.18, 24, 24);
    const mat = new THREE.MeshStandardMaterial({
      color, emissive: color.clone().multiplyScalar(0.3),
      roughness: 0.4, metalness: 0.7,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.copy(pos);
    this.root.add(mesh);

    // soft point light
    const light = new THREE.PointLight(color, 0.6, 2.2);
    light.position.copy(pos);
    this.root.add(light);

    // text label
    let labelObj = null;
    if (this.labelGroup && anchor.name) {
      const div = document.createElement('div');
      div.className = 'anchor-label';
      div.textContent = anchor.name;
      div.style.cssText =
        'background:rgba(15,18,28,0.85);border:1px solid rgba(255,255,255,0.15);' +
        'border-radius:4px;padding:2px 6px;font-size:10px;color:#cfd9ff;' +
        'font-family:Inter,system-ui,sans-serif;pointer-events:none;white-space:nowrap;';
      labelObj = new CSS2DObject(div);
      labelObj.position.copy(pos).y += 0.35;
      this.labelGroup.add(labelObj);
    }

    // ray (will be set in _rebuildAllRays)
    const rayGeo = new THREE.BufferGeometry().setFromPoints([pos, this.laptopPos.clone()]);
    const rayMat = new THREE.LineBasicMaterial({
      color, transparent: true, opacity: 0.22,
    });
    const ray = new THREE.Line(rayGeo, rayMat);
    this.root.add(ray);

    // Broadcast field: concentric rings centered on the anchor showing where
    // its signal reaches at different RSSI thresholds. Built on first display.
    const fieldRings = [];

    this.anchors.set(id, {
      anchor, mesh, light, ray, labelObj, color, fieldRings,
      baseEmissive: 0.3, emissiveBoost: 0,
      rayBaseOpacity: 0.22, rayBoost: 0,
      lastDistance: null,
      rssiSmoothed: null,     // EWMA state; first sample sets it
      rawRssi: null,          // most recent raw value (for debug display)
      sampleCount: 0,
    });
    this._updateBroadcastField(this.anchors.get(id));
  }

  _baseLabelText(anchor) {
    return anchor.name || anchor.id || 'anchor';
  }

  _updateLabel(entry) {
    if (!entry.labelObj || !entry.labelObj.element) return;
    const base = this._baseLabelText(entry.anchor);
    const d = entry.lastDistance;
    const pinned = entry.anchor.auto_position === false;
    const distTxt = d != null ? ` · ${d.toFixed(1)}m` : '';
    const modeTxt = pinned ? ' [pin]' : (d != null ? ' [auto]' : '');
    // Optional: show raw vs smoothed RSSI for debugging multipath
    let dbgTxt = '';
    if (!pinned && entry.rawRssi != null && entry.rssiSmoothed != null
        && entry.sampleCount > 1) {
      const diff = Math.abs(entry.rawRssi - entry.rssiSmoothed);
      // Only show debug if there's meaningful smoothing happening (>1 dB delta)
      if (diff >= 1) {
        dbgTxt = ` (raw ${entry.rawRssi.toFixed(0)}dB → smooth ${entry.rssiSmoothed.toFixed(0)}dB)`;
      }
    }
    entry.labelObj.element.textContent = base + distTxt + modeTxt + dbgTxt;
  }

  _createLaptopMarker(laptop) {
    if (this.laptopMesh) this.root.remove(this.laptopMesh);
    const geo = new THREE.BoxGeometry(0.4, 0.05, 0.28);
    const mat = new THREE.MeshStandardMaterial({
      color: _toThreeColor(laptop.color || '#ffd60a'),
      emissive: _toThreeColor(laptop.color || '#ffd60a').multiplyScalar(0.25),
      roughness: 0.3, metalness: 0.6,
    });
    this.laptopMesh = new THREE.Mesh(geo, mat);
    this.laptopMesh.position.copy(this.laptopPos);
    this.root.add(this.laptopMesh);

    if (this.labelGroup && laptop.name) {
      const div = document.createElement('div');
      div.className = 'anchor-label laptop-label';
      div.textContent = laptop.name;
      div.style.cssText =
        'background:rgba(40,28,0,0.9);border:1px solid #ffd60a;border-radius:4px;' +
        'padding:2px 6px;font-size:10px;color:#ffd60a;font-weight:600;' +
        'font-family:Inter,system-ui,sans-serif;pointer-events:none;white-space:nowrap;';
      const labelObj = new CSS2DObject(div);
      labelObj.position.copy(this.laptopPos).y += 0.5;
      this.labelGroup.add(labelObj);
    }
  }

  _rebuildAllRays() {
    for (const entry of this.anchors.values()) {
      const start = entry.mesh.position;
      const end = this.laptopPos;
      const positions = new Float32Array([start.x, start.y, start.z, end.x, end.y, end.z]);
      entry.ray.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      entry.ray.geometry.attributes.position.needsUpdate = true;
    }
  }

  /**
   * Process a radar event from the WebSocket.
   * @param {object} ev - { anchor_id, kind, state, value, extra, ... }
   */
  handleEvent(ev) {
    if (!ev) return;
    if (ev.kind === 'tag_fired') {
      this._spawnTagFlash(ev.extra?.name || 'Tag');
      return;
    }
    if (ev.kind === 'presence_snapshot') {
      this._updatePeople(ev.extra?.people || [], ev.extra);
      return;
    }
    if (!ev.anchor_id) return;
    const entry = this.anchors.get(ev.anchor_id);
    if (!entry) return;

    // Track for the "people in house" aggregator: labeled BLE devices + any
    // anchor that's in a motion-like state.
    this._trackForPresenceAggregate(ev, entry);

    // Auto-position by distance from RSSI (only if user hasn't pinned it).
    if (ev.value != null && entry.anchor.auto_position !== false) {
      let rssiDbm = null;
      const src = ev.source || '';
      if (src.startsWith('wifi_multi_ap') || src.startsWith('wifi_signal')) {
        rssiDbm = signalPctToRssi(ev.value);
      } else if (src === 'ble') {
        rssiDbm = ev.value;          // already dBm
      }
      if (rssiDbm != null && Number.isFinite(rssiDbm)) {
        // EWMA: smooth the noisy RSSI before computing distance, otherwise
        // multipath fading makes static objects appear to teleport ±5m.
        const alpha = entry.anchor.smoothing_alpha
                     ?? SMOOTHING_ALPHA[entry.anchor.type]
                     ?? SMOOTHING_ALPHA_DEFAULT;
        entry.rawRssi = rssiDbm;
        entry.sampleCount = (entry.sampleCount || 0) + 1;
        if (entry.rssiSmoothed == null) {
          entry.rssiSmoothed = rssiDbm;
        } else {
          entry.rssiSmoothed = alpha * rssiDbm + (1 - alpha) * entry.rssiSmoothed;
        }
        const dist = rssiToDistance(
          entry.rssiSmoothed,
          entry.anchor.type,
          entry.anchor.tx_power_at_1m,   // per-anchor calibration if set
        );
        // Deadband: skip visual update for sub-perceptible changes
        const deadband = entry.anchor.deadband_m
                      ?? DEADBAND_M[entry.anchor.type]
                      ?? DEADBAND_DEFAULT;
        if (entry.lastDistance != null &&
            Math.abs(dist - entry.lastDistance) < deadband) {
          // still refresh the label text so debug numbers update
          entry.rawRssi = rssiDbm;
          this._updateLabel(entry);
        } else {
          this._setAnchorRadius(entry, dist);
        }
      }
    }

    const isMotion = ev.state === 'motion' || ev.state === 'moving' || ev.state === 'approaching';
    if (isMotion) {
      entry.emissiveBoost = 1.0;
      entry.rayBoost = 1.0;
      this._spawnHeatSphere(entry);
    }
  }

  /**
   * Move an anchor along the line laptop->anchor so its distance to laptop = `distance`.
   * Preserves angular position (the user-set direction).
   */
  _setAnchorRadius(entry, distance) {
    const dx = entry.mesh.position.x - this.laptopPos.x;
    const dz = entry.mesh.position.z - this.laptopPos.z;
    const r0 = Math.sqrt(dx * dx + dz * dz);
    let nx, nz;
    if (r0 < 0.01) {
      // No prior angle — default to +X (east of laptop) so it's visible
      nx = 1; nz = 0;
    } else {
      nx = dx / r0; nz = dz / r0;
    }
    entry.mesh.position.x = this.laptopPos.x + nx * distance;
    entry.mesh.position.z = this.laptopPos.z + nz * distance;
    entry.light.position.copy(entry.mesh.position);
    if (entry.labelObj) {
      entry.labelObj.position.copy(entry.mesh.position);
      entry.labelObj.position.y += 0.35;
    }
    entry.anchor.x = entry.mesh.position.x;
    entry.anchor.z = entry.mesh.position.z;
    entry.lastDistance = distance;
    this._moveFieldRings(entry);
    this._updateLabel(entry);
    this._rebuildAllRays();
  }

  /**
   * Draw 3 concentric "broadcast field" rings on the floor centered on the
   * anchor itself (not on the laptop). Each ring marks where the signal
   * would be at a given RSSI threshold (close / medium / far edge).
   * The laptop is just a point inside these fields.
   */
  _updateBroadcastField(entry) {
    const type = entry.anchor?.type || 'wifi_ap';
    const ringThresholds = [-55, -70, -85];   // close, medium, edge of useful coverage
    const opacities    = [0.30, 0.18, 0.10];

    // Remove old rings
    if (entry.fieldRings && entry.fieldRings.length) {
      for (const r of entry.fieldRings) {
        this.root.remove(r);
        r.geometry.dispose();
        r.material.dispose();
      }
    }
    entry.fieldRings = [];

    for (let i = 0; i < ringThresholds.length; i++) {
      const radius = rssiToDistance(ringThresholds[i], type);
      const geo = new THREE.RingGeometry(
        Math.max(0.05, radius - 0.06), radius + 0.06, 96,
      );
      const mat = new THREE.MeshBasicMaterial({
        color: entry.color, transparent: true,
        opacity: opacities[i], side: THREE.DoubleSide,
      });
      const ring = new THREE.Mesh(geo, mat);
      ring.rotation.x = -Math.PI / 2;
      ring.position.copy(entry.mesh.position);
      ring.position.y = 0.005 + i * 0.001;  // tiny vertical stagger to avoid z-fighting
      this.root.add(ring);
      entry.fieldRings.push(ring);
    }
  }

  _moveFieldRings(entry) {
    if (!entry.fieldRings) return;
    for (const r of entry.fieldRings) {
      r.position.copy(entry.mesh.position);
      r.position.y = 0.005;
    }
  }

  _spawnHeatSphere(entry) {
    const mid = entry.mesh.position.clone().add(this.laptopPos).multiplyScalar(0.5);
    const geo = new THREE.SphereGeometry(0.35, 16, 16);
    const mat = new THREE.MeshBasicMaterial({
      color: entry.color, transparent: true, opacity: 0.55,
    });
    const sphere = new THREE.Mesh(geo, mat);
    sphere.position.copy(mid);
    this.root.add(sphere);
    this.heats.push({ sphere, born: performance.now(), life: HEAT_FADE_MS });
  }

  _spawnTagFlash(name) {
    if (!this.labelGroup) return;
    const div = document.createElement('div');
    div.className = 'tag-flash';
    div.textContent = `🚨  ${name}`;
    div.style.cssText =
      'background:rgba(220,30,60,0.95);border:2px solid #fff;border-radius:6px;' +
      'padding:6px 14px;font-size:16px;font-weight:700;color:#fff;' +
      'font-family:Inter,system-ui,sans-serif;pointer-events:none;' +
      'box-shadow:0 0 20px rgba(220,30,60,0.7);transform:translate(-50%,-50%);';
    const obj = new CSS2DObject(div);
    obj.position.copy(this.laptopPos).y += 2.5;
    this.labelGroup.add(obj);
    this.tagFlashes.push({ obj, born: performance.now(), life: TAG_FLASH_MS });
  }

  /** Call every frame from the existing animation loop. */
  tick() {
    const now = performance.now();
    const dt = Math.max(0.001, (now - this._lastUpdate) / 1000);
    this._lastUpdate = now;

    // Decay anchor highlights
    for (const entry of this.anchors.values()) {
      if (entry.emissiveBoost > 0) {
        entry.emissiveBoost = Math.max(0, entry.emissiveBoost - dt * 0.7);
        const totalEmissive = entry.baseEmissive + entry.emissiveBoost * 1.2;
        entry.mesh.material.emissive.copy(entry.color).multiplyScalar(totalEmissive);
        entry.mesh.material.needsUpdate = true;
      }
      if (entry.rayBoost > 0) {
        entry.rayBoost = Math.max(0, entry.rayBoost - dt * 0.6);
        entry.ray.material.opacity = entry.rayBaseOpacity + entry.rayBoost * 0.7;
        entry.ray.material.needsUpdate = true;
      }
      // Subtle anchor pulse
      const s = 1 + Math.sin(now * 0.003) * 0.04;
      entry.mesh.scale.set(s, s, s);
    }

    // Fade heatmap spheres
    this.heats = this.heats.filter((h) => {
      const t = (now - h.born) / h.life;
      if (t >= 1) {
        this.root.remove(h.sphere);
        h.sphere.geometry.dispose();
        h.sphere.material.dispose();
        return false;
      }
      h.sphere.material.opacity = (1 - t) * 0.55;
      h.sphere.scale.setScalar(1 + t * 1.4);
      return true;
    });

    // Animate people markers + cull stale ones
    for (const [label, entry] of [...this.people]) {
      const age = now - entry.lastSeen;
      if (entry.markStale || age > this._personStaleMs) {
        // Fade out then remove
        const opacity = Math.max(0, 1 - (age - this._personStaleMs) / 800);
        entry.mesh.material.opacity = opacity;
        entry.mesh.material.transparent = true;
        if (entry.ring) entry.ring.material.opacity = 0.45 * opacity;
        if (entry.labelObj?.element) entry.labelObj.element.style.opacity = String(opacity);
        if (opacity <= 0.01) {
          this.root.remove(entry.mesh);
          if (entry.ring) this.root.remove(entry.ring);
          if (entry.light) this.root.remove(entry.light);
          if (entry.labelObj) {
            this.labelGroup?.remove(entry.labelObj);
            entry.labelObj.element?.remove();
          }
          this.people.delete(label);
        }
        continue;
      }
      // Smooth lerp toward target position
      if (entry.target) {
        entry.current.lerp(entry.target, Math.min(1, dt * 4));
        entry.mesh.position.set(entry.current.x, 0.55, entry.current.z);
        if (entry.ring) {
          entry.ring.position.set(entry.current.x, 0.01, entry.current.z);
        }
        if (entry.light) {
          entry.light.position.set(entry.current.x, 1.4, entry.current.z);
        }
        if (entry.labelObj) {
          entry.labelObj.position.set(entry.current.x, 1.8, entry.current.z);
        }
        // Subtle breathing pulse
        const breath = 1 + Math.sin(now * 0.004) * 0.03;
        entry.mesh.scale.set(breath, 1, breath);
      }
    }

    // Fade tag flashes
    this.tagFlashes = this.tagFlashes.filter((f) => {
      const t = (now - f.born) / f.life;
      if (t >= 1) {
        if (this.labelGroup) this.labelGroup.remove(f.obj);
        f.obj.element?.remove();
        return false;
      }
      const op = 1 - t;
      f.obj.element.style.opacity = String(op);
      return true;
    });
  }

  // ─── People (from camera presence_snapshot) ─────────────────────
  /**
   * Update the visible people in the 3D scene from a camera snapshot.
   * @param {Array} people  [{label, x_m, y_m, score, identified}]
   * @param {object} extra  (count, fps, ...)
   */
  _updatePeople(people, extra) {
    const now = performance.now();
    const seen = new Set();

    for (const p of people) {
      const label = p.label || `#${p.track_id ?? '?'}`;
      seen.add(label);
      let entry = this.people.get(label);
      // 3D position: floor_mapper gives meters (x_m, y_m). In scene XYZ:
      //   x = x_m (right), z = y_m (forward, distance), y = 0.05 (just above floor)
      const tx = p.x_m, tz = p.y_m;
      if (!entry) {
        entry = this._createPersonMarker(label, p);
        entry.current = new THREE.Vector3(tx, 0.05, tz);
        entry.mesh.position.copy(entry.current);
      }
      entry.target = new THREE.Vector3(tx, 0.05, tz);
      entry.lastSeen = now;
      entry.score = p.score;
      entry.identified = !!p.identified;
    }

    // Mark stale people (will be removed by tick)
    for (const [label, entry] of this.people) {
      if (!seen.has(label)) entry.markStale = true;
    }

    // Feed the in-house aggregator with this camera snapshot
    this._cameraVisible = people.map((p) => ({
      label: p.label, x_m: p.x_m, y_m: p.y_m,
      score: p.score, identified: !!p.identified,
      lastSeen: now,
    }));

    this._renderPresencePanel(people, extra);
    this._renderPeopleInHouse();
  }

  _personColor(label, identified) {
    const palette = {
      'Yo':     [0x30d158, 0x30d158],
      'Tata':   [0x00d4ff, 0x00d4ff],
      'Abuela': [0xff375f, 0xff375f],
      'Karen':  [0xffd60a, 0xffd60a],
    };
    if (palette[label]) return new THREE.Color(palette[label][0]);
    return new THREE.Color(identified ? 0xbf5af2 : 0x9999aa);
  }

  _createPersonMarker(label, p) {
    const color = this._personColor(label, p.identified);

    // Standing pill (capsule): body
    const bodyGeo = new THREE.CapsuleGeometry(0.18, 0.9, 6, 12);
    const bodyMat = new THREE.MeshStandardMaterial({
      color, emissive: color.clone().multiplyScalar(0.4),
      roughness: 0.35, metalness: 0.2,
    });
    const mesh = new THREE.Mesh(bodyGeo, bodyMat);
    mesh.position.y = 0.55;
    this.root.add(mesh);

    // Foot ring (where they "stand")
    const ringGeo = new THREE.RingGeometry(0.32, 0.42, 32);
    const ringMat = new THREE.MeshBasicMaterial({
      color, transparent: true, opacity: 0.55, side: THREE.DoubleSide,
    });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 0.01;
    this.root.add(ring);

    // Floating soft light
    const light = new THREE.PointLight(color, 0.7, 2.5);
    light.position.set(0, 1.4, 0);
    this.root.add(light);

    // CSS2D name label
    let labelObj = null;
    if (this.labelGroup) {
      const div = document.createElement('div');
      div.className = 'person-label';
      div.style.cssText =
        'background:rgba(15,20,30,0.9);border:1.5px solid currentColor;' +
        'border-radius:6px;padding:3px 10px;font-size:13px;font-weight:700;' +
        'color:#fff;font-family:Inter,system-ui,sans-serif;pointer-events:none;' +
        'white-space:nowrap;box-shadow:0 0 10px rgba(0,0,0,0.4);transform:translate(-50%,-100%);';
      div.style.color = '#' + color.getHexString();
      div.textContent = label;
      labelObj = new CSS2DObject(div);
      labelObj.position.y = 1.8;
      this.labelGroup.add(labelObj);
    }

    const entry = {
      label, mesh, ring, light, labelObj, color,
      target: null, current: new THREE.Vector3(),
      lastSeen: performance.now(), markStale: false,
      score: p.score, identified: !!p.identified,
    };
    this.people.set(label, entry);
    return entry;
  }

  /**
   * Track per-event signals for the "people in house" aggregator.
   * Camera presence is handled separately via _updatePeople().
   */
  _trackForPresenceAggregate(ev, entry) {
    const now = performance.now();
    const anchor = entry?.anchor;
    if (!anchor) return;

    // BLE labeled device: keep its last RSSI and last_seen
    if ((ev.source === 'ble') && ev.extra?.label) {
      const mac = anchor.match?.mac || anchor.id;
      this._bleLabeled.set(mac, {
        label: ev.extra.label,
        state: ev.state,
        rssi: ev.value,
        lastSeen: now,
      });
    }

    // Anchor with sustained motion-like state
    if (ev.state === 'motion' || ev.state === 'moving' || ev.state === 'approaching') {
      this._anchorMotion.set(ev.anchor_id, {
        name: anchor.name || ev.anchor_id,
        source: ev.source,
        state: ev.state,
        lastSeen: now,
      });
    }

    this._renderPeopleInHouse();
  }

  /** Load room polygons once so we can label positions by room name. */
  async _loadRooms() {
    if (this._roomsLoaded) return;
    this._roomsLoaded = true;
    try {
      const r = await fetch('/api/floor-plan');
      if (!r.ok) return;
      const data = await r.json();
      this._rooms = (data.rooms || []).map((room) => {
        // Accept either {x, z, width, length} or {bounds: {x_min, x_max, z_min, z_max}}
        const b = room.bounds || {};
        return {
          name: room.name || room.id || 'room',
          x_min: b.x_min ?? room.x ?? 0,
          x_max: b.x_max ?? ((room.x ?? 0) + (room.width ?? 0)),
          z_min: b.z_min ?? room.z ?? 0,
          z_max: b.z_max ?? ((room.z ?? 0) + (room.length ?? 0)),
        };
      });
    } catch (e) {
      console.warn('[overlay] _loadRooms failed', e);
    }
  }

  _roomForPosition(x_m, z_m) {
    for (const r of this._rooms) {
      if (x_m >= r.x_min && x_m <= r.x_max && z_m >= r.z_min && z_m <= r.z_max) {
        return r.name;
      }
    }
    return null;
  }

  _renderPeopleInHouse() {
    this._loadRooms();

    if (!this._peopleInHousePanel) {
      const div = document.createElement('div');
      div.id = 'people-in-house-panel';
      div.style.cssText = [
        'position:absolute',
        'top:14px',
        'left:50%',
        'transform:translateX(-50%)',
        'background:linear-gradient(135deg,rgba(15,20,40,0.97),rgba(30,15,55,0.97))',
        'border:2px solid rgba(120,180,255,0.5)',
        'border-radius:18px',
        'padding:14px 24px',
        'font-family:Inter,system-ui,sans-serif',
        'color:#fff',
        'min-width:340px',
        'max-width:540px',
        'pointer-events:none',
        'z-index:15',
        'box-shadow:0 10px 36px rgba(0,0,0,0.65), 0 0 24px rgba(80,120,255,0.18)',
        'text-align:center',
      ].join(';');
      const container = document.querySelector('.view-container') || document.body;
      container.style.position = container.style.position || 'relative';
      container.appendChild(div);
      this._peopleInHousePanel = div;
    }
    const now = performance.now();

    const visible = this._cameraVisible.filter((p) => now - p.lastSeen < this._personStaleMs);
    const visibleNames = visible.map((p) => p.label);

    const bleActive = [];
    for (const [mac, v] of this._bleLabeled) {
      if (now - v.lastSeen > this._bleStaleMs) continue;
      if (visibleNames.includes(v.label)) continue;
      bleActive.push({ label: v.label, rssi: v.rssi });
    }

    const motionZones = [];
    for (const [aid, m] of this._anchorMotion) {
      if (now - m.lastSeen > this._motionStaleMs) continue;
      motionZones.push({ name: m.name, source: m.source });
    }

    const total = visible.length + bleActive.length + (motionZones.length > 0 ? 1 : 0);

    // Build the people list with ROOM names
    const visibleHtml = visible.map((p) => {
      const col = '#' + this._personColor(p.label, p.identified).getHexString();
      const room = this._roomForPosition(p.x_m, p.y_m);
      const where = room
        ? `<span style="opacity:0.85">en <b>${room}</b></span>`
        : `<span style="opacity:0.55">(${p.x_m.toFixed(1)}m, ${p.y_m.toFixed(1)}m)</span>`;
      return `<div style="display:flex;align-items:center;justify-content:center;gap:8px;margin:3px 0;font-size:15px">
        <span style="font-size:18px;color:${col}">●</span>
        <span style="color:${col};font-weight:700">${p.label}</span>
        ${where}
      </div>`;
    }).join('');

    const bleHtml = bleActive.map((b) => {
      return `<div style="display:flex;align-items:center;justify-content:center;gap:8px;margin:3px 0;font-size:14px;color:#9fc4ff">
        <span style="font-size:16px">📱</span>
        <span style="font-weight:600">${b.label}</span>
        <span style="opacity:0.6">cerca (${b.rssi}dBm)</span>
      </div>`;
    }).join('');

    const motionHtml = motionZones.length > 0
      ? `<div style="margin-top:6px;padding-top:8px;border-top:1px dashed rgba(255,255,255,0.18);font-size:12px;color:#ffb0b0">
           ⚡ Movimiento sin identificar:
           ${motionZones.map((m) => `<div style="margin-top:2px">· ${m.name}</div>`).join('')}
         </div>`
      : '';

    const countColor = total === 0 ? '#888' : (total === 1 ? '#fff' : '#ffd060');

    // Diagnose: when count is 0, look at system status and tell user WHY
    let helpHtml = '';
    if (total === 0) {
      const s = this._lastSystemStatus;
      const camOk = s?.camera?.fresh;
      const csiOk = s?.esp32_csi?.fresh;
      const bleOk = s?.ble_autobroadcast?.fresh;
      const checks = [];
      checks.push(camOk
        ? '<div style="color:#3f6">✓ Cámara publicando</div>'
        : '<div style="color:#f88">✗ Cámara NO publica — corré <code style="background:#222;padding:1px 4px;border-radius:3px">camera_detector radar --broadcast http://127.0.0.1:8000</code></div>');
      checks.push(csiOk
        ? '<div style="color:#3f6">✓ ESP32 CSI publicando</div>'
        : '<div style="color:#f88">✗ ESP32 NO publica — corré <code style="background:#222;padding:1px 4px;border-radius:3px">csi_reader live --port COM7 --broadcast http://127.0.0.1:8000</code></div>');
      checks.push(bleOk
        ? '<div style="color:#3f6">✓ BLE escaneando</div>'
        : '<div style="color:#888">○ BLE sin samples (normal si nada labelado está cerca)</div>');
      helpHtml = `
        <div style="margin-top:14px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.15);
                    text-align:left;font-size:11px;line-height:1.6">
          <div style="opacity:0.7;margin-bottom:6px;font-weight:600;letter-spacing:0.5px">Por qué no veo nada:</div>
          ${checks.join('')}
          <div style="margin-top:8px;opacity:0.55;font-size:10px">
            Tip: la forma más fácil es doble-click a <b>start.bat</b> en la carpeta del proyecto.
          </div>
        </div>`;
    }

    // Honest "what this system can / can't" line, shown only when total > 0
    let honestyHtml = '';
    if (total > 0) {
      honestyHtml = `
        <div style="margin-top:10px;padding-top:8px;border-top:1px dashed rgba(255,255,255,0.12);
                    text-align:left;font-size:10px;opacity:0.55;line-height:1.5">
          ⚠ Con 1 cámara + 1 ESP32: posición exacta solo de quienes están en cámara.
          Gente atrás de muros = "anónimo", sin posición precisa.
        </div>`;
    }

    const summary = total === 0
      ? `<div style="font-size:34px;font-weight:800;color:#888">·</div>
         <div style="opacity:0.6;font-size:11px;letter-spacing:1px">NADA DETECTADO TODAVÍA</div>`
      : `<div style="font-size:44px;font-weight:900;color:${countColor};line-height:1">${total}</div>
         <div style="opacity:0.8;font-size:11px;letter-spacing:2px;margin-top:2px;margin-bottom:10px">PERSONAS EN LA CASA</div>`;

    this._peopleInHousePanel.innerHTML = `
      ${summary}
      <div style="text-align:left;padding-left:8px">${visibleHtml}${bleHtml}${motionHtml}</div>
      ${helpHtml}
      ${honestyHtml}
    `;
  }

  /** Poll system status every 2s and render a small panel top-left. */
  async _pollSystemStatus() {
    if (!this._statusPanel) {
      const div = document.createElement('div');
      div.id = 'system-status-panel';
      div.style.cssText = [
        'position:absolute', 'top:14px', 'left:14px',
        'background:rgba(15,18,28,0.92)', 'border:1px solid rgba(255,255,255,0.12)',
        'border-radius:8px', 'padding:8px 12px',
        'font-family:Inter,system-ui,sans-serif', 'color:#cfd9ff', 'font-size:11px',
        'pointer-events:none', 'z-index:14', 'box-shadow:0 4px 12px rgba(0,0,0,0.5)',
        'min-width:200px',
      ].join(';');
      const container = document.querySelector('.view-container') || document.body;
      container.style.position = container.style.position || 'relative';
      container.appendChild(div);
      this._statusPanel = div;
    }
    try {
      const r = await fetch('/api/system_status');
      if (!r.ok) return;
      this._lastSystemStatus = await r.json();
    } catch (e) {
      this._lastSystemStatus = null;
    }
    const s = this._lastSystemStatus;
    if (!s) {
      this._statusPanel.innerHTML = '<span style="color:#f66">⚠ Backend no responde</span>';
      return;
    }
    const dot = (fresh) => fresh
      ? '<span style="color:#3f6">●</span>'
      : '<span style="color:#666">○</span>';
    const age = (a) => (a == null ? '—' : `${a.toFixed(0)}s`);
    this._statusPanel.innerHTML = `
      <div style="font-weight:700;font-size:10px;letter-spacing:1px;opacity:0.7;margin-bottom:5px">SISTEMA</div>
      <div>${dot(true)} Backend</div>
      <div>${dot(s.camera.fresh)} Cámara <span style="opacity:0.5">(${age(s.camera.last_snapshot_ago)})</span></div>
      <div>${dot(s.esp32_csi.fresh)} ESP32 CSI <span style="opacity:0.5">(${age(s.esp32_csi.last_event_ago)})</span></div>
      <div>${dot(s.ble_autobroadcast.fresh)} BLE <span style="opacity:0.5">(${age(s.ble_autobroadcast.last_event_ago)})</span></div>
      <div>${dot(s.wifi_autobroadcast.fresh)} Wi-Fi <span style="opacity:0.5">(${age(s.wifi_autobroadcast.last_event_ago)})</span></div>
    `;
  }

  _renderPresencePanel(people, extra) {
    if (!this._presencePanel) {
      const div = document.createElement('div');
      div.id = 'presence-panel';
      div.style.cssText =
        'position:absolute;top:60px;left:12px;background:rgba(15,20,30,0.92);' +
        'border:1px solid rgba(255,255,255,0.12);border-radius:8px;padding:10px 14px;' +
        'font-family:Inter,system-ui,sans-serif;color:#cfd9ff;font-size:12px;' +
        'min-width:200px;max-width:280px;pointer-events:none;z-index:10;' +
        'box-shadow:0 4px 16px rgba(0,0,0,0.4);';
      // Find a sensible container. The 3D view container works.
      const container = document.querySelector('.view-container') || document.body;
      container.style.position = container.style.position || 'relative';
      container.appendChild(div);
      this._presencePanel = div;
    }
    const count = extra?.count ?? people.length;
    const fps = extra?.fps;
    const fpsTxt = fps != null ? ` <span style="opacity:0.5">(camara ${fps.toFixed(1)} fps)</span>` : '';
    const rows = people.map((p) => {
      const col = '#' + this._personColor(p.label, p.identified).getHexString();
      const scoreTxt = p.score > 0 ? ` <span style="opacity:0.65">${(p.score * 100).toFixed(0)}%</span>` : '';
      return `<div style="margin:4px 0;color:${col};font-weight:600">
        ● ${p.label}${scoreTxt}
        <span style="opacity:0.55;font-weight:400">— (${p.x_m.toFixed(1)}m, ${p.y_m.toFixed(1)}m)</span>
      </div>`;
    }).join('');
    this._presencePanel.innerHTML = `
      <div style="font-weight:700;font-size:13px;color:#fff;margin-bottom:6px">
        👁 Visibles en cámara: ${count}${fpsTxt}
      </div>
      ${rows || '<div style="opacity:0.5">(nadie a la vista)</div>'}
    `;
  }

  dispose() {
    if (this.scene) this.scene.remove(this.root);
    for (const e of this.anchors.values()) {
      e.mesh.geometry.dispose();
      e.mesh.material.dispose();
      e.ray.geometry.dispose();
      e.ray.material.dispose();
      if (e.fieldRings) {
        for (const r of e.fieldRings) {
          r.geometry.dispose();
          r.material.dispose();
        }
      }
    }
    this.anchors.clear();
    this.heats = [];
    this.tagFlashes = [];
  }

  // ─── Drag-to-move anchors on the floor plane ─────────────────────
  /**
   * Enable click-and-drag editing. Persists to /api/anchors on drop.
   * @param {THREE.PerspectiveCamera} camera
   * @param {OrbitControls} controls
   * @param {HTMLElement} domElement - canvas or container that receives events
   */
  enableDragging(camera, controls, domElement) {
    this._camera = camera;
    this._orbit = controls;
    this._dragCanvas = domElement;
    this._raycaster = new THREE.Raycaster();
    this._dragPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0); // floor y=0
    this._mouseNDC = new THREE.Vector2();
    this._dragTarget = null;
    this._dragPending = false;
    this._saveTimer = null;

    domElement.addEventListener('pointerdown', (e) => this._onPointerDown(e));
    window.addEventListener('pointermove', (e) => this._onPointerMove(e));
    window.addEventListener('pointerup', (e) => this._onPointerUp(e));
    // Hover cursor feedback
    domElement.addEventListener('pointermove', (e) => this._onPointerHover(e));
    console.log('[RadarOverlay] Drag enabled. Click+arrastra anchors para reposicionarlos.');
  }

  _setMouseNDC(e) {
    const rect = this._dragCanvas.getBoundingClientRect();
    this._mouseNDC.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    this._mouseNDC.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  }

  _draggableMeshes() {
    const meshes = [];
    for (const entry of this.anchors.values()) meshes.push(entry.mesh);
    if (this.laptopMesh) meshes.push(this.laptopMesh);
    return meshes;
  }

  _onPointerHover(e) {
    if (this._dragTarget) return;
    this._setMouseNDC(e);
    this._raycaster.setFromCamera(this._mouseNDC, this._camera);
    const hits = this._raycaster.intersectObjects(this._draggableMeshes());
    this._dragCanvas.style.cursor = hits.length ? 'grab' : '';
  }

  _onPointerDown(e) {
    if (e.button !== 0) return;
    this._setMouseNDC(e);
    this._raycaster.setFromCamera(this._mouseNDC, this._camera);
    const hits = this._raycaster.intersectObjects(this._draggableMeshes());
    if (hits.length === 0) return;
    const hitMesh = hits[0].object;
    // Find which anchor (or laptop) was hit
    if (hitMesh === this.laptopMesh) {
      this._dragTarget = { kind: 'laptop' };
    } else {
      for (const [id, entry] of this.anchors) {
        if (entry.mesh === hitMesh) {
          this._dragTarget = { kind: 'anchor', id, entry };
          break;
        }
      }
    }
    if (this._dragTarget) {
      if (this._orbit) this._orbit.enabled = false;
      this._dragCanvas.style.cursor = 'grabbing';
      e.preventDefault();
    }
  }

  _onPointerMove(e) {
    if (!this._dragTarget) return;
    this._setMouseNDC(e);
    this._raycaster.setFromCamera(this._mouseNDC, this._camera);
    const target = new THREE.Vector3();
    if (!this._raycaster.ray.intersectPlane(this._dragPlane, target)) return;

    if (this._dragTarget.kind === 'laptop') {
      this.laptopPos.x = target.x;
      this.laptopPos.z = target.z;
      if (this.laptopMesh) this.laptopMesh.position.copy(this.laptopPos);
    } else {
      const { entry } = this._dragTarget;
      entry.mesh.position.x = target.x;
      entry.mesh.position.z = target.z;
      entry.light.position.copy(entry.mesh.position);
      if (entry.labelObj) {
        entry.labelObj.position.copy(entry.mesh.position);
        entry.labelObj.position.y += 0.35;
      }
      entry.anchor.x = target.x;
      entry.anchor.z = target.z;
      // User pinned this anchor; auto-distance no longer overrides it
      entry.anchor.auto_position = false;
      // Broadcast field rings stay centered on the anchor — move them too
      this._moveFieldRings(entry);
      this._updateLabel(entry);
    }
    this._rebuildAllRays();
    this._dragPending = true;
  }

  _onPointerUp(e) {
    if (!this._dragTarget) return;
    this._dragTarget = null;
    if (this._orbit) this._orbit.enabled = true;
    this._dragCanvas.style.cursor = '';
    if (this._dragPending) {
      this._dragPending = false;
      this._scheduleSave();
    }
  }

  _scheduleSave() {
    clearTimeout(this._saveTimer);
    this._saveTimer = setTimeout(() => this._saveConfig(), 400);
  }

  async _saveConfig() {
    if (!this._cfg) return;
    const payload = {
      laptop: {
        ...(this._cfg.laptop || {}),
        x: +this.laptopPos.x.toFixed(2),
        y: +this.laptopPos.y.toFixed(2),
        z: +this.laptopPos.z.toFixed(2),
      },
      anchors: Array.from(this.anchors.values()).map((e) => ({
        ...e.anchor,
        x: +e.anchor.x.toFixed(2),
        y: +e.anchor.y.toFixed(2),
        z: +e.anchor.z.toFixed(2),
      })),
      model: this._cfg.model,
    };
    try {
      const r = await fetch('/api/anchors', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const body = await r.json();
      console.log('[RadarOverlay] Anchors saved:', body);
    } catch (err) {
      console.error('[RadarOverlay] Save failed:', err);
    }
  }
}
