/**
 * Three.js 3D House Scene — Wi-Fi Radar · Digital Twin
 * Renders the house floor plan in 3D with person tracking, trails, and room labels.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

// ─── Room color palette ────────────────────────────────────────
const ROOM_COLORS = {
  Sala: 0x00d4ff,
  Cocina: 0xff9500,
  'Habitación': 0xbf5af2,
  'Habitación Principal': 0xbf5af2,
  'Habitación 2': 0x5e5ce6,
  Comedor: 0x30d158,
  'Baño': 0xff375f,
  'Baño 2': 0xff6482,
  Pasillo: 0x636366,
  Garaje: 0x8e8e93,
  Entrada: 0xffd60a,
  Oficina: 0x64d2ff,
  Lavandería: 0xac8e68,
  default: 0x48484a,
};

function getRoomColor(name) {
  for (const [key, color] of Object.entries(ROOM_COLORS)) {
    if (name.toLowerCase().includes(key.toLowerCase())) return color;
  }
  return ROOM_COLORS.default;
}

export class HouseScene3D {
  /**
   * @param {HTMLCanvasElement} canvas
   */
  constructor(canvas) {
    this.canvas = canvas;
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.cssRenderer = null;
    this.controls = null;
    this.animationId = null;

    // Scene groups
    this.houseGroup = new THREE.Group();
    this.labelGroup = new THREE.Group();

    // Person tracking
    this.personMesh = null;
    this.personRing = null;
    this.personLight = null;
    this.trailLine = null;
    this.trailPositions = [];
    this.persons = new Map();
    this.maxTrail = 50;

    // Smooth movement
    this._targetPos = new THREE.Vector3(0, 0.25, 0);
    this._currentPos = new THREE.Vector3(0, 0.25, 0);
    this._lerpFactor = 0.08;

    // Camera preset target
    this._cameraTargetPos = null;
    this._cameraTargetLookAt = null;
    this._cameraLerping = false;
    this._cameraPreset = 'isometric';

    // Floor plan bounds
    this._floorWidth = 10;
    this._floorLength = 11;
    this._centerX = 5;
    this._centerZ = 5.5;

    // Router
    this.routerMesh = null;

    // Clock
    this._clock = new THREE.Clock();

    this.init();
  }

  init() {
    const container = this.canvas.parentElement;
    const w = container.clientWidth;
    const h = container.clientHeight - (container.querySelector('.view-header')?.offsetHeight || 0);

    // ─── Scene ──────────────────────────────
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0a0a0f);
    this.scene.fog = new THREE.FogExp2(0x0a0a0f, 0.035);

    // ─── Camera ─────────────────────────────
    this.camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 200);
    this.camera.position.set(12, 14, 16);
    this.camera.lookAt(this._centerX, 0, this._centerZ);

    // ─── Renderer ───────────────────────────
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: false,
    });
    this.renderer.setSize(w, h);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;

    // ─── CSS2D Renderer ─────────────────────
    this.cssRenderer = new CSS2DRenderer();
    this.cssRenderer.setSize(w, h);
    this.cssRenderer.domElement.style.position = 'absolute';
    this.cssRenderer.domElement.style.top = (container.querySelector('.view-header')?.offsetHeight || 0) + 'px';
    this.cssRenderer.domElement.style.left = '0';
    this.cssRenderer.domElement.style.pointerEvents = 'none';
    container.appendChild(this.cssRenderer.domElement);

    // ─── Controls ───────────────────────────
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.maxPolarAngle = Math.PI / 2.05;
    this.controls.minDistance = 3;
    this.controls.maxDistance = 45;
    this.controls.target.set(this._centerX, 0, this._centerZ);

    // ─── Lights ─────────────────────────────
    const hemiLight = new THREE.HemisphereLight(0x334466, 0x112233, 0.6);
    this.scene.add(hemiLight);

    const dirLight = new THREE.DirectionalLight(0xffeedd, 0.8);
    dirLight.position.set(15, 20, 10);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 2048;
    dirLight.shadow.mapSize.height = 2048;
    dirLight.shadow.camera.near = 0.5;
    dirLight.shadow.camera.far = 60;
    dirLight.shadow.camera.left = -20;
    dirLight.shadow.camera.right = 20;
    dirLight.shadow.camera.top = 20;
    dirLight.shadow.camera.bottom = -20;
    dirLight.shadow.bias = -0.0005;
    this.scene.add(dirLight);

    const ambientLight = new THREE.AmbientLight(0x222244, 0.3);
    this.scene.add(ambientLight);

    // ─── Grid ───────────────────────────────
    const gridHelper = new THREE.GridHelper(40, 40, 0x1a1a2e, 0x111122);
    gridHelper.position.y = -0.01;
    this.scene.add(gridHelper);

    // ─── Ground plane ───────────────────────
    const groundGeo = new THREE.PlaneGeometry(60, 60);
    const groundMat = new THREE.MeshStandardMaterial({
      color: 0x080810,
      roughness: 0.95,
      metalness: 0.0,
    });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.02;
    ground.receiveShadow = true;
    this.scene.add(ground);

    // ─── House Group ────────────────────────
    this.scene.add(this.houseGroup);
    this.scene.add(this.labelGroup);

    // ─── Person Marker ──────────────────────
    this._createPersonMarker();

    // ─── Trail ──────────────────────────────
    this._createTrail();

    // ─── Start animation ────────────────────
    this.animate();
  }

  _createPersonMarker() {
    // Main sphere
    const geo = new THREE.SphereGeometry(0.25, 32, 32);
    const mat = new THREE.MeshBasicMaterial({ color: 0xff2d55 });
    this.personMesh = new THREE.Mesh(geo, mat);
    this.personMesh.position.copy(this._currentPos);
    this.personMesh.visible = false;
    this.scene.add(this.personMesh);

    // Pulsing ring
    const ringGeo = new THREE.RingGeometry(0.35, 0.45, 32);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0xff2d55,
      transparent: true,
      opacity: 0.5,
      side: THREE.DoubleSide,
    });
    this.personRing = new THREE.Mesh(ringGeo, ringMat);
    this.personRing.rotation.x = -Math.PI / 2;
    this.personRing.position.copy(this._currentPos);
    this.personRing.position.y = 0.05;
    this.personRing.visible = false;
    this.scene.add(this.personRing);

    // Point light for glow
    this.personLight = new THREE.PointLight(0xff2d55, 2, 3);
    this.personLight.position.copy(this._currentPos);
    this.personLight.visible = false;
    this.scene.add(this.personLight);
  }

  _createTrail() {
    const maxPoints = 200;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(maxPoints * 3);
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setDrawRange(0, 0);

    const material = new THREE.LineBasicMaterial({
      color: 0xff2d55,
      transparent: true,
      opacity: 0.4,
    });

    this.trailLine = new THREE.Line(geometry, material);
    this.trailLine.frustumCulled = false;
    this.scene.add(this.trailLine);
  }

  _updateTrail() {
    if (this.trailPositions.length < 2) return;

    const positions = this.trailLine.geometry.attributes.position.array;
    const count = Math.min(this.trailPositions.length, 200);

    for (let i = 0; i < count; i++) {
      const p = this.trailPositions[this.trailPositions.length - count + i];
      positions[i * 3] = p.x;
      positions[i * 3 + 1] = p.y;
      positions[i * 3 + 2] = p.z;
    }

    this.trailLine.geometry.attributes.position.needsUpdate = true;
    this.trailLine.geometry.setDrawRange(0, count);
  }

  /**
   * Load a floor plan JSON and generate 3D geometry.
   * @param {object} data - Floor plan data with walls, rooms, routers
   */
  loadFloorPlan(data) {
    // Clear existing house
    this.houseGroup.clear();

    // Remove old CSS2D labels
    while (this.labelGroup.children.length) {
      const child = this.labelGroup.children[0];
      if (child.element) child.element.remove();
      this.labelGroup.remove(child);
    }

    if (!data) return;

    // Update bounds
    this._floorWidth = data.width || 10;
    this._floorLength = data.length || data.height || 11;
    this._centerX = this._floorWidth / 2;
    this._centerZ = this._floorLength / 2;

    // Update controls target
    this.controls.target.set(this._centerX, 0, this._centerZ);

    // ─── Rooms / floor tiles ────────────────
    if (data.rooms) {
      data.rooms.forEach((room) => {
        this._createRoom(room);
      });
    }

    // ─── Walls ──────────────────────────────
    if (data.walls) {
      data.walls.forEach((wall) => {
        this._createWall(wall);
      });
    }

    // ─── Doors ──────────────────────────────
    if (data.doors) {
      data.doors.forEach((door) => {
        this._createDoor(door);
      });
    }

    // ─── Router ─────────────────────────────
    if (data.router) {
      this._createRouter(data.router);
    }
    if (data.routers) {
      data.routers.forEach((r) => this._createRouter(r));
    }

    // Reset camera
    this.setCameraPreset('isometric');
  }

  _createRoom(room) {
    const x = room.x || 0;
    const z = room.y || 0;
    const w = room.width || 2;
    const d = room.height || room.length || 2;
    const color = getRoomColor(room.name || '');

    // Floor
    const floorGeo = new THREE.PlaneGeometry(w, d);
    const floorMat = new THREE.MeshStandardMaterial({
      color: color,
      transparent: true,
      opacity: 0.12,
      roughness: 0.8,
      side: THREE.DoubleSide,
    });
    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(x + w / 2, 0.01, z + d / 2);
    floor.receiveShadow = true;
    this.houseGroup.add(floor);

    // Room edge highlight
    const edgeGeo = new THREE.EdgesGeometry(new THREE.PlaneGeometry(w, d));
    const edgeMat = new THREE.LineBasicMaterial({
      color: color,
      transparent: true,
      opacity: 0.3,
    });
    const edges = new THREE.LineSegments(edgeGeo, edgeMat);
    edges.rotation.x = -Math.PI / 2;
    edges.position.set(x + w / 2, 0.02, z + d / 2);
    this.houseGroup.add(edges);

    // CSS2D Label
    if (room.name) {
      const div = document.createElement('div');
      div.className = 'label-3d';
      div.textContent = room.name;
      div.style.borderColor = '#' + new THREE.Color(color).getHexString();
      const label = new CSS2DObject(div);
      label.position.set(x + w / 2, 0.5, z + d / 2);
      this.labelGroup.add(label);
    }
  }

  _createWall(wall) {
    const x1 = wall.x1 ?? wall.start?.[0] ?? 0;
    const z1 = wall.y1 ?? wall.start?.[1] ?? 0;
    const x2 = wall.x2 ?? wall.end?.[0] ?? 0;
    const z2 = wall.y2 ?? wall.end?.[1] ?? 0;
    const thickness = wall.thickness || 0.15;
    const height = wall.height || 2.6;

    const dx = x2 - x1;
    const dz = z2 - z1;
    const length = Math.sqrt(dx * dx + dz * dz);
    if (length < 0.01) return;

    const angle = Math.atan2(dz, dx);

    const geo = new THREE.BoxGeometry(length, height, thickness);
    const mat = new THREE.MeshPhysicalMaterial({
      color: 0x2c2c3a,
      roughness: 0.7,
      metalness: 0.1,
      transparent: true,
      opacity: 0.85,
      clearcoat: 0.1,
    });

    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(
      (x1 + x2) / 2,
      height / 2,
      (z1 + z2) / 2,
    );
    mesh.rotation.y = -angle;
    mesh.castShadow = true;
    mesh.receiveShadow = true;

    this.houseGroup.add(mesh);

    // Wall top edge glow
    const edgeGeo = new THREE.EdgesGeometry(geo);
    const edgeMat = new THREE.LineBasicMaterial({
      color: 0x444466,
      transparent: true,
      opacity: 0.3,
    });
    const edgeMesh = new THREE.LineSegments(edgeGeo, edgeMat);
    edgeMesh.position.copy(mesh.position);
    edgeMesh.rotation.copy(mesh.rotation);
    this.houseGroup.add(edgeMesh);
  }

  _createDoor(door) {
    const x = door.x ?? 0;
    const z = door.y ?? 0;
    const w = door.width || 0.9;
    const h = 2.1;
    const rotation = door.rotation || 0;

    const geo = new THREE.BoxGeometry(w, h, 0.08);
    const mat = new THREE.MeshPhysicalMaterial({
      color: 0x00d4ff,
      roughness: 0.4,
      metalness: 0.3,
      transparent: true,
      opacity: 0.15,
      emissive: 0x00d4ff,
      emissiveIntensity: 0.1,
    });

    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(x, h / 2, z);
    mesh.rotation.y = rotation;
    this.houseGroup.add(mesh);
  }

  _createRouter(router) {
    const x = router.x ?? 0;
    const z = router.y ?? 0;
    const y = router.z ?? 1.5;

    // Remove old router if exists
    if (this.routerMesh) {
      this.houseGroup.remove(this.routerMesh);
    }

    const geo = new THREE.SphereGeometry(0.15, 16, 16);
    const mat = new THREE.MeshBasicMaterial({ color: 0x00ff88 });
    this.routerMesh = new THREE.Mesh(geo, mat);
    this.routerMesh.position.set(x, y, z);
    this.houseGroup.add(this.routerMesh);

    // Glow
    const glowGeo = new THREE.SphereGeometry(0.25, 16, 16);
    const glowMat = new THREE.MeshBasicMaterial({
      color: 0x00ff88,
      transparent: true,
      opacity: 0.15,
    });
    const glow = new THREE.Mesh(glowGeo, glowMat);
    glow.position.copy(this.routerMesh.position);
    this.houseGroup.add(glow);

    // Small light
    const light = new THREE.PointLight(0x00ff88, 1, 4);
    light.position.copy(this.routerMesh.position);
    this.houseGroup.add(light);
  }

  /**
   * Update the person's tracked position.
   * @param {number} x - X coordinate in meters
   * @param {number} y - Y coordinate in meters (maps to Z in 3D)
   * @param {number} [z=0.25] - Height (maps to Y in 3D)
   * @param {number} [confidence=0] - 0-1 confidence value
   */
  updatePersonPosition(x, y, z = 0.25, confidence = 0) {
    this._targetPos.set(x, z, y);

    this.personMesh.visible = true;
    this.personRing.visible = true;
    this.personLight.visible = true;

    // Add to trail
    this.trailPositions.push(new THREE.Vector3(x, z, y));
    if (this.trailPositions.length > this.maxTrail) {
      this.trailPositions.shift();
    }
    this._updateTrail();

    // Update confidence visual (ring scale)
    const confScale = 1 + (1 - confidence) * 2;
    this.personRing.scale.set(confScale, confScale, 1);
    this.personRing.material.opacity = 0.15 + confidence * 0.35;
  }

  updatePersons(persons) {
    const seen = new Set();

    persons.forEach((person, index) => {
      const id = String(person.id ?? index + 1);
      const x = person.x ?? person.position?.x ?? 0;
      const y = person.y ?? person.position?.y ?? 0;
      const z = person.z ?? person.position?.z ?? 0.25;
      const confidence = person.confidence ?? 0;
      const marker = this._getPersonMarker(id);

      marker.target.set(x, z, y);
      marker.confidence = confidence;
      marker.mesh.visible = true;
      marker.ring.visible = true;
      marker.light.visible = true;

      const confScale = 1 + (1 - confidence) * 2;
      marker.ring.scale.set(confScale, confScale, 1);
      marker.ring.material.opacity = 0.15 + confidence * 0.35;
      seen.add(id);
    });

    this.persons.forEach((marker, id) => {
      if (!seen.has(id)) {
        marker.mesh.visible = false;
        marker.ring.visible = false;
        marker.light.visible = false;
      }
    });
  }

  _getPersonMarker(id) {
    if (this.persons.has(id)) {
      return this.persons.get(id);
    }

    const color = this._personColor(id);
    const geo = new THREE.SphereGeometry(0.25, 32, 32);
    const mat = new THREE.MeshBasicMaterial({ color });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.visible = false;
    this.scene.add(mesh);

    const ringGeo = new THREE.RingGeometry(0.35, 0.45, 32);
    const ringMat = new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: 0.5,
      side: THREE.DoubleSide,
    });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.rotation.x = -Math.PI / 2;
    ring.visible = false;
    this.scene.add(ring);

    const light = new THREE.PointLight(color, 1.5, 3);
    light.visible = false;
    this.scene.add(light);

    const marker = {
      mesh,
      ring,
      light,
      current: new THREE.Vector3(0, 0.25, 0),
      target: new THREE.Vector3(0, 0.25, 0),
      confidence: 0,
    };
    this.persons.set(id, marker);
    return marker;
  }

  _personColor(id) {
    const colors = {
      1: 0xff2d55,
      2: 0x00d4ff,
      3: 0x00ff88,
    };
    return colors[id] || 0xffb000;
  }

  /**
   * Switch camera to a preset position.
   * @param {'top'|'isometric'|'follow'} preset
   */
  setCameraPreset(preset) {
    this._cameraPreset = preset;
    const cx = this._centerX;
    const cz = this._centerZ;
    const maxDim = Math.max(this._floorWidth, this._floorLength);

    switch (preset) {
      case 'top':
        this._cameraTargetPos = new THREE.Vector3(cx, maxDim * 1.3, cz);
        this._cameraTargetLookAt = new THREE.Vector3(cx, 0, cz);
        break;

      case 'isometric':
        this._cameraTargetPos = new THREE.Vector3(
          cx + maxDim * 0.8,
          maxDim * 0.9,
          cz + maxDim * 0.8,
        );
        this._cameraTargetLookAt = new THREE.Vector3(cx, 0, cz);
        break;

      case 'follow':
        // Follow will be handled in animate loop
        break;
    }

    if (preset !== 'follow') {
      this._cameraLerping = true;
    }
  }

  /** Animation loop */
  animate() {
    this.animationId = requestAnimationFrame(() => this.animate());

    const delta = this._clock.getDelta();
    const elapsed = this._clock.getElapsedTime();

    // Smooth person position
    this._currentPos.x = THREE.MathUtils.lerp(this._currentPos.x, this._targetPos.x, this._lerpFactor);
    this._currentPos.y = THREE.MathUtils.lerp(this._currentPos.y, this._targetPos.y, this._lerpFactor);
    this._currentPos.z = THREE.MathUtils.lerp(this._currentPos.z, this._targetPos.z, this._lerpFactor);

    if (this.personMesh.visible) {
      this.personMesh.position.copy(this._currentPos);
      this.personLight.position.copy(this._currentPos);
      this.personRing.position.set(this._currentPos.x, 0.05, this._currentPos.z);

      // Pulse ring rotation
      this.personRing.rotation.z = elapsed * 0.5;

      // Gentle bob for the person
      this.personMesh.position.y = this._currentPos.y + Math.sin(elapsed * 2) * 0.03;
    }

    this.persons.forEach((marker, id) => {
      marker.current.x = THREE.MathUtils.lerp(marker.current.x, marker.target.x, this._lerpFactor);
      marker.current.y = THREE.MathUtils.lerp(marker.current.y, marker.target.y, this._lerpFactor);
      marker.current.z = THREE.MathUtils.lerp(marker.current.z, marker.target.z, this._lerpFactor);

      if (marker.mesh.visible) {
        marker.mesh.position.copy(marker.current);
        marker.mesh.position.y = marker.current.y + Math.sin(elapsed * 2 + Number(id)) * 0.03;
        marker.light.position.copy(marker.current);
        marker.ring.position.set(marker.current.x, 0.05, marker.current.z);
        marker.ring.rotation.z = elapsed * 0.5;
      }
    });

    // Camera smooth transitions
    if (this._cameraLerping && this._cameraTargetPos) {
      this.camera.position.lerp(this._cameraTargetPos, 0.04);
      this.controls.target.lerp(this._cameraTargetLookAt, 0.04);

      if (this.camera.position.distanceTo(this._cameraTargetPos) < 0.05) {
        this._cameraLerping = false;
      }
    }

    // Follow mode
    if (this._cameraPreset === 'follow' && this.personMesh.visible) {
      const followTarget = new THREE.Vector3(
        this._currentPos.x + 5,
        8,
        this._currentPos.z + 5,
      );
      this.camera.position.lerp(followTarget, 0.03);
      this.controls.target.lerp(this._currentPos, 0.05);
    }

    // Router pulse
    if (this.routerMesh) {
      const scale = 1 + Math.sin(elapsed * 3) * 0.1;
      this.routerMesh.scale.set(scale, scale, scale);
    }

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.cssRenderer.render(this.scene, this.camera);
  }

  /** Handle canvas resize */
  resize() {
    const container = this.canvas.parentElement;
    const headerH = container.querySelector('.view-header')?.offsetHeight || 0;
    const w = container.clientWidth;
    const h = container.clientHeight - headerH;

    if (w <= 0 || h <= 0) return;

    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    this.cssRenderer.setSize(w, h);
  }

  /** Clean up */
  dispose() {
    cancelAnimationFrame(this.animationId);
    this.controls?.dispose();
    this.renderer?.dispose();
    if (this.cssRenderer?.domElement?.parentElement) {
      this.cssRenderer.domElement.parentElement.removeChild(this.cssRenderer.domElement);
    }
  }
}
