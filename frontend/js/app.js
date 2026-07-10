/**
 * App Entry Point — Wi-Fi Radar · Digital Twin
 * Main bootstrap module that initializes all components and connects them.
 */

import { WebSocketClient } from './websocket-client.js';
import { HouseScene3D } from './three-scene.js';
import { FloorPlan2D } from './floor-plan-2d.js';
import { FloorPlanEditor } from './floor-plan-editor.js';
import { DashboardControls } from './controls.js';
import { RadarOverlay } from './radar-overlay.js';
import { HouseModelLoader } from './model-loader.js';

// ─── App Initialization ─────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  console.log('[App] Wi-Fi Radar · Digital Twin — Initializing…');

  // ─── 1. Create WebSocket Client ─────────────────────
  const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
  const wsClient = new WebSocketClient(wsUrl);

  // ─── 2. Create 3D Scene ─────────────────────────────
  const canvas3d = document.getElementById('three-canvas');
  let scene3d = null;

  try {
    scene3d = new HouseScene3D(canvas3d);
    console.log('[App] 3D scene initialized');
  } catch (err) {
    console.error('[App] Failed to initialize 3D scene:', err);
    // Create a minimal stub so the rest of the app doesn't crash
    scene3d = {
      loadFloorPlan() {},
      updatePersonPosition() {},
      setCameraPreset() {},
      resize() {},
      dispose() {},
      maxTrail: 50,
    };
  }

  // ─── 3. Create 2D Floor Plan ────────────────────────
  const canvas2d = document.getElementById('floor-plan-canvas');
  const floorPlan2d = new FloorPlan2D(canvas2d);
  console.log('[App] 2D floor plan initialized');

  // ─── 4. Create Floor Plan Editor ────────────────────
  // The editor shares the 2D canvas but is activated/deactivated via tabs
  const editor = new FloorPlanEditor(canvas2d);
  console.log('[App] Floor plan editor initialized');

  // ─── 5. Create Dashboard Controls ──────────────────
  const controls = new DashboardControls(wsClient, scene3d, floorPlan2d, editor);
  controls.init();
  console.log('[App] Dashboard controls initialized');

  // ─── 6. Load Initial Floor Plan ─────────────────────
  await controls.loadInitialFloorPlan();
  console.log('[App] Initial floor plan loaded');

  // ─── 6.5 Radar overlay (anchors + rays + motion pulses) ───
  let radarOverlay = null;
  let modelLoader = null;
  if (scene3d?.scene && scene3d?.labelGroup) {
    try {
      radarOverlay = new RadarOverlay(scene3d.scene, scene3d.labelGroup);
      modelLoader = new HouseModelLoader(scene3d.scene);

      const anchorsResp = await fetch('/api/anchors');
      const anchorsCfg = anchorsResp.ok ? await anchorsResp.json() : null;
      if (anchorsCfg) {
        radarOverlay.setConfig(anchorsCfg);
        await modelLoader.tryLoad(anchorsCfg.model);
      }

      // Click + drag para mover anchors en el piso. Guarda solo a /api/anchors.
      if (scene3d.camera && scene3d.controls && scene3d.renderer) {
        radarOverlay.enableDragging(
          scene3d.camera, scene3d.controls, scene3d.renderer.domElement,
        );
      }

      // Hook radar events from WebSocket
      wsClient.addEventListener('radar-event', (e) => {
        radarOverlay.handleEvent(e.detail);
      });

      // Render the people-in-house panel once so it appears immediately
      // (even if no events have arrived yet)
      radarOverlay._renderPeopleInHouse();

      // Poll system status every 2s for the SISTEMA panel
      const poll = () => radarOverlay._pollSystemStatus();
      poll();
      setInterval(poll, 2000);

      // Patch into the existing scene animate loop via a tick hook
      const origAnimate = scene3d.animate?.bind(scene3d);
      if (origAnimate) {
        // The scene already calls requestAnimationFrame -> animate(). We hook tick
        // by wrapping its render call. Easier: just spin our own RAF for the overlay.
        function overlayTick() {
          radarOverlay.tick();
          requestAnimationFrame(overlayTick);
        }
        requestAnimationFrame(overlayTick);
      }
      console.log('[App] Radar overlay initialized');
    } catch (err) {
      console.error('[App] Radar overlay init failed:', err);
    }
  }

  // ─── 7. Connect WebSocket ──────────────────────────
  wsClient.connect();
  console.log('[App] WebSocket connecting…');

  // ─── 8. Window Resize Handler ──────────────────────
  let resizeTimer = null;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      scene3d.resize();
      floorPlan2d.resize();
      if (editor.active) {
        editor.resize();
      }
    }, 100);
  });

  // Trigger initial resize after a tick (layout stabilization)
  requestAnimationFrame(() => {
    scene3d.resize();
    floorPlan2d.resize();
  });

  // ─── 9. Splash / Ready Message ─────────────────────
  console.log(
    '%c Wi-Fi Radar · Digital Twin %c Ready ',
    'background: linear-gradient(135deg, #00d4ff, #0099cc); color: #000; padding: 4px 12px; border-radius: 4px 0 0 4px; font-weight: 700;',
    'background: #1a1a2e; color: #00d4ff; padding: 4px 12px; border-radius: 0 4px 4px 0;',
  );

  // ─── 10. Expose for debugging ──────────────────────
  window.__radar = {
    wsClient,
    scene3d,
    floorPlan2d,
    editor,
    controls,
  };
});
