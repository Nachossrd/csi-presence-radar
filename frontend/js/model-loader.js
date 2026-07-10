/**
 * House model loader — drops the user's Polycam / Luma / RealityCapture scan
 * of their house into the existing 3D scene when one is available.
 *
 * Supports both .glb / .gltf (modern, single file) and .obj + .mtl + textures
 * (what Polycam free-tier exports without LiDAR).
 *
 * Set the path in data/anchors.json -> model.file. Relative to /data, so:
 *   "models/31_5_2026/31_5_2026.obj"
 *   "house_model.glb"
 *
 * Adjust offset / rotation_y_deg / scale to align the scan with the
 * procedural floor plan (origin of scans is arbitrary).
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { MTLLoader } from 'three/addons/loaders/MTLLoader.js';

export class HouseModelLoader {
  constructor(scene) {
    this.scene = scene;
    this.group = new THREE.Group();
    this.group.name = 'HouseModel';
    this.scene.add(this.group);
    this.loaded = false;
  }

  async tryLoad(modelCfg) {
    if (!modelCfg) return false;
    const file = modelCfg.file || 'house_model.glb';
    const url = `/data/${file}`;

    if (!await this._exists(url)) {
      console.log(`[model] ${url} no existe, usando escena procedural.`);
      return false;
    }

    try {
      if (file.toLowerCase().endsWith('.obj')) {
        await this._loadObj(url);
      } else if (file.toLowerCase().endsWith('.glb') ||
                 file.toLowerCase().endsWith('.gltf')) {
        await this._loadGltf(url);
      } else {
        console.warn(`[model] Formato no soportado: ${file}. Usa .obj o .glb`);
        return false;
      }
      this._applyTransform(modelCfg);
      this.loaded = true;
      console.log(`[model] Cargado ${url}`);
      return true;
    } catch (err) {
      console.error(`[model] Error cargando ${url}:`, err);
      return false;
    }
  }

  async _exists(url) {
    try {
      const r = await fetch(url, { method: 'HEAD' });
      return r.ok;
    } catch {
      return false;
    }
  }

  async _loadGltf(url) {
    const loader = new GLTFLoader();
    const gltf = await loader.loadAsync(url);
    this.group.add(gltf.scene);
  }

  /**
   * OBJ + MTL + textures. The .mtl references textures by relative path,
   * so we must set the base path on both loaders.
   */
  async _loadObj(url) {
    const lastSlash = url.lastIndexOf('/');
    const basePath = url.slice(0, lastSlash + 1);
    const objName = url.slice(lastSlash + 1);
    const mtlName = objName.replace(/\.obj$/i, '.mtl');
    const mtlUrl = basePath + mtlName;

    // Try to load the MTL first for materials/textures
    let materials = null;
    if (await this._exists(mtlUrl)) {
      try {
        const mtlLoader = new MTLLoader();
        mtlLoader.setPath(basePath);
        materials = await new Promise((resolve, reject) => {
          mtlLoader.load(mtlName, (m) => { m.preload(); resolve(m); }, undefined, reject);
        });
      } catch (err) {
        console.warn(`[model] MTL cargo con error, sigo sin texturas:`, err);
      }
    }

    const objLoader = new OBJLoader();
    if (materials) objLoader.setMaterials(materials);
    objLoader.setPath(basePath);

    const obj = await new Promise((resolve, reject) => {
      objLoader.load(objName, resolve, undefined, reject);
    });

    // Polycam .obj scans often come without normals -> lighting looks flat.
    // Compute them if missing.
    obj.traverse((child) => {
      if (child.isMesh && child.geometry) {
        if (!child.geometry.attributes.normal) {
          child.geometry.computeVertexNormals();
        }
        // Make double-sided so we see interior walls when camera is inside
        if (child.material) {
          if (Array.isArray(child.material)) {
            child.material.forEach((m) => { m.side = THREE.DoubleSide; });
          } else {
            child.material.side = THREE.DoubleSide;
          }
        }
        child.castShadow = false;
        child.receiveShadow = true;
      }
    });
    this.group.add(obj);
  }

  _applyTransform(cfg) {
    const off = cfg.offset || {};
    this.group.position.set(off.x || 0, off.y || 0, off.z || 0);
    if (cfg.rotation_y_deg) {
      this.group.rotation.y = (cfg.rotation_y_deg * Math.PI) / 180;
    }
    if (cfg.scale && cfg.scale !== 1) {
      this.group.scale.setScalar(cfg.scale);
    }
  }

  setVisible(visible) {
    this.group.visible = visible;
  }

  dispose() {
    if (this.scene) this.scene.remove(this.group);
  }
}
