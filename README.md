# 📡 CSI Presence Radar

> Prototipo de radar de presencia y localización *indoor* que busca depender **menos** de cámaras: combina **Wi-Fi CSI** (Channel State Information) de un ESP32, escaneo **BLE/Wi-Fi multi-AP**, visión por computador opcional y un **gemelo digital 3D** del espacio para detectar y ubicar personas. Proyecto de investigación / prueba de concepto.

**Por qué este proyecto:** para investigar alternativas de detección de presencia **menos invasivas que las cámaras** — usando las señales de radio que ya atraviesan un espacio (Wi-Fi, BLE) en lugar de grabar imagen de forma continua.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-websockets-009688?logo=fastapi&logoColor=white">
  <img alt="ESP32" src="https://img.shields.io/badge/ESP32-CSI-E7352C?logo=espressif&logoColor=white">
  <img alt="Three.js" src="https://img.shields.io/badge/Three.js-3D-000000?logo=three.js&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

---

## 🎯 Idea

Detectar **dónde hay personas** dentro de un espacio usando las señales de radio que ya lo atraviesan — sin instalar cámaras en cada habitación. El movimiento de una persona perturba la *Channel State Information* del Wi-Fi; cruzando eso con triangulación BLE/Wi-Fi multi-AP y un mapa 2D/3D del lugar, el sistema estima posición y estado (quieto / moviéndose / ausente).

## ✨ Componentes

- **Firmware ESP32** (`esp32/`, `csi_radar_v01/`) — captura CSI cruda del Wi-Fi y la envía por serie/red al backend.
- **Backend (FastAPI + WebSockets)** (`backend/`):
  - `csi_reader` / `wifi_radar` / `multi_ap_radar` — lectura CSI y radar Wi-Fi multi-AP
  - `ble_radar` / `wifi_scanner` / `wifi_localizer` — escaneo y trilateración BLE/Wi-Fi
  - `camera_detector` / `face_identifier` — visión y detección opcional (YOLO)
  - `floor_mapper` / `scanner_3d` / `digital_twin` — mapa del espacio y gemelo digital
  - `state_classifier` — clasificación de estado de presencia
  - `event_broadcast` / `dashboard_listener` — streaming de eventos al frontend
- **Frontend** (`frontend/`) — dashboard web con planta 2D editable y escena **3D (Three.js)** en vivo por WebSocket.

## 🏗️ Arquitectura

```
  ESP32 (CSI)  ─┐
  BLE/Wi-Fi     ├─▶  Backend FastAPI  ──WebSocket──▶  Frontend 3D / 2D
  Cámara (opc.) ─┘   (radar + fusión + gemelo digital)   (Three.js)
```

## 🚀 Uso

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn backend.api:app --host 127.0.0.1 --port 8000

# Firmware ESP32
#  1. Abre csi_radar_v01/csi_radar_v01.ino en Arduino IDE / PlatformIO
#  2. Configura WIFI_SSID y WIFI_PASS (placeholders en el código)
#  3. Flashea a un ESP32 con soporte CSI

# Frontend: sirve la carpeta frontend/ (o ábrela vía el dashboard del backend)
```

Los modelos YOLO (`yolov8n.pt` / `yolov8s.pt`) no se incluyen; descárgalos de Ultralytics y colócalos en la raíz.

## 🔒 Privacidad — importante

Este proyecto trabaja con datos potencialmente **biométricos y personales** (rostros, embeddings, MACs de dispositivos, plano de una vivienda). Por diseño, **la carpeta `data/`, los embeddings y los modelos están excluidos del repositorio** (`.gitignore`). Ejecuta el sistema **solo en espacios propios y con consentimiento** de las personas involucradas. Las credenciales Wi-Fi del firmware son placeholders — coloca las tuyas localmente.

## 📊 Benchmarks

> ⚠️ **Pendiente de medición.** El rendimiento depende fuertemente del entorno RF (nº de APs, materiales, interferencia), así que estas métricas solo tienen sentido con su metodología y entorno declarados. Objetivo:

| Métrica | Cómo se mide | Valor |
|---------|--------------|-------|
| FPS de detección | frames CSI procesados / seg en el backend | *por medir* |
| Latencia evento → dashboard | timestamp del evento vs. recepción en el frontend (WebSocket) | *por medir* |
| Precisión de zona | detecciones correctas de zona / total, en un layout etiquetado | *por medir* |
| Error de localización | distancia media (m) entre posición estimada y real | *por medir* |

*Entorno de referencia: por definir (nº de APs, tamaño del espacio, hardware).*

## ⚠️ Limitaciones

- **Prueba de concepto**, no un sistema de seguridad ni de vigilancia certificado.
- La precisión depende de la **calibración** y del **entorno RF**; muros gruesos, metal e interferencia degradan la señal.
- Requiere **hardware específico** (ESP32 con soporte CSI y varios APs) y suele necesitar re-calibración si cambia el layout.
- La localización por trilateración BLE/Wi-Fi es **aproximada** (orden de metros), no centimétrica.
- Trabaja con datos potencialmente **biométricos/personales**: úsalo solo en espacios propios y con consentimiento (ver sección de privacidad).

## 🛠️ Stack

`Python` · `FastAPI` · `WebSockets` · `NumPy` · `OpenCV / YOLO` · `ESP32 (Arduino/CSI)` · `Three.js` · `JavaScript`

---

## ✍️ Autor

**Nacho** — [@Nachossrd](https://github.com/Nachossrd)

## 📄 Licencia

MIT — ver [`LICENSE`](LICENSE).
