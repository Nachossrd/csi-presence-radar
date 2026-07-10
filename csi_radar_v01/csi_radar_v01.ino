// CSI Wall Radar - ESP32-S3 MVP
// Conecta a Wi-Fi, habilita CSI, envia amplitud + varianza por USB-Serial.
// Output (CSV): timestamp_ms,rssi_dbm,n_subc,amp_mean,amp_var,channel,src_mac
//
// Pipeline: ESP32 -> USB-Serial -> Python parser -> FastAPI dashboard

#include <WiFi.h>
#include "esp_wifi.h"
#include "esp_log.h"
#include <math.h>

// ====== CAMBIA ESTOS ======
const char* WIFI_SSID = "TU_WIFI_SSID";        // SSID de tu Wi-Fi de casa
const char* WIFI_PASS = "TU_WIFI_PASSWORD";   // password
// ==========================

static volatile uint32_t pkt_count = 0;

void IRAM_ATTR csi_cb(void *ctx, wifi_csi_info_t *info) {
  if (!info || info->len <= 0) return;
  pkt_count++;

  // CSI: array de int8_t como pares (I, Q) por subcarrier
  // len = bytes totales; cada subcarrier ocupa 2 bytes (I, Q signed)
  int n_subc = info->len / 2;

  double sum = 0.0, sum_sq = 0.0;
  for (int i = 0; i < n_subc; i++) {
    int8_t I = info->buf[i * 2];
    int8_t Q = info->buf[i * 2 + 1];
    double amp = sqrt((double)(I * I + Q * Q));
    sum += amp;
    sum_sq += amp * amp;
  }
  double mean = sum / n_subc;
  double var = (sum_sq / n_subc) - (mean * mean);

  // MAC del origen del paquete
  const uint8_t *mac = info->mac;

  // Imprimir CSV: ts,rssi,n_subc,amp_mean,amp_var,channel,mac
  Serial.printf("%lu,%d,%d,%.2f,%.2f,%u,%02x:%02x:%02x:%02x:%02x:%02x\n",
                millis(),
                (int)info->rx_ctrl.rssi,
                n_subc,
                mean,
                var,
                (unsigned)info->rx_ctrl.channel,
                mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n[boot] ESP32-S3 CSI radar v0.1");
  Serial.printf("[boot] PSRAM size: %u\n", ESP.getPsramSize());

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("[wifi] connecting");
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(500);
  }
  Serial.printf("\n[wifi] OK ip=%s rssi=%d ch=%u\n",
                WiFi.localIP().toString().c_str(),
                WiFi.RSSI(),
                WiFi.channel());

  // Configurar y habilitar CSI
  wifi_csi_config_t cfg = {};
  cfg.lltf_en = true;
  cfg.htltf_en = true;
  cfg.stbc_htltf2_en = true;
  cfg.ltf_merge_en = true;
  cfg.channel_filter_en = 0;
  cfg.manu_scale = 0;
  cfg.shift = 0;
  esp_wifi_set_csi_config(&cfg);
  esp_wifi_set_csi_rx_cb(csi_cb, NULL);
  esp_wifi_set_csi(true);

  Serial.println("[csi] CSI habilitado. Imprimiendo CSV por cada paquete recibido:");
  Serial.println("ts_ms,rssi,n_subc,amp_mean,amp_var,channel,src_mac");
}

void loop() {
  // Cada 5s: heartbeat con tasa de paquetes recibidos
  static uint32_t last = 0;
  if (millis() - last > 5000) {
    Serial.printf("[stats] pkt_count=%lu (last 5s)\n", pkt_count);
    pkt_count = 0;
    last = millis();
  }
  delay(10);
}
