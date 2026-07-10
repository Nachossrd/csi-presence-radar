// CSI Wall Radar - ESP32-S3 MVP (v0.2)
// Modo promiscuo: NO se conecta a Wi-Fi, solo escucha el canal y captura
// CSI de TODOS los paquetes 802.11 que pasen por el aire.
//
// Output (CSV por USB-Serial @ 115200):
//   ts_ms,rssi,n_subc,amp_mean,amp_var,channel,src_mac

#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "nvs_flash.h"
#include <math.h>

// Tu Movistar esta en canal 11 (chequeado con netsh wlan show interfaces).
static const uint8_t CHANNEL = 11;

static volatile uint32_t pkt_count = 0;
static volatile uint32_t csi_count = 0;

void IRAM_ATTR csi_cb(void *ctx, wifi_csi_info_t *info) {
  if (!info || info->len <= 0) return;
  csi_count++;

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

  const uint8_t *mac = info->mac;
  Serial.printf("%lu,%d,%d,%.2f,%.2f,%u,%02x:%02x:%02x:%02x:%02x:%02x\n",
                millis(),
                (int)info->rx_ctrl.rssi,
                n_subc,
                mean,
                var,
                (unsigned)info->rx_ctrl.channel,
                mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
}

void IRAM_ATTR sniffer_cb(void *buf, wifi_promiscuous_pkt_type_t type) {
  pkt_count++;
}

void setup() {
  Serial.begin(115200);
  delay(700);
  Serial.println("\n[boot] ESP32-S3 CSI radar v0.2 (promiscuous, full init)");
  Serial.printf("[boot] PSRAM=%u flash=%u\n", ESP.getPsramSize(), ESP.getFlashChipSize());

  // NVS init (requerido por esp_wifi_init en algunos paths)
  esp_err_t nvs_err = nvs_flash_init();
  if (nvs_err == ESP_ERR_NVS_NO_FREE_PAGES || nvs_err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    nvs_flash_erase();
    nvs_flash_init();
  }

  // TCP/IP stack + event loop (necesario para que esp_wifi_start no falle)
  esp_netif_init();
  esp_event_loop_create_default();

  // Wi-Fi init con config default
  wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
  esp_err_t e1 = esp_wifi_init(&cfg);
  Serial.printf("[init] wifi_init: %d\n", (int)e1);
  esp_wifi_set_storage(WIFI_STORAGE_RAM);

  // MODE_NULL: sin STA, sin AP — el radio queda libre para promiscuo
  esp_err_t e2 = esp_wifi_set_mode(WIFI_MODE_NULL);
  Serial.printf("[init] set_mode(NULL): %d\n", (int)e2);

  esp_err_t e3 = esp_wifi_start();
  Serial.printf("[init] wifi_start: %d\n", (int)e3);

  // Filter: capturar MGMT + DATA + CTRL + MISC
  wifi_promiscuous_filter_t filter = {};
  filter.filter_mask = WIFI_PROMIS_FILTER_MASK_ALL;
  esp_wifi_set_promiscuous_filter(&filter);
  esp_wifi_set_promiscuous_rx_cb(sniffer_cb);

  // CSI config
  wifi_csi_config_t ccfg = {};
  ccfg.lltf_en = true;
  ccfg.htltf_en = true;
  ccfg.stbc_htltf2_en = true;
  ccfg.ltf_merge_en = true;
  ccfg.channel_filter_en = 0;
  ccfg.manu_scale = 0;
  ccfg.shift = 0;
  esp_err_t e4 = esp_wifi_set_csi_config(&ccfg);
  Serial.printf("[init] set_csi_config: %d\n", (int)e4);
  esp_wifi_set_csi_rx_cb(csi_cb, NULL);

  esp_err_t e5 = esp_wifi_set_channel(CHANNEL, WIFI_SECOND_CHAN_NONE);
  Serial.printf("[init] set_channel(%u): %d\n", CHANNEL, (int)e5);

  esp_err_t e6 = esp_wifi_set_promiscuous(true);
  Serial.printf("[init] set_promiscuous(true): %d\n", (int)e6);

  esp_err_t e7 = esp_wifi_set_csi(true);
  Serial.printf("[init] set_csi(true): %d\n", (int)e7);

  Serial.printf("[csi] LISTO. canal=%u\n", CHANNEL);
  Serial.println("ts_ms,rssi,n_subc,amp_mean,amp_var,channel,src_mac");
}

void loop() {
  static uint32_t last = 0;
  if (millis() - last > 5000) {
    Serial.printf("[stats] pkts_5s=%lu csi_5s=%lu\n",
                  (unsigned long)pkt_count, (unsigned long)csi_count);
    pkt_count = 0;
    csi_count = 0;
    last = millis();
  }
  delay(10);
}
