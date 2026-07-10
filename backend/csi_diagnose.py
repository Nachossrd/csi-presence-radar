"""CSI diagnostic: capture 30s of raw samples and analyze all detection modes.

Outputs:
  - Overall stats for amp_mean and amp_var
  - Top 5 MACs by packet count
  - Per-MAC stats for the top sources
  - Would-have-fired analysis: simulate the detector under multiple parameter
    sets and report when each one would have entered "motion" state.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import Counter, defaultdict, deque
from typing import List

import serial

from backend.csi_reader import parse_line, CSISample, MAD_FLOOR


def simulate(values: List[float], baseline_n: int,
             k_sigma: float, window: int, threshold: float) -> tuple[int, int, float]:
    """Run the same robust detector over a pre-recorded sample list.

    Returns (motion_frames, total_frames, peak_anomaly_rate).
    """
    if len(values) < baseline_n + window:
        return 0, 0, 0.0
    baseline = values[:baseline_n]
    median = statistics.median(baseline)
    mad = max(MAD_FLOOR, 1.4826 * statistics.median([abs(x - median) for x in baseline]))

    win: deque = deque(maxlen=window)
    motion_frames = 0
    total_frames = 0
    peak_anom = 0.0
    for v in values[baseline_n:]:
        win.append(v)
        if len(win) >= window // 2:
            anoms = sum(1 for x in win if abs(x - median) > k_sigma * mad)
            rate = anoms / len(win)
            peak_anom = max(peak_anom, rate)
            total_frames += 1
            if rate >= threshold:
                motion_frames += 1
    return motion_frames, total_frames, peak_anom


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="COM7")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--duration", type=float, default=30.0)
    args = p.parse_args()

    print(f"Capturando {args.duration:.0f}s de CSI desde {args.port}...")
    s = serial.Serial(args.port, args.baud, timeout=1.0)
    s.setDTR(False); time.sleep(0.1); s.setDTR(True); time.sleep(0.5)

    samples: List[CSISample] = []
    t_end = time.time() + args.duration
    while time.time() < t_end:
        line = s.readline().decode("utf-8", errors="replace").strip()
        if not line:
            continue
        sm = parse_line(line)
        if sm:
            samples.append(sm)
    s.close()

    n = len(samples)
    if n < 30:
        print(f"Solo {n} samples capturados. ESP32 no esta enviando.")
        sys.exit(1)

    print(f"\nCapturados {n} samples en {args.duration:.0f}s "
          f"= {n/args.duration:.1f} Hz")

    # Overall stats
    amp_means = [x.amp_mean for x in samples]
    amp_vars = [x.amp_var for x in samples]
    rssis = [x.rssi for x in samples]

    def stats(vals):
        return (min(vals), statistics.median(vals), max(vals),
                statistics.fmean(vals), statistics.stdev(vals) if len(vals) > 1 else 0)

    am_min, am_med, am_max, am_avg, am_std = stats(amp_means)
    av_min, av_med, av_max, av_avg, av_std = stats(amp_vars)
    rs_min, rs_med, rs_max, rs_avg, rs_std = stats(rssis)

    print(f"\n=== Stats globales ===")
    print(f"  amp_mean  min={am_min:7.2f}  median={am_med:7.2f}  max={am_max:8.2f}  avg={am_avg:7.2f}  std={am_std:.2f}")
    print(f"  amp_var   min={av_min:7.2f}  median={av_med:7.2f}  max={av_max:8.2f}  avg={av_avg:7.2f}  std={av_std:.2f}")
    print(f"  rssi      min={rs_min:7d}  median={rs_med:7.1f}  max={rs_max:8d}  avg={rs_avg:7.1f}")

    # Top MACs by frequency
    mac_counts = Counter(x.src_mac for x in samples)
    top_macs = mac_counts.most_common(8)
    print(f"\n=== Top 8 MACs por frecuencia ===")
    print(f"  MACs unicas: {len(mac_counts)}")
    for mac, cnt in top_macs:
        rate_hz = cnt / args.duration
        print(f"  {cnt:5d} pkts ({rate_hz:5.1f} Hz)  {mac}")

    # Per-MAC stats for top 3
    print(f"\n=== Stats per-MAC para los top 3 ===")
    for mac, cnt in top_macs[:3]:
        mac_samples = [x for x in samples if x.src_mac == mac]
        if len(mac_samples) < 5:
            continue
        m_am = [x.amp_mean for x in mac_samples]
        m_av = [x.amp_var for x in mac_samples]
        print(f"  {mac}  ({len(mac_samples)} pkts)")
        print(f"    amp_mean  med={statistics.median(m_am):6.2f}  range=[{min(m_am):.1f}, {max(m_am):.1f}]")
        print(f"    amp_var   med={statistics.median(m_av):6.2f}  range=[{min(m_av):.1f}, {max(m_av):.1f}]")

    # Detection simulation across multiple parameter sets
    print(f"\n=== Simulacion: que detector hubiera disparado motion? ===")
    baseline_n = min(150, n // 3)  # use first ~5s as calibration baseline
    configs = [
        ("amp_mean k=2.5 w=30 t=15%", amp_means, 2.5, 30, 0.15),
        ("amp_mean k=4.0 w=60 t=30%", amp_means, 4.0, 60, 0.30),  # original defaults
        ("amp_var  k=2.5 w=30 t=15%", amp_vars,  2.5, 30, 0.15),  # new defaults
        ("amp_var  k=2.0 w=20 t=10%", amp_vars,  2.0, 20, 0.10),  # very sensitive
        ("amp_var  k=3.0 w=20 t=20%", amp_vars,  3.0, 20, 0.20),
    ]
    print(f"  {'config':<35} motion_frames/total  peak_anom%")
    for name, vals, k, w, t in configs:
        mf, tf, peak = simulate(vals, baseline_n, k, w, t)
        if tf == 0:
            print(f"  {name:<35} not enough samples")
        else:
            pct = 100 * mf / tf
            tag = "  <-- TRIGGERED" if mf > 0 else ""
            print(f"  {name:<35} {mf:5d}/{tf:<5d} ({pct:5.1f}%)   peak={peak*100:4.1f}%{tag}")

    # Per-MAC simulation for the most active MAC (likely the router)
    if top_macs:
        top_mac, top_cnt = top_macs[0]
        mac_samples = [x for x in samples if x.src_mac == top_mac]
        if len(mac_samples) >= 50:
            mac_am = [x.amp_mean for x in mac_samples]
            mac_av = [x.amp_var for x in mac_samples]
            base_n = min(15, len(mac_samples) // 3)
            print(f"\n=== Simulacion FILTRADA a {top_mac} ({top_cnt} pkts) ===")
            for metric_name, vals in [("amp_mean", mac_am), ("amp_var", mac_av)]:
                for k, w, t in [(2.5, 10, 0.20), (2.0, 8, 0.15)]:
                    mf, tf, peak = simulate(vals, base_n, k, w, t)
                    if tf > 0:
                        tag = "  <-- TRIGGERED" if mf > 0 else ""
                        print(f"  {metric_name} k={k} w={w} t={int(t*100)}%   "
                              f"{mf}/{tf}  peak={peak*100:.1f}%{tag}")

    print()
    print("=== Recomendacion ===")
    print("La config que mas frames de motion detecto Y tiene peak_anom alto")
    print("es la mejor. Si NINGUNA disparo, significa que durante esta captura")
    print("no hubo motion suficiente cerca del ESP32, o la baseline absorbio")
    print("el ruido de fondo.")


if __name__ == "__main__":
    main()
