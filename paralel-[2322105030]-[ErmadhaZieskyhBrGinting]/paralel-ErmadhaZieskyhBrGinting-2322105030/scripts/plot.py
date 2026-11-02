#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot.py - Membuat grafik dari results/benchmark.csv.

Grafik yang dihasilkan (folder results/):
  waktu_<filter>.png        waktu eksekusi (median +- std) vs jumlah thread
  speedup_<filter>.png      speedup vs thread + garis ideal + kurva Amdahl
  efficiency_<filter>.png   efisiensi vs thread
  io_vs_compute_<filter>.png  stacked bar baca | komputasi | tulis
  varian_serial.png         waktu serial per varian (analisis cache locality)
  ukuran_<filter>.png       perbandingan ukuran citra (scaling), bila ada >1 ukuran
  amdahl.csv                estimasi serial fraction hasil fitting Amdahl

Estimasi serial fraction f (Amdahl's Law):
    S(p) = 1 / (f + (1 - f) / p)
Nilai f dicari dengan least squares (grid search f in [0,1], langkah 1e-4)
terhadap speedup terukur pada p = 1, 2, 4, 8, ...; speedup maksimum teoretis = 1/f.

Pemakaian:
    python3 scripts/plot.py [--csv results/benchmark.csv] [--outdir results]
                            [--size 2000x2000] [--border replicate]
                            [--schedule static]
Prasyarat: python3, numpy, matplotlib (pip install numpy matplotlib).
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")  # backend tanpa layar, aman di server/WSL
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

METHOD_LABEL = {"serial": "Serial", "pthreads": "Pthreads", "openmp": "OpenMP"}
METHOD_STYLE = {"pthreads": ("-", "o"), "openmp": ("--", "s")}
VARIANT_COLOR = {
    "direct": "tab:blue",
    "separable": "tab:green",
    "tiled": "tab:orange",
    "column": "tab:red",
}
AMDAHL_GRID_STEP = 1e-4
AMDAHL_EXTRAPOLATE_FACTOR = 2  # kurva prediksi sampai 2x thread terbesar yang diukur
MS = 1000.0
DPI = 150


def load_rows(path):
    """Baca CSV benchmark menjadi list of dict dengan tipe numerik."""
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            r["thread"] = int(r["thread"])
            for key in ("waktu_median_s", "waktu_std_s", "waktu_min_s", "speedup",
                        "efficiency", "baca_s", "tulis_s"):
                r[key] = float(r[key])
            rows.append(r)
    return rows


def select(rows, **conds):
    """Filter baris berdasarkan kesamaan kolom."""
    return [r for r in rows if all(r[k] == v for k, v in conds.items())]


def series_by_method_variant(rows):
    """Kelompokkan baris paralel menjadi {(metode, varian): [baris urut thread]}."""
    groups = defaultdict(list)
    for r in rows:
        if r["metode"] != "serial":
            groups[(r["metode"], r["varian"])].append(r)
    for key in groups:
        groups[key].sort(key=lambda r: r["thread"])
    return groups


def fit_amdahl(threads, speedups):
    """Cari serial fraction f yang meminimalkan galat kuadrat. Return (f, rmse)."""
    p = np.asarray(threads, dtype=float)
    s = np.asarray(speedups, dtype=float)
    grid = np.arange(0.0, 1.0 + AMDAHL_GRID_STEP, AMDAHL_GRID_STEP)
    pred = 1.0 / (grid[:, None] + (1.0 - grid[:, None]) / p[None, :])
    err = np.sum((pred - s[None, :]) ** 2, axis=1)
    best = int(np.argmin(err))
    return float(grid[best]), float(np.sqrt(err[best] / len(p)))


def amdahl_curve(f, p_values):
    p = np.asarray(p_values, dtype=float)
    return 1.0 / (f + (1.0 - f) / p)


def set_thread_axis(ax, threads):
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(set(threads)))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Jumlah thread")
    ax.grid(True, alpha=0.3)


def save(fig, outdir, name):
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print("  ->", path)


def plot_time(rows, flt, title_suffix, outdir):
    groups = series_by_method_variant(rows)
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(8, 5.2))
    all_threads = []
    for (method, variant), rs in sorted(groups.items()):
        ls, mk = METHOD_STYLE[method]
        x = [r["thread"] for r in rs]
        all_threads += x
        ax.errorbar(x, [r["waktu_median_s"] * MS for r in rs],
                    yerr=[r["waktu_std_s"] * MS for r in rs], color=VARIANT_COLOR[variant],
                    linestyle=ls, marker=mk, capsize=3,
                    label=f"{METHOD_LABEL[method]} - {variant}")
    set_thread_axis(ax, all_threads)
    ax.set_yscale("log")
    ax.set_ylabel("Waktu komputasi, median (ms) - skala log")
    ax.set_title(f"Waktu eksekusi - {flt} ({title_suffix})")
    ax.legend(fontsize=8, ncol=2)
    save(fig, outdir, f"waktu_{flt}.png")


def plot_speedup_eff(rows, flt, title_suffix, outdir, amdahl_out):
    groups = series_by_method_variant(rows)
    if not groups:
        return
    max_threads = max(r["thread"] for rs in groups.values() for r in rs)
    p_ext = np.arange(1, max_threads * AMDAHL_EXTRAPOLATE_FACTOR + 1)

    fig_s, ax_s = plt.subplots(figsize=(8, 5.2))
    fig_e, ax_e = plt.subplots(figsize=(8, 5.2))
    all_threads = []
    for (method, variant), rs in sorted(groups.items()):
        ls, mk = METHOD_STYLE[method]
        x = [r["thread"] for r in rs]
        all_threads += x
        color = VARIANT_COLOR[variant]
        label = f"{METHOD_LABEL[method]} - {variant}"
        ax_s.plot(x, [r["speedup"] for r in rs], color=color, linestyle=ls, marker=mk, label=label)
        ax_e.plot(x, [r["efficiency"] * 100 for r in rs], color=color, linestyle=ls, marker=mk,
                  label=label)

        f, rmse = fit_amdahl(x, [r["speedup"] for r in rs])
        amdahl_out.append({
            "filter": flt, "metode": method, "varian": variant,
            "serial_fraction": f, "speedup_maks_teoretis": (1.0 / f) if f > 0 else float("inf"),
            "rmse_fit": rmse,
        })
        if variant == "direct":  # kurva prediksi hanya untuk varian utama agar grafik terbaca
            ax_s.plot(p_ext, amdahl_curve(f, p_ext), color=color, linestyle=":", alpha=0.8,
                      label=f"Amdahl {METHOD_LABEL[method]} (f={f:.3f})")

    ax_s.plot(p_ext, p_ext, color="black", linestyle="-.", linewidth=1, label="Ideal (S = p)")
    set_thread_axis(ax_s, all_threads + [int(p_ext[-1])])
    ax_s.set_yscale("log", base=2)
    ax_s.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_s.set_ylabel("Speedup (T_serial / T_paralel)")
    ax_s.set_title(f"Speedup - {flt} ({title_suffix})")
    ax_s.legend(fontsize=7, ncol=2)
    save(fig_s, outdir, f"speedup_{flt}.png")

    ax_e.axhline(100, color="black", linestyle="-.", linewidth=1, label="Ideal (100%)")
    set_thread_axis(ax_e, all_threads)
    ax_e.set_ylabel("Efisiensi (%) = Speedup / p x 100")
    ax_e.set_title(f"Efisiensi - {flt} ({title_suffix})")
    ax_e.legend(fontsize=8, ncol=2)
    save(fig_e, outdir, f"efficiency_{flt}.png")


def plot_io_compute(rows, flt, title_suffix, outdir):
    sel = select(rows, varian="direct")
    serial = [r for r in sel if r["metode"] == "serial"]
    if not serial:
        return
    cats, read_ms, comp_ms, write_ms = [], [], [], []

    def add(label, r):
        cats.append(label)
        read_ms.append(r["baca_s"] * MS)
        comp_ms.append(r["waktu_median_s"] * MS)
        write_ms.append(r["tulis_s"] * MS)

    add("Serial", serial[0])
    for r in sorted((r for r in sel if r["metode"] != "serial"),
                    key=lambda r: (r["thread"], r["metode"])):
        add(f"{METHOD_LABEL[r['metode']]}\n{r['thread']}T", r)

    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(cats) + 3), 5.2))
    idx = np.arange(len(cats))
    ax.bar(idx, read_ms, label="Baca file", color="tab:blue")
    ax.bar(idx, comp_ms, bottom=read_ms, label="Komputasi", color="tab:orange")
    ax.bar(idx, write_ms, bottom=np.array(read_ms) + np.array(comp_ms), label="Tulis file",
           color="tab:green")
    for i in idx:
        total = read_ms[i] + comp_ms[i] + write_ms[i]
        io_share = (read_ms[i] + write_ms[i]) / total * 100 if total > 0 else 0
        ax.text(i, total, f"I/O {io_share:.0f}%", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(idx)
    ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylabel("Waktu (ms)")
    ax.set_title(f"I/O vs komputasi - {flt} ({title_suffix})")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    save(fig, outdir, f"io_vs_compute_{flt}.png")


def plot_variants(rows, outdir, title_suffix):
    sel = [r for r in rows if r["metode"] == "serial"]
    filters = sorted({r["filter"] for r in sel})
    variants = [v for v in VARIANT_COLOR if any(r["varian"] == v for r in sel)]
    if not sel:
        return
    fig, ax = plt.subplots(figsize=(8, 5.2))
    width = 0.8 / max(len(variants), 1)
    for vi, v in enumerate(variants):
        vals = []
        for flt in filters:
            m = [r for r in sel if r["filter"] == flt and r["varian"] == v]
            vals.append(m[0]["waktu_median_s"] * MS if m else 0.0)
        ax.bar(np.arange(len(filters)) + vi * width, vals, width, label=v,
               color=VARIANT_COLOR[v])
    ax.set_xticks(np.arange(len(filters)) + width * (len(variants) - 1) / 2)
    ax.set_xticklabels(filters)
    ax.set_ylabel("Waktu serial, median (ms)")
    ax.set_title(f"Pengaruh varian/pola akses memori - serial ({title_suffix})")
    ax.legend(title="Varian")
    ax.grid(True, axis="y", alpha=0.3)
    save(fig, outdir, "varian_serial.png")


def plot_sizes(rows, flt, outdir, border, schedule):
    sizes = sorted({r["ukuran"] for r in rows})
    if len(sizes) < 2:
        return
    fig, ax = plt.subplots(figsize=(8, 5.2))
    all_threads = []
    for size in sizes:
        sel = select(rows, filter=flt, varian="direct", border=border, ukuran=size)
        for method in ("pthreads", "openmp"):
            rs = sorted((r for r in sel if r["metode"] == method
                         and (method != "openmp" or r["jadwal"] == schedule)),
                        key=lambda r: r["thread"])
            if not rs:
                continue
            ls, mk = METHOD_STYLE[method]
            x = [r["thread"] for r in rs]
            all_threads += x
            ax.plot(x, [r["waktu_median_s"] * MS for r in rs], linestyle=ls, marker=mk,
                    label=f"{METHOD_LABEL[method]} - {size}")
    if not all_threads:
        plt.close(fig)
        return
    set_thread_axis(ax, all_threads)
    ax.set_yscale("log")
    ax.set_ylabel("Waktu komputasi, median (ms) - skala log")
    ax.set_title(f"Scaling ukuran citra - {flt} (direct, {border})")
    ax.legend(fontsize=8)
    save(fig, outdir, f"ukuran_{flt}.png")


def main():
    ap = argparse.ArgumentParser(description="Plot hasil benchmark konvolusi paralel")
    ap.add_argument("--csv", default="results/benchmark.csv")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--size", default="2000x2000", help="ukuran citra utama untuk grafik")
    ap.add_argument("--border", default="replicate", choices=["zero", "replicate"])
    ap.add_argument("--schedule", default="static", help="jadwal OpenMP untuk grafik utama")
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        sys.exit(f"File {args.csv} tidak ditemukan. Jalankan dulu: make bench")
    os.makedirs(args.outdir, exist_ok=True)
    rows = load_rows(args.csv)
    if not rows:
        sys.exit("CSV kosong.")

    # Data utama: ukuran, border, dan jadwal OpenMP tertentu (baris non-OpenMP berjadwal '-').
    main_rows = [r for r in rows if r["ukuran"] == args.size and r["border"] == args.border
                 and r["jadwal"] in ("-", args.schedule)]
    if not main_rows:
        sys.exit(f"Tidak ada data untuk ukuran={args.size}, border={args.border}, "
                 f"jadwal={args.schedule}.")

    suffix = f"{args.size}, border {args.border}, jadwal {args.schedule}"
    amdahl_rows = []
    print("Membuat grafik:")
    for flt in sorted({r["filter"] for r in main_rows}):
        sub = select(main_rows, filter=flt)
        plot_time(sub, flt, suffix, args.outdir)
        plot_speedup_eff(sub, flt, suffix, args.outdir, amdahl_rows)
        plot_io_compute(sub, flt, suffix, args.outdir)
        plot_sizes(rows, flt, args.outdir, args.border, args.schedule)
    plot_variants(main_rows, args.outdir, suffix)

    if amdahl_rows:
        path = os.path.join(args.outdir, "amdahl.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(amdahl_rows[0].keys()))
            w.writeheader()
            w.writerows(amdahl_rows)
        print("  ->", path)
        print("\nEstimasi serial fraction (Amdahl):")
        for a in amdahl_rows:
            print(f"  {a['filter']:<9} {a['metode']:<9} {a['varian']:<10} f = {a['serial_fraction']:.4f}"
                  f"  -> S_maks = {a['speedup_maks_teoretis']:.2f}  (RMSE fit {a['rmse_fit']:.3f})")


if __name__ == "__main__":
    main()
