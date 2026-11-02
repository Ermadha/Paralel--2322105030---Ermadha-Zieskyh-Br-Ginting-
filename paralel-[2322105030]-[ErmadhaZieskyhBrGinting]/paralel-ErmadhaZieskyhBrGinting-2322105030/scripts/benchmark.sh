#!/usr/bin/env bash
# ============================================================================
# benchmark.sh - Menjalankan seluruh skenario benchmark dan menulis
#                results/benchmark.csv (dipakai oleh scripts/plot.py).
#
# Variabel lingkungan (opsional):
#   REPEAT   jumlah pengulangan terukur per konfigurasi   (default 5)
#   THREADS  daftar thread, mis. "1,2,4,8"                 (default 1,2,4,8)
#   SKIP_4K  isi "1" untuk melewati uji scaling 4000x4000  (default kosong)
#
# Contoh:  REPEAT=10 THREADS=1,2,4,8 bash scripts/benchmark.sh
#
# Catatan reproduksibilitas: jalankan saat mesin idle, tutup aplikasi berat,
# dan (di laptop) pastikan mode daya "performance" / charger terpasang.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

REPEAT="${REPEAT:-5}"
THREADS="${THREADS:-1,2,4,8}"
CSV="results/benchmark.csv"
BIN="build/convolve"
OUT="data/out.pgm"
IMG_2K="data/synthetic_2000.pgm"
IMG_4K="data/synthetic_4000.pgm"

make -s all images
mkdir -p results data
rm -f "$CSV"

# run <argumen tambahan...> : satu eksekusi convolve yang menambah baris ke CSV
run() {
    echo
    echo ">>> $*"
    "$BIN" --no-color --csv "$CSV" -n "$REPEAT" -t "$THREADS" -o "$OUT" "$@"
}

echo "== A. Semua varian pada 2000x2000, border replicate (jadwal static) =="
for f in gaussian sobel; do
    for v in direct tiled column; do
        run -i "$IMG_2K" -f "$f" -v "$v" -b replicate
    done
done
run -i "$IMG_2K" -f gaussian -v separable -b replicate

echo "== B. Mode border zero-padding =="
for f in gaussian sobel; do
    run -i "$IMG_2K" -f "$f" -v direct -b zero
done
run -i "$IMG_2K" -f gaussian -v separable -b zero

echo "== C. Jadwal OpenMP (dynamic / guided, dengan & tanpa chunk) =="
for s in dynamic guided; do
    run -i "$IMG_2K" -f gaussian -v direct -b replicate -m openmp -s "$s"
    run -i "$IMG_2K" -f gaussian -v direct -b replicate -m openmp -s "$s" -c 16
done
run -i "$IMG_2K" -f gaussian -v direct -b replicate -m openmp --collapse

if [[ "${SKIP_4K:-}" != "1" ]]; then
    echo "== D. Uji scaling 4000x4000 =="
    for f in gaussian sobel; do
        run -i "$IMG_4K" -f "$f" -v direct -b replicate
    done
    run -i "$IMG_4K" -f gaussian -v separable -b replicate
fi

echo
echo "Selesai. Hasil CSV: $CSV ($(($(wc -l < "$CSV") - 1)) baris data)"
