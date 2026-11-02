#!/usr/bin/env bash
# ============================================================================
# loc.sh - Menghitung LOC per implementasi (untuk analisis kompleksitas kode).
#
# Definisi:
#   LOC kode      = baris bukan-kosong dan bukan-komentar
#   Komentar      = baris yang diawali // atau /* ... */ (termasuk isi blok)
#   Kosong        = baris hanya berisi spasi
# Catatan: komentar di akhir baris kode tetap dihitung sebagai kode.
# Selain LOC, dihitung jumlah "titik paralelisasi":
#   - pthread_* : baris berisi pemanggilan API Pthreads (create/join/...) -> manual
#                 (baris komentar tidak dihitung)
#   - pragma omp: baris direktif #pragma omp (komentar tidak dihitung) -> deklaratif
# Keluaran: tabel di terminal + results/loc.csv
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p results
CSV="results/loc.csv"
echo "berkas,peran,loc_kode,komentar,kosong,total,pthread_api,pragma_omp" > "$CSV"

count_file() {
    awk '
    BEGIN { in_block = 0; code = 0; cmt = 0; blank = 0 }
    {
        line = $0
        gsub(/^[ \t]+|[ \t]+$/, "", line)
        if (line == "") { blank++; next }
        if (in_block) { cmt++; if (index(line, "*/") > 0) in_block = 0; next }
        if (substr(line, 1, 2) == "//") { cmt++; next }
        if (substr(line, 1, 2) == "/*") { cmt++; if (index(line, "*/") == 0) in_block = 1; next }
        code++
    }
    END { printf "%d %d %d\n", code, cmt, blank }' "$1"
}

printf "%-24s %-26s %8s %9s %7s %7s %9s %11s\n" \
    "Berkas" "Peran" "LOC kode" "Komentar" "Kosong" "Total" "pthread_*" "pragma omp"
printf '%.0s-' {1..110}; echo

report() {
    local file="$1" role="$2"
    read -r code cmt blank < <(count_file "$file")
    local total=$((code + cmt + blank))
    local pth omp
    pth=$(grep -Ec '^[^*/]*pthread_[a-z_]+\(' "$file" || true)
    omp=$(grep -Ec '^[[:space:]]*#pragma omp' "$file" || true)
    printf "%-24s %-26s %8d %9d %7d %7d %9d %11d\n" \
        "$file" "$role" "$code" "$cmt" "$blank" "$total" "$pth" "$omp"
    echo "$file,$role,$code,$cmt,$blank,$total,$pth,$omp" >> "$CSV"
}

report src/serial.c        "Serial (baseline)"
report src/pthreads_impl.c "POSIX Threads"
report src/openmp_impl.c   "OpenMP"
report src/kernels.c       "Inti komputasi (bersama)"
report src/pgm_io.c        "I/O PGM (bersama)"
report src/main.c          "Driver/CLI/benchmark"
report tests/test_small.c  "Unit test"
report scripts/gen_image.c "Generator citra"

echo
echo "CSV: $CSV"
