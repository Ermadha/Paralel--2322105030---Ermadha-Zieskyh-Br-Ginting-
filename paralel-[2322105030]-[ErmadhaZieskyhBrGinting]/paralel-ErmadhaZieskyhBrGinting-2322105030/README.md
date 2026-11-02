# Image Convolution Paralel: Serial vs POSIX Threads vs OpenMP

Proyek UTS Pemrograman Paralel (Opsi B: *Image Convolution / Blur Filter*).
Repositori: `paralel-ErmadhaZieskyhBrGinting-2322105030`

## 1. Deskripsi

Program C (C11) yang menerapkan filter konvolusi pada citra grayscale
(minimal 2000x2000 piksel) dalam format PGM, tanpa library eksternal:

- **Filter**: Gaussian blur 5x5 dan deteksi tepi Sobel (dipilih lewat CLI).
- **Implementasi**: (1) serial baseline, (2) POSIX Threads, (3) OpenMP.
- **Border**: `zero` (zero-padding) dan `replicate` (jepit ke tepi).
- **Varian tambahan**: Gaussian *separable* (1D + 1D), *tiling* (cache-blocking),
  dan traversal *column-major* (pembanding cache locality).
- **Benchmark otomatis**: warm-up, N pengulangan, median + simpangan baku,
  CSV, grafik matplotlib, estimasi serial fraction (Amdahl), penghitung LOC.
- **Verifikasi otomatis**: hasil Pthreads dan OpenMP dibandingkan dengan serial
  piksel demi piksel (harus 0 piksel berbeda) + unit test 10x10 yang dapat
  dihitung manual.

Seluruh komputasi memakai aritmatika integer (kernel Gaussian bulat berjumlah
256; Sobel bulat) sehingga ketiga implementasi menghasilkan keluaran
**identik bit-per-bit**.

### Tiga tantangan khusus (dijelaskan di komentar kode)

| Tantangan | Lokasi penjelasan |
|-----------|-------------------|
| a. Dependensi piksel di border & strategi penanganan | `src/kernels.c` (header file, `fetch_pixel`, jalur interior vs border) |
| b. Pembagian workload (row band, halo, sisa baris) | `src/pthreads_impl.c` (`partition_rows`), `src/openmp_impl.c`, `src/kernels.c` (halo separable) |
| c. I/O-bound vs compute-bound | `src/main.c` (waktu baca, komputasi, tulis diukur terpisah) |

## 2. Prasyarat

- Linux/WSL/macOS dengan `gcc` yang mendukung C11 dan OpenMP (`-fopenmp`), serta `make`
  (di macOS gunakan `gcc-NN` dari Homebrew, bukan `clang` bawaan)
- `python3` dengan `numpy` dan `matplotlib` (hanya untuk `make plot`):
  `pip install numpy matplotlib`
- `bash` (untuk skrip `benchmark.sh` dan `loc.sh`)

Ubuntu/Debian: `sudo apt install build-essential python3-numpy python3-matplotlib`

## 3. Cara compile

```bash
make            # membangun build/convolve dan build/gen_image
make test       # membangun & menjalankan unit test
```

Flag kompilasi: `-std=c11 -Wall -Wextra -O2 -fopenmp -pthread` (bersih tanpa warning).

## 4. Cara run

```bash
# 1) Buat citra uji sintetis 2000x2000 dan 4000x4000 (reproducible, srand(42))
make images
# atau manual:
build/gen_image 2000 2000 data/synthetic_2000.pgm

# 2) Gaussian blur, semua metode, thread 1,2,4,8
build/convolve -i data/synthetic_2000.pgm -f gaussian -b replicate -t 1,2,4,8

# 3) Sobel, zero-padding, hanya OpenMP 4 thread, jadwal dynamic chunk 16
build/convolve -i data/synthetic_2000.pgm -f sobel -b zero -m openmp -t 4 -s dynamic -c 16

# 4) Gaussian separable, simpan hasil ke file dan tulis CSV
build/convolve -i data/synthetic_2000.pgm -f gaussian -v separable -o hasil.pgm --csv results/coba.csv

# 5) Keluaran P2 (ASCII)
build/convolve -i data/synthetic_2000.pgm --ascii -o hasil_ascii.pgm
```

Kode keluar: `0` sukses, `1` error argumen/IO/memori, `2` verifikasi gagal.

## 5. Argumen CLI

| Opsi | Nilai | Default | Keterangan |
|------|-------|---------|------------|
| `-i, --input` | path PGM | wajib | citra masuk (P5 atau P2) |
| `-o, --output` | path | `out.pgm` | citra hasil (P5; P2 jika `--ascii`) |
| `-f, --filter` | `gaussian`, `sobel` | `gaussian` | jenis filter |
| `-v, --variant` | `direct`, `separable`, `tiled`, `column` | `direct` | `separable` hanya Gaussian |
| `-b, --border` | `zero`, `replicate` | `replicate` | penanganan border |
| `-m, --method` | `serial`, `pthreads`, `openmp`, `all` | `all` | metode yang dilaporkan (serial selalu dijalankan sebagai baseline) |
| `-t, --threads` | daftar, mis. `4` atau `1,2,4,8` | `1,2,4,8` | jumlah thread (maks 256, maks 16 item) |
| `-n, --repeat` | 1..1000 | `5` | pengulangan terukur |
| `-w, --warmup` | 0..1000 | `1` | warm-up run (dibuang) |
| `-s, --schedule` | `static`, `dynamic`, `guided` | `static` | jadwal OpenMP |
| `-c, --chunk` | >= 0 | `0` | chunk OpenMP (0 = default runtime) |
| `--collapse` | - | mati | `collapse(2)` OpenMP pada varian `direct` |
| `--csv` | path | - | tambahkan baris hasil ke CSV |
| `--ascii` | - | mati | tulis keluaran P2 |
| `--no-color` | - | - | matikan warna terminal |
| `--no-verify` | - | - | lewati verifikasi |
| `-h, --help` | - | - | bantuan |

Skema CSV: `metode,filter,varian,border,ukuran,thread,jadwal,waktu_median_s,waktu_std_s,waktu_min_s,speedup,efficiency,baca_s,tulis_s,piksel_beda,beda_vs_direct`.
`speedup = median_serial / median_metode`, `efficiency = speedup / thread`.

## 6. Reproduksi benchmark

```bash
make bench                         # menjalankan semua skenario -> results/benchmark.csv
REPEAT=10 THREADS=1,2,4,8 make bench   # pengulangan lebih banyak
make plot                          # grafik PNG + results/amdahl.csv di folder results/
make loc                           # tabel LOC per implementasi -> results/loc.csv
```

Skenario `scripts/benchmark.sh`: (A) semua varian pada 2000x2000 border replicate,
(B) border zero-padding, (C) jadwal OpenMP dynamic/guided (+ chunk 16) dan collapse,
(D) scaling 4000x4000 (lewati dengan `SKIP_4K=1`).

Grafik yang dibuat `scripts/plot.py` (folder `results/`): `waktu_*.png`,
`speedup_*.png` (garis ideal + kurva Amdahl), `efficiency_*.png`,
`io_vs_compute_*.png` (stacked bar), `varian_serial.png`, `ukuran_*.png`.

Tips pengukuran yang adil: jalankan saat mesin idle, hubungkan charger/mode
performa, tutup aplikasi berat, laporkan spesifikasi mesin (CPU, jumlah core
fisik/logis, RAM, versi gcc) di laporan. Jumlah thread di atas jumlah core
logis tidak akan menambah speedup. I/O diukur dengan file di page cache OS.

## 7. Struktur repo

```
paralel-ErmadhaZieskyhBrGinting-2322105030/
├── src/
│   ├── common.h          tipe data, konstanta, deklarasi
│   ├── pgm_io.c          alokasi citra, baca/tulis PGM (P5 & P2)
│   ├── kernels.c         Gaussian 5x5 (2D & separable), Sobel, border, tiling
│   ├── serial.c          implementasi serial (baseline)
│   ├── pthreads_impl.c   implementasi POSIX Threads (row band + sisa)
│   ├── openmp_impl.c     implementasi OpenMP (schedule/collapse via CLI)
│   └── main.c            CLI, benchmark, statistik, verifikasi, tabel, CSV
├── tests/
│   └── test_small.c      unit test 10x10 + ekuivalensi + partisi + PGM I/O
├── scripts/
│   ├── gen_image.c       generator citra sintetis (gradien+lingkaran+noise)
│   ├── benchmark.sh      menjalankan seluruh skenario benchmark
│   ├── plot.py           grafik matplotlib + fitting Amdahl
│   └── loc.sh            penghitung LOC per implementasi
├── results/              CSV dan grafik hasil benchmark
├── data/                 citra uji (dibuat oleh make images; tidak di-commit)
├── Makefile              target: all, test, bench, plot, clean, loc (+ images, distclean)
└── README.md
```

## 8. Catatan desain singkat

- **Satu kernel untuk semua**: serial, Pthreads, dan OpenMP memanggil `kernel_band()`/
  `kernel_region()` yang sama; perbedaan waktu murni dari cara paralelisasi.
- **Tanpa mutex**: region tulis tiap thread (baris/blok/tile) saling lepas dan input
  read-only; sinkronisasi hanya `pthread_join` / barrier implisit OpenMP.
- **False sharing**: argumen thread diselaraskan 64 byte; buffer sementara per thread
  di-padding ke kelipatan cache line.
- **Separable + halo**: tiap thread/blok menghitung ulang 2R baris halo ke buffer privat
  sehingga tidak perlu barrier antara pass horizontal dan vertikal.
- **Timing**: `clock_gettime(CLOCK_MONOTONIC)` untuk serial & Pthreads (mencakup
  create+join), `omp_get_wtime()` untuk OpenMP; alokasi buffer dilakukan sebelum timer.
