/**
 * @file common.h
 * @brief Definisi bersama (tipe data, konstanta, deklarasi fungsi) untuk proyek
 *        Image Convolution paralel: Serial vs POSIX Threads vs OpenMP.
 *
 * Seluruh modul (pgm_io.c, kernels.c, serial.c, pthreads_impl.c,
 * openmp_impl.c, main.c, dan tests/test_small.c) meng-include header ini.
 *
 * Keputusan desain penting:
 *  - Seluruh komputasi memakai ARITMATIKA INTEGER (kernel Gaussian bulat dengan
 *    jumlah 256, Sobel bulat). Hasilnya deterministik sehingga ketiga
 *    implementasi harus IDENTIK piksel demi piksel (tidak ada selisih
 *    pembulatan floating-point akibat urutan penjumlahan yang berbeda).
 *  - Citra disimpan row-major, 1 byte per piksel (grayscale 8-bit).
 */
#ifndef COMMON_H
#define COMMON_H

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L /* untuk clock_gettime, dsb. */
#endif

#include <stddef.h>
#include <stdint.h>
#include <time.h>

/* ------------------------------------------------------------------------ */
/* Konstanta                                                                 */
/* ------------------------------------------------------------------------ */

#define PIXEL_MAX 255            /* nilai keabuan maksimum (8-bit)            */
#define GAUSS_SIZE 5             /* ukuran kernel Gaussian 5x5                */
#define GAUSS_RADIUS 2           /* radius = GAUSS_SIZE / 2                   */
#define GAUSS_SHIFT 8            /* jumlah bobot 2D = 256 = 2^8               */
#define GAUSS_ROUND 128          /* setengah dari 256, untuk pembulatan       */
#define SOBEL_SIZE 3             /* ukuran kernel Sobel 3x3                   */
#define SOBEL_RADIUS 1           /* radius Sobel                              */

#define CACHE_LINE_SIZE 64       /* ukuran cache line (byte) x86-64 umum      */
#define TILE_H 64                /* tinggi tile untuk varian cache-blocking   */
#define TILE_W 256               /* lebar tile (256 byte = 4 cache line)      */
#define STRIP_W 256              /* lebar strip kolom untuk OpenMP collapse(2)*/
#define SEP_BLOCK_ROWS 64        /* tinggi blok baris pada Gaussian separable */

#define MAX_DIMENSION 65536      /* batas sanity lebar/tinggi citra           */
#define MAX_THREADS 256          /* batas atas jumlah thread dari CLI         */
#define MAX_THREAD_LIST 16       /* maksimum item pada daftar "-t 1,2,4,8"    */

#define NS_PER_SEC 1.0e9         /* konversi nanodetik -> detik               */

/* ------------------------------------------------------------------------ */
/* Tipe data                                                                 */
/* ------------------------------------------------------------------------ */

/** Citra grayscale 8-bit, row-major: piksel (y, x) = data[y * width + x]. */
typedef struct {
    int width;      /**< lebar (jumlah kolom)                     */
    int height;     /**< tinggi (jumlah baris)                    */
    int maxval;     /**< nilai maksimum pada header PGM           */
    uint8_t *data;  /**< buffer piksel (selaras 64 byte)          */
} Image;

/** Jenis filter. */
typedef enum { FILTER_GAUSSIAN = 0, FILTER_SOBEL = 1 } FilterType;

/** Varian algoritma/pola traversal. */
typedef enum {
    VARIANT_DIRECT = 0,    /**< konvolusi 2D penuh, traversal row-major       */
    VARIANT_SEPARABLE = 1, /**< Gaussian separable (1D horizontal + vertikal) */
    VARIANT_TILED = 2,     /**< konvolusi 2D dengan cache-blocking (tiling)   */
    VARIANT_COLUMN = 3     /**< traversal column-major (sengaja tidak ramah cache) */
} Variant;

/** Mode penanganan border. */
typedef enum { BORDER_ZERO = 0, BORDER_REPLICATE = 1 } BorderMode;

/** Jenis penjadwalan loop OpenMP. */
typedef enum { SCHED_STATIC = 0, SCHED_DYNAMIC = 1, SCHED_GUIDED = 2 } SchedKind;

/** Konfigurasi satu kali konvolusi. */
typedef struct {
    FilterType filter;
    Variant variant;
    BorderMode border;
    int threads;     /**< jumlah thread (diabaikan oleh serial)        */
    SchedKind sched; /**< jadwal OpenMP                                */
    int chunk;       /**< ukuran chunk OpenMP (0 = default runtime)    */
    int collapse;    /**< 1 = pakai collapse(2) pada varian direct     */
} ConvConfig;

/* ------------------------------------------------------------------------ */
/* Utilitas waktu                                                            */
/* ------------------------------------------------------------------------ */

/** Waktu monotonic dalam detik (CLOCK_MONOTONIC, tidak terpengaruh jam sistem). */
static inline double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / NS_PER_SEC;
}

/* ------------------------------------------------------------------------ */
/* pgm_io.c : alokasi citra & baca/tulis PGM                                 */
/* ------------------------------------------------------------------------ */

/** Alokasi citra (buffer selaras 64 byte, diisi nol). Return 0 jika sukses. */
int image_alloc(Image *img, int width, int height);

/** Bebaskan buffer citra dan reset field. Aman dipanggil berulang. */
void image_free(Image *img);

/** Baca PGM P5 (biner) atau P2 (ASCII). Return 0 jika sukses, -1 jika gagal. */
int pgm_read(const char *path, Image *img);

/** Tulis PGM. ascii=0 -> P5 (biner), ascii=1 -> P2 (ASCII). Return 0/-1. */
int pgm_write(const char *path, const Image *img, int ascii);

/** Jumlah piksel yang berbeda antara dua citra; -1 jika dimensi berbeda. */
long image_diff_count(const Image *a, const Image *b);

/* ------------------------------------------------------------------------ */
/* kernels.c : inti komputasi (dipakai bersama oleh ketiga implementasi)     */
/* ------------------------------------------------------------------------ */

const char *filter_name(FilterType f);
const char *variant_name(Variant v);
const char *border_name(BorderMode b);
const char *sched_name(SchedKind s);

/** Validasi konfigurasi (mis. separable hanya untuk Gaussian). Return 0 jika valid. */
int config_validate(const ConvConfig *cfg);

/** Jumlah elemen uint16_t buffer sementara per thread (0 bila tidak perlu). */
size_t kernel_tmp_elems(const ConvConfig *cfg, int width);

/** Hitung wilayah persegi [y0,y1) x [x0,x1) dengan konvolusi 2D langsung. */
void kernel_region(const Image *in, Image *out, FilterType filter,
                   BorderMode border, int y0, int y1, int x0, int x1);

/** Hitung pita baris [y0,y1) sesuai cfg->variant. tmp wajib untuk separable. */
void kernel_band(const Image *in, Image *out, const ConvConfig *cfg,
                 int y0, int y1, uint16_t *tmp);

/* ------------------------------------------------------------------------ */
/* Tiga implementasi. Waktu komputasi (detik) dikembalikan lewat *elapsed.   */
/* Return 0 jika sukses, -1 jika gagal (pesan error dicetak ke stderr).      */
/* ------------------------------------------------------------------------ */

int run_serial(const Image *in, Image *out, const ConvConfig *cfg, double *elapsed);
int run_pthreads(const Image *in, Image *out, const ConvConfig *cfg, double *elapsed);
int run_openmp(const Image *in, Image *out, const ConvConfig *cfg, double *elapsed);

/** Partisi baris: thread `tid` dari `nthreads` mendapat baris [*y0, *y1). */
void partition_rows(int height, int nthreads, int tid, int *y0, int *y1);

#endif /* COMMON_H */
