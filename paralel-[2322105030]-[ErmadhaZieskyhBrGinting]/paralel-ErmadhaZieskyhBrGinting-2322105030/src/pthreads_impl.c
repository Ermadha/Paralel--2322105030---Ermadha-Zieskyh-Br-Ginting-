/**
 * @file pthreads_impl.c
 * @brief Implementasi paralel dengan POSIX Threads (pendekatan eksplisit/manual).
 *
 * ==========================================================================
 * TANTANGAN (b): PEMBAGIAN WORKLOAD YANG TIDAK TRIVIAL
 * ==========================================================================
 * Citra dibagi menjadi PITA BARIS (row band) berurutan, satu pita per thread.
 *
 *   tinggi H, jumlah thread T:  base = H / T,  rem = H % T
 *   thread t mendapat (base + 1) baris bila t < rem, selain itu base baris.
 *
 * Contoh H = 10, T = 3 -> base = 3, rem = 1:
 *   thread 0: baris [0,4)  (4 baris, kebagian sisa)
 *   thread 1: baris [4,7)  (3 baris)
 *   thread 2: baris [7,10) (3 baris)
 * Sisa (rem) disebar satu baris ke tiap thread pertama sehingga selisih beban
 * antar-thread paling besar SATU baris (bukan menumpuk di thread terakhir).
 * Bila T > H, jumlah thread dikurangi menjadi H (tidak ada pita kosong).
 *
 * HALO / OVERLAP REGION:
 *   Untuk menghitung baris di tepi pita, kernel (radius R) memerlukan R baris
 *   input milik pita tetangga. Citra INPUT bersifat read-only dan dibagi
 *   bersama oleh semua thread, jadi baris halo itu cukup DIBACA langsung dari
 *   input (tanpa menyalin). Daerah baca dua pita saling tumpang-tindih sebesar
 *   2R baris, tetapi daerah TULIS (output) tetap saling lepas (disjoint).
 *   Untuk varian separable, baris halo dihitung ulang ke buffer privat tiap
 *   thread (lihat kernels.c) -> tanpa barrier.
 *
 * ==========================================================================
 * MEKANISME SINKRONISASI
 * ==========================================================================
 *  - TIDAK ADA mutex untuk output: setiap thread menulis hanya ke baris
 *    [y0, y1) miliknya; region tulis antar-thread disjoint, input read-only,
 *    sehingga tidak ada data race (kondisi Bernstein terpenuhi).
 *  - Satu-satunya sinkronisasi adalah pthread_join() oleh thread utama sebagai
 *    barrier akhir; join juga menjamin semua tulisan thread terlihat (happens-
 *    before) oleh thread utama sebelum hasil dibaca/diverifikasi.
 *
 * ==========================================================================
 * MENGHINDARI FALSE SHARING
 * ==========================================================================
 *  - Struct argumen tiap thread diselaraskan 64 byte (_Alignas) sehingga
 *    ThreadArg milik thread berbeda tidak berbagi cache line (thread menulis
 *    `thread_seconds` di akhir).
 *  - Buffer sementara privat (separable) di-padding ke kelipatan cache line.
 *  - Batas pita jatuh pada batas baris; paling banyak satu cache line di
 *    perbatasan dua pita yang dapat dibagi (bukan di loop panas) dan buffer
 *    output sendiri selaras 64 byte -> dampak diabaikan.
 */
#include "common.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/** Argumen satu thread pekerja. Anggota pertama memaksa alignment 64 byte. */
typedef struct {
    _Alignas(CACHE_LINE_SIZE) const Image *in; /**< citra input (read-only, shared)  */
    Image *out;                                 /**< citra output (tulis disjoint)    */
    const ConvConfig *cfg;                      /**< konfigurasi (read-only, shared)  */
    int y0;                                     /**< baris awal pita (inklusif)       */
    int y1;                                     /**< baris akhir pita (eksklusif)     */
    uint16_t *tmp;                              /**< buffer privat (separable) / NULL */
    double thread_seconds;                      /**< waktu kerja thread (diisi thread)*/
} ThreadArg;

/* sizeof harus kelipatan 64 agar elemen array berurutan tidak berbagi cache line. */
_Static_assert(sizeof(ThreadArg) % CACHE_LINE_SIZE == 0,
               "ThreadArg harus kelipatan ukuran cache line");

void partition_rows(int height, int nthreads, int tid, int *y0, int *y1) {
    const int base = height / nthreads;
    const int rem = height % nthreads;
    /* thread 0..rem-1 mendapat satu baris ekstra -> offset awal ikut bergeser */
    const int extra_before = tid < rem ? tid : rem;
    *y0 = tid * base + extra_before;
    *y1 = *y0 + base + (tid < rem ? 1 : 0);
}

/** Fungsi entri thread: hitung pita baris milik thread ini. */
static void *worker(void *param) {
    ThreadArg *arg = (ThreadArg *)param;
    double t_start = now_seconds();
    kernel_band(arg->in, arg->out, arg->cfg, arg->y0, arg->y1, arg->tmp);
    arg->thread_seconds = now_seconds() - t_start;
    return NULL;
}

int run_pthreads(const Image *in, Image *out, const ConvConfig *cfg, double *elapsed) {
    int nthreads = cfg->threads;
    int status = -1;
    ThreadArg *args = NULL;
    pthread_t *tids = NULL;
    uint16_t *tmp_pool = NULL;
    int created = 0;

    if (nthreads < 1) {
        fprintf(stderr, "run_pthreads: jumlah thread harus >= 1\n");
        return -1;
    }
    if (nthreads > in->height) {
        nthreads = in->height; /* tidak ada pita kosong */
    }

    args = aligned_alloc(CACHE_LINE_SIZE, (size_t)nthreads * sizeof *args);
    tids = malloc((size_t)nthreads * sizeof *tids);
    if (args == NULL || tids == NULL) {
        fprintf(stderr, "run_pthreads: gagal alokasi argumen thread\n");
        goto cleanup;
    }
    memset(args, 0, (size_t)nthreads * sizeof *args);

    /* Buffer sementara privat per thread (hanya varian separable). */
    size_t tmp_elems = kernel_tmp_elems(cfg, in->width);
    size_t tmp_stride = 0;
    if (tmp_elems > 0) {
        const size_t line_elems = CACHE_LINE_SIZE / sizeof(uint16_t);
        tmp_stride = (tmp_elems + line_elems - 1) / line_elems * line_elems; /* padding */
        size_t bytes = (size_t)nthreads * tmp_stride * sizeof *tmp_pool;
        bytes = (bytes + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE * CACHE_LINE_SIZE;
        tmp_pool = aligned_alloc(CACHE_LINE_SIZE, bytes);
        if (tmp_pool == NULL) {
            fprintf(stderr, "run_pthreads: gagal alokasi buffer sementara\n");
            goto cleanup;
        }
        memset(tmp_pool, 0, bytes); /* first-touch di luar timer */
    }

    /* Isi argumen: pembagian pita baris + sisa. */
    for (int t = 0; t < nthreads; t++) {
        args[t].in = in;
        args[t].out = out;
        args[t].cfg = cfg;
        partition_rows(in->height, nthreads, t, &args[t].y0, &args[t].y1);
        args[t].tmp = tmp_pool != NULL ? tmp_pool + (size_t)t * tmp_stride : NULL;
    }

    /* Waktu mencakup pembuatan + join thread (overhead nyata pendekatan manual). */
    double t_start = now_seconds();
    int create_failed = 0;
    for (int t = 0; t < nthreads; t++) {
        int rc = pthread_create(&tids[t], NULL, worker, &args[t]);
        if (rc != 0) {
            fprintf(stderr, "run_pthreads: pthread_create gagal (kode %d)\n", rc);
            create_failed = 1;
            break;
        }
        created++;
    }
    for (int t = 0; t < created; t++) { /* barrier akhir + jaminan visibilitas memori */
        pthread_join(tids[t], NULL);
    }
    double t_end = now_seconds();

    if (!create_failed) {
        *elapsed = t_end - t_start;
        status = 0;
    }

cleanup:
    free(tmp_pool);
    free(tids);
    free(args);
    return status;
}
