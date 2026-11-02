/**
 * @file openmp_impl.c
 * @brief Implementasi paralel dengan OpenMP (pendekatan compiler directive).
 *
 * Pembagian kerja diserahkan ke runtime OpenMP lewat
 *   #pragma omp parallel for schedule(runtime)
 * dengan jadwal (static / dynamic / guided) dan ukuran chunk dipilih lewat CLI
 * (omp_set_schedule). Pembagian sisa saat tinggi tidak habis dibagi jumlah
 * thread ditangani otomatis oleh runtime (static: selisih maksimum 1 iterasi;
 * dynamic/guided: thread yang selesai lebih cepat mengambil chunk berikutnya).
 *
 * ==========================================================================
 * SCOPE VARIABEL (shared / private)
 * ==========================================================================
 *  - SHARED (default untuk variabel yang dideklarasikan di luar region):
 *      in, out, filter, border, w, h, nstrips, ... Semuanya read-only di dalam
 *      region, kecuali isi out->data yang ditulis pada indeks DISJOINT per
 *      iterasi -> aman tanpa critical/atomic.
 *  - PRIVATE: variabel indeks loop (y, x, s, ty, tx, b) otomatis private; var
 *      yang dideklarasikan DI DALAM loop body (x0, x1, y_end, ...) juga private
 *      (hidup di stack tiap thread). Pada collapse(2) kedua indeks loop private.
 *  - PRIVATE per-thread (manual): buffer sementara separable `my_tmp`,
 *      dipilih berdasarkan omp_get_thread_num() dari pool yang dialokasikan
 *      sebelum timer.
 *
 * SINKRONISASI: tidak ada critical/atomic/reduction karena tidak ada variabel
 * yang ditulis bersama. Hanya ada barrier implisit di akhir `omp for`.
 *
 * Waktu diukur dengan omp_get_wtime() (monotonic, resolusi tinggi).
 */
#include "common.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/** Petakan SchedKind proyek -> omp_sched_t. */
static omp_sched_t to_omp_sched(SchedKind kind) {
    switch (kind) {
    case SCHED_DYNAMIC: return omp_sched_dynamic;
    case SCHED_GUIDED: return omp_sched_guided;
    case SCHED_STATIC: return omp_sched_static;
    }
    return omp_sched_static;
}

int run_openmp(const Image *in, Image *out, const ConvConfig *cfg, double *elapsed) {
    if (cfg->threads < 1) {
        fprintf(stderr, "run_openmp: jumlah thread harus >= 1\n");
        return -1;
    }

    const int w = in->width;
    const int h = in->height;
    const FilterType filter = cfg->filter;
    const BorderMode border = cfg->border;

    omp_set_dynamic(0);                     /* paksa tepat `threads` thread       */
    omp_set_num_threads(cfg->threads);
    omp_set_schedule(to_omp_sched(cfg->sched), cfg->chunk); /* dipakai schedule(runtime) */

    /* Buffer privat per thread untuk varian separable (dialokasi sebelum timer). */
    uint16_t *tmp_pool = NULL;
    size_t tmp_elems = kernel_tmp_elems(cfg, w);
    size_t tmp_stride = 0;
    if (tmp_elems > 0) {
        const size_t line_elems = CACHE_LINE_SIZE / sizeof(uint16_t);
        tmp_stride = (tmp_elems + line_elems - 1) / line_elems * line_elems; /* hindari false sharing */
        size_t bytes = (size_t)cfg->threads * tmp_stride * sizeof *tmp_pool;
        bytes = (bytes + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE * CACHE_LINE_SIZE;
        tmp_pool = aligned_alloc(CACHE_LINE_SIZE, bytes);
        if (tmp_pool == NULL) {
            fprintf(stderr, "run_openmp: gagal alokasi buffer sementara\n");
            return -1;
        }
        memset(tmp_pool, 0, bytes);
    }

    const int nstrips = (w + STRIP_W - 1) / STRIP_W;       /* jumlah strip kolom   */
    const int ntiles_y = (h + TILE_H - 1) / TILE_H;        /* jumlah tile vertikal */
    const int ntiles_x = (w + TILE_W - 1) / TILE_W;        /* jumlah tile horizontal */
    const int nblocks = (h + SEP_BLOCK_ROWS - 1) / SEP_BLOCK_ROWS;

    double t_start = omp_get_wtime();

    switch (cfg->variant) {
    case VARIANT_DIRECT:
        if (cfg->collapse) {
            /*
             * collapse(2): ruang iterasi (baris x strip kolom) diratakan menjadi
             * h*nstrips iterasi independen -> granularitas lebih halus, berguna
             * saat jumlah baris kecil dibanding jumlah thread.
             */
#pragma omp parallel for collapse(2) schedule(runtime)
            for (int y = 0; y < h; y++) {
                for (int s = 0; s < nstrips; s++) {
                    int x0 = s * STRIP_W;                       /* private */
                    int x1 = x0 + STRIP_W < w ? x0 + STRIP_W : w; /* private */
                    kernel_region(in, out, filter, border, y, y + 1, x0, x1);
                }
            }
        } else {
            /* Satu iterasi = satu baris penuh (row-major, ramah cache). */
#pragma omp parallel for schedule(runtime)
            for (int y = 0; y < h; y++) {
                kernel_region(in, out, filter, border, y, y + 1, 0, w);
            }
        }
        break;

    case VARIANT_TILED:
        /* Satu iterasi = satu tile; collapse(2) pada indeks tile (ty, tx). */
#pragma omp parallel for collapse(2) schedule(runtime)
        for (int ty = 0; ty < ntiles_y; ty++) {
            for (int tx = 0; tx < ntiles_x; tx++) {
                int y0 = ty * TILE_H;
                int y1 = y0 + TILE_H < h ? y0 + TILE_H : h;
                int x0 = tx * TILE_W;
                int x1 = x0 + TILE_W < w ? x0 + TILE_W : w;
                kernel_region(in, out, filter, border, y0, y1, x0, x1);
            }
        }
        break;

    case VARIANT_COLUMN:
        /* Satu iterasi = satu kolom penuh (pola akses sengaja tidak ramah cache). */
#pragma omp parallel for schedule(runtime)
        for (int x = 0; x < w; x++) {
            kernel_region(in, out, filter, border, 0, h, x, x + 1);
        }
        break;

    case VARIANT_SEPARABLE:
        /*
         * Satu iterasi = satu blok SEP_BLOCK_ROWS baris. Tiap thread memakai
         * buffer privat `my_tmp` dan menghitung ulang baris halo-nya sendiri,
         * sehingga tidak perlu barrier di antara pass horizontal & vertikal.
         */
#pragma omp parallel
        {
            uint16_t *my_tmp = tmp_pool + (size_t)omp_get_thread_num() * tmp_stride; /* private */
#pragma omp for schedule(runtime)
            for (int b = 0; b < nblocks; b++) {
                int y0 = b * SEP_BLOCK_ROWS;
                int y1 = y0 + SEP_BLOCK_ROWS < h ? y0 + SEP_BLOCK_ROWS : h;
                kernel_band(in, out, cfg, y0, y1, my_tmp);
            }
        }
        break;
    }

    double t_end = omp_get_wtime();

    free(tmp_pool);
    *elapsed = t_end - t_start;
    return 0;
}
