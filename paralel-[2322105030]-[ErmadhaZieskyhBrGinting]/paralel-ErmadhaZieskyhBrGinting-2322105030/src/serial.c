/**
 * @file serial.c
 * @brief Implementasi SERIAL (baseline): satu thread memproses seluruh citra.
 *
 * Seluruh pekerjaan dipanggil lewat kernel_band() untuk pita [0, tinggi),
 * sehingga baseline memakai kode komputasi yang sama dengan versi paralel.
 * Buffer sementara (varian separable) dialokasikan dan di-"touch" SEBELUM
 * timer dimulai agar biaya malloc/page-fault tidak ikut terukur.
 *
 * Waktu diukur dengan clock_gettime(CLOCK_MONOTONIC) (via now_seconds()).
 */
#include "common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int run_serial(const Image *in, Image *out, const ConvConfig *cfg, double *elapsed) {
    uint16_t *tmp = NULL;
    size_t tmp_elems = kernel_tmp_elems(cfg, in->width);

    if (tmp_elems > 0) {
        tmp = malloc(tmp_elems * sizeof *tmp);
        if (tmp == NULL) {
            fprintf(stderr, "run_serial: gagal alokasi buffer sementara\n");
            return -1;
        }
        memset(tmp, 0, tmp_elems * sizeof *tmp); /* first-touch di luar timer */
    }

    double t_start = now_seconds();
    kernel_band(in, out, cfg, 0, in->height, tmp);
    double t_end = now_seconds();

    free(tmp);
    *elapsed = t_end - t_start;
    return 0;
}
