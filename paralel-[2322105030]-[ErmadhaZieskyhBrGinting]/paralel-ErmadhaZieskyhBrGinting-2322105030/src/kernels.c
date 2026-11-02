/**
 * @file kernels.c
 * @brief Inti komputasi konvolusi: Gaussian 5x5 (2D & separable) dan Sobel,
 *        dengan dua mode border. Dipakai bersama oleh serial, Pthreads, OpenMP.
 *
 * Karena SEMUA implementasi memanggil fungsi yang sama di file ini, perbedaan
 * performa antar-implementasi murni berasal dari cara PARALELISASI, bukan dari
 * perbedaan rumus. Itu juga menjamin hasil identik piksel demi piksel.
 *
 * ==========================================================================
 * TANTANGAN (a): DEPENDENSI ANTAR-PIKSEL DI AREA BORDER
 * ==========================================================================
 * Untuk menghitung piksel (y, x) dengan kernel radius R, dibutuhkan piksel
 * input pada jendela [y-R, y+R] x [x-R, x+R]. Pada piksel dalam jarak R dari
 * tepi citra, sebagian jendela jatuh DI LUAR citra. Strategi:
 *   - BORDER_ZERO      : piksel di luar citra dianggap bernilai 0.
 *   - BORDER_REPLICATE : koordinat di luar dijepit (clamp) ke tepi terdekat.
 * Agar cepat, tiap piksel diklasifikasi:
 *   - INTERIOR (jendela seluruhnya di dalam citra) -> jalur cepat tanpa cek.
 *   - BORDER   (jendela menyentuh luar citra)      -> jalur lambat fetch_pixel().
 * Hanya sebagian kecil piksel (sekitar 4R/N) yang memakai jalur lambat.
 * Input bersifat READ-ONLY selama komputasi, sehingga dependensi border ini
 * tidak menimbulkan race condition; hanya perlu penanganan nilai di luar citra.
 *
 * Aritmatika integer murni:
 *   Gaussian : bobot bulat (jumlah 256), hasil = (sum + 128) >> 8
 *   Sobel    : magnitudo = round(sqrt(gx^2 + gy^2)), di-clamp ke [0, 255]
 */
#include "common.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------------ */
/* Koefisien kernel                                                          */
/* ------------------------------------------------------------------------ */

/** Kernel Gaussian 1D binomial [1 4 6 4 1] (jumlah = 16). */
static const int GAUSS_1D[GAUSS_SIZE] = {1, 4, 6, 4, 1};

/** Kernel Gaussian 2D = GAUSS_1D (x) GAUSS_1D, jumlah = 256 = 2^GAUSS_SHIFT. */
static const int GAUSS_2D[GAUSS_SIZE][GAUSS_SIZE] = {
    {1, 4, 6, 4, 1},
    {4, 16, 24, 16, 4},
    {6, 24, 36, 24, 6},
    {4, 16, 24, 16, 4},
    {1, 4, 6, 4, 1},
};

/** Kernel Sobel horizontal (turunan arah x). */
static const int SOBEL_GX[SOBEL_SIZE][SOBEL_SIZE] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1},
};

/** Kernel Sobel vertikal (turunan arah y). */
static const int SOBEL_GY[SOBEL_SIZE][SOBEL_SIZE] = {
    {-1, -2, -1},
    {0, 0, 0},
    {1, 2, 1},
};

/* ------------------------------------------------------------------------ */
/* Nama untuk keluaran / parsing                                             */
/* ------------------------------------------------------------------------ */

const char *filter_name(FilterType f) {
    return f == FILTER_GAUSSIAN ? "gaussian" : "sobel";
}

const char *variant_name(Variant v) {
    switch (v) {
    case VARIANT_DIRECT: return "direct";
    case VARIANT_SEPARABLE: return "separable";
    case VARIANT_TILED: return "tiled";
    case VARIANT_COLUMN: return "column";
    }
    return "?";
}

const char *border_name(BorderMode b) {
    return b == BORDER_ZERO ? "zero" : "replicate";
}

const char *sched_name(SchedKind s) {
    switch (s) {
    case SCHED_STATIC: return "static";
    case SCHED_DYNAMIC: return "dynamic";
    case SCHED_GUIDED: return "guided";
    }
    return "?";
}

int config_validate(const ConvConfig *cfg) {
    if (cfg->variant == VARIANT_SEPARABLE && cfg->filter != FILTER_GAUSSIAN) {
        fprintf(stderr, "Varian 'separable' hanya berlaku untuk filter gaussian "
                        "(Sobel versi 2D penuh).\n");
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------------ */
/* Akses piksel dengan penanganan border                                     */
/* ------------------------------------------------------------------------ */

static inline int clamp_int(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/**
 * Ambil piksel (y, x); jika di luar citra terapkan mode border.
 * Inilah satu-satunya tempat aturan border diimplementasikan.
 */
static inline int fetch_pixel(const Image *img, int y, int x, BorderMode border) {
    if (y < 0 || y >= img->height || x < 0 || x >= img->width) {
        if (border == BORDER_ZERO) {
            return 0;
        }
        y = clamp_int(y, 0, img->height - 1); /* replicate: jepit ke tepi */
        x = clamp_int(x, 0, img->width - 1);
    }
    return img->data[(size_t)y * (size_t)img->width + (size_t)x];
}

/* ------------------------------------------------------------------------ */
/* Satu piksel keluaran: Gaussian 5x5 dan Sobel                              */
/* ------------------------------------------------------------------------ */

static inline uint8_t gauss_pixel(const Image *in, BorderMode border, int y, int x) {
    const int w = in->width;
    const int h = in->height;
    int sum = 0;

    if (y >= GAUSS_RADIUS && y < h - GAUSS_RADIUS &&
        x >= GAUSS_RADIUS && x < w - GAUSS_RADIUS) {
        /* Jalur cepat: seluruh jendela 5x5 berada di dalam citra. */
        for (int j = 0; j < GAUSS_SIZE; j++) {
            const uint8_t *row = in->data + (size_t)(y + j - GAUSS_RADIUS) * (size_t)w +
                                 (size_t)(x - GAUSS_RADIUS);
            for (int i = 0; i < GAUSS_SIZE; i++) {
                sum += GAUSS_2D[j][i] * row[i];
            }
        }
    } else {
        /* Jalur border: setiap akses melewati fetch_pixel(). */
        for (int j = 0; j < GAUSS_SIZE; j++) {
            for (int i = 0; i < GAUSS_SIZE; i++) {
                sum += GAUSS_2D[j][i] *
                       fetch_pixel(in, y + j - GAUSS_RADIUS, x + i - GAUSS_RADIUS, border);
            }
        }
    }
    return (uint8_t)((sum + GAUSS_ROUND) >> GAUSS_SHIFT);
}

static inline uint8_t sobel_pixel(const Image *in, BorderMode border, int y, int x) {
    const int w = in->width;
    const int h = in->height;
    int gx = 0;
    int gy = 0;

    if (y >= SOBEL_RADIUS && y < h - SOBEL_RADIUS &&
        x >= SOBEL_RADIUS && x < w - SOBEL_RADIUS) {
        for (int j = 0; j < SOBEL_SIZE; j++) {
            const uint8_t *row = in->data + (size_t)(y + j - SOBEL_RADIUS) * (size_t)w +
                                 (size_t)(x - SOBEL_RADIUS);
            for (int i = 0; i < SOBEL_SIZE; i++) {
                gx += SOBEL_GX[j][i] * row[i];
                gy += SOBEL_GY[j][i] * row[i];
            }
        }
    } else {
        for (int j = 0; j < SOBEL_SIZE; j++) {
            for (int i = 0; i < SOBEL_SIZE; i++) {
                int p = fetch_pixel(in, y + j - SOBEL_RADIUS, x + i - SOBEL_RADIUS, border);
                gx += SOBEL_GX[j][i] * p;
                gy += SOBEL_GY[j][i] * p;
            }
        }
    }
    /* sqrt(double) deterministik untuk input yang sama -> hasil identik di semua thread. */
    int mag = (int)(sqrt((double)(gx * gx + gy * gy)) + 0.5);
    return (uint8_t)clamp_int(mag, 0, PIXEL_MAX);
}

/* ------------------------------------------------------------------------ */
/* Konvolusi 2D langsung pada sebuah wilayah persegi                         */
/* ------------------------------------------------------------------------ */

void kernel_region(const Image *in, Image *out, FilterType filter,
                   BorderMode border, int y0, int y1, int x0, int x1) {
    const size_t w = (size_t)in->width;
    if (filter == FILTER_GAUSSIAN) {
        for (int y = y0; y < y1; y++) {
            for (int x = x0; x < x1; x++) {
                out->data[(size_t)y * w + (size_t)x] = gauss_pixel(in, border, y, x);
            }
        }
    } else {
        for (int y = y0; y < y1; y++) {
            for (int x = x0; x < x1; x++) {
                out->data[(size_t)y * w + (size_t)x] = sobel_pixel(in, border, y, x);
            }
        }
    }
}

/* ------------------------------------------------------------------------ */
/* Gaussian separable: konvolusi 1D horizontal lalu 1D vertikal              */
/* ------------------------------------------------------------------------ */
/*
 * Karena kernel 2D = GAUSS_1D (x) GAUSS_1D, konvolusi 2D dapat dipecah:
 *   out(y,x) = [ sum_j K[j] * ( sum_i K[i] * p(y+j-R, x+i-R) ) + 128 ] >> 8
 * Biaya per piksel turun dari 25 perkalian menjadi 5 + 5 = 10 perkalian.
 * Karena memakai integer tanpa pembulatan antara, hasilnya SAMA PERSIS dengan
 * versi 2D penuh (baik zero-padding maupun replicate, sebab aturan border
 * bersifat terpisah per sumbu).
 *
 * Pass horizontal menghasilkan buffer sementara `tmp` (uint16: maks 255*16).
 * Pass vertikal membutuhkan baris tmp dari y-R .. y+R, jadi pita [y0, y1)
 * memerlukan HALO R baris di atas dan di bawah. Strategi halo di sini:
 * setiap pita MENGHITUNG ULANG baris halo-nya sendiri ke buffer privat
 * (redundansi 2R baris per blok) sehingga TIDAK butuh barrier/sinkronisasi
 * antar-thread. Pemrosesan per blok SEP_BLOCK_ROWS baris menjaga buffer tmp
 * tetap kecil (muat di cache L2) -> ini juga bentuk cache-blocking.
 */

/** Pass horizontal untuk baris y (boleh di luar citra: halo atas/bawah). */
static void sep_hpass(const Image *in, BorderMode border, int y, uint16_t *dst) {
    const int w = in->width;
    const int h = in->height;

    if (y < 0 || y >= h) {
        if (border == BORDER_ZERO) { /* baris di luar citra seluruhnya nol */
            memset(dst, 0, (size_t)w * sizeof *dst);
            return;
        }
        y = clamp_int(y, 0, h - 1); /* replicate: pakai baris tepi terdekat */
    }
    const uint8_t *row = in->data + (size_t)y * (size_t)w;
    for (int x = 0; x < w; x++) {
        int sum = 0;
        if (x >= GAUSS_RADIUS && x < w - GAUSS_RADIUS) {
            for (int i = 0; i < GAUSS_SIZE; i++) {
                sum += GAUSS_1D[i] * row[x + i - GAUSS_RADIUS];
            }
        } else {
            for (int i = 0; i < GAUSS_SIZE; i++) {
                sum += GAUSS_1D[i] * fetch_pixel(in, y, x + i - GAUSS_RADIUS, border);
            }
        }
        dst[x] = (uint16_t)sum;
    }
}

/**
 * Pass vertikal untuk baris keluaran [y0, y1).
 * tmp[0] bersesuaian dengan baris citra `first` (= y0 - R).
 */
static void sep_vpass(const uint16_t *tmp, int first, int width, Image *out, int y0, int y1) {
    for (int y = y0; y < y1; y++) {
        const uint16_t *rows[GAUSS_SIZE];
        for (int j = 0; j < GAUSS_SIZE; j++) {
            rows[j] = tmp + (size_t)(y + j - GAUSS_RADIUS - first) * (size_t)width;
        }
        uint8_t *dst = out->data + (size_t)y * (size_t)width;
        for (int x = 0; x < width; x++) {
            int sum = 0;
            for (int j = 0; j < GAUSS_SIZE; j++) {
                sum += GAUSS_1D[j] * rows[j][x];
            }
            dst[x] = (uint8_t)((sum + GAUSS_ROUND) >> GAUSS_SHIFT);
        }
    }
}

/** Gaussian separable untuk pita [y0, y1), diproses per blok SEP_BLOCK_ROWS. */
static void sep_band(const Image *in, Image *out, BorderMode border,
                     int y0, int y1, uint16_t *tmp) {
    const int w = in->width;
    for (int by = y0; by < y1; by += SEP_BLOCK_ROWS) {
        int bend = by + SEP_BLOCK_ROWS < y1 ? by + SEP_BLOCK_ROWS : y1;
        int first = by - GAUSS_RADIUS;                  /* awal halo atas   */
        int last = bend + GAUSS_RADIUS;                 /* akhir halo bawah (eksklusif) */
        for (int y = first; y < last; y++) {
            sep_hpass(in, border, y, tmp + (size_t)(y - first) * (size_t)w);
        }
        sep_vpass(tmp, first, w, out, by, bend);
    }
}

size_t kernel_tmp_elems(const ConvConfig *cfg, int width) {
    if (cfg->variant != VARIANT_SEPARABLE) {
        return 0;
    }
    return (size_t)(SEP_BLOCK_ROWS + 2 * GAUSS_RADIUS) * (size_t)width;
}

/* ------------------------------------------------------------------------ */
/* Pita baris: pintu masuk tunggal untuk serial / pthreads / openmp          */
/* ------------------------------------------------------------------------ */

void kernel_band(const Image *in, Image *out, const ConvConfig *cfg,
                 int y0, int y1, uint16_t *tmp) {
    const int w = in->width;

    switch (cfg->variant) {
    case VARIANT_DIRECT:
        /* Row-major: inner loop menyusuri x -> akses memori berurutan (ramah cache). */
        kernel_region(in, out, cfg->filter, cfg->border, y0, y1, 0, w);
        break;

    case VARIANT_TILED:
        /*
         * Cache-blocking: citra dipecah menjadi tile TILE_H x TILE_W agar
         * jendela kerja (TILE_H + 2R baris x TILE_W + 2R kolom) tetap di cache
         * dan baris input yang berdekatan dipakai ulang sebelum terusir.
         */
        for (int ty = y0; ty < y1; ty += TILE_H) {
            int ye = ty + TILE_H < y1 ? ty + TILE_H : y1;
            for (int tx = 0; tx < w; tx += TILE_W) {
                int xe = tx + TILE_W < w ? tx + TILE_W : w;
                kernel_region(in, out, cfg->filter, cfg->border, ty, ye, tx, xe);
            }
        }
        break;

    case VARIANT_COLUMN:
        /*
         * Traversal column-major (sengaja buruk): inner loop menyusuri y,
         * lompatan alamat sebesar `width` byte per langkah -> miss cache/TLB.
         * Dipakai sebagai pembanding untuk analisis cache locality.
         */
        for (int x = 0; x < w; x++) {
            kernel_region(in, out, cfg->filter, cfg->border, y0, y1, x, x + 1);
        }
        break;

    case VARIANT_SEPARABLE:
        sep_band(in, out, cfg->border, y0, y1, tmp);
        break;
    }
}
