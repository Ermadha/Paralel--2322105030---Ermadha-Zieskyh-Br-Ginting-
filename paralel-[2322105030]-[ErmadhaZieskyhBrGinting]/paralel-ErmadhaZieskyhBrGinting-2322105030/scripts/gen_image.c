/**
 * @file gen_image.c
 * @brief Generator citra uji sintetis PGM (P5) yang REPRODUCIBLE.
 *
 * Isi citra: gradien diagonal + dua lingkaran (terang dan gelap) + noise acak.
 * Noise memakai srand(42) sehingga hasilnya identik di setiap pemanggilan
 * (pada implementasi libc yang sama). Tepi lingkaran memberi pekerjaan nyata
 * bagi Sobel; noise memberi pekerjaan bagi Gaussian.
 *
 * Pemakaian: gen_image LEBAR TINGGI OUTPUT.pgm
 *   contoh : gen_image 2000 2000 data/synthetic_2000.pgm
 *            gen_image 4000 4000 data/synthetic_4000.pgm
 *
 * Kompilasi (lihat Makefile): digabung dengan src/pgm_io.c.
 */
#include "common.h"

#include <stdio.h>
#include <stdlib.h>

#define SEED 42                  /* seed tetap -> citra reproducible          */
#define GRADIENT_MAX 150         /* nilai maksimum gradien latar              */
#define BRIGHT_CIRCLE_VALUE 230  /* intensitas lingkaran terang               */
#define DARK_CIRCLE_VALUE 25     /* intensitas lingkaran gelap                */
#define NOISE_AMPLITUDE 12       /* noise acak uniform dalam [-12, +12]       */
#define BRIGHT_RADIUS_DIV 4      /* jari-jari terang = min(W,H) / 4           */
#define DARK_RADIUS_DIV 8        /* jari-jari gelap  = min(W,H) / 8           */
#define DARK_CENTER_DIV 3        /* pusat lingkaran gelap di (W/3, H/3)       */

static int clamp_pixel(int v) {
    return v < 0 ? 0 : (v > PIXEL_MAX ? PIXEL_MAX : v);
}

/** Apakah (x, y) berada di dalam lingkaran berpusat (cx, cy) berjari-jari r. */
static int in_circle(long x, long y, long cx, long cy, long r) {
    long dx = x - cx;
    long dy = y - cy;
    return dx * dx + dy * dy <= r * r;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Pemakaian: %s LEBAR TINGGI OUTPUT.pgm\n", argv[0]);
        return EXIT_FAILURE;
    }
    char *end_w = NULL;
    char *end_h = NULL;
    long w = strtol(argv[1], &end_w, 10);
    long h = strtol(argv[2], &end_h, 10);
    if (*end_w != '\0' || *end_h != '\0' || w < 1 || h < 1 || w > MAX_DIMENSION ||
        h > MAX_DIMENSION) {
        fprintf(stderr, "Ukuran tidak valid (1..%d)\n", MAX_DIMENSION);
        return EXIT_FAILURE;
    }

    Image img = {0, 0, 0, NULL};
    if (image_alloc(&img, (int)w, (int)h) != 0) {
        return EXIT_FAILURE;
    }

    const long min_side = w < h ? w : h;
    srand(SEED);
    for (long y = 0; y < h; y++) {
        for (long x = 0; x < w; x++) {
            /* 1) gradien diagonal: 0 di kiri-atas -> GRADIENT_MAX di kanan-bawah */
            int value = (int)((x * GRADIENT_MAX / (w > 1 ? w - 1 : 1) +
                               y * GRADIENT_MAX / (h > 1 ? h - 1 : 1)) / 2);
            /* 2) lingkaran terang di tengah, lingkaran gelap di kiri-atas */
            if (in_circle(x, y, w / 2, h / 2, min_side / BRIGHT_RADIUS_DIV)) {
                value = BRIGHT_CIRCLE_VALUE;
            }
            if (in_circle(x, y, w / DARK_CENTER_DIV, h / DARK_CENTER_DIV,
                          min_side / DARK_RADIUS_DIV)) {
                value = DARK_CIRCLE_VALUE;
            }
            /* 3) noise reproducible dari srand(SEED) */
            value += rand() % (2 * NOISE_AMPLITUDE + 1) - NOISE_AMPLITUDE;
            img.data[(size_t)y * (size_t)w + (size_t)x] = (uint8_t)clamp_pixel(value);
        }
    }

    int rc = pgm_write(argv[3], &img, 0);
    image_free(&img);
    if (rc != 0) {
        return EXIT_FAILURE;
    }
    printf("Citra sintetis %ldx%ld ditulis ke %s\n", w, h, argv[3]);
    return EXIT_SUCCESS;
}
