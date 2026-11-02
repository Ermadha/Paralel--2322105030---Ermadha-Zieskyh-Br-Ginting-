/**
 * @file pgm_io.c
 * @brief Alokasi citra serta pembacaan/penulisan format PGM tanpa library
 *        eksternal. Mendukung P5 (biner) dan P2 (ASCII).
 *
 * Format PGM (Portable GrayMap):
 *   <magic> <lebar> <tinggi> <maxval> <data piksel>
 *   - magic  : "P5" (data biner, 1 byte/piksel) atau "P2" (data teks desimal)
 *   - header dipisahkan whitespace; baris yang diawali '#' adalah komentar
 *   - pada P5, tepat SATU karakter whitespace memisahkan maxval dan data
 *
 * Batasan: hanya maxval <= 255 (8-bit). PGM 16-bit ditolak dengan pesan jelas.
 */
#include "common.h"

#include <ctype.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PGM_MAGIC_BINARY '5'     /* "P5" */
#define PGM_MAGIC_ASCII '2'      /* "P2" */
#define ASCII_VALUES_PER_LINE 16 /* jumlah nilai per baris pada keluaran P2 */

/* ------------------------------------------------------------------------ */
/* Alokasi                                                                   */
/* ------------------------------------------------------------------------ */

int image_alloc(Image *img, int width, int height) {
    if (width <= 0 || height <= 0 || width > MAX_DIMENSION || height > MAX_DIMENSION) {
        fprintf(stderr, "image_alloc: dimensi tidak valid (%d x %d)\n", width, height);
        return -1;
    }
    size_t bytes = (size_t)width * (size_t)height;
    /* aligned_alloc (C11) mensyaratkan ukuran kelipatan alignment. */
    size_t alloc_bytes = (bytes + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE * CACHE_LINE_SIZE;
    uint8_t *buf = aligned_alloc(CACHE_LINE_SIZE, alloc_bytes);
    if (buf == NULL) {
        fprintf(stderr, "image_alloc: gagal mengalokasikan %zu byte\n", alloc_bytes);
        return -1;
    }
    memset(buf, 0, alloc_bytes); /* "first touch" di thread utama + inisialisasi */
    img->width = width;
    img->height = height;
    img->maxval = PIXEL_MAX;
    img->data = buf;
    return 0;
}

void image_free(Image *img) {
    free(img->data);
    img->data = NULL;
    img->width = 0;
    img->height = 0;
}

long image_diff_count(const Image *a, const Image *b) {
    if (a->width != b->width || a->height != b->height) {
        return -1;
    }
    long diff = 0;
    size_t n = (size_t)a->width * (size_t)a->height;
    for (size_t i = 0; i < n; i++) {
        if (a->data[i] != b->data[i]) {
            diff++;
        }
    }
    return diff;
}

/* ------------------------------------------------------------------------ */
/* Pembacaan                                                                 */
/* ------------------------------------------------------------------------ */

/**
 * Baca satu bilangan bulat non-negatif dari stream PGM.
 * Melewati whitespace dan komentar ('#' sampai akhir baris). Tepat satu
 * karakter pemisah setelah angka ikut terbaca (sesuai spesifikasi P5).
 * @return 0 jika sukses, -1 jika format salah / EOF.
 */
static int read_pgm_int(FILE *f, int *value) {
    int c;
    for (;;) {
        c = fgetc(f);
        if (c == EOF) {
            return -1;
        }
        if (c == '#') { /* komentar: buang sampai akhir baris */
            while ((c = fgetc(f)) != EOF && c != '\n') {
            }
            if (c == EOF) {
                return -1;
            }
            continue;
        }
        if (!isspace(c)) {
            break;
        }
    }
    if (!isdigit(c)) {
        return -1;
    }
    long v = 0;
    while (isdigit(c)) {
        v = v * 10 + (c - '0');
        if (v > INT_MAX) {
            return -1;
        }
        c = fgetc(f);
    }
    if (c == '#') {
        ungetc(c, f); /* komentar tepat setelah angka: biarkan dibaca berikutnya */
    }
    *value = (int)v;
    return 0;
}

int pgm_read(const char *path, Image *img) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        perror(path);
        return -1;
    }

    int rc = -1;
    int width = 0, height = 0, maxval = 0;
    int magic_p = fgetc(f);
    int magic_n = fgetc(f);
    if (magic_p != 'P' || (magic_n != PGM_MAGIC_BINARY && magic_n != PGM_MAGIC_ASCII)) {
        fprintf(stderr, "%s: bukan file PGM P5/P2\n", path);
        goto done;
    }
    if (read_pgm_int(f, &width) != 0 || read_pgm_int(f, &height) != 0 ||
        read_pgm_int(f, &maxval) != 0) {
        fprintf(stderr, "%s: header PGM rusak\n", path);
        goto done;
    }
    if (maxval < 1 || maxval > PIXEL_MAX) {
        fprintf(stderr, "%s: maxval %d tidak didukung (harus 1..%d)\n", path, maxval, PIXEL_MAX);
        goto done;
    }
    if (image_alloc(img, width, height) != 0) {
        goto done;
    }
    img->maxval = maxval;

    size_t n = (size_t)width * (size_t)height;
    if (magic_n == PGM_MAGIC_BINARY) {
        if (fread(img->data, 1, n, f) != n) {
            fprintf(stderr, "%s: data P5 kurang dari %zu byte\n", path, n);
            image_free(img);
            goto done;
        }
    } else {
        for (size_t i = 0; i < n; i++) {
            int v;
            if (read_pgm_int(f, &v) != 0 || v > maxval) {
                fprintf(stderr, "%s: nilai piksel P2 ke-%zu tidak valid\n", path, i);
                image_free(img);
                goto done;
            }
            img->data[i] = (uint8_t)v;
        }
    }
    rc = 0;
done:
    fclose(f);
    return rc;
}

/* ------------------------------------------------------------------------ */
/* Penulisan                                                                 */
/* ------------------------------------------------------------------------ */

int pgm_write(const char *path, const Image *img, int ascii) {
    FILE *f = fopen(path, "wb");
    if (f == NULL) {
        perror(path);
        return -1;
    }
    size_t n = (size_t)img->width * (size_t)img->height;
    int ok = 1;

    /* Keluaran selalu memakai maxval 255. */
    fprintf(f, "P%c\n%d %d\n%d\n", ascii ? PGM_MAGIC_ASCII : PGM_MAGIC_BINARY,
            img->width, img->height, PIXEL_MAX);
    if (ascii) {
        for (size_t i = 0; i < n; i++) {
            int last_in_line = ((i + 1) % ASCII_VALUES_PER_LINE == 0) || (i + 1 == n);
            fprintf(f, "%u%c", (unsigned)img->data[i], last_in_line ? '\n' : ' ');
        }
    } else if (fwrite(img->data, 1, n, f) != n) {
        ok = 0;
    }
    if (ferror(f)) {
        ok = 0;
    }
    if (fclose(f) != 0) {
        ok = 0;
    }
    if (!ok) {
        fprintf(stderr, "%s: gagal menulis file\n", path);
        return -1;
    }
    return 0;
}
