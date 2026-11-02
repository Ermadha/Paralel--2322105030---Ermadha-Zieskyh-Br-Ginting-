/**
 * @file test_small.c
 * @brief Unit test kecil (citra 10x10) dengan hasil yang dapat dihitung manual,
 *        plus uji ekuivalensi Serial vs Pthreads vs OpenMP.
 *
 * Cara hitung manual (Gaussian, kernel bulat jumlah 256, hasil = (sum+128)>>8):
 *  - Citra konstan 100, replicate  : sum = 100*256 -> hasil 100 di semua piksel.
 *  - Zero-padding, sudut (0,0)     : hanya bobot kernel baris/kolom 2..4 yang
 *    jatuh di dalam citra: (36+24+6)+(24+16+4)+(6+4+1) = 121
 *    -> (121*100 + 128) >> 8 = 12228 >> 8 = 47.
 *  - Zero-padding, tepi atas (0,5) : jumlah bobot 3 baris = 96+64+16 = 176
 *    -> (17600 + 128) >> 8 = 69.
 *  - Zero-padding, (0,1)           : (24+36+24+6)+(16+24+16+4)+(4+6+4+1)=165
 *    -> (16500 + 128) >> 8 = 64.
 *  - Impuls 255 di (5,5): hasil (5,5) = (36*255+128)>>8 = 36,
 *    (5,6) = 24, (4,4) = 16, (3,3) = 1, (5,7) = 6, (2,2) = 0 (di luar jendela).
 *  - Sobel, tangga vertikal 0|10 di kolom 5: gx = (1+2+1)*10 = 40, gy = 0 -> 40.
 *  - Sobel zero-padding (0,4): gx = 30, gy = 10 -> round(sqrt(1000)) = 32.
 *
 * Kode keluar 0 bila semua pemeriksaan lulus, 1 bila ada yang gagal.
 */
#include "common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_checks = 0;
static int g_failures = 0;

#define CHECK(cond, msg)                                                          \
    do {                                                                          \
        g_checks++;                                                               \
        if (!(cond)) {                                                            \
            g_failures++;                                                         \
            printf("  [GAGAL] %s (tests/test_small.c baris %d)\n", (msg), __LINE__); \
        }                                                                         \
    } while (0)

#define SMALL_SIZE 10
#define RNG_MULT 1664525u
#define RNG_INC 1013904223u
#define TEST_FILE_P5 "test_tmp_p5.pgm"
#define TEST_FILE_P2 "test_tmp_p2.pgm"
#define TEST_FILE_COMMENT "test_tmp_comment.pgm"

static Image make_image(int w, int h, uint8_t fill) {
    Image img = {0, 0, 0, NULL};
    if (image_alloc(&img, w, h) != 0) {
        fprintf(stderr, "make_image gagal\n");
        exit(EXIT_FAILURE);
    }
    memset(img.data, fill, (size_t)w * (size_t)h);
    return img;
}

static void fill_random(Image *img, uint32_t seed) {
    uint32_t state = seed;
    for (size_t i = 0; i < (size_t)img->width * (size_t)img->height; i++) {
        state = state * RNG_MULT + RNG_INC;
        img->data[i] = (uint8_t)(state >> 24);
    }
}

static int px(const Image *img, int y, int x) {
    return img->data[(size_t)y * (size_t)img->width + (size_t)x];
}

static ConvConfig make_cfg(FilterType f, Variant v, BorderMode b, int threads) {
    ConvConfig cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.filter = f;
    cfg.variant = v;
    cfg.border = b;
    cfg.threads = threads;
    cfg.sched = SCHED_STATIC;
    return cfg;
}

/** Jalankan serial dan kembalikan citra hasil (caller wajib image_free). */
static Image run_ref(const Image *in, const ConvConfig *cfg) {
    Image out = make_image(in->width, in->height, 0);
    double t;
    if (run_serial(in, &out, cfg, &t) != 0) {
        fprintf(stderr, "run_serial gagal\n");
        exit(EXIT_FAILURE);
    }
    return out;
}

/* ------------------------------------------------------------------------ */

static void test_gaussian_manual(void) {
    printf("[1] Gaussian 10x10 - hasil hitung manual\n");
    ConvConfig rep = make_cfg(FILTER_GAUSSIAN, VARIANT_DIRECT, BORDER_REPLICATE, 1);
    ConvConfig zero = make_cfg(FILTER_GAUSSIAN, VARIANT_DIRECT, BORDER_ZERO, 1);

    Image flat = make_image(SMALL_SIZE, SMALL_SIZE, 100);
    Image r = run_ref(&flat, &rep);
    int all_100 = 1;
    for (int y = 0; y < SMALL_SIZE; y++)
        for (int x = 0; x < SMALL_SIZE; x++)
            if (px(&r, y, x) != 100) all_100 = 0;
    CHECK(all_100, "citra konstan + replicate harus tetap 100 di semua piksel");

    Image z = run_ref(&flat, &zero);
    CHECK(px(&z, 0, 0) == 47, "zero-padding sudut (0,0) harus 47");
    CHECK(px(&z, 0, 5) == 69, "zero-padding tepi atas (0,5) harus 69");
    CHECK(px(&z, 0, 1) == 64, "zero-padding (0,1) harus 64");
    CHECK(px(&z, 5, 5) == 100, "piksel interior (5,5) harus 100");

    Image imp = make_image(SMALL_SIZE, SMALL_SIZE, 0);
    imp.data[5 * SMALL_SIZE + 5] = PIXEL_MAX;
    Image i = run_ref(&imp, &rep);
    CHECK(px(&i, 5, 5) == 36, "impuls: (5,5) harus 36");
    CHECK(px(&i, 5, 6) == 24, "impuls: (5,6) harus 24");
    CHECK(px(&i, 4, 4) == 16, "impuls: (4,4) harus 16");
    CHECK(px(&i, 3, 3) == 1, "impuls: (3,3) harus 1");
    CHECK(px(&i, 5, 7) == 6, "impuls: (5,7) harus 6");
    CHECK(px(&i, 2, 2) == 0, "impuls: (2,2) di luar jendela harus 0");

    image_free(&flat); image_free(&r); image_free(&z); image_free(&imp); image_free(&i);
}

static void test_sobel_manual(void) {
    printf("[2] Sobel 10x10 - hasil hitung manual\n");
    Image step = make_image(SMALL_SIZE, SMALL_SIZE, 0);
    for (int y = 0; y < SMALL_SIZE; y++)
        for (int x = SMALL_SIZE / 2; x < SMALL_SIZE; x++)
            step.data[y * SMALL_SIZE + x] = 10;

    ConvConfig rep = make_cfg(FILTER_SOBEL, VARIANT_DIRECT, BORDER_REPLICATE, 1);
    ConvConfig zero = make_cfg(FILTER_SOBEL, VARIANT_DIRECT, BORDER_ZERO, 1);
    Image r = run_ref(&step, &rep);
    Image z = run_ref(&step, &zero);

    CHECK(px(&r, 5, 4) == 40, "tepi vertikal (5,4) harus 40");
    CHECK(px(&r, 5, 5) == 40, "tepi vertikal (5,5) harus 40");
    CHECK(px(&r, 5, 3) == 0, "area datar kiri (5,3) harus 0");
    CHECK(px(&r, 5, 6) == 0, "area datar kanan (5,6) harus 0");
    CHECK(px(&r, 0, 4) == 40, "replicate: tepi atas (0,4) harus 40");
    CHECK(px(&z, 0, 4) == 32, "zero-padding: (0,4) harus round(sqrt(1000)) = 32");

    Image big = make_image(SMALL_SIZE, SMALL_SIZE, 0);
    for (int y = 0; y < SMALL_SIZE; y++)
        for (int x = SMALL_SIZE / 2; x < SMALL_SIZE; x++)
            big.data[y * SMALL_SIZE + x] = PIXEL_MAX;
    Image b = run_ref(&big, &rep);
    CHECK(px(&b, 5, 4) == PIXEL_MAX, "magnitudo > 255 harus di-clamp ke 255");

    image_free(&step); image_free(&r); image_free(&z); image_free(&big); image_free(&b);
}

/* ------------------------------------------------------------------------ */

static void test_partition(void) {
    printf("[3] Partisi baris (sisa saat tinggi tidak habis dibagi thread)\n");
    int y0, y1;
    partition_rows(10, 3, 0, &y0, &y1);
    CHECK(y0 == 0 && y1 == 4, "H=10,T=3: thread 0 harus [0,4)");
    partition_rows(10, 3, 1, &y0, &y1);
    CHECK(y0 == 4 && y1 == 7, "H=10,T=3: thread 1 harus [4,7)");
    partition_rows(10, 3, 2, &y0, &y1);
    CHECK(y0 == 7 && y1 == 10, "H=10,T=3: thread 2 harus [7,10)");

    int ok = 1;
    for (int h = 1; h <= 60; h++) {
        for (int t = 1; t <= 12 && t <= h; t++) {
            int expect_start = 0;
            int min_rows = h, max_rows = 0;
            for (int id = 0; id < t; id++) {
                partition_rows(h, t, id, &y0, &y1);
                if (y0 != expect_start || y1 <= y0) ok = 0; /* tanpa celah/tumpang-tindih */
                expect_start = y1;
                if (y1 - y0 < min_rows) min_rows = y1 - y0;
                if (y1 - y0 > max_rows) max_rows = y1 - y0;
            }
            if (expect_start != h || max_rows - min_rows > 1) ok = 0; /* tertutup & seimbang */
        }
    }
    CHECK(ok, "partisi harus menutup [0,H) tanpa celah dan selisih beban <= 1 baris");
}

/* ------------------------------------------------------------------------ */

/** Bandingkan Pthreads & OpenMP (berbagai konfigurasi) dengan serial. */
static void check_equivalence(int w, int h, uint32_t seed) {
    static const FilterType FILTERS[] = {FILTER_GAUSSIAN, FILTER_SOBEL};
    static const Variant VARIANTS[] = {VARIANT_DIRECT, VARIANT_SEPARABLE, VARIANT_TILED,
                                       VARIANT_COLUMN};
    static const BorderMode BORDERS[] = {BORDER_ZERO, BORDER_REPLICATE};
    static const int THREADS[] = {1, 2, 3, 4, 8, 20};
    static const SchedKind SCHEDS[] = {SCHED_STATIC, SCHED_DYNAMIC, SCHED_GUIDED};
    static const int CHUNKS[] = {0, 3};

    Image in = make_image(w, h, 0);
    fill_random(&in, seed);
    long bad = 0;
    long runs = 0;

    for (size_t fi = 0; fi < 2; fi++) {
        ConvConfig base_cfg = make_cfg(FILTERS[fi], VARIANT_DIRECT, BORDER_ZERO, 1);
        for (size_t bi = 0; bi < 2; bi++) {
            base_cfg.border = BORDERS[bi];
            Image ref = run_ref(&in, &base_cfg); /* acuan: serial, 2D langsung */
            Image out = make_image(w, h, 0);
            double t;

            for (size_t vi = 0; vi < 4; vi++) {
                ConvConfig cfg = base_cfg;
                cfg.variant = VARIANTS[vi];
                if (cfg.variant == VARIANT_SEPARABLE && cfg.filter != FILTER_GAUSSIAN) {
                    continue; /* separable hanya untuk Gaussian */
                }

                memset(out.data, 0, (size_t)w * (size_t)h);
                run_serial(&in, &out, &cfg, &t);
                bad += image_diff_count(&out, &ref) != 0;
                runs++;

                for (size_t ti = 0; ti < sizeof THREADS / sizeof THREADS[0]; ti++) {
                    cfg.threads = THREADS[ti];
                    memset(out.data, 0, (size_t)w * (size_t)h);
                    run_pthreads(&in, &out, &cfg, &t);
                    bad += image_diff_count(&out, &ref) != 0;
                    runs++;

                    for (size_t si = 0; si < 3; si++) {
                        for (size_t ci = 0; ci < 2; ci++) {
                            for (int col = 0; col < 2; col++) {
                                cfg.sched = SCHEDS[si];
                                cfg.chunk = CHUNKS[ci];
                                cfg.collapse = col;
                                memset(out.data, 0, (size_t)w * (size_t)h);
                                run_openmp(&in, &out, &cfg, &t);
                                bad += image_diff_count(&out, &ref) != 0;
                                runs++;
                            }
                        }
                    }
                    cfg.sched = SCHED_STATIC;
                    cfg.chunk = 0;
                    cfg.collapse = 0;
                }
            }
            image_free(&ref);
            image_free(&out);
        }
    }
    char msg[128];
    snprintf(msg, sizeof msg, "ekuivalensi %dx%d: %ld dari %ld konfigurasi berbeda dari acuan",
             w, h, bad, runs);
    printf("    citra %3dx%-3d : %ld konfigurasi diuji, %ld berbeda\n", w, h, runs, bad);
    CHECK(bad == 0, msg);
    image_free(&in);
}

static void test_equivalence(void) {
    printf("[4] Ekuivalensi Serial = Pthreads = OpenMP = semua varian (piksel berbeda harus 0)\n");
    check_equivalence(SMALL_SIZE, SMALL_SIZE, 1u); /* 10x10: T=3 -> sisa 1 baris */
    check_equivalence(13, 17, 2u);                 /* ukuran ganjil/prima                  */
    check_equivalence(70, 300, 3u);                /* melewati batas tile & blok separable */
    check_equivalence(3, 3, 4u);                   /* lebih kecil dari kernel 5x5          */
    check_equivalence(1, 9, 5u);                   /* lebar 1 piksel                       */
}

/* ------------------------------------------------------------------------ */

static void test_pgm_io(void) {
    printf("[5] Baca/tulis PGM (P5, P2, komentar)\n");
    Image img = make_image(SMALL_SIZE, SMALL_SIZE, 0);
    fill_random(&img, 7u);

    Image back = {0, 0, 0, NULL};
    CHECK(pgm_write(TEST_FILE_P5, &img, 0) == 0, "tulis P5");
    CHECK(pgm_read(TEST_FILE_P5, &back) == 0, "baca P5");
    CHECK(image_diff_count(&img, &back) == 0, "round-trip P5 harus identik");
    image_free(&back);

    CHECK(pgm_write(TEST_FILE_P2, &img, 1) == 0, "tulis P2");
    CHECK(pgm_read(TEST_FILE_P2, &back) == 0, "baca P2");
    CHECK(image_diff_count(&img, &back) == 0, "round-trip P2 harus identik");
    image_free(&back);

    FILE *f = fopen(TEST_FILE_COMMENT, "w");
    CHECK(f != NULL, "buat file uji komentar");
    if (f != NULL) {
        fputs("P2\n# komentar header\n3 2 # komentar lain\n255\n0 1 2\n3 4 5\n", f);
        fclose(f);
        CHECK(pgm_read(TEST_FILE_COMMENT, &back) == 0, "baca P2 dengan komentar");
        CHECK(back.width == 3 && back.height == 2, "dimensi dari header ber-komentar");
        CHECK(back.data != NULL && px(&back, 1, 2) == 5 && px(&back, 0, 1) == 1,
              "nilai piksel P2 ber-komentar");
        image_free(&back);
    }

    f = fopen(TEST_FILE_COMMENT, "w");
    if (f != NULL) {
        fputs("P6\n1 1\n255\nx", f); /* bukan PGM -> harus ditolak */
        fclose(f);
        fprintf(stderr, "(pesan error berikut disengaja untuk uji file tidak valid)\n");
        CHECK(pgm_read(TEST_FILE_COMMENT, &back) != 0, "magic selain P5/P2 harus ditolak");
    }

    remove(TEST_FILE_P5);
    remove(TEST_FILE_P2);
    remove(TEST_FILE_COMMENT);
    image_free(&img);
}

int main(void) {
    printf("=== Unit test Image Convolution ===\n");
    test_gaussian_manual();
    test_sobel_manual();
    test_partition();
    test_equivalence();
    test_pgm_io();
    printf("\nHasil: %d pemeriksaan, %d gagal -> %s\n", g_checks, g_failures,
           g_failures == 0 ? "LULUS" : "GAGAL");
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
