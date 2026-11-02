/**
 * @file main.c
 * @brief Program utama: parsing CLI, benchmark Serial vs Pthreads vs OpenMP,
 *        verifikasi correctness otomatis, tabel berwarna, dan keluaran CSV.
 *
 * Alur:
 *  1. Baca citra PGM `repeat` kali  -> waktu BACA (median)         [tantangan c]
 *  2. Serial: warm-up + `repeat` kali ukur -> median/std/min       (baseline)
 *  3. Untuk tiap jumlah thread: Pthreads dan OpenMP dengan prosedur sama;
 *     bandingkan keluaran dengan serial piksel demi piksel.
 *  4. Tulis citra hasil `repeat` kali -> waktu TULIS (median)      [tantangan c]
 *  5. Cetak tabel, ringkasan I/O-bound vs compute-bound, dan tulis CSV.
 *
 * Kode keluar: 0 sukses; 1 error argumen/IO/memori; 2 verifikasi GAGAL
 * (ada piksel berbeda antara implementasi paralel dan serial).
 */
#include "common.h"

#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ------------------------------------------------------------------------ */
/* Konstanta                                                                 */
/* ------------------------------------------------------------------------ */

#define DEFAULT_REPEAT 5
#define DEFAULT_WARMUP 1
#define DEFAULT_OUTPUT "out.pgm"
#define MAX_REPEAT 1000
#define MS_PER_SEC 1000.0
#define PERCENT 100.0
#define EFFICIENCY_GOOD 0.80  /* efisiensi >= 80% diwarnai hijau  */
#define EFFICIENCY_OK 0.50    /* efisiensi >= 50% diwarnai kuning */
#define SCHEDULE_TEXT_LEN 64
#define EXIT_VERIFY_FAILED 2

enum { METHOD_SERIAL = 1, METHOD_PTHREADS = 2, METHOD_OPENMP = 4, METHOD_ALL = 7 };

/* ------------------------------------------------------------------------ */
/* Warna terminal (ANSI)                                                     */
/* ------------------------------------------------------------------------ */

static const char *C_RESET = "";
static const char *C_BOLD = "";
static const char *C_RED = "";
static const char *C_GREEN = "";
static const char *C_YELLOW = "";
static const char *C_CYAN = "";

static void enable_colors(int on) {
    if (on) {
        C_RESET = "\033[0m";
        C_BOLD = "\033[1m";
        C_RED = "\033[31m";
        C_GREEN = "\033[32m";
        C_YELLOW = "\033[33m";
        C_CYAN = "\033[36m";
    }
}

/* ------------------------------------------------------------------------ */
/* Opsi CLI                                                                  */
/* ------------------------------------------------------------------------ */

typedef struct {
    const char *input;
    const char *output;
    const char *csv_path;
    ConvConfig cfg;
    unsigned methods;
    int thread_list[MAX_THREAD_LIST];
    int thread_count;
    int repeat;
    int warmup;
    int ascii;
    int use_color;
    int verify;
} Options;

static void print_usage(const char *prog) {
    printf(
        "Pemakaian: %s -i FILE.pgm [opsi]\n\n"
        "Konvolusi citra paralel (Serial vs POSIX Threads vs OpenMP).\n\n"
        "Opsi:\n"
        "  -i, --input FILE      citra masuk PGM P5/P2 (WAJIB)\n"
        "  -o, --output FILE     citra hasil, P5 (default: %s)\n"
        "  -f, --filter NAMA     gaussian | sobel            (default: gaussian)\n"
        "  -v, --variant NAMA    direct | separable | tiled | column (default: direct)\n"
        "                        separable hanya untuk gaussian\n"
        "  -b, --border MODE     zero | replicate            (default: replicate)\n"
        "  -m, --method NAMA     serial | pthreads | openmp | all (default: all)\n"
        "  -t, --threads LIST    daftar thread, mis. 4 atau 1,2,4,8 (default: 1,2,4,8)\n"
        "  -n, --repeat N        jumlah pengulangan terukur  (default: %d)\n"
        "  -w, --warmup N        jumlah warm-up run          (default: %d)\n"
        "  -s, --schedule JENIS  static | dynamic | guided (OpenMP; default: static)\n"
        "  -c, --chunk N         ukuran chunk OpenMP, 0 = default (default: 0)\n"
        "      --collapse        OpenMP collapse(2) pada varian direct\n"
        "      --csv FILE        tambahkan hasil ke FILE CSV\n"
        "      --ascii           tulis citra hasil sebagai P2 (ASCII)\n"
        "      --no-color        matikan warna terminal\n"
        "      --no-verify       lewati verifikasi correctness\n"
        "  -h, --help            tampilkan bantuan ini\n",
        prog, DEFAULT_OUTPUT, DEFAULT_REPEAT, DEFAULT_WARMUP);
}

/** Parse bilangan bulat dalam [lo, hi]. Return 0 jika valid. */
static int parse_int(const char *text, long lo, long hi, int *out) {
    char *end = NULL;
    long v = strtol(text, &end, 10);
    if (end == text || *end != '\0' || v < lo || v > hi) {
        return -1;
    }
    *out = (int)v;
    return 0;
}

/** Parse "1,2,4,8" ke dalam opt->thread_list. Return 0 jika valid. */
static int parse_thread_list(const char *text, Options *opt) {
    char buf[256];
    if (strlen(text) >= sizeof buf) {
        return -1;
    }
    strcpy(buf, text);
    opt->thread_count = 0;
    char *save = NULL;
    for (char *tok = strtok_r(buf, ",", &save); tok != NULL; tok = strtok_r(NULL, ",", &save)) {
        int v;
        if (opt->thread_count >= MAX_THREAD_LIST || parse_int(tok, 1, MAX_THREADS, &v) != 0) {
            return -1;
        }
        opt->thread_list[opt->thread_count++] = v;
    }
    return opt->thread_count > 0 ? 0 : -1;
}

enum { OPT_COLLAPSE = 1000, OPT_CSV, OPT_ASCII, OPT_NO_COLOR, OPT_NO_VERIFY };

/** Parse argumen CLI. Return 0 sukses, 1 bila --help, -1 bila error. */
static int parse_args(int argc, char **argv, Options *opt) {
    static const struct option long_opts[] = {
        {"input", required_argument, NULL, 'i'},   {"output", required_argument, NULL, 'o'},
        {"filter", required_argument, NULL, 'f'},  {"variant", required_argument, NULL, 'v'},
        {"border", required_argument, NULL, 'b'},  {"method", required_argument, NULL, 'm'},
        {"threads", required_argument, NULL, 't'}, {"repeat", required_argument, NULL, 'n'},
        {"warmup", required_argument, NULL, 'w'},  {"schedule", required_argument, NULL, 's'},
        {"chunk", required_argument, NULL, 'c'},   {"collapse", no_argument, NULL, OPT_COLLAPSE},
        {"csv", required_argument, NULL, OPT_CSV}, {"ascii", no_argument, NULL, OPT_ASCII},
        {"no-color", no_argument, NULL, OPT_NO_COLOR},
        {"no-verify", no_argument, NULL, OPT_NO_VERIFY},
        {"help", no_argument, NULL, 'h'},          {NULL, 0, NULL, 0}};

    int c;
    while ((c = getopt_long(argc, argv, "i:o:f:v:b:m:t:n:w:s:c:h", long_opts, NULL)) != -1) {
        switch (c) {
        case 'i': opt->input = optarg; break;
        case 'o': opt->output = optarg; break;
        case 'f':
            if (strcmp(optarg, "gaussian") == 0) opt->cfg.filter = FILTER_GAUSSIAN;
            else if (strcmp(optarg, "sobel") == 0) opt->cfg.filter = FILTER_SOBEL;
            else { fprintf(stderr, "Filter tidak dikenal: %s\n", optarg); return -1; }
            break;
        case 'v':
            if (strcmp(optarg, "direct") == 0) opt->cfg.variant = VARIANT_DIRECT;
            else if (strcmp(optarg, "separable") == 0) opt->cfg.variant = VARIANT_SEPARABLE;
            else if (strcmp(optarg, "tiled") == 0) opt->cfg.variant = VARIANT_TILED;
            else if (strcmp(optarg, "column") == 0) opt->cfg.variant = VARIANT_COLUMN;
            else { fprintf(stderr, "Varian tidak dikenal: %s\n", optarg); return -1; }
            break;
        case 'b':
            if (strcmp(optarg, "zero") == 0) opt->cfg.border = BORDER_ZERO;
            else if (strcmp(optarg, "replicate") == 0) opt->cfg.border = BORDER_REPLICATE;
            else { fprintf(stderr, "Mode border tidak dikenal: %s\n", optarg); return -1; }
            break;
        case 'm':
            if (strcmp(optarg, "serial") == 0) opt->methods = METHOD_SERIAL;
            else if (strcmp(optarg, "pthreads") == 0) opt->methods = METHOD_PTHREADS;
            else if (strcmp(optarg, "openmp") == 0) opt->methods = METHOD_OPENMP;
            else if (strcmp(optarg, "all") == 0) opt->methods = METHOD_ALL;
            else { fprintf(stderr, "Metode tidak dikenal: %s\n", optarg); return -1; }
            break;
        case 't':
            if (parse_thread_list(optarg, opt) != 0) {
                fprintf(stderr, "Daftar thread tidak valid: %s (contoh: 1,2,4,8)\n", optarg);
                return -1;
            }
            break;
        case 'n':
            if (parse_int(optarg, 1, MAX_REPEAT, &opt->repeat) != 0) {
                fprintf(stderr, "--repeat harus 1..%d\n", MAX_REPEAT);
                return -1;
            }
            break;
        case 'w':
            if (parse_int(optarg, 0, MAX_REPEAT, &opt->warmup) != 0) {
                fprintf(stderr, "--warmup harus 0..%d\n", MAX_REPEAT);
                return -1;
            }
            break;
        case 's':
            if (strcmp(optarg, "static") == 0) opt->cfg.sched = SCHED_STATIC;
            else if (strcmp(optarg, "dynamic") == 0) opt->cfg.sched = SCHED_DYNAMIC;
            else if (strcmp(optarg, "guided") == 0) opt->cfg.sched = SCHED_GUIDED;
            else { fprintf(stderr, "Jadwal OpenMP tidak dikenal: %s\n", optarg); return -1; }
            break;
        case 'c':
            if (parse_int(optarg, 0, INT_MAX, &opt->cfg.chunk) != 0) {
                fprintf(stderr, "--chunk harus bilangan bulat >= 0\n");
                return -1;
            }
            break;
        case OPT_COLLAPSE: opt->cfg.collapse = 1; break;
        case OPT_CSV: opt->csv_path = optarg; break;
        case OPT_ASCII: opt->ascii = 1; break;
        case OPT_NO_COLOR: opt->use_color = 0; break;
        case OPT_NO_VERIFY: opt->verify = 0; break;
        case 'h': return 1;
        default: return -1;
        }
    }
    if (opt->input == NULL) {
        fprintf(stderr, "Argumen -i/--input wajib diisi.\n");
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------------ */
/* Statistik                                                                 */
/* ------------------------------------------------------------------------ */

typedef struct {
    double median;
    double stddev; /**< simpangan baku sampel (n-1); 0 bila n == 1 */
    double min;
} Stats;

static int compare_double(const void *a, const void *b) {
    double x = *(const double *)a;
    double y = *(const double *)b;
    return (x > y) - (x < y);
}

/** Hitung median, std, min dari n sampel (array asli tidak diubah). */
static int compute_stats(const double *samples, int n, Stats *st) {
    double *sorted = malloc((size_t)n * sizeof *sorted);
    if (sorted == NULL) {
        fprintf(stderr, "compute_stats: gagal alokasi memori\n");
        return -1;
    }
    memcpy(sorted, samples, (size_t)n * sizeof *sorted);
    qsort(sorted, (size_t)n, sizeof *sorted, compare_double);
    st->median = (n % 2 == 1) ? sorted[n / 2] : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
    st->min = sorted[0];

    double mean = 0.0;
    for (int i = 0; i < n; i++) mean += samples[i];
    mean /= n;
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (samples[i] - mean) * (samples[i] - mean);
    st->stddev = n > 1 ? sqrt(ss / (n - 1)) : 0.0;
    free(sorted);
    return 0;
}

/* ------------------------------------------------------------------------ */
/* Pengukuran                                                                */
/* ------------------------------------------------------------------------ */

/** Tipe pointer fungsi untuk ketiga implementasi (run_serial/pthreads/openmp). */
typedef int (*RunFn)(const Image *, Image *, const ConvConfig *, double *);

/** Kosongkan citra keluaran agar verifikasi tidak "tertipu" hasil run sebelumnya. */
static void clear_image(Image *img) {
    memset(img->data, 0, (size_t)img->width * (size_t)img->height);
}

/**
 * Ukur waktu komputasi: `warmup` run dibuang (mengisi cache, page-fault,
 * frekuensi CPU, pembuatan tim thread OpenMP), lalu `repeat` run dicatat.
 * Setelah selesai, `out` berisi hasil run terakhir untuk diverifikasi.
 */
static int measure_compute(RunFn fn, const Image *in, Image *out, const ConvConfig *cfg,
                           int warmup, int repeat, double *samples) {
    double t = 0.0;
    for (int i = 0; i < warmup; i++) {
        clear_image(out);
        if (fn(in, out, cfg, &t) != 0) return -1;
    }
    for (int i = 0; i < repeat; i++) {
        clear_image(out);
        if (fn(in, out, cfg, &t) != 0) return -1;
        samples[i] = t;
    }
    return 0;
}

/** Ukur waktu BACA file (tantangan c). Citra terakhir yang terbaca disimpan di *img. */
static int measure_read(const char *path, int repeat, Image *img, double *samples) {
    for (int i = 0; i < repeat; i++) {
        if (img->data != NULL) image_free(img);
        double t_start = now_seconds();
        if (pgm_read(path, img) != 0) return -1;
        samples[i] = now_seconds() - t_start;
    }
    return 0;
}

/** Ukur waktu TULIS file (tantangan c). */
static int measure_write(const char *path, const Image *img, int ascii, int repeat,
                         double *samples) {
    for (int i = 0; i < repeat; i++) {
        double t_start = now_seconds();
        if (pgm_write(path, img, ascii) != 0) return -1;
        samples[i] = now_seconds() - t_start;
    }
    return 0;
}

/* ------------------------------------------------------------------------ */
/* Pelaporan: tabel terminal & CSV                                           */
/* ------------------------------------------------------------------------ */

/** Satu baris hasil. */
typedef struct {
    const char *method;
    int threads;
    Stats st;
    double speedup;
    double efficiency;
    long diff_serial; /**< piksel berbeda vs serial (varian sama); -1 = tidak diverifikasi */
    long diff_direct; /**< piksel berbeda vs serial varian 'direct'; -1 = tidak diverifikasi */
} Row;

static void print_table_header(void) {
    printf("%s%-9s %7s  %-10s %11s %9s %9s %9s %10s %9s%s\n", C_BOLD, "Metode", "Thread",
           "Varian", "Median(ms)", "Std(ms)", "Min(ms)", "Speedup", "Efisiensi", "Beda px",
           C_RESET);
}

static void print_row(const Options *opt, const Row *r) {
    const char *eff_color = r->efficiency >= EFFICIENCY_GOOD ? C_GREEN
                            : r->efficiency >= EFFICIENCY_OK ? C_YELLOW : C_RED;
    long worst = r->diff_serial > r->diff_direct ? r->diff_serial : r->diff_direct;

    printf("%-9s %7d  %-10s %11.3f %9.3f %9.3f %s%8.2fx%s %s%9.1f%%%s ", r->method, r->threads,
           variant_name(opt->cfg.variant), r->st.median * MS_PER_SEC, r->st.stddev * MS_PER_SEC,
           r->st.min * MS_PER_SEC, C_CYAN, r->speedup, C_RESET, eff_color,
           r->efficiency * PERCENT, C_RESET);
    if (worst < 0) {
        printf("%9s\n", "n/a");
    } else {
        printf("%s%9ld%s\n", worst == 0 ? C_GREEN : C_RED, worst, C_RESET);
    }
}

/** Susun teks jadwal OpenMP untuk kolom CSV ("-" untuk metode non-OpenMP). */
static void format_schedule(const Options *opt, const char *method, char *buf, size_t n) {
    if (strcmp(method, "openmp") != 0) {
        snprintf(buf, n, "-");
        return;
    }
    size_t len = (size_t)snprintf(buf, n, "%s", sched_name(opt->cfg.sched));
    if (opt->cfg.chunk > 0 && len < n) {
        len += (size_t)snprintf(buf + len, n - len, ",%d", opt->cfg.chunk);
    }
    if (opt->cfg.collapse && opt->cfg.variant == VARIANT_DIRECT && len < n) {
        snprintf(buf + len, n - len, "+collapse");
    }
}

static const char *CSV_HEADER =
    "metode,filter,varian,border,ukuran,thread,jadwal,waktu_median_s,waktu_std_s,"
    "waktu_min_s,speedup,efficiency,baca_s,tulis_s,piksel_beda,beda_vs_direct\n";

static void write_csv_row(FILE *csv, const Options *opt, const Image *in, const Row *r,
                          double read_s, double write_s) {
    char schedule[SCHEDULE_TEXT_LEN];
    format_schedule(opt, r->method, schedule, sizeof schedule);
    fprintf(csv, "%s,%s,%s,%s,%dx%d,%d,%s,%.9f,%.9f,%.9f,%.6f,%.6f,%.9f,%.9f,%ld,%ld\n",
            r->method, filter_name(opt->cfg.filter), variant_name(opt->cfg.variant),
            border_name(opt->cfg.border), in->width, in->height, r->threads, schedule,
            r->st.median, r->st.stddev, r->st.min, r->speedup, r->efficiency, read_s, write_s,
            r->diff_serial, r->diff_direct);
}

/** Pembuka CSV: tulis header bila file masih kosong. */
static FILE *open_csv(const char *path) {
    FILE *f = fopen(path, "a");
    if (f == NULL) {
        perror(path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    if (ftell(f) == 0) {
        fputs(CSV_HEADER, f);
    }
    return f;
}

/** Cetak ringkasan analisis I/O-bound vs compute-bound (tantangan c). */
static void print_io_summary(double read_s, double compute_s, double write_s,
                             const char *label) {
    double io = read_s + write_s;
    double total = io + compute_s;
    double share = total > 0.0 ? io / total : 0.0;
    const char *verdict = compute_s > io ? "COMPUTE-BOUND" : "I/O-BOUND";
    const char *color = compute_s > io ? C_GREEN : C_YELLOW;
    printf("  %-26s baca %8.3f ms | hitung %8.3f ms | tulis %8.3f ms | I/O %5.1f%% -> %s%s%s\n",
           label, read_s * MS_PER_SEC, compute_s * MS_PER_SEC, write_s * MS_PER_SEC,
           share * PERCENT, color, verdict, C_RESET);
}

/* ------------------------------------------------------------------------ */
/* main                                                                      */
/* ------------------------------------------------------------------------ */

int main(int argc, char **argv) {
    Options opt;
    memset(&opt, 0, sizeof opt);
    opt.output = DEFAULT_OUTPUT;
    opt.cfg.filter = FILTER_GAUSSIAN;
    opt.cfg.variant = VARIANT_DIRECT;
    opt.cfg.border = BORDER_REPLICATE;
    opt.cfg.sched = SCHED_STATIC;
    opt.methods = METHOD_ALL;
    opt.thread_list[0] = 1;
    opt.thread_list[1] = 2;
    opt.thread_list[2] = 4;
    opt.thread_list[3] = 8;
    opt.thread_count = 4;
    opt.repeat = DEFAULT_REPEAT;
    opt.warmup = DEFAULT_WARMUP;
    opt.use_color = 1;
    opt.verify = 1;

    int parsed = parse_args(argc, argv, &opt);
    if (parsed == 1) {
        print_usage(argv[0]);
        return EXIT_SUCCESS;
    }
    if (parsed < 0) {
        fprintf(stderr, "Gunakan --help untuk melihat daftar opsi.\n");
        return EXIT_FAILURE;
    }
    if (config_validate(&opt.cfg) != 0) {
        return EXIT_FAILURE;
    }
    enable_colors(opt.use_color && isatty(STDOUT_FILENO));

    int exit_code = EXIT_FAILURE;
    Image in = {0, 0, 0, NULL};
    Image out_serial = {0, 0, 0, NULL};
    Image out_par = {0, 0, 0, NULL};
    Image out_ref = {0, 0, 0, NULL};
    FILE *csv = NULL;
    double *samples = malloc((size_t)opt.repeat * sizeof *samples);
    double *read_samples = malloc((size_t)opt.repeat * sizeof *read_samples);
    double *write_samples = malloc((size_t)opt.repeat * sizeof *write_samples);
    if (samples == NULL || read_samples == NULL || write_samples == NULL) {
        fprintf(stderr, "Gagal alokasi memori untuk sampel waktu\n");
        goto cleanup;
    }

    /* --- 1. Waktu BACA ---------------------------------------------------- */
    if (measure_read(opt.input, opt.repeat, &in, read_samples) != 0) goto cleanup;
    Stats read_st;
    if (compute_stats(read_samples, opt.repeat, &read_st) != 0) goto cleanup;

    if (image_alloc(&out_serial, in.width, in.height) != 0 ||
        image_alloc(&out_par, in.width, in.height) != 0) {
        goto cleanup;
    }
    int need_ref = opt.verify && opt.cfg.variant != VARIANT_DIRECT;
    if (need_ref && image_alloc(&out_ref, in.width, in.height) != 0) goto cleanup;

    if (in.width < GAUSS_SIZE || in.height < GAUSS_SIZE) {
        fprintf(stderr, "Peringatan: citra %dx%d lebih kecil dari kernel; seluruh piksel "
                        "diproses lewat jalur border.\n", in.width, in.height);
    }
    if (opt.csv_path != NULL) {
        csv = open_csv(opt.csv_path);
        if (csv == NULL) goto cleanup;
    }

    printf("%s=== Image Convolution Paralel ===%s\n", C_BOLD, C_RESET);
    printf("Citra   : %s (%dx%d, %.2f MP)\n", opt.input, in.width, in.height,
           (double)in.width * in.height / 1.0e6);
    printf("Konfigurasi: filter=%s varian=%s border=%s | repeat=%d warmup=%d | OpenMP jadwal=%s "
           "chunk=%d collapse=%d\n\n",
           filter_name(opt.cfg.filter), variant_name(opt.cfg.variant),
           border_name(opt.cfg.border), opt.repeat, opt.warmup, sched_name(opt.cfg.sched),
           opt.cfg.chunk, opt.cfg.collapse);

    /* --- 2. Serial (baseline; selalu dijalankan untuk speedup & verifikasi) - */
    ConvConfig cfg = opt.cfg;
    cfg.threads = 1;
    if (measure_compute(run_serial, &in, &out_serial, &cfg, opt.warmup, opt.repeat, samples) != 0)
        goto cleanup;
    Stats serial_st;
    if (compute_stats(samples, opt.repeat, &serial_st) != 0) goto cleanup;

    /* Verifikasi silang: varian non-direct harus sama dengan 2D langsung (row-major). */
    long cross_serial = opt.verify ? 0 : -1;
    if (need_ref) {
        ConvConfig ref_cfg = cfg;
        double ignored = 0.0;
        ref_cfg.variant = VARIANT_DIRECT;
        if (run_serial(&in, &out_ref, &ref_cfg, &ignored) != 0) goto cleanup;
        cross_serial = image_diff_count(&out_serial, &out_ref);
    }

    /* --- 3. Waktu TULIS (hasil serial identik dengan hasil paralel) ------- */
    if (measure_write(opt.output, &out_serial, opt.ascii, opt.repeat, write_samples) != 0)
        goto cleanup;
    Stats write_st;
    if (compute_stats(write_samples, opt.repeat, &write_st) != 0) goto cleanup;

    /* --- 4. Tabel hasil --------------------------------------------------- */
    print_table_header();
    long total_mismatch = 0;
    double best_median = serial_st.median;
    char best_label[SCHEDULE_TEXT_LEN];
    snprintf(best_label, sizeof best_label, "serial");

    Row serial_row = {"serial", 1, serial_st, 1.0, 1.0, opt.verify ? 0 : -1, cross_serial};
    if (cross_serial > 0) total_mismatch += cross_serial;
    if (opt.methods & METHOD_SERIAL) {
        print_row(&opt, &serial_row);
        if (csv != NULL) write_csv_row(csv, &opt, &in, &serial_row, read_st.median, write_st.median);
    }

    static const struct { const char *name; RunFn fn; unsigned flag; } METHODS[] = {
        {"pthreads", run_pthreads, METHOD_PTHREADS},
        {"openmp", run_openmp, METHOD_OPENMP},
    };

    for (int ti = 0; ti < opt.thread_count; ti++) {
        for (size_t mi = 0; mi < sizeof METHODS / sizeof METHODS[0]; mi++) {
            if (!(opt.methods & METHODS[mi].flag)) continue;

            ConvConfig pcfg = opt.cfg;
            pcfg.threads = opt.thread_list[ti];
            if (measure_compute(METHODS[mi].fn, &in, &out_par, &pcfg, opt.warmup, opt.repeat,
                                samples) != 0) {
                goto cleanup;
            }
            Row row;
            row.method = METHODS[mi].name;
            row.threads = pcfg.threads;
            if (compute_stats(samples, opt.repeat, &row.st) != 0) goto cleanup;
            row.speedup = serial_st.median / row.st.median;
            row.efficiency = row.speedup / row.threads;
            row.diff_serial = -1;
            row.diff_direct = -1;
            if (opt.verify) { /* bandingkan piksel demi piksel */
                row.diff_serial = image_diff_count(&out_par, &out_serial);
                row.diff_direct = need_ref ? image_diff_count(&out_par, &out_ref) : row.diff_serial;
                if (row.diff_serial != 0 || row.diff_direct != 0) total_mismatch++;
            }
            print_row(&opt, &row);
            if (csv != NULL) write_csv_row(csv, &opt, &in, &row, read_st.median, write_st.median);
            if (row.st.median < best_median) {
                best_median = row.st.median;
                snprintf(best_label, sizeof best_label, "%s x%d", row.method, row.threads);
            }
        }
    }

    /* --- 5. Ringkasan I/O vs compute (tantangan c) ------------------------ */
    printf("\n%sAnalisis I/O-bound vs compute-bound (median):%s\n", C_BOLD, C_RESET);
    print_io_summary(read_st.median, serial_st.median, write_st.median, "serial");
    if (strcmp(best_label, "serial") != 0) {
        print_io_summary(read_st.median, best_median, write_st.median, best_label);
    }
    printf("  Catatan: I/O diukur dengan file dalam page cache OS (setelah pembacaan pertama);\n"
           "  pada disk dingin porsi I/O sebenarnya lebih besar.\n");

    if (opt.verify) {
        printf("\n%sVerifikasi correctness:%s ", C_BOLD, C_RESET);
        if (total_mismatch == 0) {
            printf("%sLULUS%s - semua keluaran identik piksel demi piksel (piksel berbeda = 0)\n",
                   C_GREEN, C_RESET);
        } else {
            printf("%sGAGAL%s - ditemukan ketidakcocokan (%ld kasus)\n", C_RED, C_RESET,
                   total_mismatch);
        }
    }
    printf("Citra hasil ditulis ke: %s (%s)\n", opt.output, opt.ascii ? "P2 ASCII" : "P5 biner");

    exit_code = total_mismatch == 0 ? EXIT_SUCCESS : EXIT_VERIFY_FAILED;

cleanup:
    if (csv != NULL) fclose(csv);
    image_free(&in);
    image_free(&out_serial);
    image_free(&out_par);
    image_free(&out_ref);
    free(samples);
    free(read_samples);
    free(write_samples);
    return exit_code;
}
