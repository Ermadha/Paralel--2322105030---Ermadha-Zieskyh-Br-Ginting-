# ==========================================================================
# Makefile - Image Convolution Paralel (Serial vs Pthreads vs OpenMP)
# Target: all, test, bench, plot, clean, loc  (+ images, distclean, help)
# ==========================================================================
CC       = gcc
CFLAGS   = -std=c11 -Wall -Wextra -O2 -D_POSIX_C_SOURCE=200809L -fopenmp -pthread -Isrc
LDFLAGS  = -fopenmp -pthread
LDLIBS   = -lm

BUILD    = build
LIB_SRC  = src/pgm_io.c src/kernels.c src/serial.c src/pthreads_impl.c src/openmp_impl.c
LIB_OBJ  = $(LIB_SRC:src/%.c=$(BUILD)/%.o)

CONVOLVE = $(BUILD)/convolve
TESTBIN  = $(BUILD)/test_small
GENBIN   = $(BUILD)/gen_image
IMG2000  = data/synthetic_2000.pgm
IMG4000  = data/synthetic_4000.pgm

.PHONY: all test bench plot loc images clean distclean help

all: $(CONVOLVE) $(GENBIN)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: src/%.c src/common.h | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(CONVOLVE): $(BUILD)/main.o $(LIB_OBJ)
	$(CC) $(LDFLAGS) $^ -o $@ $(LDLIBS)

$(GENBIN): scripts/gen_image.c $(BUILD)/pgm_io.o src/common.h | $(BUILD)
	$(CC) $(CFLAGS) scripts/gen_image.c $(BUILD)/pgm_io.o -o $@ $(LDLIBS)

$(TESTBIN): tests/test_small.c $(LIB_OBJ) src/common.h | $(BUILD)
	$(CC) $(CFLAGS) tests/test_small.c $(LIB_OBJ) -o $@ $(LDFLAGS) $(LDLIBS)

# Citra uji sintetis (reproducible, srand(42))
$(IMG2000): $(GENBIN)
	mkdir -p data
	$(GENBIN) 2000 2000 $@

$(IMG4000): $(GENBIN)
	mkdir -p data
	$(GENBIN) 4000 4000 $@

images: $(IMG2000) $(IMG4000)

test: $(TESTBIN)
	$(TESTBIN)

bench: all images
	bash scripts/benchmark.sh

plot:
	python3 scripts/plot.py

loc:
	bash scripts/loc.sh

clean:
	rm -rf $(BUILD) out.pgm data/out.pgm test_tmp_*.pgm

distclean: clean
	rm -rf data results/*.csv results/*.png

help:
	@echo "Target: all | test | images | bench | plot | loc | clean | distclean"
