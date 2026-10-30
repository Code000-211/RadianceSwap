# tools/swap-proto/Makefile
# Standalone benchmark build for ROCm gfx1201 (R9700)
# Not hooked into main CMake build.

HIPCC ?= /opt/rocm/bin/hipcc
ARCH ?= gfx1201
CXXFLAGS ?= -O3 --offload-arch=$(ARCH) -Wall
LDFLAGS ?= -lrocblas

TARGET = measure_swap

all: $(TARGET)

$(TARGET): measure_swap.cpp
	$(HIPCC) $(CXXFLAGS) $< $(LDFLAGS) -o $@

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
