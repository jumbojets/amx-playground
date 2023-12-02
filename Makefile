build_dir := ./build
CC := clang
CFLAGS := -O3
TARGETS := matmul play perf

.PHONY: $(TARGETS) clean

help:
	@echo "Available targets: $(TARGETS)"

$(TARGETS): %: $(build_dir)/%.out
	./$(build_dir)/$@

$(build_dir)/%.out: %.c amx.h | $(build_dir)
	$(CC) $(CFLAGS) -o $@ $<

$(build_dir):
	mkdir -p $(build_dir)

clean:
	rm -rf $(build_dir)/*
