build_dir = ./build

play: play.out
	${build_dir}/play

play.out: play.c amx.h | ${build_dir}
	clang -O3 -o ${build_dir}/play play.c

matmul: matmul.out
	${build_dir}/matmul

matmul.out: matmul.c amx.h | ${build_dir}
	clang -O3 -o ${build_dir}/matmul matmul.c

perf: perf.out
	${build_dir}/perf

perf.out: perf.c amx.h | ${build_dir}
	clang -O3 -o ${build_dir}/perf perf.c

clean:
	rm -rf $(build_dir)/*

${build_dir}:
	mkdir ${build_dir}
