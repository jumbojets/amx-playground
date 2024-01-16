# amx-playpen

This subdirectory serves as a place where I was experimenting with the Apple M1's [AMX coprocessor](https://github.com/corsix/amx).

The most notable file is matmul.c and can be run with `make matmul`. This implementation uses the AMX coprocessor to compute the matrix multiplication of 2 `NxN` matrices (note `N` must be a multiple of 32). With fp16, the implementation currently achieves about 2870 GFLOP/s peak on a single core.

There are several key improvements that can be made.

- [x] use other half of the z-register. only half of the z register pool is being utilized
- [ ] be more cache aware. this implementation does not scale above N=1024 very well, probably due to cache blow up
- [ ] load consecutive 128 bytes into consecutive AMX registers, would cut down on the number of loads. not sure if this has practical importance for this example
- [ ] multithreaded
