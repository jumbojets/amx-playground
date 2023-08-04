from test import sample

if __name__ == '__main__':
  # TODO: print FLOPS
  print("scaling N")
  for i in range(10):
    mtl_t, mps_t = sample(2**i, 128, 4, 16)
    print(f"N={2**i:3d}, HW=128, C=4, F=16:\t\tmtl {mtl_t:.3f}s, mps: {mps_t:.3f}s, {mps_t/mtl_t:.3f}x")

  print("\nscaling HW")
  for i in range(10):
    mtl_t, mps_t = sample(32, 32*(i+1), 4, 16)
    print(f"N=32, HW={32*i:3d}, C=4, F=16:\t\tmtl {mtl_t:.3f}s, mps: {mps_t:.3f}s, {mps_t/mtl_t:.3f}x")

  print("\nscaling C")
  for i in range(10):
    try: mtl_t, mps_t = sample(32, 128, 2**i, 16) # ???
    except: continue
    print(f"N=32, HW=128, C={2**i:3d}, F=16:\t\tmtl {mtl_t:.3f}s, mps: {mps_t:.3f}s, {mps_t/mtl_t:.3f}x")

  print("\nscaling F")
  for i in range(10):
    mtl_t, mps_t = sample(32, 128, 4, 2**i)
    print(f"N=32, HW=128, C=4, F={2**i:3d}:\t\tmtl {mtl_t:.3f}s, mps: {mps_t:.3f}s, {mps_t/mtl_t:.3f}x")
