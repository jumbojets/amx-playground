// Benchmarked on Apple M1 Max (Gen 7)
// https://github.com/philipturner/metal-benchmarks
// 32 cores, 1.286 Ghz, 10617 GFLOPS32, 5308 GIPS, 512K L2, 48M L3
// each core, 64 KB, 3072 max threads, 64 KB SM, 12 KB i-cache, 8 KB d-cache
// 16 shared banks of size 4 B
// 128 B cache line
// Each core

// IDEAS
// for large channel size, fold across multiple threads in an workgroup
// instead of allocating $F,$C on stack/sm, try to make it blocks that can be treated as params
// take that even furthur: nothing inside of kernel should depend on input size, just parameters

#include <metal_stdlib>
#include <metal_matrix>

using namespace metal;

// AHWxAHW is the shape of the acc matrix
#define AHW ($BHW/2)

// fs: F, C, 3, 3
// out: F, C, 4, 4
kernel void filter_transform(
    device       float *out, 
    device const float *fs,
    uint id [[thread_position_in_grid]]) {
  float3x3 alu0;
  for (uint i = 0; i < 3; i++) {
    for (uint j = 0; j < 3; j++) {
      alu0[j][i] = fs[(id*9)+i+(j*3)]; // transpose on load
    }
  }
  float4x3 alu1 = float4x3(alu0[0], 0.5*(alu0[0]+alu0[1]+alu0[2]), 0.5*(alu0[0]-alu0[1]+alu0[2]), alu0[2]);
  float3x4 alu2 = transpose(alu1);
  float4x4 alu3 = float4x4(alu2[0], 0.5*(alu2[0]+alu2[1]+alu2[2]), 0.5*(alu2[0]-alu2[1]+alu2[2]), alu2[2]);
  for (uint i = 0; i < 4; i++) {
    for (uint j = 0; j < 4; j++) {
      out[(id*16)+i+(j*4)] = alu3[i][j];
    }
  }
}

float4x4 load_float4x4(device const float *data, uint W) {
  float4x4 result;
  for (uint i = 0; i < 4; i++) {
    for (uint j = 0; j < 4; j++) {
      result[i][j] = data[i+(j*W)];
    }
  }
  return result;
}

void write_float2x2(device float *out, uint W, float2x2 data) {
  for (uint i = 0; i < 2; i++) {
    for (uint j = 0; j < 2; j++) {
      out[i+(j*W)] = data[i][j];
    }
  }
}

// ims: N, C, HW, HW
// fs:  F, C, 4, 4
// out: N, F, HW-2, HW-2
kernel void conv(
    device       float *out,
    device const float *ims,
    device const float *fs,
    uint3 local_size [[threads_per_threadgroup]],
    uint3        gid [[threadgroup_position_in_grid]],
    uint3        lid [[thread_position_in_threadgroup]]) {
  uint idx = gid.x*local_size.x + lid.x;
  uint idy = gid.y*local_size.y + lid.y;
  uint idz = gid.z*local_size.z + lid.z;

  ims += (idx*$BHW)+(idy*$BHW*$HW)+(idz*$HW*$HW*$C);
  out += (idx*$BHW)+(idy*$BHW*($HW-2))+(idz*($HW-2)*($HW-2)*$F);

  const bool is_last_col = (idx+1)*$BHW == $HW;
  const bool is_last_row = (idy+1)*$BHW == $HW;

  // threadgroup float4x4 filters[$C][$F];
  // for (uint i = lid.x; i < $F*$C; i+=local_size.x) {
  //   uint c = i % $C;
  //   uint f = i / $C;
  //   filters[c][f] = load_float4x4(fs+(c*16)+(f*16*$C), 4);
  // }
  // threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint f = 0; f < $F; f++) {
    float4x4 acc[AHW][AHW];
    for (uint i = 0; i < AHW; i++) {
      for (uint j = 0; j < AHW; j++) {
        acc[i][j] = float4x4(0.0);
      }
    }

    for (uint c = 0; c < $C; c++) {
      float4x4 filter = load_float4x4(fs+(c*16)+(f*16*$C), 4);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      // float4x4 filter = filters[c][f];

      // input transformation, fused prod and acc with filter
      for (uint i = 0; i < AHW; i++) {
        if (i == (AHW-1) && is_last_col) {
          break;
        }

        for (uint j = 0; j < AHW; j++) {
          if (j == (AHW-1) && is_last_row) {
            break;
          }
          float4x4 alu0 = load_float4x4(ims+(i*2)+(j*2*$HW)+(c*$HW*$HW), $HW);
          threadgroup_barrier(mem_flags::mem_threadgroup);
          float4x4 alu1 = transpose(alu0);
          alu0 = float4x4(alu1[0]-alu1[2], alu1[1]+alu1[2], -alu1[1]+alu1[2], alu1[1]-alu1[3]);
          alu1 = transpose(alu0);
          alu0 = float4x4(alu1[0]-alu1[2], alu1[1]+alu1[2], -alu1[1]+alu1[2], alu1[1]-alu1[3]);
          for (uint k = 0; k < 4; k++) {
            acc[i][j][k] = fma(alu0[k], filter[k], acc[i][j][k]);
          }
        }
      }
    }

    // output transformation
    for (uint i = 0; i < AHW; i++) {
      if (i == (AHW-1) && is_last_col) {
        break;
      }

      for (uint j = 0; j < AHW; j++) {
        if (j == (AHW-1) && is_last_row) {
          break;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float4x4 alu0 = transpose(acc[i][j]);
        float2x4 alu1 = float2x4(alu0[0]+alu0[1]+alu0[2], alu0[1]-alu0[2]-alu0[3]);
        float4x2 alu2 = transpose(alu1);
        float2x2 alu3 = float2x2(alu2[0]+alu2[1]+alu2[2], alu2[1]-alu2[2]-alu2[3]);
        write_float2x2(out+(i*2)+(j*2*($HW-2))+(f*($HW-2)*($HW-2)), $HW-2, alu3);
      }
    }
  }
}

#define RLOOP $RLOOP
#define RLCL ($C/RLOOP)

// ims: N, C, HW, HW
// fs:  F, C, 4, 4
// out: N, F, HW-2, HW-2
kernel void conv2(
    device       float *out,
    device const float *ims,
    device const float *fs,
    uint3 local_size [[threads_per_threadgroup]],
    uint3        gid [[threadgroup_position_in_grid]],
    uint3        lid [[thread_position_in_threadgroup]]) {
  uint idx = gid.x*local_size.x + lid.x;
  uint idy = gid.y*local_size.y + lid.y;
  uint idz = gid.z*local_size.z;
  uint rlcl_idx = lid.z;
  
  ims += (idx*$BHW)+(idy*$BHW*$HW)+(idz*$HW*$HW*$C);
  out += (idx*$BHW)+(idy*$BHW*($HW-2))+(idz*($HW-2)*($HW-2)*$F);
  fs += rlcl_idx*16*RLOOP; // TODO: get a filter offset into channel

  const bool is_last_col = (idx+1)*$BHW == $HW;
  const bool is_last_row = (idy+1)*$BHW == $HW;

  for (uint f = 0; f < $F; f++) { // filter

    threadgroup float4x4 partial_accs[RLCL][AHW][AHW];
    for (int i = 0; i < AHW; i++) {
      for (int j = 0; j < AHW; j++) {
        partial_accs[rlcl_idx][i][j] = float4x4(0.0);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint c = 0; c < RLOOP; c++) { // channel reduce

      float4x4 filter = load_float4x4(fs+(c*16)+(f*16*$C), 4);
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // input transformation, fused prod and acc with filter
      for (uint i = 0; i < AHW; i++) {
        if (i == (AHW-1) && is_last_col) {
          break;
        }

        for (uint j = 0; j < AHW; j++) {
          if (j == (AHW-1) && is_last_row) {
            break;
          }

          float4x4 alu0 = load_float4x4(ims+(i*2)+(j*2*$HW)+(c*$HW*$HW), $HW);
          threadgroup_barrier(mem_flags::mem_threadgroup);
          float4x4 alu1 = transpose(alu0);
          alu0 = float4x4(alu1[0]-alu1[2], alu1[1]+alu1[2], -alu1[1]+alu1[2], alu1[1]-alu1[3]);
          alu1 = transpose(alu0);
          alu0 = float4x4(alu1[0]-alu1[2], alu1[1]+alu1[2], -alu1[1]+alu1[2], alu1[1]-alu1[3]);
          for (uint k = 0; k < 4; k++) {
            partial_accs[rlcl_idx][i][j][k] = fma(alu0[k], filter[k], partial_accs[rlcl_idx][i][j][k]);
          }
        }
      }
    }

    for (uint x = rlcl_idx; x < AHW*AHW; x += local_size.z) {
      
      threadgroup_barrier(mem_flags::mem_threadgroup); 
      
      uint i = x % AHW;
      uint j = x / AHW;
      if ((i == (AHW-1) && is_last_col) || (j == (AHW-1) && is_last_row)) {
        continue;
      }
      
      float4x4 acc = float4x4(0.0);
      for (uint k = 0; k < RLCL; k++) {
        acc += partial_accs[k][i][j];
      }

      // output transformation 
      float4x4 alu0 = transpose(acc);
      float2x4 alu1 = float2x4(alu0[0]+alu0[1]+alu0[2], alu0[1]-alu0[2]-alu0[3]);
      float4x2 alu2 = transpose(alu1);
      float2x2 alu3 = float2x2(alu2[0]+alu2[1]+alu2[2], alu2[1]-alu2[2]-alu2[3]);

      write_float2x2(out+(i*2)+(j*2*($HW-2))+(f*($HW-2)*($HW-2)), $HW-2, alu3);
    }
  }
}
