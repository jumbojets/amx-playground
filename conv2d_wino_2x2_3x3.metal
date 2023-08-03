#include <metal_stdlib>
#include <metal_matrix>

using namespace metal;

// fs: F, C, 3, 3
// out: F, C, 4, 4
kernel void filter_transform(device float *out, device const float *fs, uint id [[thread_position_in_grid]]) { 
  float3x3 alu0;
  for (uint i = 0; i < 3; i++) {
    for (uint j = 0; j < 3; j++) {
      alu0[j][i] = fs[id*9+i+j*3]; // tranpose on load
    }
  }
  float4x3 alu1 = float4x3(alu0[0], 0.5*(alu0[0]+alu0[1]+alu0[2]), 0.5*(alu0[0]-alu0[1]+alu0[2]), alu0[2]);
  float3x4 alu2 = transpose(alu1);
  float4x4 alu3 = float4x4(alu2[0], 0.5*(alu2[0]+alu2[1]+alu2[2]), 0.5*(alu2[0]-alu2[1]+alu2[2]), alu2[2]);
  for (uint i = 0; i < 4; i++) {
    for (uint j = 0; j < 4; j++) {
      out[id*16+i+4*j] = alu3[i][j];
    }
  }
}

float4x4 load_float4x4(device const float *data, uint D) {
  float4x4 result;
  for (uint i = 0; i < 4; i++) {
    for (uint j = 0; j < 4; j++) {
      result[i][j] = data[i+j*D];
    }
  }
}

void write_float2x2(device float *out, uint D, float2x2 data) {
  for (uint i = 0; i < 2; i++) {
    for (uint j = 0; j < 2; j++) {
      out[i+D*j] = data[i][j];
    }
  }
}

// ims: N, C, HW, HW
// fs:  F, C, 4, 4
// out: N, F, HW-2, HW-2
kernel void conv(device float *out,
                 device const float *ims,
                 device const float *fs,
                 uint2 local_size [[threads_per_threadgroup]],
                 uint2 gid [[threadgroup_position_in_grid]],
                 uint2 lid [[thread_position_in_threadgroup]]) {
  // for now, lets assume N = F = 1
  uint idx = gid.x*local_size.x + lid.x;
  uint idy = gid.y*local_size.y + lid.y;

  ims += idx*16 + idy*16*$HW;
  fs += 0; // This could get tricky with multiple filters
  out += idx*16 + idy*16*($HW-2);

  bool is_last_col = (idx+1)*16 == $HW;
  bool is_last_row = (idx+1)*16 == $HW;

  float4x4 acc[8][8];
  for (uint i = 0; i < 8; i++) {
    for (uint j = 0; j < 8; j++) {
      acc[i][j] = float4x4(0.0);
    }
  }

  for (uint c = 0; c < $C; c++) {
    // TODO: handle multiple filters
    float4x4 f = load_float4x4(fs+c*16, 4);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // input transformation, fused prod and acc with filter
    for (uint i = 0; i < 16; i+=2) {
      if (i == 15 && is_last_col) {
        break;
      }

      for (uint j = 0; j < 16; j+=2) {
        if (j == 15 && is_last_row) {
          break;
        }
        float4x4 alu0 = load_float4x4(ims+i+j*$HW, $HW);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float4x4 alu1;
        alu1[0] =  alu0[0] - alu0[2];
        alu1[1] =  alu0[1] + alu0[2];
        alu1[2] = -alu0[1] + alu0[2];
        alu1[3] =  alu0[1] - alu0[3];
        alu0 = transpose(alu1);
        alu1[0] =  alu0[0] - alu0[2];
        alu1[1] =  alu0[1] + alu0[2];
        alu1[2] = -alu0[1] + alu0[2];
        alu1[3] =  alu0[1] - alu0[3];
        for (uint k = 0; k < 4; k++) {
          acc[i][j][k] = fma(alu1[k], f[k], acc[i][j][k]);
        }
      }
    }
  }

  // output transformation
  for (uint i = 0; i < 16; i+=2) {
    if (i == 15 && is_last_col) {
      break;
    } 

    for (uint j = 0; j < 16; j+=2) {
      if (j == 15 && is_last_row) {
        break;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float4x4 alu0 = acc[i][j];
      float2x4 alu1 = float2x4(alu0[0] - alu0[3], -alu0[1]);
      float4x2 alu2 = transpose(alu1);
      float2x2 alu3 = float2x2(alu2[0] - alu2[3], -alu2[1]);
      write_float2x2(out+i+j*($HW-2), $HW-2, alu3);
    }
  }
}
