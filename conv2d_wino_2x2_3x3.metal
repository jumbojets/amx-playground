#include <metal_stdlib>
#include <metal_matrix>

using namespace metal;

kernel void filter_transform(device float *out, device const float *fs, uint id [[thread_position_in_grid]]) { 
  // layout
  // fs: F, C, 3, 3
  // out: F, C, 4, 4

  float3x3 tmp0; for (uint i = 0; i < 9; i++) {
    tmp0[i] = fs[id*9+i];
  }
  float4x3 tmp1 = float4x3(tmp0[0], 0.5*(tmp0[0]+tmp0[1]+tmp0[2]), 0.5*(tmp0[0]-tmp0[1]+tmp0[2]), tmp0[2]);
  float3x4 tmp2 = transpose(tmp1);
  float4x4 tmp3 = float4x4(tmp2[0], 0.5*(tmp2[0]+tmp2[1]+tmp2[2]), 0.5*(tmp2[0]-tmp2[1]+tmp2[2]), tmp2[2]);
  for (uint i = 0; i < 4; i++) {
    for (uint j = 0; j < 4; j++) {
      out[id*16+i+4*j] = tmp3[i][j];
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

kernel void conv(device float *out,
                 device const float *ims,
                 device const float *fs,
                 uint2 local_size [[threads_per_threadgroup]],
                 uint2 gid [[threadgroup_position_in_grid]],
                 uint2 lid [[thread_position_in_threadgroup]]) {
  // layout
  // ims: N, C, HW, HW
  // fs:  F, C, 4, 4
  // out: N, F, HW-2, HW-2

  // for now, lets assume N = F = 1

  uint idx = gid.x*local_size.x + lid.x;
  uint idy = gid.y*local_size.y + lid.y;

  ims += idx*16 + idy*$HW*16;
  fs += 0; // This could get tricky with multiple filters
  out += idx*16 + idy*($HW-2)*16;

  float4x4 acc[8][8];
  for (uint i = 0; i < 8; i++) {
    for (uint j = 0; j < 8; j++) {
      acc[i][j] = float4x4(0.0);
    }
  }

  for (uint c = 0; c < $C; c++) {
    // TODO: handle multiple filters
    float4x4 f = load_float4x4(fs + c*16, 4);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // input transformation, fused prod and acc with filter
    for (uint i = 0; i < 8; i++) {
      for (uint j = 0; j < 8; j++) {
        for (uint k = 0; k < 2; k++) { // overlapped block
          float4x4 in = load_float4x4(ims + i*4+k*2 + j*4*$HW, $HW);
          threadgroup_barrier(mem_flags::mem_threadgroup);
          in[0] =  in[0] - in[2];
          in[0] =  in[1] + in[2];
          in[0] = -in[1] + in[2];
          in[0] =  in[1] - in[3];
          in = transpose(in);
          in[0] =  in[0] - in[2];
          in[0] =  in[1] + in[2];
          in[0] = -in[1] + in[2];
          in[0] =  in[1] - in[3];
          for (uint k = 0; k < 4; k++) {
            acc[i][j][k] = fma(in[k], f[i][j][k], acc[i][j][k]);
          }
        }
      }
    }
  }

  // output transformation
  for (uint i = 0; i < 8; i++) {
    for (uint j = 0; j < 8; i++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float4x4 tmp0 = acc[i][j];
      float2x4 tmp1 = float2x4(tmp0[0] - tmp0[3], -tmp0[1]);
      float4x2 tmp2 = transpose(tmp1);
      float2x2 tmp3 = float2x2(tmp2[0] - tmp2[3], -tmp2[1]);
      write_float2x2(out + i*? + j*?, $HW-2, tmp3);
    }
  }
}
