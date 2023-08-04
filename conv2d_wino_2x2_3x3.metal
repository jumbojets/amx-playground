#include <metal_stdlib>
#include <metal_matrix>

using namespace metal;

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

float4x4 load_float4x4(device const float *data, uint D) {
  float4x4 result;
  for (uint i = 0; i < 4; i++) {
    for (uint j = 0; j < 4; j++) {
      result[i][j] = data[i+(j*D)];
    }
  }
  return result;
}

void write_float2x2(device float *out, uint D, float2x2 data) {
  for (uint i = 0; i < 2; i++) {
    for (uint j = 0; j < 2; j++) {
      out[i+(j*D)] = data[i][j];
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

  ims += (idx*16)+(idy*16*$HW)+(idz*$HW*$HW*$C);
  out += (idx*16)+(idy*16*($HW-2))+(idz*($HW-2)*($HW-2)*$F);

  const bool is_last_col = (idx+1)*16 == $HW;
  const bool is_last_row = (idy+1)*16 == $HW;

  for (uint f = 0; f < $F; f++) {
    float4x4 acc[8][8];
    for (uint i = 0; i < 8; i++) {
      for (uint j = 0; j < 8; j++) {
        acc[i][j] = float4x4(0.0);
      }
    }

    for (uint c = 0; c < $C; c++) {
      float4x4 filter = load_float4x4(fs+(c*16)+(f*16*$C), 4);

      threadgroup_barrier(mem_flags::mem_threadgroup);

      // input transformation, fused prod and acc with filter
      for (uint i = 0; i < 8; i++) {
        if (i == 7 && is_last_col) {
          break;
        }

        for (uint j = 0; j < 8; j++) {
          if (j == 7 && is_last_row) {
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
    for (uint i = 0; i < 8; i++) {
      if (i == 7 && is_last_col) {
        break;
      }

      for (uint j = 0; j < 8; j++) {
        if (j == 7 && is_last_row) {
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
