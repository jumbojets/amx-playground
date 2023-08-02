#include <metal_stdlib>
#include <metal_matrix>

using namespace metal;

kernel void filter_transform(device float *out, device const float *fs, uint3 id [[thread_position_in_grid]]) { 
  // layout
  // fs: F, C, 3, 3
  // out: F, C, 4, 4

  if (id < {F*C}) {
    constant float3x3* tmp0 = reinterpret_cast<constant float3x3*>(fs+id*9);
    float3x4 tmp1 = float3x4(tmp0[0], 0.5*(tmp0[0]+tmp0[1]+tmp0[2]), 0.5*(tmp0[0]-tmp0[1]+tmp0[2]), tmp0[2]);
    float4x3 tmp2 = transpose(tmp1);
    float4x4 tmp3 = float3x4(tmp2[0], 0.5*(tmp2[0]+tmp2[1]+tmp2[2]), 0.5*(tmp2[0]-tmp2[1]+tmp2[2]), tmp2[2]);
    for (uint i = 0; i < 16; i++) {
      out[fs+id*16+i] = tmp3[i];
    }
  }
}

kernel void conv(device float *out, device const float *ims, device const float *fs, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) { 
  // layout
  // ims: N, C, HW, HW
  // fs:  F, C, 4, 4
  // out: N, F, HW, HW  

  // lets assume, N = F = 1 and ignore them at first.
  
  ims += ???;
  fs += ???;
  out += ???;

  float4x4 acc[8][8];
  for (uint i = 0; i < 8; i++) {
    for (uint j = 0; i < 8; j++) {
      acc[i][j] = float4x4(0);
    }
  }

  for (uint c = 0; c < {C}; c++) {
    // TODO: load filter into f
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // input transformation, fused prod and acc with filter
    for (uint i = 0; i < 8; i++) {
      for (uint j = 0; j < 8; j++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float4x4 in;
        simdgroup_load(in, ims+i*8+j*({HW}*8), {HW}, ulong2(0, 0)); // TODO: this load isnt for float4x4
        in[0] =  in[0] - in[2];
        in[0] =  in[1] + in[2];
        in[0] = -in[1] + in[2];
        in[0] =  in[1] - in[3];
        in = transpose(in);
        in[0] =  in[0] - in[2];
        in[0] =  in[1] + in[2];
        in[0] = -in[1] + in[2];
        in[0] =  in[1] - in[3];
        acc[i][j] = fma(in, f[i][j], acc[i][j]);
      }
    }
  }

  // output transformation
  for (uint i = 0; i < 8; i++) {
    for (uint j = 0; j < 8; i++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float4x4 tmp0 = acc[i][j];
      float4x2 tmp1 = float4x2(tmp0[0] - tmp0[3], -tmp0[1]);
      float2x4 tmp2 = transpose(tmp1);
      float2x2 tmp3 = float2x2(tmp2[0] - tmp2[3], -tmp2[1]);
      // TODO: write tmp3 to out
    }
  }
}
