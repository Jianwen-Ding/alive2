; agx-lower-air on calls-agx.src.ll.
target datalayout = "e-m:e-p:64:64-p1:64:64-p3:32:32-p4:64:64-p5:32:32-i1:8-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-v16:16-v32:32-v64:64-v96:32-v128:128-n16:32:64-S32-A5-G1"
target triple = "agx"

define agx_kernel void @k(ptr addrspace(1) %p) {
entry:
  %v = load float, ptr addrspace(1) %p, align 4
  %y.i = fmul float %v, 3.000000e+00
  store float %y.i, ptr addrspace(1) %p, align 4
  ret void
}
