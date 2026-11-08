; agx-lower-air on atomics-agx.src.ll.
target datalayout = "e-m:e-p:64:64-p1:64:64-p3:32:32-p4:64:64-p5:32:32-i1:8-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-v16:16-v32:32-v64:64-v96:32-v128:128-n16:32:64-S32-A5-G1"
target triple = "agx"

@tg = internal addrspace(3) global [64 x i32] poison, align 4

define agx_kernel void @k(ptr addrspace(1) %ctr, ptr addrspace(1) %out, ptr addrspace(4) noalias noundef nonnull readonly byref(i32) align 4 dereferenceable(4) %v) {
entry:
  %v.val = load i32, ptr addrspace(4) %v, align 4
  %cmpxchg.expected = alloca i32, align 4, addrspace(5)
  %old1 = atomicrmw add ptr addrspace(1) %ctr, i32 %v.val monotonic, align 4
  %m2 = atomicrmw umax ptr addrspace(3) @tg, i32 %old1 syncscope("workgroup") seq_cst, align 4
  store i32 %m2, ptr addrspace(5) %cmpxchg.expected, align 4
  %0 = load i32, ptr addrspace(5) %cmpxchg.expected, align 4
  %1 = cmpxchg ptr addrspace(1) %ctr, i32 %0, i32 0 acq_rel monotonic, align 4
  %2 = extractvalue { i32, i1 } %1, 0
  store i32 %2, ptr addrspace(5) %cmpxchg.expected, align 4
  %3 = icmp eq i32 %2, %m2
  %4 = insertvalue { i32, i1 } poison, i32 %2, 0
  %5 = insertvalue { i32, i1 } %4, i1 %3, 1
  %f = extractvalue { i32, i1 } %5, 0
  store i32 %f, ptr addrspace(1) %out, align 4
  ret void
}
