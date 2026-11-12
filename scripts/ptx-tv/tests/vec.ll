; Vector loads and stores, and an atomic.
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @vec(ptr addrspace(1) %p) {
  %v = load <4 x i32>, ptr addrspace(1) %p, align 16
  %e0 = extractelement <4 x i32> %v, i32 0
  %e3 = extractelement <4 x i32> %v, i32 3
  %s = add i32 %e0, %e3
  %q = getelementptr i32, ptr addrspace(1) %p, i64 4
  %w = insertelement <2 x i32> poison, i32 %s, i32 0
  %w2 = insertelement <2 x i32> %w, i32 %e3, i32 1
  store <2 x i32> %w2, ptr addrspace(1) %q, align 8
  %old = atomicrmw add ptr addrspace(1) %p, i32 1 monotonic
  %r = getelementptr i32, ptr addrspace(1) %p, i64 6
  store i32 %old, ptr addrspace(1) %r, align 4
  ret void
}
