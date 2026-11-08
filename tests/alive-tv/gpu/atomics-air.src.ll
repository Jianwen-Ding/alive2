; Atomics of device and threadgroup memory and a compare-exchange (upstream);
; its AIR calls AIR's atomic functions between fences.
@tg = internal addrspace(3) global [64 x i32] poison, align 4

define void @k(ptr addrspace(1) %ctr, ptr addrspace(1) %out, i32 %v) "air-kernel" {
  %old = atomicrmw add ptr addrspace(1) %ctr, i32 %v monotonic
  %m = atomicrmw umax ptr addrspace(3) @tg, i32 %old seq_cst
  %x = cmpxchg ptr addrspace(1) %ctr, i32 %m, i32 0 acq_rel monotonic
  %f = extractvalue { i32, i1 } %x, 0
  store i32 %f, ptr addrspace(1) %out
  ret void
}
