; The OpenCL kernel of opencl-air.src.ll compared with its AGX IR: the barrier
; between threadgroup-scope fences.
target triple = "spir64"
declare i64 @_Z13get_global_idj(i32)
declare i64 @_Z12get_local_idj(i32)
declare i64 @_Z14get_local_sizej(i32)
declare i64 @_Z12get_group_idj(i32)
declare void @_Z7barrierj(i32)

@tg = internal addrspace(3) global [64 x i32] poison, align 4

define spir_kernel void @k(ptr addrspace(1) %out) {
  %g = call i64 @_Z13get_global_idj(i32 0)
  %l = call i64 @_Z12get_local_idj(i32 1)
  %s = call i64 @_Z14get_local_sizej(i32 2)
  %b = call i64 @_Z12get_group_idj(i32 0)
  %g32 = trunc i64 %g to i32
  %slot = getelementptr i32, ptr addrspace(3) @tg, i64 %l
  store i32 %g32, ptr addrspace(3) %slot
  call void @_Z7barrierj(i32 1)
  %v = load i32, ptr addrspace(3) @tg
  %s32 = trunc i64 %s to i32
  %b32 = trunc i64 %b to i32
  %sum = add i32 %v, %s32
  %sum2 = add i32 %sum, %b32
  %p = getelementptr i32, ptr addrspace(1) %out, i64 %g
  store i32 %sum2, ptr addrspace(1) %p
  ret void
}
