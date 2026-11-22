; Each thread of a CTA writes its value to shared memory, and after the
; barrier reads its mirror's.
target triple = "nvptx64-nvidia-cuda"

@sh = internal addrspace(3) global [256 x i32] undef, align 4

define ptx_kernel void @shared(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %t = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %t64 = zext i32 %t to i64
  %pi = getelementptr i32, ptr addrspace(1) %in, i64 %t64
  %v = load i32, ptr addrspace(1) %pi, align 4
  %ps = getelementptr [256 x i32], ptr addrspace(3) @sh, i64 0, i64 %t64
  store i32 %v, ptr addrspace(3) %ps, align 4
  call void @llvm.nvvm.barrier0()
  %r = sub i32 255, %t
  %r64 = zext i32 %r to i64
  %pr = getelementptr [256 x i32], ptr addrspace(3) @sh, i64 0, i64 %r64
  %w = load i32, ptr addrspace(3) %pr, align 4
  %po = getelementptr i32, ptr addrspace(1) %out, i64 %t64
  store i32 %w, ptr addrspace(1) %po, align 4
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare void @llvm.nvvm.barrier0()
