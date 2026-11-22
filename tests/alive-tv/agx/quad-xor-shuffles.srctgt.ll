; AGXCodeGenPrepare quadXorShuffles (L15-E304): a SIMD-group xor shuffle by
; 1, 2 or 3 reads a lane of the thread's quad, so it is the quad-group's.
declare i32 @llvm.agx.simd.shuffle.xor.i32(i32, i32)
declare i32 @llvm.agx.quad.shuffle.xor.i32(i32, i32)

define void @src(ptr addrspace(1) %out, i32 %v) "air-kernel" {
  %a = call i32 @llvm.agx.simd.shuffle.xor.i32(i32 %v, i32 1)
  %b = call i32 @llvm.agx.simd.shuffle.xor.i32(i32 %a, i32 2)
  %c = call i32 @llvm.agx.simd.shuffle.xor.i32(i32 %b, i32 3)
  store i32 %c, ptr addrspace(1) %out, align 4
  ret void
}

define void @tgt(ptr addrspace(1) %out, i32 %v) "air-kernel" {
  %a = call i32 @llvm.agx.quad.shuffle.xor.i32(i32 %v, i32 1)
  %b = call i32 @llvm.agx.quad.shuffle.xor.i32(i32 %a, i32 2)
  %c = call i32 @llvm.agx.quad.shuffle.xor.i32(i32 %b, i32 3)
  store i32 %c, ptr addrspace(1) %out, align 4
  ret void
}
