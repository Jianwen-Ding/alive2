; AGXCodeGenPrepare lanePermutations (L15-E307): an index shuffle of an
; index of the lane that is one shuffle's source lanes is that shuffle:
; (lane + 1) & 31 the rotate up by 31, (lane & -4) | ((lane + 1) & 3) the
; quad-group's rotate up by 3, 31 - lane the xor by 31, lane ^ 1 the
; quad-group's xor by 1.
declare i32 @llvm.agx.thread.index.in.simdgroup()
declare i32 @llvm.agx.simd.shuffle.i32(i32, i32)
declare i32 @llvm.agx.simd.shuffle.rotate.up.i32(i32, i32)
declare i32 @llvm.agx.quad.shuffle.rotate.up.i32(i32, i32)
declare i32 @llvm.agx.simd.shuffle.xor.i32(i32, i32)
declare i32 @llvm.agx.quad.shuffle.xor.i32(i32, i32)

define void @src(ptr addrspace(1) %out, i32 %v) "air-kernel" {
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %a = add i32 %lane, 1
  %i0 = and i32 %a, 31
  %r0 = call i32 @llvm.agx.simd.shuffle.i32(i32 %v, i32 %i0)
  %b = and i32 %a, 3
  %c = and i32 %lane, -4
  %i1 = or i32 %b, %c
  %r1 = call i32 @llvm.agx.simd.shuffle.i32(i32 %r0, i32 %i1)
  %i2 = sub i32 31, %lane
  %r2 = call i32 @llvm.agx.simd.shuffle.i32(i32 %r1, i32 %i2)
  %i3 = xor i32 %lane, 1
  %r3 = call i32 @llvm.agx.simd.shuffle.i32(i32 %r2, i32 %i3)
  store i32 %r3, ptr addrspace(1) %out, align 4
  ret void
}

define void @tgt(ptr addrspace(1) %out, i32 %v) "air-kernel" {
  %r0 = call i32 @llvm.agx.simd.shuffle.rotate.up.i32(i32 %v, i32 31)
  %r1 = call i32 @llvm.agx.quad.shuffle.rotate.up.i32(i32 %r0, i32 3)
  %r2 = call i32 @llvm.agx.simd.shuffle.xor.i32(i32 %r1, i32 31)
  %r3 = call i32 @llvm.agx.quad.shuffle.xor.i32(i32 %r2, i32 1)
  store i32 %r3, ptr addrspace(1) %out, align 4
  ret void
}
