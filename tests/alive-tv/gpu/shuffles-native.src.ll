; MLIR's gpu.shuffle up / down / xor by constant offsets, as convert-gpu-to-air
; lowered them before (the lane arithmetic and an index shuffle), and as it
; lowers them now (shuffles-native.tgt.ll: AIR's shuffle_up, shuffle_down and
; shuffle_xor, by MSL's definition of them).
declare i32 @air.thread_index_in_simdgroup() nounwind willreturn memory(none)
declare i32 @air.simd_shuffle.u.i32(i32, i16) convergent nounwind willreturn memory(none)
define void @k(ptr addrspace(1) %out, i32 %v) "air-kernel" {
  %lane0 = call i32 @air.thread_index_in_simdgroup()
  %add0 = add i32 %lane0, 32
  %end0 = and i32 %add0, -32
  %src0 = sub i32 %lane0, 2
  %start0 = and i32 %lane0, -32
  %ge0 = icmp sge i32 %src0, %start0
  %lt0 = icmp slt i32 %src0, %end0
  %valid0 = and i1 %ge0, %lt0
  %sel0 = select i1 %valid0, i32 %src0, i32 %lane0
  %l160 = trunc i32 %sel0 to i16
  %sh0 = call i32 @air.simd_shuffle.u.i32(i32 %v, i16 %l160)
  %r0 = select i1 %valid0, i32 %sh0, i32 %v
  %lane1 = call i32 @air.thread_index_in_simdgroup()
  %add1 = add i32 %lane1, 32
  %end1 = and i32 %add1, -32
  %src1 = add i32 %lane1, 2
  %start1 = and i32 %lane1, -32
  %ge1 = icmp sge i32 %src1, %start1
  %lt1 = icmp slt i32 %src1, %end1
  %valid1 = and i1 %ge1, %lt1
  %sel1 = select i1 %valid1, i32 %src1, i32 %lane1
  %l161 = trunc i32 %sel1 to i16
  %sh1 = call i32 @air.simd_shuffle.u.i32(i32 %v, i16 %l161)
  %r1 = select i1 %valid1, i32 %sh1, i32 %v
  %lane2 = call i32 @air.thread_index_in_simdgroup()
  %add2 = add i32 %lane2, 8
  %end2 = and i32 %add2, -8
  %src2 = xor i32 %lane2, 2
  %start2 = and i32 %lane2, -8
  %ge2 = icmp sge i32 %src2, %start2
  %lt2 = icmp slt i32 %src2, %end2
  %valid2 = and i1 %ge2, %lt2
  %sel2 = select i1 %valid2, i32 %src2, i32 %lane2
  %l162 = trunc i32 %sel2 to i16
  %sh2 = call i32 @air.simd_shuffle.u.i32(i32 %v, i16 %l162)
  %r2 = select i1 %valid2, i32 %sh2, i32 %v
  store i32 %r0, ptr addrspace(1) %out, align 4
  %p1 = getelementptr inbounds nuw i32, ptr addrspace(1) %out, i64 1
  store i32 %r1, ptr addrspace(1) %p1, align 4
  %p2 = getelementptr inbounds nuw i32, ptr addrspace(1) %out, i64 2
  store i32 %r2, ptr addrspace(1) %p2, align 4
  ret void
}
