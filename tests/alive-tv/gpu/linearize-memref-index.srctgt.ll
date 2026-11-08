; convert-gpu-to-air linearizes the indices of a memref access of static
; shape, non-negative static strides and offset, whose last element is
; within 2^32 of the base, in 32 bits (linearize-indices): each index is
; below its dimension's size (an access out of bounds is undefined: the
; assumes), so offset + sum(index_k * stride_k) is at most the last
; element's offset and its 32-bit computation is exact. Here
; memref<4x8x16xf32, strided<[256, 16, 1], offset: 7>> (last element 1030).

define i64 @src(i64 %i, i64 %j, i64 %k) {
  %ib = icmp ult i64 %i, 4
  call void @llvm.assume(i1 %ib)
  %jb = icmp ult i64 %j, 8
  call void @llvm.assume(i1 %jb)
  %kb = icmp ult i64 %k, 16
  call void @llvm.assume(i1 %kb)
  %a = mul i64 %i, 256
  %b = mul i64 %j, 16
  %s0 = add i64 7, %a
  %s1 = add i64 %s0, %b
  %s2 = add i64 %s1, %k
  ret i64 %s2
}

define i64 @tgt(i64 %i, i64 %j, i64 %k) {
  %i32 = trunc i64 %i to i32
  %j32 = trunc i64 %j to i32
  %k32 = trunc i64 %k to i32
  %a = mul i32 %i32, 256
  %b = mul i32 %j32, 16
  %s0 = add i32 7, %a
  %s1 = add i32 %s0, %b
  %s2 = add i32 %s1, %k32
  %r = zext i32 %s2 to i64
  ret i64 %r
}

declare void @llvm.assume(i1)
