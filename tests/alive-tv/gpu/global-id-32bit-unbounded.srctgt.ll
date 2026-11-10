; Without the bound the 32-bit register is not the index: a grid dimension of
; 2^16 threads or more (2^32 at full width) wraps it.
define i32 @src(i16 %block, i16 %dim, i16 %tid) {
  %b = zext i16 %block to i32
  %d = zext i16 %dim to i32
  %t = zext i16 %tid to i32
  %m = mul i32 %b, %d
  %g = add i32 %m, %t
  %ok = icmp ult i32 %g, 131072
  call void @llvm.assume(i1 %ok)
  ret i32 %g
}

define i32 @tgt(i16 %block, i16 %dim, i16 %tid) {
  %m = mul i16 %block, %dim
  %g = add i16 %m, %tid
  %z = zext i16 %g to i32
  ret i32 %z
}

declare void @llvm.assume(i1)
; ERROR: Value mismatch
