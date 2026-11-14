; AGXStraightLineSRPass (L15-E291): gep i8 (gep [4 x i8], b, zext t), c, c / 4
; past the device accesses' 16-bit immediate index, is gep [4 x i8], b,
; zext(t + c / 4), the 32-bit add not wrapping by t's range (here t < 2^16,
; as a thread position bounded by the grid).

define ptr @src(ptr %b, i32 %t0) {
  %t = and i32 %t0, 65535
  %z = zext i32 %t to i64
  %p = getelementptr inbounds nuw [4 x i8], ptr %b, i64 %z
  %g = getelementptr inbounds i8, ptr %p, i64 262144
  ret ptr %g
}

define ptr @tgt(ptr %b, i32 %t0) {
  %t = and i32 %t0, 65535
  %i = add nuw i32 %t, 65536
  %z = zext i32 %i to i64
  %g = getelementptr inbounds [4 x i8], ptr %b, i64 %z
  ret ptr %g
}
