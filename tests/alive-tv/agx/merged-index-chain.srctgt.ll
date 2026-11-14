; AGXStraightLineSRPass (L15-E303), the address arithmetic: b + 4x + 4y with
; x, y non-negative and x + y below 2^32 (here x < 2^20, y < 2^14, as the
; thread bounds give) is b + 4 * zext(trunc x + trunc y).

define i64 @src(i64 %b, i32 %p, i32 %q) {
  %pm = and i32 %p, 1048575
  %qm = and i32 %q, 16383
  %x = zext i32 %pm to i64
  %y = zext i32 %qm to i64
  %x4 = shl i64 %x, 2
  %y4 = shl i64 %y, 2
  %t = add i64 %b, %x4
  %g = add i64 %t, %y4
  ret i64 %g
}

define i64 @tgt(i64 %b, i32 %p, i32 %q) {
  %pm = and i32 %p, 1048575
  %qm = and i32 %q, 16383
  %s = add nuw i32 %pm, %qm
  %z = zext i32 %s to i64
  %z4 = shl i64 %z, 2
  %g = add i64 %b, %z4
  ret i64 %g
}
