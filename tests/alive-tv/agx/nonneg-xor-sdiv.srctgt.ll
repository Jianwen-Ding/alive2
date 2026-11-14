; AGXCodeGenPrepare (L15-E292): sdiv (x ^ (x >>s 31)), c, c > 0 (here 7), is
; udiv of the dividend's low 31 bits: the dividend is never negative.

define i32 @src(i32 %x) {
  %s = ashr i32 %x, 31
  %v = xor i32 %x, %s
  %q = sdiv i32 %v, 7
  ret i32 %q
}

define i32 @tgt(i32 %x) {
  %s = ashr i32 %x, 31
  %v = xor i32 %x, %s
  %m = and i32 %v, 2147483647
  %q = udiv i32 %m, 7
  ret i32 %q
}
