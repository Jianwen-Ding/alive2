; AGXCodeGenPrepare (L15-E294): the floor modulo r < 0 ? r + c : r of
; r = srem x, c (c > 0, here 7; MLIR's affine mod) is x - c * floor(x / c),
; the floor quotient s ^ ((x ^ s) / c) divided unsigned (E292), x frozen
; (used three times).

define i32 @src(i32 %x) {
  %r = srem i32 %x, 7
  %n = icmp slt i32 %r, 0
  %a = add nsw i32 %r, 7
  %m = select i1 %n, i32 %a, i32 %r
  ret i32 %m
}

define i32 @tgt(i32 %x0) {
  %x = freeze i32 %x0
  %s = ashr i32 %x, 31
  %v = xor i32 %x, %s
  %v1 = and i32 %v, 2147483647
  %q = udiv i32 %v1, 7
  %f = xor i32 %q, %s
  %fc = mul i32 %f, 7
  %m = sub i32 %x, %fc
  ret i32 %m
}
