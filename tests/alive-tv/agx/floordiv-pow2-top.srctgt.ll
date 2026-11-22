; As floordiv-pow2, by 2^30.
define i32 @src(i32 %x) {
  %q = sdiv i32 %x, 1073741824
  %m = mul i32 %q, 1073741824
  %ne = icmp ne i32 %x, %m
  %neg = icmp slt i32 %x, 0
  %c = and i1 %ne, %neg
  %q1 = add i32 %q, -1
  %r = select i1 %c, i32 %q1, i32 %q
  ret i32 %r
}

define i32 @tgt(i32 %x) {
  %r = ashr i32 %x, 30
  ret i32 %r
}
