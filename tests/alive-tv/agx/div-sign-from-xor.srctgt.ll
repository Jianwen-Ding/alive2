; AGX CodeGenPrepare (L15-E324): signed division's quotient sign as
; (x ^ d) <s 0, not (x <s 0) xor (d <s 0).

define i32 @src(i32 %x, i32 %d, i32 %q) {
  %xn = icmp slt i32 %x, 0
  %dn = icmp slt i32 %d, 0
  %neg = xor i1 %xn, %dn
  %nq = sub i32 0, %q
  %r = select i1 %neg, i32 %nq, i32 %q
  ret i32 %r
}

define i32 @tgt(i32 %x, i32 %d, i32 %q) {
  %s = xor i32 %x, %d
  %neg = icmp slt i32 %s, 0
  %nq = sub i32 0, %q
  %r = select i1 %neg, i32 %nq, i32 %q
  ret i32 %r
}
