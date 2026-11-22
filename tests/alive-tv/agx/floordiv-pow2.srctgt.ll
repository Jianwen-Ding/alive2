; convert-gpu-to-air (L15-E290): arith.floordivsi by 2^k is the arithmetic
; shift by k (here k = 3 and the top 2^30), and ceildivsi by 2^k that shift
; plus 1 where a low bit is set; src is arith's expansion (ArithToLLVM).

define i32 @src(i32 %x) {
  %q = sdiv i32 %x, 8
  %m = mul i32 %q, 8
  %ne = icmp ne i32 %x, %m
  %neg = icmp slt i32 %x, 0
  %c = and i1 %ne, %neg
  %q1 = add i32 %q, -1
  %r = select i1 %c, i32 %q1, i32 %q
  ret i32 %r
}

define i32 @tgt(i32 %x) {
  %r = ashr i32 %x, 3
  ret i32 %r
}
