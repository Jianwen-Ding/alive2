; AGX soft double (L15-E327): x * c for a constant power of two c = +-2^C
; takes a short path where x is normal and so is the product (exponent
; fields within 1..2046), or x is zero: C added to x's exponent field (the
; sign flipped for a negative c), or the zero with the product's sign; the
; general multiplication (a slow path) otherwise. Each case under its
; condition: c = 0.5 (x's field within 2..2046) and c = -4 (1..2044); and
; a factor y whose significand field is known zero (bits built as e << 52):
; y's exponent field (less the bias) added and the signs combined, where both
; fields and the product's are within 1..2046, or x is zero and y normal.

define double @src_half(double %x) {
  %b = bitcast double %x to i64
  %eb = lshr i64 %b, 52
  %e = and i64 %eb, 2047
  %el = sub i64 %e, 2
  %ok = icmp ult i64 %el, 2045
  call void @llvm.assume(i1 %ok)
  %r = fmul double %x, 0.5
  ret double %r
}

define double @tgt_half(double %x) {
  %b = bitcast double %x to i64
  %eb = lshr i64 %b, 52
  %e = and i64 %eb, 2047
  %el = sub i64 %e, 2
  %ok = icmp ult i64 %el, 2045
  call void @llvm.assume(i1 %ok)
  %s = add i64 %b, -4503599627370496
  %f = bitcast i64 %s to double
  ret double %f
}

define double @src_half_zero(double %x) {
  %b = bitcast double %x to i64
  %mag = and i64 %b, 9223372036854775807
  %z = icmp eq i64 %mag, 0
  call void @llvm.assume(i1 %z)
  %r = fmul double %x, 0.5
  ret double %r
}

define double @tgt_half_zero(double %x) {
  %b = bitcast double %x to i64
  %mag = and i64 %b, 9223372036854775807
  %z = icmp eq i64 %mag, 0
  call void @llvm.assume(i1 %z)
  %f = bitcast i64 %b to double
  ret double %f
}

define double @src_neg4(double %x) {
  %b = bitcast double %x to i64
  %eb = lshr i64 %b, 52
  %e = and i64 %eb, 2047
  %el = sub i64 %e, 1
  %ok = icmp ult i64 %el, 2044
  call void @llvm.assume(i1 %ok)
  %r = fmul double %x, -4.0
  ret double %r
}

define double @tgt_neg4(double %x) {
  %b = bitcast double %x to i64
  %eb = lshr i64 %b, 52
  %e = and i64 %eb, 2047
  %el = sub i64 %e, 1
  %ok = icmp ult i64 %el, 2044
  call void @llvm.assume(i1 %ok)
  %a = add i64 %b, 9007199254740992
  %s = xor i64 %a, -9223372036854775808
  %f = bitcast i64 %s to double
  ret double %f
}

define double @src_neg4_zero(double %x) {
  %b = bitcast double %x to i64
  %mag = and i64 %b, 9223372036854775807
  %z = icmp eq i64 %mag, 0
  call void @llvm.assume(i1 %z)
  %r = fmul double %x, -4.0
  ret double %r
}

define double @tgt_neg4_zero(double %x) {
  %b = bitcast double %x to i64
  %mag = and i64 %b, 9223372036854775807
  %z = icmp eq i64 %mag, 0
  call void @llvm.assume(i1 %z)
  %s = xor i64 %b, -9223372036854775808
  %f = bitcast i64 %s to double
  ret double %f
}

define double @src_pow2(double %x, i64 %e) {
  %yb = shl i64 %e, 52
  %y = bitcast i64 %yb to double
  %b = bitcast double %x to i64
  %exs = lshr i64 %b, 52
  %ex = and i64 %exs, 2047
  %eys = lshr i64 %yb, 52
  %ey = and i64 %eys, 2047
  %exm = sub i64 %ex, 1
  %okx = icmp ult i64 %exm, 2046
  %eym = sub i64 %ey, 1
  %oky = icmp ult i64 %eym, 2046
  %es = add i64 %ex, %ey
  %er = sub i64 %es, 1024
  %okr = icmp ult i64 %er, 2046
  call void @llvm.assume(i1 %okx)
  call void @llvm.assume(i1 %oky)
  call void @llvm.assume(i1 %okr)
  %r = fmul double %x, %y
  ret double %r
}

define double @tgt_pow2(double %x, i64 %e) {
  %yb = shl i64 %e, 52
  %b = bitcast double %x to i64
  %exs = lshr i64 %b, 52
  %ex = and i64 %exs, 2047
  %eys = lshr i64 %yb, 52
  %ey = and i64 %eys, 2047
  %exm = sub i64 %ex, 1
  %okx = icmp ult i64 %exm, 2046
  %eym = sub i64 %ey, 1
  %oky = icmp ult i64 %eym, 2046
  %es = add i64 %ex, %ey
  %er = sub i64 %es, 1024
  %okr = icmp ult i64 %er, 2046
  call void @llvm.assume(i1 %okx)
  call void @llvm.assume(i1 %oky)
  call void @llvm.assume(i1 %okr)
  %sy = and i64 %yb, -9223372036854775808
  %my = xor i64 %yb, %sy
  %d = sub i64 %my, 4607182418800017408
  %a = add i64 %b, %d
  %rb = xor i64 %a, %sy
  %r = bitcast i64 %rb to double
  ret double %r
}

define double @src_pow2_zero(double %x, i64 %e) {
  %yb = shl i64 %e, 52
  %y = bitcast i64 %yb to double
  %b = bitcast double %x to i64
  %mag = and i64 %b, 9223372036854775807
  %z = icmp eq i64 %mag, 0
  %eys = lshr i64 %yb, 52
  %ey = and i64 %eys, 2047
  %eym = sub i64 %ey, 1
  %oky = icmp ult i64 %eym, 2046
  call void @llvm.assume(i1 %z)
  call void @llvm.assume(i1 %oky)
  %r = fmul double %x, %y
  ret double %r
}

define double @tgt_pow2_zero(double %x, i64 %e) {
  %yb = shl i64 %e, 52
  %b = bitcast double %x to i64
  %mag = and i64 %b, 9223372036854775807
  %z = icmp eq i64 %mag, 0
  %eys = lshr i64 %yb, 52
  %ey = and i64 %eys, 2047
  %eym = sub i64 %ey, 1
  %oky = icmp ult i64 %eym, 2046
  call void @llvm.assume(i1 %z)
  call void @llvm.assume(i1 %oky)
  %sy = and i64 %yb, -9223372036854775808
  %rb = xor i64 %b, %sy
  %r = bitcast i64 %rb to double
  ret double %r
}

declare void @llvm.assume(i1)
