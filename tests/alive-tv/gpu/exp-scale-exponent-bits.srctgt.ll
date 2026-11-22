; f64 exp (L15-E326): y 2^k1 for y in [0.5, 2) and k1 in [-538, 515] is
; exact and normal, so it is k1 added to y's exponent field (an integer
; addition for a double multiplication by a power of two); a NaN y is kept.

define double @src(double %y, i64 %k1) {
  %ylo = fcmp oge double %y, 0.5
  %yhi = fcmp olt double %y, 2.0
  %isnan = fcmp uno double %y, %y
  %inr = and i1 %ylo, %yhi
  %ok = or i1 %inr, %isnan
  call void @llvm.assume(i1 %ok)
  %klo = icmp sge i64 %k1, -538
  %khi = icmp sle i64 %k1, 515
  call void @llvm.assume(i1 %klo)
  call void @llvm.assume(i1 %khi)
  %e = add i64 %k1, 1023
  %pb = shl i64 %e, 52
  %p = bitcast i64 %pb to double
  %r = fmul double %y, %p
  ret double %r
}

define double @tgt(double %y, i64 %k1) {
  %ylo = fcmp oge double %y, 0.5
  %yhi = fcmp olt double %y, 2.0
  %isnan = fcmp uno double %y, %y
  %inr = and i1 %ylo, %yhi
  %ok = or i1 %inr, %isnan
  call void @llvm.assume(i1 %ok)
  %klo = icmp sge i64 %k1, -538
  %khi = icmp sle i64 %k1, 515
  call void @llvm.assume(i1 %klo)
  call void @llvm.assume(i1 %khi)
  %yb = bitcast double %y to i64
  %kb = shl i64 %k1, 52
  %sb = add i64 %yb, %kb
  %s = bitcast i64 %sb to double
  %r = select i1 %isnan, double %y, double %s
  ret double %r
}

declare void @llvm.assume(i1)
