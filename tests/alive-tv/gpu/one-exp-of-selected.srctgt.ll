; TEST-ARGS: -smt-to=60000
; convert-gpu-to-air's f64 cosh and sinh (L15-E309): something of e = exp(|x|)
; below 709 and of exp(|x| / 2) from 709, one exp of the argument the range
; picks. (The arms are e + 1 and -w here, for the solver: the shape is what
; changes.)

define double @src(double noundef %x) {
  %a = call double @llvm.fabs.f64(double %x)
  %e = call double @ex(double %a)
  %near = fadd double %e, 1.0
  %a2 = fmul double %a, 0.5
  %w = call double @ex(double %a2)
  %far = fneg double %w
  %c = fcmp olt double %a, 709.0
  %r = select i1 %c, double %near, double %far
  ret double %r
}

define double @tgt(double noundef %x) {
  %a = call double @llvm.fabs.f64(double %x)
  %c = fcmp olt double %a, 709.0
  %a2 = fmul double %a, 0.5
  %arg = select i1 %c, double %a, double %a2
  %e = call double @ex(double %arg)
  %near = fadd double %e, 1.0
  %far = fneg double %e
  %r = select i1 %c, double %near, double %far
  ret double %r
}

declare double @llvm.fabs.f64(double)
; the exp: any pure function of its argument
declare double @ex(double) memory(none) nounwind willreturn
