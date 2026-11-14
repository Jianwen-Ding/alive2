; TEST-ARGS: -smt-to=60000
; convert-gpu-to-air's asinh / acosh (L15-E308): log1p(t)'s log of 1 + t
; where |x| < 4096 and log(|x|) + ln 2 elsewhere, one log of the argument
; the range picks. (log1p's lu * t / (u - 1) is lu * t here, for the
; solver: the shape is what changes.)

define float @src(float noundef %x, float noundef %t) {
  %a = call float @llvm.fabs.f32(float %x)
  %u = fadd float %t, 1.0
  %lu = call float @lg(float %u)
  %q = fmul float %lu, %t
  %one = fcmp oeq float %u, 1.0
  %qs = select i1 %one, float %t, float %q
  %at = call float @llvm.fabs.f32(float %t)
  %small = fcmp olt float %at, 0.5
  %near = select i1 %small, float %qs, float %lu
  %la = call float @lg(float %a)
  %far = fadd float %la, 0x3FE62E4300000000
  %c = fcmp olt float %a, 4096.0
  %r = select i1 %c, float %near, float %far
  ret float %r
}

define float @tgt(float noundef %x, float noundef %t) {
  %a = call float @llvm.fabs.f32(float %x)
  %u = fadd float %t, 1.0
  %c = fcmp olt float %a, 4096.0
  %arg = select i1 %c, float %u, float %a
  %lu = call float @lg(float %arg)
  %q = fmul float %lu, %t
  %one = fcmp oeq float %u, 1.0
  %qs = select i1 %one, float %t, float %q
  %at = call float @llvm.fabs.f32(float %t)
  %small = fcmp olt float %at, 0.5
  %near = select i1 %small, float %qs, float %lu
  %far = fadd float %lu, 0x3FE62E4300000000
  %r = select i1 %c, float %near, float %far
  ret float %r
}

declare float @llvm.fabs.f32(float)
; the log: any pure function of its argument
declare float @lg(float) memory(none) nounwind willreturn
