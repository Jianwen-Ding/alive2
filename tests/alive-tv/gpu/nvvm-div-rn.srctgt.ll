; llvm.nvvm.div.rn.f (div.rn.f32) and llvm.nvvm.d2f.rn (cvt.rn.f32.f64):
; IEEE division and narrowing, rounding to nearest even.
; TEST-ARGS: -disable-undef-input
target triple = "nvptx64-nvidia-cuda"

declare float @llvm.nvvm.div.rn.f(float, float)
declare float @llvm.nvvm.d2f.rn(double)

define float @src(float %a, double %b) {
  %n = call float @llvm.nvvm.d2f.rn(double %b)
  %q = call float @llvm.nvvm.div.rn.f(float %a, float %n)
  ret float %q
}

define float @tgt(float %a, double %b) {
  %n = fptrunc double %b to float
  %q = fdiv float %a, %n
  ret float %q
}
