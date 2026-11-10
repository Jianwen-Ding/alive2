; llvm.minimum orders -0.0 below +0.0; air.fmin is minnum, which does not: the
; portable AIR that only added the NaN select (before air-nan-min-max ordered
; the zeros there too) could return +0.0 for minimum(-0.0, +0.0).
target triple = "air64-apple-macosx26.0.0"

declare float @llvm.minimum.f32(float, float)
declare float @air.fmin.f32(float, float)

define float @src(float %x, float %y) {
  %n = call float @llvm.minimum.f32(float %x, float %y)
  ret float %n
}

define float @tgt(float %x, float %y) {
  %n = call float @air.fmin.f32(float %x, float %y)
  %nan_check = fcmp uno float %x, %y
  %r = select i1 %nan_check, float 0x7FF8000000000000, float %n
  ret float %r
}

; TEST-ARGS: -disable-undef-input
; ERROR: Value mismatch
