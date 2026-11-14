; AGXCodeGenPrepare (L15-E281): as converted-minmax-num, a chain through a
; minimum of a variable (unsigned; a zero bound of either sign).

define i16 @src(float %x, float %y) {
  %m = call float @llvm.maximum.f32(float %x, float -0.0)
  %n = call float @llvm.minimum.f32(float %m, float %y)
  %r = fptoui float %n to i16
  ret i16 %r
}

define i16 @tgt(float %x, float %y) {
  %m = call float @llvm.maxnum.f32(float %x, float -0.0)
  %n = call float @llvm.minnum.f32(float %m, float %y)
  %r = fptoui float %n to i16
  ret i16 %r
}

declare float @llvm.maximum.f32(float, float)
declare float @llvm.minimum.f32(float, float)
declare float @llvm.maxnum.f32(float, float)
declare float @llvm.minnum.f32(float, float)
