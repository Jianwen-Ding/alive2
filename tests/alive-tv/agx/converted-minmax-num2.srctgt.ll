; AGXCodeGenPrepare (L15-E281): as converted-minmax-num, of two variables
; (whose zeros' signs may differ).

define i16 @src(float %x, float %y) {
  %m = call float @llvm.maximum.f32(float %x, float %y)
  %r = fptosi float %m to i16
  ret i16 %r
}

define i16 @tgt(float %x, float %y) {
  %m = call float @llvm.maxnum.f32(float %x, float %y)
  %r = fptosi float %m to i16
  ret i16 %r
}

declare float @llvm.maximum.f32(float, float)
declare float @llvm.maxnum.f32(float, float)
