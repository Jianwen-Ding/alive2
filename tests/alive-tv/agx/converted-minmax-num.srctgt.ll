; AGXCodeGenPrepare (L15-E281): a maximum / minimum whose only use is a
; float-to-integer conversion is the maxnum / minnum (equal where no operand
; is a NaN, but for a zero's sign, which converts to 0 either way; a NaN made
; the conversion poison).

define i8 @src(float %x) {
  %m = call float @llvm.maximum.f32(float %x, float -128.0)
  %n = call float @llvm.minimum.f32(float %m, float 127.0)
  %r = fptosi float %n to i8
  ret i8 %r
}

define i8 @tgt(float %x) {
  %m = call float @llvm.maxnum.f32(float %x, float -128.0)
  %n = call float @llvm.minnum.f32(float %m, float 127.0)
  %r = fptosi float %n to i8
  ret i8 %r
}

declare float @llvm.maximum.f32(float, float)
declare float @llvm.minimum.f32(float, float)
declare float @llvm.maxnum.f32(float, float)
declare float @llvm.minnum.f32(float, float)
