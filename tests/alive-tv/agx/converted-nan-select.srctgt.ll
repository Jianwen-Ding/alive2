; AGXCodeGenPrepare (L15-E281): a select of a NaN constant into a
; float-to-integer conversion (MLIR's maximumf: maxnum and a NaN where an
; operand is one) is the other arm: the NaN's conversion is poison.

define i8 @src(float %x) {
  %m = call float @llvm.maxnum.f32(float %x, float -128.0)
  %c = fcmp uno float %x, 0.0
  %n = call float @llvm.minnum.f32(float %m, float 127.0)
  %s = select i1 %c, float 0x7FF8000000000000, float %n
  %r = fptosi float %s to i8
  ret i8 %r
}

define i8 @tgt(float %x) {
  %m = call float @llvm.maxnum.f32(float %x, float -128.0)
  %n = call float @llvm.minnum.f32(float %m, float 127.0)
  %r = fptosi float %n to i8
  ret i8 %r
}

declare float @llvm.maxnum.f32(float, float)
declare float @llvm.minnum.f32(float, float)
