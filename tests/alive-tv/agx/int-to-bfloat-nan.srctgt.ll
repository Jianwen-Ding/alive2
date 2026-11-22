; AGX FP_TO_BF16 lowering (L15-E312): an integer converted to float is never
; a NaN, so the rounding to bfloat needs no NaN test (here the test alone, of an i16: the solver times out on i32).

define i32 @src(i16 %w) {
  %g = uitofp i16 %w to float
  %y = bitcast float %g to i32
  %q = or i32 %y, 4194304
  %nan = fcmp uno float %g, %g
  %s = select i1 %nan, i32 %q, i32 %y
  ret i32 %s
}

define i32 @tgt(i16 %w) {
  %g = uitofp i16 %w to float
  %y = bitcast float %g to i32
  ret i32 %y
}
