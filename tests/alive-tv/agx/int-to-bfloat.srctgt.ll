; AGX FP_TO_BF16 lowering (L15-E312): an integer of at most 8 significant
; bits (an i8) converted to float is a bfloat exactly, so the rounding to
; bfloat (and its NaN test: a converted integer is never a NaN) is the
; float's top half.

define i16 @src(i8 %b) {
  %f = sitofp i8 %b to float
  %x = bitcast float %f to i32
  %h = lshr i32 %x, 16
  %l = and i32 %h, 1
  %k = add i32 %l, 32767
  %r = add i32 %x, %k
  %q = or i32 %x, 4194304
  %nan = fcmp uno float %f, %f
  %s = select i1 %nan, i32 %q, i32 %r
  %t = lshr i32 %s, 16
  %o = trunc i32 %t to i16
  ret i16 %o
}

define i16 @tgt(i8 %b) {
  %f = sitofp i8 %b to float
  %x = bitcast float %f to i32
  %t = lshr i32 %x, 16
  %o = trunc i32 %t to i16
  ret i16 %o
}
