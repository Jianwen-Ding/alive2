; AGX DAG combine (L15-E311): a byte with a zero high byte sign-extended in a
; 16-bit register is (x ^ 128) - 128, on the ALU.

define i16 @src(i8 %b) {
  %x = zext i8 %b to i16
  %s = shl i16 %x, 8
  %r = ashr i16 %s, 8
  ret i16 %r
}

define i16 @tgt(i8 %b) {
  %x = zext i8 %b to i16
  %f = xor i16 %x, 128
  %r = sub i16 %f, 128
  ret i16 %r
}
