; convert-gpu-to-air computes the index of a memref access of a static
; dimension of at most 2^32 elements in 32 bits: the index is below the
; size (an access out of bounds is undefined: the assume), so it is its
; value mod 2^32, which additions and multiplications commute with. Here at
; a quarter of the widths (index i16, dimension 2^4, narrowed to i8), as Z3
; does not decide the wider multiplications in time; the argument does not
; depend on them.
; TEST-ARGS: -smt-to=600000

define i16 @src(i8 %tx, i8 %ty, i8 %bx, i8 %by, i16 %ld) {
  %txw = zext i8 %tx to i16
  %tyw = zext i8 %ty to i16
  %bxw = zext i8 %bx to i16
  %byw = zext i8 %by to i16
  %x0 = mul i16 %bxw, 4
  %x = add i16 %x0, %txw
  %y0 = mul i16 %byw, 4
  %y = add i16 %y0, %tyw
  %row = mul i16 %y, %ld
  %i = add i16 %row, %x
  %ib = icmp ult i16 %i, 16
  call void @llvm.assume(i1 %ib)
  ret i16 %i
}

define i16 @tgt(i8 %tx, i8 %ty, i8 %bx, i8 %by, i16 %ld) {
  %ldn = trunc i16 %ld to i8
  %x0 = mul i8 %bx, 4
  %x = add i8 %x0, %tx
  %y0 = mul i8 %by, 4
  %y = add i8 %y0, %ty
  %row = mul i8 %y, %ldn
  %i = add i8 %row, %x
  %iw = zext i8 %i to i16
  ret i16 %iw
}

declare void @llvm.assume(i1)
