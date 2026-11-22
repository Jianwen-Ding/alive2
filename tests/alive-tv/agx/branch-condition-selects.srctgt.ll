; AGXCodeGenPrepare (L15-E738): a branch on c or (a p b), p true at
; equality, is a branch on (c ? b : a) p b; on c and (a q b), q false at
; equality, on (c ? a : b) q b, b frozen (used twice: an undef b would be
; two values). (Plain or / and: a poison operand of either makes the
; original poison. At 8 bits: the identity does not depend on the width.)

define i1 @src(i8 %x, i8 %w, i8 %y, i8 %h) {
  %ox = icmp sge i8 %x, %w
  %oy = icmp sge i8 %y, %h
  %r = or i1 %ox, %oy
  ret i1 %r
}

define i1 @tgt(i8 %x, i8 %w, i8 %y, i8 %h) {
  %ox = icmp sge i8 %x, %w
  %hf = freeze i8 %h
  %s = select i1 %ox, i8 %hf, i8 %y
  %r = icmp sge i8 %s, %hf
  ret i1 %r
}
