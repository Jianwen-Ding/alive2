; AGXCodeGenPrepare (L15-E738): c and (a < b) is (c ? a : b) < b (false at
; equality), b frozen. At 8 bits: the identity does not depend on the width.

define i1 @src(i8 %x, i8 %w, i8 %y, i8 %h) {
  %ix = icmp ult i8 %x, %w
  %iy = icmp slt i8 %y, %h
  %r = and i1 %ix, %iy
  ret i1 %r
}

define i1 @tgt(i8 %x, i8 %w, i8 %y, i8 %h) {
  %ix = icmp ult i8 %x, %w
  %hf = freeze i8 %h
  %s = select i1 %ix, i8 %y, i8 %hf
  %r = icmp slt i8 %s, %hf
  ret i1 %r
}
