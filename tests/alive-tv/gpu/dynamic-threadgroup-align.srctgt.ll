; AGX's dynamic threadgroup memory (a zero-sized threadgroup variable without a
; definition) has the alignment the variable declares: InferAlignment may raise
; an access of it to that alignment.
target triple = "agx"

@dyn = external addrspace(3) global [0 x i8], align 16

define void @src(i64 %v) {
  store i64 %v, ptr addrspace(3) @dyn, align 8
  ret void
}

define void @tgt(i64 %v) {
  store i64 %v, ptr addrspace(3) @dyn, align 16
  ret void
}
