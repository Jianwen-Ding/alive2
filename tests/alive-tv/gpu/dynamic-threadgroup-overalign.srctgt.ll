; An alignment beyond the one the dynamic threadgroup variable declares is not
; known.
target triple = "agx"

@dyn = external addrspace(3) global [0 x i8], align 16

define void @src(i64 %v) {
  store i64 %v, ptr addrspace(3) @dyn, align 8
  ret void
}

define void @tgt(i64 %v) {
  store i64 %v, ptr addrspace(3) @dyn, align 32
  ret void
}

; ERROR: Source is more defined than target
