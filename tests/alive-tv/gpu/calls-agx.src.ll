; A non-kernel function that a kernel calls: AGX inlines it, a call being its
; body (the canonical form inlines it too).


define internal float @helper(float %x) noinline {
  %y = fmul float %x, 3.0
  ret float %y
}

define void @k(ptr addrspace(1) %p) "air-kernel" {
  %v = load float, ptr addrspace(1) %p
  %r = call float @helper(float %v)
  store float %r, ptr addrspace(1) %p
  ret void
}
