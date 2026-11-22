; Float arithmetic in registers and memory.
target triple = "nvptx64-nvidia-cuda"

define float @fma_like(float %a, float %b) {
  %m = fmul float %a, %b
  %r = fadd float %m, %a
  ret float %r
}

define ptx_kernel void @scale(float %a, ptr addrspace(1) %p) {
  %v = load float, ptr addrspace(1) %p, align 4
  %r = fmul float %v, %a
  store float %r, ptr addrspace(1) %p, align 4
  ret void
}
