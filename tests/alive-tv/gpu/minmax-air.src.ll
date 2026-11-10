; llvm.minimum and llvm.maximum (upstream): -0.0 is below +0.0; their portable
; AIR orders the zeros with a select around air.fmin / air.fmax (minnum).
declare float @llvm.minimum.f32(float, float)
declare float @llvm.maximum.f32(float, float)
define float @f(float %x, float %y) {
  %n = call float @llvm.minimum.f32(float %x, float %y)
  %m = call float @llvm.maximum.f32(float %n, float %y)
  ret float %m
}
define void @k(ptr addrspace(1) %out, ptr addrspace(1) %in) "air-kernel" {
  %x = load float, ptr addrspace(1) %in
  %p = getelementptr float, ptr addrspace(1) %in, i32 1
  %y = load float, ptr addrspace(1) %p
  %r = call float @f(float %x, float %y)
  store float %r, ptr addrspace(1) %out
  ret void
}
; TEST-ARGS: -func=k
