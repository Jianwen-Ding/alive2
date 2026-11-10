; convert-gpu-to-air lowers gpu.subgroup_reduce maximumf (minimumf) of f32,
; which AIR's float SIMD-group maximum (a maxnum) does not compute, to AIR's
; signed i32 maximum (minimum) of the keys b ^ ((b >>s 31) & 0x7fffffff) of
; the bits b of the frozen float (used twice), which order the floats with
; -0.0 below +0.0, the key of the
; result decoded the same way, and a NaN where the SIMD-group's or of
; "is NaN" is set. Each step of the reduction is this, for two lanes.

define float @src(float %x, float %y) {
  %m = call float @llvm.maximum.f32(float %x, float %y)
  ret float %m
}

define float @tgt(float %x0, float %y0) {
  %x = freeze float %x0
  %y = freeze float %y0
  %bx = bitcast float %x to i32
  %sx = ashr i32 %bx, 31
  %mx = and i32 %sx, 2147483647
  %kx = xor i32 %bx, %mx
  %by = bitcast float %y to i32
  %sy = ashr i32 %by, 31
  %my = and i32 %sy, 2147483647
  %ky = xor i32 %by, %my
  %k = call i32 @llvm.smax.i32(i32 %kx, i32 %ky)
  %s = ashr i32 %k, 31
  %mk = and i32 %s, 2147483647
  %b = xor i32 %k, %mk
  %r = bitcast i32 %b to float
  %nx = fcmp uno float %x, 0.0
  %nxi = zext i1 %nx to i32
  %ny = fcmp uno float %y, 0.0
  %nyi = zext i1 %ny to i32
  %n = or i32 %nxi, %nyi
  %isnan = icmp ne i32 %n, 0
  %res = select i1 %isnan, float 0x7FF8000000000000, float %r
  ret float %res
}

declare float @llvm.maximum.f32(float, float)
declare i32 @llvm.smax.i32(i32, i32)
