; convert-gpu-to-air lowers gpu.subgroup_reduce maximumf of f32 to AIR's
; signed i32 maximum of the keys b ^ ((b >>s 31) & 0x7fffffff) of the bits b
; of each value, a NaN's replaced by that of the quiet NaN 0x7fc00000,
; 0x7fc00000, which is above every other float's (+inf's is 0x7f800000), so the
; maximum is a NaN's key wherever there is one, and it decodes to that quiet
; NaN; no or of "is NaN" (L15-E276). Each step of the reduction is this, for
; two lanes.

define float @src(float %x, float %y) {
  %m = call float @llvm.maximum.f32(float %x, float %y)
  ret float %m
}

define float @tgt(float %x1, float %y1) {
  %x0 = freeze float %x1
  %y0 = freeze float %y1
  %nx = fcmp uno float %x0, 0.0
  %ny = fcmp uno float %y0, 0.0
  %bx = bitcast float %x0 to i32
  %sx = ashr i32 %bx, 31
  %mx = and i32 %sx, 2147483647
  %kx0 = xor i32 %bx, %mx
  %kx = select i1 %nx, i32 2143289344, i32 %kx0
  %by = bitcast float %y0 to i32
  %sy = ashr i32 %by, 31
  %my = and i32 %sy, 2147483647
  %ky0 = xor i32 %by, %my
  %ky = select i1 %ny, i32 2143289344, i32 %ky0
  %k = call i32 @llvm.smax.i32(i32 %kx, i32 %ky)
  %s = ashr i32 %k, 31
  %mk = and i32 %s, 2147483647
  %b = xor i32 %k, %mk
  %r = bitcast i32 %b to float
  ret float %r
}

declare float @llvm.maximum.f32(float, float)
declare i32 @llvm.smax.i32(i32, i32)
