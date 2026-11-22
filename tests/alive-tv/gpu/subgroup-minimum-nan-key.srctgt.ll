; convert-gpu-to-air lowers gpu.subgroup_reduce minimumf of f32 to AIR's
; signed i32 minimum of the keys b ^ ((b >>s 31) & 0x7fffffff) of the bits b
; of each value, a NaN's replaced by that of 0xffffffff, 0x80000000, which is
; below every other float's, so the minimum is that key wherever there is a
; NaN, which gives the quiet NaN; no or of "is NaN" (L15-E276). Each step of
; the reduction is this, for two lanes.

define float @src(float %x, float %y) {
  %m = call float @llvm.minimum.f32(float %x, float %y)
  ret float %m
}

define float @tgt(float %x1, float %y1) {
  %x0 = freeze float %x1
  %y0 = freeze float %y1
  %bx = bitcast float %x0 to i32
  %nx = fcmp uno float %x0, 0.0
  %by = bitcast float %y0 to i32
  %ny = fcmp uno float %y0, 0.0
  %sx = ashr i32 %bx, 31
  %mx = and i32 %sx, 2147483647
  %kx0 = xor i32 %bx, %mx
  %kx = select i1 %nx, i32 -2147483648, i32 %kx0
  %sy = ashr i32 %by, 31
  %my = and i32 %sy, 2147483647
  %ky0 = xor i32 %by, %my
  %ky = select i1 %ny, i32 -2147483648, i32 %ky0
  %k = call i32 @llvm.smin.i32(i32 %kx, i32 %ky)
  %s = ashr i32 %k, 31
  %mk = and i32 %s, 2147483647
  %b = xor i32 %k, %mk
  %r = bitcast i32 %b to float
  %isnan = icmp eq i32 %k, -2147483648
  %res = select i1 %isnan, float 0x7FF8000000000000, float %r
  ret float %res
}

declare float @llvm.minimum.f32(float, float)
declare i32 @llvm.smin.i32(i32, i32)
