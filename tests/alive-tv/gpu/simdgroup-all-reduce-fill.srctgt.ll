; convert-gpu-to-air lowers a gpu.all_reduce of whole SIMD-groups to each
; SIMD-group's reduction of the partials (simdgroup-all-reduce): lane l
; reads partial l, and the lanes past the SIMD-group count read the identity
; of an addition, multiplication or xor (-0.0 or 0, 1, 0) or partial 0 of a
; minimum, maximum, and or or. Whatever the order of the reduction, an
; identity lane meets another (-0.0 + -0.0 is -0.0, 1.0 * 1.0 is 1.0) or a
; sum (product) s of partials, which like the partials is a result of float
; arithmetic and so not a signaling NaN (the assumes: ordered or quiet),
; which it leaves as it is; a maximum (here maxnum and umin) of partials p0,
; p1 with another lane of p0 is the maximum of p0 and p1; NaNs compare as one
; (their payloads are any NaN's). The partials are values in memory: noundef.

define {float, float, float, float, i32, float, i32} @src(float noundef %s, float noundef %p, i32 noundef %i, float noundef %g0, float noundef %g1, i32 noundef %u0, i32 noundef %u1) {
  %qsb = bitcast float %s to i32
  %qsm = and i32 %qsb, 4194304
  %qsq = icmp ne i32 %qsm, 0
  %qsn = fcmp ord float %s, 0.0
  %qsok = or i1 %qsq, %qsn
  call void @llvm.assume(i1 %qsok)
  %qpb = bitcast float %p to i32
  %qpm = and i32 %qpb, 4194304
  %qpq = icmp ne i32 %qpm, 0
  %qpn = fcmp ord float %p, 0.0
  %qpok = or i1 %qpq, %qpn
  call void @llvm.assume(i1 %qpok)
  %qg0b = bitcast float %g0 to i32
  %qg0m = and i32 %qg0b, 4194304
  %qg0q = icmp ne i32 %qg0m, 0
  %qg0n = fcmp ord float %g0, 0.0
  %qg0ok = or i1 %qg0q, %qg0n
  call void @llvm.assume(i1 %qg0ok)
  %qg1b = bitcast float %g1 to i32
  %qg1m = and i32 %qg1b, 4194304
  %qg1q = icmp ne i32 %qg1m, 0
  %qg1n = fcmp ord float %g1, 0.0
  %qg1ok = or i1 %qg1q, %qg1n
  call void @llvm.assume(i1 %qg1ok)
  %snan = fcmp uno float %s, 0.0
  %sc = select i1 %snan, float 0x7FF8000000000000, float %s
  %pnan = fcmp uno float %p, 0.0
  %pc = select i1 %pnan, float 0x7FF8000000000000, float %p
  %fmax = call float @llvm.maxnum.f32(float %g0, float %g1)
  %umin = call i32 @llvm.umin.i32(i32 %u0, i32 %u1)
  %fnan = fcmp uno float %fmax, 0.0
  %fmaxc = select i1 %fnan, float 0x7FF8000000000000, float %fmax
  %r0 = insertvalue {float, float, float, float, i32, float, i32} poison, float %sc, 0
  %r1 = insertvalue {float, float, float, float, i32, float, i32} %r0, float %pc, 1
  %r2 = insertvalue {float, float, float, float, i32, float, i32} %r1, float -0.0, 2
  %r3 = insertvalue {float, float, float, float, i32, float, i32} %r2, float 1.0, 3
  %r4 = insertvalue {float, float, float, float, i32, float, i32} %r3, i32 %i, 4
  %r5 = insertvalue {float, float, float, float, i32, float, i32} %r4, float %fmaxc, 5
  %r6 = insertvalue {float, float, float, float, i32, float, i32} %r5, i32 %umin, 6
  ret {float, float, float, float, i32, float, i32} %r6
}

define {float, float, float, float, i32, float, i32} @tgt(float noundef %s0, float noundef %p0, i32 noundef %i, float noundef %g0, float noundef %g1, i32 noundef %u0, i32 noundef %u1) {
  %qsb = bitcast float %s0 to i32
  %qsm = and i32 %qsb, 4194304
  %qsq = icmp ne i32 %qsm, 0
  %qsn = fcmp ord float %s0, 0.0
  %qsok = or i1 %qsq, %qsn
  call void @llvm.assume(i1 %qsok)
  %qpb = bitcast float %p0 to i32
  %qpm = and i32 %qpb, 4194304
  %qpq = icmp ne i32 %qpm, 0
  %qpn = fcmp ord float %p0, 0.0
  %qpok = or i1 %qpq, %qpn
  call void @llvm.assume(i1 %qpok)
  %s = fadd float %s0, -0.0
  %p = fmul float %p0, 1.0
  %zz = fadd float -0.0, -0.0
  %oo = fmul float 1.0, 1.0
  %ii = add i32 %i, 0
  %snan = fcmp uno float %s, 0.0
  %sc = select i1 %snan, float 0x7FF8000000000000, float %s
  %pnan = fcmp uno float %p, 0.0
  %pc = select i1 %pnan, float 0x7FF8000000000000, float %p
  %qg0b = bitcast float %g0 to i32
  %qg0m = and i32 %qg0b, 4194304
  %qg0q = icmp ne i32 %qg0m, 0
  %qg0n = fcmp ord float %g0, 0.0
  %qg0ok = or i1 %qg0q, %qg0n
  call void @llvm.assume(i1 %qg0ok)
  %qg1b = bitcast float %g1 to i32
  %qg1m = and i32 %qg1b, 4194304
  %qg1q = icmp ne i32 %qg1m, 0
  %qg1n = fcmp ord float %g1, 0.0
  %qg1ok = or i1 %qg1q, %qg1n
  call void @llvm.assume(i1 %qg1ok)
  ; lanes [p0, p1, p0, p0], reduced pairwise
  %g02 = call float @llvm.maxnum.f32(float %g0, float %g0)
  %g13 = call float @llvm.maxnum.f32(float %g1, float %g0)
  %fmax = call float @llvm.maxnum.f32(float %g02, float %g13)
  %u02 = call i32 @llvm.umin.i32(i32 %u0, i32 %u0)
  %u13 = call i32 @llvm.umin.i32(i32 %u1, i32 %u0)
  %umin = call i32 @llvm.umin.i32(i32 %u02, i32 %u13)
  %fnan = fcmp uno float %fmax, 0.0
  %fmaxc = select i1 %fnan, float 0x7FF8000000000000, float %fmax
  %r0 = insertvalue {float, float, float, float, i32, float, i32} poison, float %sc, 0
  %r1 = insertvalue {float, float, float, float, i32, float, i32} %r0, float %pc, 1
  %r2 = insertvalue {float, float, float, float, i32, float, i32} %r1, float %zz, 2
  %r3 = insertvalue {float, float, float, float, i32, float, i32} %r2, float %oo, 3
  %r4 = insertvalue {float, float, float, float, i32, float, i32} %r3, i32 %ii, 4
  %r5 = insertvalue {float, float, float, float, i32, float, i32} %r4, float %fmaxc, 5
  %r6 = insertvalue {float, float, float, float, i32, float, i32} %r5, i32 %umin, 6
  ret {float, float, float, float, i32, float, i32} %r6
}

declare float @llvm.maxnum.f32(float, float)
declare i32 @llvm.umin.i32(i32, i32)

declare void @llvm.assume(i1)
