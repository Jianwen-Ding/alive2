; AGXCodeGenPrepare rangedCompares (L15-E300): with the SIMD lane in
; [0, 32), a shuffle-up's segment test by d (here 4) is lane >= d.

define i1 @src(i32 %x) {
  %lane = and i32 %x, 31
  %start = and i32 %lane, -32
  %end = add i32 %start, 32
  %src = add i32 %lane, -4
  %lo = icmp sge i32 %src, %start
  %hi = icmp slt i32 %src, %end
  %ok = and i1 %lo, %hi
  ret i1 %ok
}

define i1 @tgt(i32 %x) {
  %lane = and i32 %x, 31
  %ok = icmp ugt i32 %lane, 3
  ret i1 %ok
}
