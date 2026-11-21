; AGXCodeGenPrepare mergeLoopExits (L15-E735): the merged exit e1 or i == n,
; n loop-invariant, as (e1 ? n : i) == n (mandelbrot's escape test and
; iteration limit).

define i1 @src(float %m, i32 %i, i32 %n) {
  %e1 = fcmp uge float %m, 4.0
  %e2 = icmp eq i32 %i, %n
  %r = or i1 %e1, %e2
  ret i1 %r
}

define i1 @tgt(float %m, i32 %i, i32 %n) {
  %e1 = fcmp uge float %m, 4.0
  %nf = freeze i32 %n
  %s = select i1 %e1, i32 %nf, i32 %i
  %r = icmp eq i32 %s, %nf
  ret i1 %r
}
