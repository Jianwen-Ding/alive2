; GPUToAIR speculate-guarded-loads (L15-E737): after the speculation,
; in && (in ? x : 0) <= t is in && x <= t: where in is false the and is
; false whatever the compare (find_first's scan step).

define i1 @src(i1 %in, float %x, float %t) {
  %v = select i1 %in, float %x, float 0.0
  %le = fcmp ole float %v, %t
  %r = and i1 %in, %le
  ret i1 %r
}

define i1 @tgt(i1 %in, float %x, float %t) {
  %le = fcmp ole float %x, %t
  %r = select i1 %in, i1 %le, i1 false
  ret i1 %r
}
