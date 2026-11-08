; AGX has no floating-point environment to change: outside strictfp, a
; constrained operation that rounds to nearest and ignores exceptions (as
; agx-lower-air restores AIR's software bfloat addition) is the operation.
target triple = "agx"

define bfloat @src(bfloat %a, bfloat %b) {
  %s = fadd bfloat %a, %b
  ret bfloat %s
}

define bfloat @tgt(bfloat %a, bfloat %b) {
  %s = call bfloat @llvm.experimental.constrained.fadd.bf16(bfloat %a, bfloat %b, metadata !"round.tonearest", metadata !"fpexcept.ignore")
  ret bfloat %s
}

declare bfloat @llvm.experimental.constrained.fadd.bf16(bfloat, bfloat, metadata, metadata)
