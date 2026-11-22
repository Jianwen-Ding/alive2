; llvm.nvvm.f2i.rn (cvt.rni.s32.f32) rounds to nearest even, not toward zero.
; TEST-ARGS: -disable-undef-input
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.f2i.rn(float)

define i32 @src(float %x) {
  %r = call i32 @llvm.nvvm.f2i.rn(float %x)
  ret i32 %r
}

define i32 @tgt(float %x) {
  %r = call i32 @llvm.fptosi.sat.i32.f32(float %x)
  ret i32 %r
}

; ERROR: Value mismatch
