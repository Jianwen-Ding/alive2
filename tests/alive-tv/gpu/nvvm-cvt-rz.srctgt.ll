; llvm.nvvm.f2i.rz is PTX's cvt.rzi.s32.f32, whose meaning comes from the PTX
; formalization (scripts/ptx-tv/ptx.formal): toward zero, saturating, NaN to
; 0, which is llvm.fptosi.sat.
; TEST-ARGS: -disable-undef-input
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.f2i.rz(float)

define i32 @src(float %x) {
  %r = call i32 @llvm.nvvm.f2i.rz(float %x)
  ret i32 %r
}

define i32 @tgt(float %x) {
  %r = call i32 @llvm.fptosi.sat.i32.f32(float %x)
  ret i32 %r
}
