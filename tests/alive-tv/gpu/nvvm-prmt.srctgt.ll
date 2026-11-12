; llvm.nvvm.prmt (prmt.b32): selector 0x1076 takes bytes 6, 7, 0 and 1 of
; {b, a}, lowest first: b's high half below a's low half.
; TEST-ARGS: -disable-undef-input
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.prmt(i32, i32, i32)

define i32 @src(i32 %a, i32 %b) {
  %r = call i32 @llvm.nvvm.prmt(i32 %a, i32 %b, i32 4214)
  ret i32 %r
}

define i32 @tgt(i32 %a, i32 %b) {
  %hi = lshr i32 %b, 16
  %lo = shl i32 %a, 16
  %r = or i32 %hi, %lo
  ret i32 %r
}
