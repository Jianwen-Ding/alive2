; llvm.nvvm.mulhi.i (mul.hi.s32): the high word of the signed product.
; TEST-ARGS: -disable-undef-input
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.mulhi.i(i32, i32)

define i32 @src(i32 %a, i32 %b) {
  %h = call i32 @llvm.nvvm.mulhi.i(i32 %a, i32 %b)
  ret i32 %h
}

define i32 @tgt(i32 %a, i32 %b) {
  %1 = sext i32 %a to i64
  %2 = sext i32 %b to i64
  %3 = mul i64 %1, %2
  %4 = lshr i64 %3, 32
  %5 = trunc i64 %4 to i32
  ret i32 %5
}
