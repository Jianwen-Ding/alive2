; CUDA's __mulhi is the high word of the signed product, which the AIR
; target's llvm-to-air-intrinsics computed from the unsigned one (zext):
; __mulhi(-1610612736, 7) is -3, not 4. (It uses sext now.)
target triple = "air64-apple-macosx26.0.0"

declare i32 @__mulhi(i32, i32)

define i32 @src(i32 %a, i32 %b) {
  %h = call i32 @__mulhi(i32 %a, i32 %b)
  ret i32 %h
}

define i32 @tgt(i32 %a, i32 %b) {
  %1 = zext i32 %a to i64
  %2 = zext i32 %b to i64
  %3 = mul i64 %1, %2
  %4 = lshr i64 %3, 32
  %5 = trunc i64 %4 to i32
  ret i32 %5
}

; ERROR: Value mismatch
