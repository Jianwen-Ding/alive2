; A barrier orders the threadgroup's memory: a store may not move above it
; (the canonical form: fences, which may change memory other threads see,
; around the threadgroup barrier).
target triple = "air64-apple-macosx26.0.0"

@tg = internal addrspace(3) global [64 x i32] poison, align 4

declare void @air.wg.barrier(i32, i32)

define void @src(ptr addrspace(1) %out, i32 %v) {
  call void @air.wg.barrier(i32 2, i32 1)
  store i32 %v, ptr addrspace(3) @tg
  %x = load i32, ptr addrspace(3) @tg
  store i32 %x, ptr addrspace(1) %out
  ret void
}

define void @tgt(ptr addrspace(1) %out, i32 %v) {
  store i32 %v, ptr addrspace(3) @tg
  call void @air.wg.barrier(i32 2, i32 1)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; ERROR: Source is more defined than target
