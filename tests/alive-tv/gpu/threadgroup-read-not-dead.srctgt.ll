; A threadgroup variable that is read is not write-only: another thread's store
; may be what the load reads after the barrier, so the store before the
; barrier may not be deleted.
target triple = "air64-apple-macosx26.0.0"

@tg = internal addrspace(3) global [64 x i32] poison, align 4

declare void @air.wg.barrier(i32, i32)

define void @src(ptr addrspace(1) %out, i32 %v, i32 %i, i32 %j) {
  %p = getelementptr i32, ptr addrspace(3) @tg, i32 %i
  store i32 %v, ptr addrspace(3) %p
  call void @air.wg.barrier(i32 2, i32 1)
  %q = getelementptr i32, ptr addrspace(3) @tg, i32 %j
  %x = load i32, ptr addrspace(3) %q
  store i32 %x, ptr addrspace(1) %out
  ret void
}

define void @tgt(ptr addrspace(1) %out, i32 %v, i32 %i, i32 %j) {
  call void @air.wg.barrier(i32 2, i32 1)
  %q = getelementptr i32, ptr addrspace(3) @tg, i32 %j
  %x = load i32, ptr addrspace(3) %q
  store i32 %x, ptr addrspace(1) %out
  ret void
}

; ERROR: Source is more defined than target
