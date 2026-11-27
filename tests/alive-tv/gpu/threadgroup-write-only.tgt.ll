; threadgroup-write-only.src.ll with the write-only @marker deleted.
target triple = "air64-apple-macosx26.0.0"

@tg = internal addrspace(3) global [64 x i32] poison, align 4

declare void @air.wg.barrier(i32, i32)

define void @k(ptr addrspace(1) %out, i32 %v, i32 %i) {
  %p = getelementptr i32, ptr addrspace(3) @tg, i32 %i
  store i32 %v, ptr addrspace(3) %p
  call void @air.wg.barrier(i32 2, i32 1)
  %q = getelementptr i32, ptr addrspace(3) @tg, i32 0
  %x = load i32, ptr addrspace(3) %q
  store i32 %x, ptr addrspace(1) %out
  ret void
}
