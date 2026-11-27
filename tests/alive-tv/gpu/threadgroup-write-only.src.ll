; A threadgroup variable the kernel only stores to is read by no thread (every
; thread runs this function): its stores are dead, and deleting the variable
; (IPSCCP, GlobalOpt) is correct, across a barrier too. Source and target are
; modules of their own, as each is when a pass is checked: a variable that
; another function of the module uses is not this kernel's alone.
target triple = "air64-apple-macosx26.0.0"

@marker = internal addrspace(3) global i32 undef, align 4
@tg = internal addrspace(3) global [64 x i32] poison, align 4

declare void @air.wg.barrier(i32, i32)

define void @k(ptr addrspace(1) %out, i32 %v, i32 %i) {
  store i32 7, ptr addrspace(3) @marker
  %p = getelementptr i32, ptr addrspace(3) @tg, i32 %i
  store i32 %v, ptr addrspace(3) %p
  call void @air.wg.barrier(i32 2, i32 1)
  %q = getelementptr i32, ptr addrspace(3) @tg, i32 0
  %x = load i32, ptr addrspace(3) %q
  store i32 %x, ptr addrspace(1) %out
  ret void
}
