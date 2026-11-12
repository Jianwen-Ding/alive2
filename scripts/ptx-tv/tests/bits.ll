; Shifts, min/max, population counts and narrow integers.
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @shifts(ptr addrspace(1) %out, i32 %b) {
  %sh = and i32 %b, 31
  %l = shl i32 %b, %sh
  %r = lshr i32 %b, 3
  %mx = call i32 @llvm.umax.i32(i32 %l, i32 %r)
  store i32 %mx, ptr addrspace(1) %out, align 4
  ret void
}

define ptx_kernel void @counts(ptr addrspace(1) %out, i32 %b, i16 %h) {
  %hs = sext i16 %h to i32
  %mn = call i32 @llvm.smin.i32(i32 %b, i32 %hs)
  %pc = call i32 @llvm.ctpop.i32(i32 %mn)
  store i32 %pc, ptr addrspace(1) %out, align 4
  ret void
}

define ptx_kernel void @wide(ptr addrspace(1) %out, i64 %a) {
  %cz = call i64 @llvm.ctlz.i64(i64 %a, i1 false)
  %s = ashr i64 %a, 7
  %x = add i64 %cz, %s
  store i64 %x, ptr addrspace(1) %out, align 8
  ret void
}

define ptx_kernel void @narrow(ptr addrspace(1) %out, i16 %h, i8 %c) {
  %c1 = add i8 %c, 3
  %p2 = getelementptr i8, ptr addrspace(1) %out, i64 16
  store i8 %c1, ptr addrspace(1) %p2, align 1
  %h1 = mul i16 %h, %h
  %p3 = getelementptr i16, ptr addrspace(1) %out, i64 9
  store i16 %h1, ptr addrspace(1) %p3, align 2
  ret void
}

; A device function.
define i32 @dev(i32 %a, i32 %b) {
  %d = sub i32 %a, %b
  %c = icmp ugt i32 %a, %b
  %r = select i1 %c, i32 %d, i32 0
  ret i32 %r
}

declare i32 @llvm.umax.i32(i32, i32)
declare i32 @llvm.smin.i32(i32, i32)
declare i32 @llvm.ctpop.i32(i32)
declare i64 @llvm.ctlz.i64(i64, i1)
