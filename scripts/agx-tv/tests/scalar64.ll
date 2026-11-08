; A 64-bit scalar argument (in a constant buffer in alive2's GPU form) beside
; buffers that may alias, in a module without a datalayout.
target triple = "agx"
define agx_kernel void @k(ptr addrspace(1) %out, ptr addrspace(1) %in, i64 %n) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %tw = zext i32 %t to i64
  %c = icmp ult i64 %tw, %n
  br i1 %c, label %body, label %done
body:
  %p = getelementptr i32, ptr addrspace(1) %in, i64 %tw
  %v = load i32, ptr addrspace(1) %p
  %q = getelementptr i32, ptr addrspace(1) %out, i64 %tw
  store i32 %v, ptr addrspace(1) %q
  br label %done
done:
  ret void
}
declare i32 @llvm.agx.thread.position.in.grid.x()
