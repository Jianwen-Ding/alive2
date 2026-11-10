; Threadgroup memory: a thread's slot written and read back, scaled
; threadgroup addresses.
target triple = "agx"
@tg = internal addrspace(3) global [64 x i32] undef, align 4
define agx_kernel void @k(ptr addrspace(1) %out, i32 %n) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %c = icmp ult i32 %t, 64
  br i1 %c, label %body, label %done
body:
  %s = getelementptr [64 x i32], ptr addrspace(3) @tg, i32 0, i32 %t
  %v = add i32 %t, %n
  store i32 %v, ptr addrspace(3) %s
  %w = load i32, ptr addrspace(3) %s
  %x = xor i32 %w, 255
  %q = getelementptr i32, ptr addrspace(1) %out, i32 %t
  store i32 %x, ptr addrspace(1) %q
  br label %done
done:
  ret void
}
declare i32 @llvm.agx.thread.position.in.threadgroup.x()
