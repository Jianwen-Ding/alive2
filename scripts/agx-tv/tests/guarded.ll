; A guarded load-modify-store: control flow, a special register, scaled
; device addresses and the wait of a load.
target triple = "agx"
define agx_kernel void @k(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %n) {
entry:
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %c = icmp slt i32 %t, %n
  br i1 %c, label %body, label %done
body:
  %p = getelementptr i32, ptr addrspace(1) %in, i32 %t
  %v = load i32, ptr addrspace(1) %p
  %m = mul i32 %v, 3
  %a = add i32 %m, %n
  %x = xor i32 %a, 255
  %q = getelementptr i32, ptr addrspace(1) %out, i32 %t
  store i32 %x, ptr addrspace(1) %q
  br label %done
done:
  ret void
}
declare i32 @llvm.agx.thread.position.in.grid.x()
