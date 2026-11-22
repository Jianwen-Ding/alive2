; y[i] = x[i] ^ 255 for the threads whose global index i is below n.
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @guarded(ptr addrspace(1) %x, ptr addrspace(1) %y, i32 %n) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ctaid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %ntid = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %m = mul i32 %ctaid, %ntid
  %i = add i32 %m, %tid
  %c = icmp slt i32 %i, %n
  br i1 %c, label %body, label %exit
body:
  %idx = sext i32 %i to i64
  %px = getelementptr i32, ptr addrspace(1) %x, i64 %idx
  %v = load i32, ptr addrspace(1) %px, align 4
  %w = xor i32 %v, 255
  %py = getelementptr i32, ptr addrspace(1) %y, i64 %idx
  store i32 %w, ptr addrspace(1) %py, align 4
  br label %exit
exit:
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
