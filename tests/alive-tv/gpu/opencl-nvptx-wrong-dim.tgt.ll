; opencl-nvptx.tgt.ll with ctaid.y for get_group_id(0): wrong.
target triple = "nvptx64-nvidia-cuda"

@tg = internal addrspace(3) global [64 x i32] poison, align 4

define ptx_kernel void @k(ptr addrspace(1) %out) {
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ntid = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %ctaid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %m = mul i32 %ctaid, %ntid
  %g32 = add i32 %m, %tid
  %ly = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %l = zext i32 %ly to i64
  %slot = getelementptr i32, ptr addrspace(3) @tg, i64 %l
  store i32 %g32, ptr addrspace(3) %slot
  call void @llvm.nvvm.barrier0()
  %v = load i32, ptr addrspace(3) @tg
  %sz = call i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
  %sum = add i32 %v, %sz
  %cy = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %sum2 = add i32 %sum, %cy
  %g = zext i32 %g32 to i64
  %p = getelementptr i32, ptr addrspace(1) %out, i64 %g
  store i32 %sum2, ptr addrspace(1) %p
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
declare void @llvm.nvvm.barrier0()
