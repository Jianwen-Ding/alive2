; NVVM intrinsics that stand for PTX instructions: alive2 gives the source
; their meaning from the PTX formalization (llvm_util/ptx_formal.cpp), and
; ptx-tv the PTX llc prints for them (ptx_formal.py): both read ptx.formal.
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @cvt(ptr addrspace(1) %out, float %f) {
  %c = call i32 @llvm.nvvm.f2i.rz(float %f)
  store i32 %c, ptr addrspace(1) %out, align 4
  ret void
}

define ptx_kernel void @prmt(ptr addrspace(1) %out, i32 %a, i32 %b) {
  %p = call i32 @llvm.nvvm.prmt(i32 %a, i32 %b, i32 4214)
  store i32 %p, ptr addrspace(1) %out, align 4
  ret void
}

declare i32 @llvm.nvvm.f2i.rz(float)
declare i32 @llvm.nvvm.prmt(i32, i32, i32)
