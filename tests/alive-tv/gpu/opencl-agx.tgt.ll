; agx-lower-air on the portable AIR of opencl-air.src.ll.
target datalayout = "e-m:e-p:64:64-p1:64:64-p3:32:32-p4:64:64-p5:32:32-i1:8-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-v16:16-v32:32-v64:64-v96:32-v128:128-n16:32:64-S32-A5-G1"
target triple = "agx"

@tg = internal addrspace(3) global [64 x i32] poison, align 4

define agx_kernel void @k(ptr addrspace(1) %out) #0 {
entry:
  %tgsize_z = call i32 @llvm.agx.threads.per.threadgroup.z()
  %tidtg_y = call i32 @llvm.agx.thread.position.in.threadgroup.y()
  %tid_x = call i32 @llvm.agx.thread.position.in.grid.x()
  %pid_x = call i32 @llvm.agx.threadgroup.position.in.grid.x()
  %0 = zext i32 %tid_x to i64
  %1 = zext i32 %tidtg_y to i64
  %2 = zext i32 %tgsize_z to i64
  %3 = zext i32 %pid_x to i64
  %g32 = trunc i64 %0 to i32
  %slot = getelementptr i32, ptr addrspace(3) @tg, i64 %1
  store i32 %g32, ptr addrspace(3) %slot, align 4
  fence syncscope("workgroup") release
  call void @llvm.agx.threadgroup.barrier()
  fence syncscope("workgroup") acquire
  %v = load i32, ptr addrspace(3) @tg, align 4
  %s32 = trunc i64 %2 to i32
  %b32 = trunc i64 %3 to i32
  %sum = add i32 %v, %s32
  %sum2 = add i32 %sum, %b32
  %p = getelementptr i32, ptr addrspace(1) %out, i64 %0
  store i32 %sum2, ptr addrspace(1) %p, align 4
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.threadgroup.position.in.grid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.thread.position.in.grid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.thread.position.in.threadgroup.y() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.threads.per.threadgroup.z() #1

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.agx.threadgroup.barrier() #2

attributes #0 = { "air-from-generic-gpu" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nofree nounwind willreturn }
