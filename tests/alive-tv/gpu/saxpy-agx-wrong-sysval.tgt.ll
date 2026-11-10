; AGX IR that reads the thread's position in its threadgroup where AIR
; reads its position in the grid.
target datalayout = "e-m:e-p:64:64-p1:64:64-p3:32:32-p4:64:64-p5:32:32-i1:8-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-v16:16-v32:32-v64:64-v96:32-v128:128-n16:32:64-S32-A5-G1"
target triple = "agx"

define agx_kernel void @saxpy(ptr addrspace(4) noalias noundef nonnull readonly byref(i32) align 4 dereferenceable(4) %alpha, ptr addrspace(1) %x, ptr addrspace(1) %y, ptr addrspace(4) noalias noundef nonnull readonly byref(i32) align 4 dereferenceable(4) %n) {
entry:
  %0 = call i32 @llvm.agx.thread.index.in.simdgroup()
  %tid_x = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %alpha.val = load i32, ptr addrspace(4) %alpha, align 4
  %n.val = load i32, ptr addrspace(4) %n, align 4
  %tmp = alloca [4 x i32], align 4, addrspace(5)
  %slot = getelementptr [4 x i32], ptr addrspace(5) %tmp, i32 0, i32 %0
  store i32 1, ptr addrspace(5) %slot, align 4
  %ok = icmp slt i32 %tid_x, %n.val
  br i1 %ok, label %body, label %done

body:                                             ; preds = %entry
  %px = getelementptr i32, ptr addrspace(1) %x, i32 %tid_x
  %py = getelementptr i32, ptr addrspace(1) %y, i32 %tid_x
  %vx = load i32, ptr addrspace(1) %px, align 4
  %vy = load i32, ptr addrspace(1) %py, align 4
  %s = load i32, ptr addrspace(5) %slot, align 4
  %ax = mul i32 %alpha.val, %vx
  %r0 = add i32 %ax, %vy
  %r = mul i32 %r0, %s
  store i32 %r, ptr addrspace(1) %py, align 4
  br label %done

done:                                             ; preds = %body, %entry
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.thread.position.in.threadgroup.x() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.thread.index.in.simdgroup() #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
