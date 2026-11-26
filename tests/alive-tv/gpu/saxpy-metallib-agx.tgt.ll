; llvm-air2agx -mcpu=agx-g14c --emit=llvm on that .metallib: agx-lower-air and
; -O3 with the AGX passes (readonly / writeonly / memory effects dropped, as
; air2agx-tv does).
target datalayout = "e-m:e-p:64:64-p1:64:64-p3:32:32-p4:64:64-p5:32:32-i1:8-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-v16:16-v32:32-v64:64-v96:32-v128:128-n16:32:64-S32-A5-G1"
target triple = "agx"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define agx_kernel void @saxpy(ptr addrspace(1) nofree captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) nofree captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) nofree captures(none) "air-buffer-no-alias" %2) local_unnamed_addr #0 {
entry:
  %3 = tail call i32 @llvm.agx.thread.position.in.grid.x()
  %4 = getelementptr inbounds nuw i8, ptr addrspace(1) %2, i64 4
  %5 = load i32, ptr addrspace(1) %4, align 4
  %6 = icmp slt i32 %3, %5
  br i1 %6, label %7, label %16

7:                                                ; preds = %entry
  %8 = load i32, ptr addrspace(1) %2, align 4
  %9 = sext i32 %3 to i64
  %10 = getelementptr [4 x i8], ptr addrspace(1) %0, i64 %9
  %11 = getelementptr [4 x i8], ptr addrspace(1) %1, i64 %9
  %12 = load i32, ptr addrspace(1) %10, align 4
  %13 = load i32, ptr addrspace(1) %11, align 4
  %14 = mul i32 %12, %8
  %15 = add i32 %14, %13
  store i32 %15, ptr addrspace(1) %11, align 4
  br label %16

16:                                               ; preds = %7, %entry
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.agx.thread.position.in.grid.x() #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn "target-cpu"="agx-g14c" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn }

!29 = distinct !{!29, !31, !"air-alias-scope-arg(2)"}
!30 = !{!29}
!31 = distinct !{!31, i1 false, !"air-alias-scopes(saxpy)"}
!32 = distinct !{!32, !31, !"air-alias-scope-arg(0)"}
!33 = distinct !{!33, !31, !"air-alias-scope-arg(1)"}
!34 = !{!32, !33}
!35 = !{!32}
!36 = !{!33, !29}
!37 = !{!33}
!38 = !{!32, !29}
