; The integer saxpy kernel (saxpy-air.src.ll) as the AIR in the .metallib the
; AIR target writes for Metal (llc -mtriple=air -filetype=obj), read back by
; llvm-air2agx, without its scoped alias metadata (scripts/agx-tv/air2agx-tv).
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @saxpy(ptr addrspace(1) "air-buffer-no-alias" %0, ptr addrspace(1) "air-buffer-no-alias" %1, ptr addrspace(1) "air-buffer-no-alias" %2, <3 x i32> %3, i32 %4) {
  %6 = extractelement <3 x i32> %3, i32 0
  %7 = getelementptr inbounds float, ptr addrspace(1) %2, i64 0
  %8 = load float, ptr addrspace(1) %7, align 4
  %9 = bitcast float %8 to i32
  %10 = getelementptr inbounds float, ptr addrspace(1) %2, i64 1
  %11 = load float, ptr addrspace(1) %10, align 4
  %12 = bitcast float %11 to i32
  %13 = alloca [4 x i32], align 4
  %14 = getelementptr [4 x i32], ptr %13, i32 0, i32 %4
  store i32 1, ptr %14, align 4
  %15 = icmp slt i32 %6, %12
  br i1 %15, label %16, label %25

16:                                               ; preds = %5
  %17 = getelementptr i32, ptr addrspace(1) %0, i32 %6
  %18 = getelementptr i32, ptr addrspace(1) %1, i32 %6
  %19 = load i32, ptr addrspace(1) %17, align 4
  %20 = load i32, ptr addrspace(1) %18, align 4
  %21 = load i32, ptr %14, align 4
  %22 = mul i32 %9, %19
  %23 = add i32 %22, %20
  %24 = mul i32 %23, %21
  store i32 %24, ptr addrspace(1) %18, align 4
  br label %25

25:                                               ; preds = %16, %5
  ret void
}

!air.kernel = !{!0}
!air.version = !{!8}
!air.language_version = !{!9}
!air.compile_options = !{!10, !11, !12}
!llvm.module.flags = !{!13, !14, !15, !16, !17, !18}

!0 = !{ptr @saxpy, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6, !7}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"x"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"y"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"_scalar_buf"}
!6 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"tid"}
!7 = !{i32 4, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"simdlane"}
!8 = !{i32 2, i32 8, i32 0}
!9 = !{!"Metal", i32 4, i32 0, i32 0}
!10 = !{!"air.compile.denorms_disable"}
!11 = !{!"air.compile.fast_math_enable"}
!12 = !{!"air.compile.framebuffer_fetch_enable"}
!13 = !{i32 7, !"air.max_device_buffers", i32 31}
!14 = !{i32 7, !"air.max_constant_buffers", i32 31}
!15 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!16 = !{i32 7, !"air.max_textures", i32 128}
!17 = !{i32 7, !"air.max_read_write_textures", i32 8}
!18 = !{i32 7, !"air.max_samplers", i32 16}
!19 = !{!20}
!20 = distinct !{!20, !21, !"air-alias-scope-arg(2)"}
!21 = distinct !{!21, i1 false, !"air-alias-scopes(saxpy)"}
!22 = !{!23, !24}
!23 = distinct !{!23, !21, !"air-alias-scope-arg(0)"}
!24 = distinct !{!24, !21, !"air-alias-scope-arg(1)"}
!25 = !{!23}
!26 = !{!24, !20}
!27 = !{!24}
!28 = !{!23, !20}
