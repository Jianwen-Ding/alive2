; The AIR target's portable passes on saxpy-air.src.ll: scalar arguments in
; constant buffers, system values as parameters described by !air.kernel.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air"

define void @saxpy(ptr addrspace(2) noalias noundef nonnull readonly align 4 dereferenceable(4) %alpha, ptr addrspace(1) %x, ptr addrspace(1) %y, ptr addrspace(2) noalias noundef nonnull readonly align 4 dereferenceable(4) %n, <3 x i32> %tid, i32 %simdlane) #0 {
entry:
  %tid_x = extractelement <3 x i32> %tid, i32 0
  %alpha.val = load i32, ptr addrspace(2) %alpha, align 4
  %n.val = load i32, ptr addrspace(2) %n, align 4
  %tmp = alloca [4 x i32], align 4
  %slot = getelementptr [4 x i32], ptr %tmp, i32 0, i32 %simdlane
  store i32 1, ptr %slot, align 4
  %ok = icmp slt i32 %tid_x, %n.val
  br i1 %ok, label %body, label %done

body:                                             ; preds = %entry
  %px = getelementptr i32, ptr addrspace(1) %x, i32 %tid_x
  %py = getelementptr i32, ptr addrspace(1) %y, i32 %tid_x
  %vx = load i32, ptr addrspace(1) %px, align 4
  %vy = load i32, ptr addrspace(1) %py, align 4
  %s = load i32, ptr %slot, align 4
  %ax = mul i32 %alpha.val, %vx
  %r0 = add i32 %ax, %vy
  %r = mul i32 %r0, %s
  store i32 %r, ptr addrspace(1) %py, align 4
  br label %done

done:                                             ; preds = %body, %entry
  ret void

; uselistorder directives
  uselistorder i32 %tid_x, { 2, 1, 0 }
}

attributes #0 = { "air-kernel" }

!air.kernel = !{!8}
!air.version = !{!9}
!air.language_version = !{!10}
!air.compile_options = !{!11, !12, !13}
!llvm.module.flags = !{!14, !15, !16, !17, !18, !19}

!0 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"alpha"}
!1 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"x"}
!2 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"y"}
!3 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"n"}
!4 = !{i32 4, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"tid"}
!5 = !{i32 5, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"simdlane"}
!6 = !{}
!7 = !{!0, !1, !2, !3, !4, !5}
!8 = !{ptr @saxpy, !6, !7}
!9 = !{i32 2, i32 8, i32 0}
!10 = !{!"Metal", i32 4, i32 0, i32 0}
!11 = !{!"air.compile.denorms_disable"}
!12 = !{!"air.compile.fast_math_enable"}
!13 = !{!"air.compile.framebuffer_fetch_enable"}
!14 = !{i32 7, !"air.max_device_buffers", i32 31}
!15 = !{i32 7, !"air.max_constant_buffers", i32 31}
!16 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!17 = !{i32 7, !"air.max_textures", i32 128}
!18 = !{i32 7, !"air.max_read_write_textures", i32 8}
!19 = !{i32 7, !"air.max_samplers", i32 16}
