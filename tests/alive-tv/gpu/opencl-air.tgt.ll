; The portable AIR of opencl-air.src.ll.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air"

@tg = internal addrspace(3) global [64 x i32] poison, align 4

define void @k(ptr addrspace(1) %out, <3 x i32> %pid, <3 x i32> %tid, <3 x i32> %tidtg, <3 x i32> %tgsize) #0 {
  %tgsize_z = extractelement <3 x i32> %tgsize, i32 2
  %tidtg_y = extractelement <3 x i32> %tidtg, i32 1
  %tid_x = extractelement <3 x i32> %tid, i32 0
  %pid_x = extractelement <3 x i32> %pid, i32 0
  %1 = zext i32 %tid_x to i64
  %2 = zext i32 %tidtg_y to i64
  %3 = zext i32 %tgsize_z to i64
  %4 = zext i32 %pid_x to i64
  %g32 = trunc i64 %1 to i32
  %slot = getelementptr i32, ptr addrspace(3) @tg, i64 %2
  store i32 %g32, ptr addrspace(3) %slot, align 4
  call void @air.wg.barrier(i32 2, i32 1)
  %v = load i32, ptr addrspace(3) @tg, align 4
  %s32 = trunc i64 %3 to i32
  %b32 = trunc i64 %4 to i32
  %sum = add i32 %v, %s32
  %sum2 = add i32 %sum, %b32
  %p = getelementptr i32, ptr addrspace(1) %out, i64 %1
  store i32 %sum2, ptr addrspace(1) %p, align 4
  ret void

; uselistorder directives
  uselistorder i64 %1, { 1, 0 }
}

declare void @air.wg.barrier(i32, i32)

attributes #0 = { "air-from-generic-gpu" "air-kernel" }

!air.kernel = !{!7}
!air.version = !{!8}
!air.language_version = !{!9}
!air.compile_options = !{!10, !11, !12}
!llvm.module.flags = !{!13, !14, !15, !16, !17, !18}

!0 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"out"}
!1 = !{i32 1, !"air.threadgroup_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"pid"}
!2 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"tid"}
!3 = !{i32 3, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint3", !"air.arg_name", !"tidtg"}
!4 = !{i32 4, !"air.threads_per_threadgroup", !"air.arg_type_name", !"uint3", !"air.arg_name", !"tgsize"}
!5 = !{}
!6 = !{!0, !1, !2, !3, !4}
!7 = !{ptr @k, !5, !6}
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
