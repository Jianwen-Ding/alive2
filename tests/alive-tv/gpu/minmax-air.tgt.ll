; The portable AIR of minmax-air.src.ll.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air"

define float @f(float %x, float %y) {
  %n = call float @air.fmin.f32(float %x, float %y)
  %1 = bitcast float %x to i32
  %2 = bitcast float %y to i32
  %3 = or i32 %1, %2
  %4 = shl i32 %3, 1
  %zero_check = icmp eq i32 %4, 0
  %5 = bitcast i32 %3 to float
  %6 = select i1 %zero_check, float %5, float %n
  %nan_check = fcmp uno float %x, %y
  %n.nan = select i1 %nan_check, float +qnan, float %6
  %m = call float @air.fmax.f32(float %n.nan, float %y)
  %7 = bitcast float %n.nan to i32
  %8 = bitcast float %y to i32
  %9 = or i32 %7, %8
  %10 = shl i32 %9, 1
  %zero_check1 = icmp eq i32 %10, 0
  %11 = and i32 %7, %8
  %12 = bitcast i32 %11 to float
  %13 = select i1 %zero_check1, float %12, float %m
  %nan_check2 = fcmp uno float %n.nan, %y
  %m.nan = select i1 %nan_check2, float +qnan, float %13
  ret float %m.nan

; uselistorder directives
  uselistorder float %x, { 1, 2, 0 }
  uselistorder float %y, { 2, 3, 0, 4, 5, 1 }
  uselistorder float %n.nan, { 1, 2, 0 }
}

define void @k(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %x = load float, ptr addrspace(1) %in, align 4
  %p = getelementptr float, ptr addrspace(1) %in, i32 1
  %y = load float, ptr addrspace(1) %p, align 4
  %r = call float @f(float %x, float %y)
  store float %r, ptr addrspace(1) %out, align 4
  ret void
}

; Function Attrs: nounwind willreturn memory(none)
declare float @air.fmin.f32(float, float) #1

; Function Attrs: nounwind willreturn memory(none)
declare float @air.fmax.f32(float, float) #1

attributes #0 = { "air-kernel" }
attributes #1 = { nounwind willreturn memory(none) }

!air.kernel = !{!4}
!air.version = !{!5}
!air.language_version = !{!6}
!air.compile_options = !{!7, !8, !9}
!llvm.module.flags = !{!10, !11, !12, !13, !14, !15}

!0 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!1 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"in"}
!2 = !{}
!3 = !{!0, !1}
!4 = !{ptr @k, !2, !3}
!5 = !{i32 2, i32 8, i32 0}
!6 = !{!"Metal", i32 4, i32 0, i32 0}
!7 = !{!"air.compile.denorms_disable"}
!8 = !{!"air.compile.fast_math_enable"}
!9 = !{!"air.compile.framebuffer_fetch_enable"}
!10 = !{i32 7, !"air.max_device_buffers", i32 31}
!11 = !{i32 7, !"air.max_constant_buffers", i32 31}
!12 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!13 = !{i32 7, !"air.max_textures", i32 128}
!14 = !{i32 7, !"air.max_read_write_textures", i32 8}
!15 = !{i32 7, !"air.max_samplers", i32 16}
