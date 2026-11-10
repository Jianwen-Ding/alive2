; AIR's atomic functions (the portable AIR of atomics-air.src.ll) and their AGX
; IR: atomicrmw and cmpxchg in the orders !agx.atomic.ordering records.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air"

@tg = internal addrspace(3) global [64 x i32] poison, align 4

declare i32 @air.atomic.global.add.s.i32(ptr addrspace(1), i32, i32, i32, i1)

declare i32 @air.atomic.local.max.u.i32(ptr addrspace(3), i32, i32, i32, i1)

declare void @air.atomic.fence(i32, i32, i32)

declare i32 @air.atomic.global.cmpxchg.weak.i32(ptr addrspace(1), ptr, i32, i32, i32, i32, i1)

define void @k(ptr addrspace(1) %ctr, ptr addrspace(1) %out, ptr addrspace(2) noalias noundef nonnull readonly align 4 dereferenceable(4) %v) #0 {
  %v.val = load i32, ptr addrspace(2) %v, align 4
  %cmpxchg.expected = alloca i32, align 4
  %old1 = call i32 @air.atomic.global.add.s.i32(ptr addrspace(1) %ctr, i32 %v.val, i32 0, i32 2, i1 false)
  call void @air.atomic.fence(i32 2, i32 5, i32 1)
  %m2 = call i32 @air.atomic.local.max.u.i32(ptr addrspace(3) @tg, i32 %old1, i32 0, i32 1, i1 false), !agx.atomic.ordering !0
  call void @air.atomic.fence(i32 2, i32 5, i32 1)
  call void @air.atomic.fence(i32 3, i32 5, i32 2)
  store i32 %m2, ptr %cmpxchg.expected, align 4
  %1 = call i32 @air.atomic.global.cmpxchg.weak.i32(ptr addrspace(1) %ctr, ptr %cmpxchg.expected, i32 0, i32 0, i32 0, i32 2, i1 false), !agx.atomic.ordering !1
  call void @air.atomic.fence(i32 3, i32 5, i32 2)
  %2 = icmp eq i32 %1, %m2
  %3 = insertvalue { i32, i1 } poison, i32 %1, 0
  %4 = insertvalue { i32, i1 } %3, i1 %2, 1
  %f = extractvalue { i32, i1 } %4, 0
  store i32 %f, ptr addrspace(1) %out, align 4
  ret void

; uselistorder directives
  uselistorder ptr addrspace(1) %ctr, { 1, 0 }
}

attributes #0 = { "air-kernel" }

!air.kernel = !{!7}
!air.version = !{!8}
!air.language_version = !{!9}
!air.compile_options = !{!10, !11, !12}
!llvm.module.flags = !{!13, !14, !15, !16, !17, !18}

!0 = !{i32 5}
!1 = !{i32 4, i32 0}
!2 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"ctr"}
!3 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"out"}
!4 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"v"}
!5 = !{}
!6 = !{!2, !3, !4}
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
; TEST-ARGS: -smt-to=60000
