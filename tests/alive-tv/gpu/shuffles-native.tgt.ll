; mlir-opt --convert-gpu-to-air and mlir-translate on the kernel of
; shuffles-native.src.ll.

; Function Attrs: convergent nounwind willreturn memory(none)
declare i32 @air.simd_shuffle_xor.u.i32(i32, i16) #0

; Function Attrs: convergent nounwind willreturn memory(none)
declare i32 @air.simd_shuffle_down.u.i32(i32, i16) #0

; Function Attrs: convergent nounwind willreturn memory(none)
declare i32 @air.simd_shuffle_up.u.i32(i32, i16) #0

; Function Attrs: nounwind willreturn memory(none)
declare i32 @air.thread_index_in_simdgroup() #1

define void @k(ptr addrspace(1) %0, i32 %1) #2 {
  %3 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } poison, ptr addrspace(1) %0, 0
  %4 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %3, ptr addrspace(1) %0, 1
  %5 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %4, i64 0, 2
  %6 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %5, i64 4, 3, 0
  %7 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %6, i64 1, 4, 0
  %8 = call range(i32 0, 32) i32 @air.thread_index_in_simdgroup()
  %9 = icmp uge i32 %8, 2
  %10 = call i32 @air.simd_shuffle_up.u.i32(i32 %1, i16 2)
  %11 = call range(i32 0, 32) i32 @air.thread_index_in_simdgroup()
  %12 = icmp ult i32 %11, 30
  %13 = call i32 @air.simd_shuffle_down.u.i32(i32 %1, i16 2)
  %14 = call i32 @air.simd_shuffle_xor.u.i32(i32 %1, i16 2)
  %15 = select i1 %9, i32 %10, i32 %1
  %16 = select i1 %12, i32 %13, i32 %1
  %17 = select i1 true, i32 %14, i32 %1
  %18 = extractvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %7, 1
  %19 = getelementptr inbounds nuw i32, ptr addrspace(1) %18, i64 0
  store i32 %15, ptr addrspace(1) %19, align 4
  %20 = extractvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %7, 1
  %21 = getelementptr inbounds nuw i32, ptr addrspace(1) %20, i64 1
  store i32 %16, ptr addrspace(1) %21, align 4
  %22 = extractvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %7, 1
  %23 = getelementptr inbounds nuw i32, ptr addrspace(1) %22, i64 2
  store i32 %17, ptr addrspace(1) %23, align 4
  ret void
}

attributes #0 = { convergent nounwind willreturn memory(none) }
attributes #1 = { nounwind willreturn memory(none) }
attributes #2 = { "air-kernel" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
