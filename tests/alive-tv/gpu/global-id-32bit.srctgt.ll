; convert-gpu-to-air lowers gpu.global_id to AIR's 32-bit thread_position_in_grid
; only where the dimension has fewer than 2^32 threads (upper_bound, or known
; sizes): then the exact index arithmetic block_id * block_dim + thread_id is
; the zero-extended register, which wraps modulo 2^32. (At half the widths:
; the 64-bit form is the same identity, which the solver takes long over.)
define i32 @src(i16 %block, i16 %dim, i16 %tid) {
  %b = zext i16 %block to i32
  %d = zext i16 %dim to i32
  %t = zext i16 %tid to i32
  %m = mul i32 %b, %d
  %g = add i32 %m, %t
  %ok = icmp ult i32 %g, 65536
  call void @llvm.assume(i1 %ok)
  ret i32 %g
}

define i32 @tgt(i16 %block, i16 %dim, i16 %tid) {
  %m = mul i16 %block, %dim
  %g = add i16 %m, %tid
  %z = zext i16 %g to i32
  ret i32 %z
}

declare void @llvm.assume(i1)
