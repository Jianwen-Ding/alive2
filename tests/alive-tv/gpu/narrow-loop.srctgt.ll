; convert-gpu-to-air runs an scf.for over index whose iterations a memref
; access keeps in bounds as a loop over an unsigned i32 offset from the
; lower bound (narrow-loops): each iteration accesses index x + iv of a
; static dimension of S <= 2^31 elements (out of bounds is undefined: the
; assume), so the offsets iv - lb stay below S, and the loop over
; o = 0 to umin(ub - lb, S) (0 for ub <= lb) by umin(step, S) has the same
; iterations. Here at an eighth of the widths (index i8, the offset i4,
; S = 8 = 2^3), unrolled past the most iterations either loop can run.
; TEST-ARGS: -src-unroll=10 -tgt-unroll=10 -smt-to=600000

define i8 @src(i8 %lb, i8 %ub, i8 %st, i8 %x) {
entry:
  %pos = icmp sgt i8 %st, 0
  call void @llvm.assume(i1 %pos)
  %c0 = icmp slt i8 %lb, %ub
  br i1 %c0, label %body, label %exit

body:
  %i = phi i8 [ %lb, %entry ], [ %inext, %body ]
  %acc = phi i8 [ 0, %entry ], [ %acc1, %body ]
  %idx = add i8 %x, %i
  %ib = icmp ult i8 %idx, 8
  call void @llvm.assume(i1 %ib)
  %h = mul i8 %acc, 5
  %acc1 = xor i8 %h, %i
  %inext = add nsw i8 %i, %st
  %c = icmp slt i8 %inext, %ub
  br i1 %c, label %body, label %exit

exit:
  %r = phi i8 [ 0, %entry ], [ %acc1, %body ]
  ret i8 %r
}

define i8 @tgt(i8 %lb, i8 %ub, i8 %st, i8 %x) {
entry:
  %gt = icmp sgt i8 %ub, %lb
  %d = sub i8 %ub, %lb
  %dm = call i8 @llvm.umin.i8(i8 %d, i8 8)
  %ext = select i1 %gt, i8 %dm, i8 0
  %stm = call i8 @llvm.umin.i8(i8 %st, i8 8)
  %e = trunc i8 %ext to i4
  %s = trunc i8 %stm to i4
  %c0 = icmp ult i4 0, %e
  br i1 %c0, label %body, label %exit

body:
  %o = phi i4 [ 0, %entry ], [ %onext, %body ]
  %acc = phi i8 [ 0, %entry ], [ %acc1, %body ]
  %ow = zext i4 %o to i8
  %i = add i8 %lb, %ow
  %h = mul i8 %acc, 5
  %acc1 = xor i8 %h, %i
  %onext = add nuw i4 %o, %s
  %c = icmp ult i4 %onext, %e
  br i1 %c, label %body, label %exit

exit:
  %r = phi i8 [ 0, %entry ], [ %acc1, %body ]
  ret i8 %r
}

declare void @llvm.assume(i1)
declare i8 @llvm.umin.i8(i8, i8)
