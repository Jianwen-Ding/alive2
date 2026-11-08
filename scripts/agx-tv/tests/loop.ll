; A loop (push_exec, while_icmp, jmp_exec_any, pop_exec) over a device array.
target triple = "agx"

define agx_kernel void @sum(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %n) {
entry:
  %c0 = icmp sgt i32 %n, 0
  br i1 %c0, label %loop, label %done
loop:
  %i = phi i32 [ 0, %entry ], [ %i1, %loop ]
  %s = phi i32 [ 0, %entry ], [ %s1, %loop ]
  %p = getelementptr i32, ptr addrspace(1) %in, i32 %i
  %v = load i32, ptr addrspace(1) %p
  %s1 = add i32 %s, %v
  %i1 = add i32 %i, 1
  %c = icmp slt i32 %i1, %n
  br i1 %c, label %loop, label %done
done:
  %r = phi i32 [ 0, %entry ], [ %s1, %loop ]
  store i32 %r, ptr addrspace(1) %out
  ret void
}
