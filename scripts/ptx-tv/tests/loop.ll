; The sum of the squares below n.
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop(ptr addrspace(1) %a, i32 %n) {
entry:
  %c0 = icmp sgt i32 %n, 0
  br i1 %c0, label %body, label %exit
body:
  %i = phi i32 [0, %entry], [%i1, %body]
  %s = phi i32 [0, %entry], [%s1, %body]
  %sq = mul i32 %i, %i
  %s1 = add i32 %s, %sq
  %i1 = add i32 %i, 1
  %c = icmp slt i32 %i1, %n
  br i1 %c, label %body, label %done
done:
  store i32 %s1, ptr addrspace(1) %a, align 4
  br label %exit
exit:
  ret void
}
