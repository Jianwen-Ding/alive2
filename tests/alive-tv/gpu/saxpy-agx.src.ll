; The integer saxpy kernel (upstream GPU IR) compared directly with its AGX IR.
declare { i32, i32, i32 } @air.thread_position_in_grid.v3i32()
declare i32 @air.thread_index_in_simdgroup()

define void @saxpy(i32 %alpha, ptr addrspace(1) %x, ptr addrspace(1) %y, i32 %n) "air-kernel" {
entry:
  %t = call { i32, i32, i32 } @air.thread_position_in_grid.v3i32()
  %i = extractvalue { i32, i32, i32 } %t, 0
  %lane = call i32 @air.thread_index_in_simdgroup()
  %tmp = alloca [4 x i32]
  %slot = getelementptr [4 x i32], ptr %tmp, i32 0, i32 %lane
  store i32 1, ptr %slot
  %ok = icmp slt i32 %i, %n
  br i1 %ok, label %body, label %done
body:
  %px = getelementptr i32, ptr addrspace(1) %x, i32 %i
  %py = getelementptr i32, ptr addrspace(1) %y, i32 %i
  %vx = load i32, ptr addrspace(1) %px
  %vy = load i32, ptr addrspace(1) %py
  %s = load i32, ptr %slot
  %ax = mul i32 %alpha, %vx
  %r0 = add i32 %ax, %vy
  %r = mul i32 %r0, %s
  store i32 %r, ptr addrspace(1) %py
  br label %done
done:
  ret void
}
