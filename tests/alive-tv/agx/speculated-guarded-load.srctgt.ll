; GPUToAIR speculate-guarded-loads (L15-E736): in = j < 256 guarding a load
; of a static row of 256 floats, v = in ? x[j] : 0, as the load at
; umin(j, 255), always within the row, and a select (the memref's elements
; are dereferenceable: its static shape).

define float @src(ptr dereferenceable(1024) align 4 %x, i64 %j) {
entry:
  %in = icmp ult i64 %j, 256
  br i1 %in, label %load, label %join
load:
  %p = getelementptr inbounds float, ptr %x, i64 %j
  %l = load float, ptr %p, align 4
  br label %join
join:
  %v = phi float [ %l, %load ], [ 0.0, %entry ]
  ret float %v
}

define float @tgt(ptr dereferenceable(1024) align 4 %x, i64 %j) {
entry:
  %in = icmp ult i64 %j, 256
  %c = call i64 @llvm.umin.i64(i64 %j, i64 255)
  %p = getelementptr inbounds float, ptr %x, i64 %c
  %l = load float, ptr %p, align 4
  %v = select i1 %in, float %l, float 0.0
  ret float %v
}

declare i64 @llvm.umin.i64(i64, i64)
