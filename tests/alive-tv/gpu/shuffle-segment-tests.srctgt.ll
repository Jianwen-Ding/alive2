; convert-gpu-to-air (L15-E306): gpu.shuffle within segments of w lanes by a
; constant, the source lane's test as the lane's position in its segment:
; down by k, lane + k < (lane & -w) + w is lane % w < w - k; up by k,
; lane - k >= (lane & -w) is lane % w >= k; idx of lane c, c in the lane's
; segment is lane & -w == c & -w (here w = 8, k = 3, c = 13).

define i3 @src(i32 %x) {
  %lane = and i32 %x, 31
  %start = and i32 %lane, -8
  %end = add i32 %start, 8
  %dsrc = add i32 %lane, 3
  %down = icmp slt i32 %dsrc, %end
  %usrc = sub i32 %lane, 3
  %up = icmp sge i32 %usrc, %start
  %ige = icmp sge i32 13, %start
  %ilt = icmp slt i32 13, %end
  %idx = and i1 %ige, %ilt
  %d = zext i1 %down to i3
  %u0 = zext i1 %up to i3
  %u = shl i3 %u0, 1
  %i0 = zext i1 %idx to i3
  %i = shl i3 %i0, 2
  %r0 = or i3 %d, %u
  %r = or i3 %r0, %i
  ret i3 %r
}

define i3 @tgt(i32 %x) {
  %lane = and i32 %x, 31
  %pos = and i32 %lane, 7
  %down = icmp ult i32 %pos, 5
  %up = icmp uge i32 %pos, 3
  %seg = and i32 %lane, -8
  %idx = icmp eq i32 %seg, 8
  %d = zext i1 %down to i3
  %u0 = zext i1 %up to i3
  %u = shl i3 %u0, 1
  %i0 = zext i1 %idx to i3
  %i = shl i3 %i0, 2
  %r0 = or i3 %d, %u
  %r = or i3 %r0, %i
  ret i3 %r
}
