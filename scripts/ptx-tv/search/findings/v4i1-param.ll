; A <4 x i1> argument: the NVPTX backend declares its .param as 1 byte (the
; vector's store size, its bits packed) but passes it one byte per element,
; at offsets 0 to 3: three of the four accesses are past the parameter.
target triple = "nvptx64-nvidia-cuda"

define i32 @third(<4 x i1> %v) {
  %e = extractelement <4 x i1> %v, i32 3
  %r = zext i1 %e to i32
  ret i32 %r
}

define i32 @caller(i32 %x) {
  %c = icmp ne i32 %x, 0
  %v = insertelement <4 x i1> zeroinitializer, i1 %c, i32 3
  %r = call i32 @third(<4 x i1> %v)
  ret i32 %r
}
