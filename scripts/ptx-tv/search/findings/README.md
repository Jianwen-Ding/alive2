Findings
========

`<N x i1>` parameters are accessed past their declared size
-----------------------------------------------------------

`v4i1-param.ll` (LLVM main of 2026-11-12, `llc -mtriple=nvptx64-nvidia-cuda
-mcpu=sm_80`; `llvm/test/CodeGen/NVPTX/param-load-store.ll`'s `test_v3i1`,
`test_v4i1` and `test_v5i1` show the same, and its CHECK lines expect it).

The backend declares a `<4 x i1>` parameter, return value and call-site
argument with the vector's store size, one byte (its bits packed):

```
.visible .func (.param .b32 func_retval0) third(
	.param .align 1 .b8 third_param_0[1]
)
	ld.param.b8 	%r1, [third_param_0+3];
...
	.param .align 1 .b8 	param0[1];
	st.param.b8 	[param0+3], %rs1;
	st.param.b8 	[param0+2], 0;
	st.param.b8 	[param0+1], 0;
	st.param.b8 	[param0], 0;
	call.uni (retval0), third, (param0);
```

but passes it one byte per element, at offsets 0 to N-1: every access but
the first is past the `.param` it is declared as. ptx-tv rejects both
functions (`third`: target more poisonous than source, its element read
from beyond its parameter; `caller`: source more defined than target, its
stores beyond the argument's bytes). What the GPU does with the bytes past
a parameter is not PTX's to say. The declaration and the accesses should
agree: declare `N` bytes (`.b8 param0[4]`), or pack the elements into the
one byte.

Reproduce:

```
$ LLC=llc OPT=opt ALIVE_TV=alive-tv scripts/ptx-tv/ptx-tv \
    scripts/ptx-tv/search/findings/v4i1-param.ll
```
