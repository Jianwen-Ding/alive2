AGX machine code: agx-tv
========================

`agx-tv` validates the AGX backend (Apple GPUs, `llc -mtriple=agx`) the way
arm-tv validates AArch64: it compiles each `agx_kernel` of an LLVM IR module
with `llc`, lifts the machine code back to LLVM IR (`agx_lift.py`) and asks
alive-tv whether the machine code refines the IR.

```
$ LLC=llc OPT=opt ALIVE_TV=alive-tv scripts/agx-tv/agx-tv kernel.ll
$ scripts/agx-tv/agx-tv --keep dir --asm edited.s kernel.ll   # lift given code
$ scripts/agx-tv/agx-tv --ieee kernel.ll                      # see below
```

`llc`, `opt` and `alive-tv` must come from an LLVM with the AGX target (and
alive2 built against it: the lifted code calls `llvm.agx.*` intrinsics,
which alive2's GPU semantics, `llvm_util/gpu.h`, gives their meaning).

What the lifted IR says
-----------------------

One thread's execution of the machine code, a function with the kernel's
signature:

* The arguments are where the kernel's `.agx.kernels` record (which the
  backend emits) says they are, in the uniform registers; the driver's
  `threadgroups_per_grid` and the constants of extension 6 too. A pointer
  argument stays a pointer: an address based on its uniforms is a
  `getelementptr` from it, so alive2 sees its provenance.
* Device buffer arguments are 16-byte aligned, as the drivers allocate them:
  the lifted function's pointer parameters say `align 16`, so a launch with
  a misaligned buffer is not checked. The GPU rounds an access's address
  down to its item size; with aligned buffers every access is then aligned
  to its item, which the lifted loads and stores say, and alive2 need not
  model bytes or the buffers' addresses (`tests/guarded.ll` 14 s -> 2 s).
  `--exact-alignment` checks every alignment, rounding the address as the
  GPU does.
* Registers hold bits, never poison: the general registers start as unknown
  values (`freeze poison`), an argument that is poison in the IR is some
  value in its uniforms, and a 16-bit half written into a register is
  frozen so that it cannot poison the other half. The nesting counter `r0l`
  starts at 0 (active).
* A by-reference argument's bytes are loaded where the code reads them (the
  driver loads them; an argument the code does not read may point nowhere).
  A register that copies a pointer argument's uniforms, half by half or in
  64-bit sums, keeps its provenance.
* `trap` is `llvm.trap` (the GPU faults the dispatch).
* Every instruction but control flow acts only for an active thread. The
  counter follows applegpu's pseudocode for `if`, `else`, `while`, `push`
  and `pop`; `jmp_exec_none` is taken when the thread is inactive and
  `jmp_exec_any` when it is active: one of the executions, the others
  differing only in masked instructions. Past a `jmp_exec_none` the counter
  is 0, so the instructions up to its next change are lifted unmasked.
* Threadgroup memory is the kernel's threadgroup variable (one per kernel;
  the backend places it at 0), addressed as BASE + 4 * size * INDEX; its
  loads write their registers at once.
* A device load writes its registers at the `wait` on its scoreboard slot,
  as the GPU does (a register read before the wait keeps its old value),
  so a missing wait shows as a wrong value.
* Float32 operations flush denormal operands and results (the function's
  `denormal_fpenv(float: preservesign)`), as the hardware's ALU does; NaN
  results are any NaN (the GPU's canonical one among them).
* Special registers are the `llvm.agx.*` intrinsics; `memory_barrier`s with
  only waits between them are one fence; `threadgroup_barrier` is
  `llvm.agx.threadgroup.barrier`.

Without the AGX feature `ieee-fp32`, the backend compiles float arithmetic
for the hardware's: denormals flushed (llvm/docs/AGXUsage.rst, "Strict
floating point"). agx-tv then checks the kernel as a function whose float
denormals flush, the contract it is compiled for (`--source-out` of
agx_lift.py writes it); `--ieee` checks it against IEEE arithmetic instead.

Where the semantics come from
-----------------------------

The instructions mean what Dougall Johnson's applegpu reference
(https://dougallj.github.io/applegpu/docs.html) and the AGX Performance
Simulator's functional model say, not what the backend's patterns assume.
`check_coverage.py` holds the lifter to the GPU itself: it lifts each program
of the simulator's coverage suite (cases recorded on an M2 Max, with their
inputs and the buffers the GPU left), runs the lifted code for every lane
(compiled with the host's clang) and compares the buffers bit for bit:

```
$ scripts/agx-tv/check_coverage.py ~/AGX-Performance-Simulator
pass 686, skip 648
```

What that pinned down: an `iadd`/`imadd` shift of 5 or more drops the
operand; `asr` sign-extends from its source's width; signed compares
sign-extend each operand from its own width; `bitrev`, `popcount` and `ffs`
(the highest set bit) work on 32 bits; a 16-bit source of a float32
operation is a half, and a 16-bit destination receives a half; `convert`'s
integer source is zero-extended unless `.sx`; a device access rounds its base
down to the item size and a shift field of 3 is 2; loads land at their
`wait`; `bitop` tables 0011 and 1100 give A.

Skipped: cross-lane operations (shuffles, ballots, reductions: lanes run one
after another there), the transcendental approximations (`rcp`, `rsqrt`,
`sin_pt_*`, `log2`, `exp2`), packed and normalized memory formats, converts
in non-default roundings, coherent accesses, and eight stores whose
neighbouring lanes overlap (the GPU writes them component by component across
the lanes; in the IR they are races between threads).

Not lifted yet: atomics, stack memory, threadgroup memory of more than one
variable, the SIMD-group
matrix and transcendental instructions, calls, and kernels with a global data
buffer or kernel record extensions other than 6.
