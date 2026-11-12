PTX: ptx-tv
===========

`ptx-tv` validates the NVPTX backend (NVIDIA GPUs, `llc -mtriple=nvptx64`)
the way arm-tv validates AArch64 and agx-tv the AGX backend: it compiles
each function of an LLVM IR module with `llc`, lifts the PTX back to LLVM IR
(`ptx_lift.py`) and asks alive-tv whether the PTX refines the IR.

```
$ LLC=llc OPT=opt ALIVE_TV=alive-tv scripts/ptx-tv/ptx-tv kernel.ll
$ scripts/ptx-tv/ptx-tv -mcpu=sm_90 kernel.ll -smt-to=300000
$ scripts/ptx-tv/ptx-tv --keep dir --ptx edited.ptx --func k kernel.ll
$ scripts/ptx-tv/tests/run.sh                 # the tests
```

alive-tv must be built from this repository: its GPU semantics
(`llvm_util/gpu.h`, the NVPTX target `gpu_nvptx.cpp`) gives NVPTX's address
spaces, special registers and barriers their meaning, in the same canonical
form as AIR's and AGX's. Any recent LLVM's `llc` and `opt` will do (the
NVPTX target is upstream).

The PTX formalization
---------------------

What each PTX instruction computes is `ptx.formal`, a formalization of the
PTX ISA's "Instruction Set" chapter in a small declarative format: one entry
per instruction family, its pattern of opcode parts, its operands and its
meaning over bit-vectors and IEEE floats in SMT-LIB's words, with the ISA's
section and wording beside it:

```
; 9.7.8.4 shl: "Shift amounts greater than the register width N are
; clamped to N".
(instruction (shl $T:bits) (d a b:32)
  (set d (ite (bvuge b W) 0 (bvshl a (zext-or-trunc b W)))))

(instruction (cvt rzi $D:int $S:float) (d:D a:S)
  (set d (fp.to-int-sat D a)))
```

The format is described at the top of `ptx.formal`; `ptx_formal.py` reads
it and interprets an entry as LLVM IR. It holds the integer and float
arithmetic, logic, shifts, bit-field and permute instructions, comparisons,
selection and conversions: an instruction no entry matches (a modifier such
as `.ftz` or `.sat` that the entries do not have) is not lifted, so the file
is the whole account of what the lifted code says those instructions do.
The lifter, `ptx_lift.py`, models the machine they run on and lifts the
instructions that act on it: registers, the parameter space, loads and
stores, atomics, barriers, fences and control flow.

What the lifted IR says
-----------------------

One thread's execution of the PTX, a function with the IR function's
signature:

* Registers hold bits, never poison: they start as unknown values
  (`freeze poison`), a parameter that is poison in the IR is some value in
  the parameter space, and every load's value is frozen. A `.b32` register
  that holds a float holds its bits.
* A 64-bit register the lifter's analysis finds to hold a pointer is a
  pointer, so that alive2 sees its provenance: one loaded from a pointer
  parameter (not frozen: an access through a poison pointer is UB in the IR
  too), a variable's address (`.shared`, `.global`, `.const`, the local
  depot), and sums of those with integers (`add`, `sub` as `getelementptr`).
  `cvta` and `cvta.to` are the identity: the canonical form has one address
  space.
* An access is aligned to its size (a vector access to the vector's), as
  the GPU requires.
* `ld.param` of a scalar parameter reads its bits (wider registers
  extended by the type); of a `byval` parameter, its memory.
  `st.param [func_retval0]` is the returned value of a `.func`.
* Predicated instructions are executed in a block of their own; `@p bra` is
  a conditional branch.
* `bar.sync` / `barrier.sync` is `llvm.nvvm.barrier.cta.sync.aligned.all`,
  `membar` and `fence.sc` / `fence.acq_rel` LLVM fences, `atom` / `red`
  `atomicrmw` / `cmpxchg` in their order (`.relaxed`, or none: monotonic);
  the special registers are `llvm.nvvm.read.ptx.sreg.*`. alive2's GPU
  semantics makes the barrier a convergent call between fences, the memory
  model sequential.
* The approximate instructions (`ex2.approx`, `rsqrt.approx`, ...) are calls
  of the NVVM intrinsics of the same meaning: uninterpreted, so that they
  verify against IR that calls those intrinsics and against nothing else.

Inputs are taken not to be undef (`-disable-undef-input`; `--undef` checks
them too): the lifted code freezes what it reads from parameters, and with
undef inputs each freeze is a quantifier the solver rarely gets past.
Poison inputs are always checked. `!tbaa`, which alive2 does not translate,
is dropped from the source (that only drops assumptions).

Not lifted yet: calls (`call`, the parameter space of a callee), `.ftz`,
`.sat` and the directed roundings (`.rz`, `.rm`, `.rp`) of float operations,
half-precision vector (`.f16x2`) arithmetic, warp shuffles and votes
(`shfl.sync`, `vote.sync`), textures and surfaces, `mul24` / `mad24`,
`.cc` carries, and named barriers with a thread count.

Floating point is slow in alive2: kernels that keep floats in memory, or
several float operations, may need `-smt-to` in minutes.

The tests
---------

`tests/run.sh` checks that six kernels' PTX verifies (a guarded store with
the global thread index, shared memory with a barrier, a loop, vector
accesses with an atomic, integer bit operations and narrow integers, a
device function, float arithmetic), and that seeded mistakes in their PTX
are rejected: a comparison, an immediate, an index scale and its sign, the
wrong buffer, a missing barrier, a shift's and a minimum's signedness, a
parameter's sign extension, a count, a store's width, float operations and
a loop's step.
