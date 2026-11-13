Searching NVPTX for bugs with ptx-tv
=====================================

`batch.py` runs ptx-tv's pipeline (llc, `ptx_lift.py`, alive-tv) over every
function of a directory of LLVM IR files; `classify.py` sorts the
mismatches by the false-alarm patterns below and prints the others, which
are for triage by hand. `fuzz_int.py` and `fuzz_mem.py` write random
modules to run it over.

```
$ git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/llvm/llvm-project.git
$ (cd llvm-project && git sparse-checkout set llvm/test/CodeGen/NVPTX)
$ LLC=llc OPT=opt ALIVE_TV=alive-tv scripts/ptx-tv/search/batch.py \
    llvm-project/llvm/test/CodeGen/NVPTX work --verify -j4 --per-file 8 \
    > results.jsonl
$ scripts/ptx-tv/search/classify.py work results.jsonl

$ mkdir corpus; for s in $(seq 1 40); do
    scripts/ptx-tv/search/fuzz_int.py $s 10 > corpus/fz$s.ll; done
$ scripts/ptx-tv/search/batch.py corpus fwork --verify > fuzz.jsonl
```

What it found (LLVM main of 2026-11-12, Z3 5.4.1)
--------------------------------------------------

No miscompilation in the NVPTX backend.

* llvm/test/CodeGen/NVPTX, up to 8 functions of each of its 663 files:
  755 verify, 223 time out (10 s), 1416 are not lifted (calls, Hopper and
  Blackwell's tensor instructions, `.ftz` / `.sat`, shuffles, packed
  half-precision arithmetic), 93 are alive2 errors (unknown results;
  intrinsics with metadata operands). The 69 mismatches are all the false
  alarms below.
* 400 random integer functions (`fuzz_int.py`, sm_70 / sm_80 / sm_90):
  374 verify, 25 time out, 1 is not lifted.
* 320 random functions over memory (`fuzz_mem.py`: mixed widths, offsets,
  under-aligned and vector accesses, shuffles): 311 verify, 9 time out.

Leads that turned out right: f32 atomic adds that flush subnormals in IEEE
code are on shared memory (which does not flush, PTX ISA 4.2 on) or carry
`!atomic.ignore.denormal.mode` (which the auto-upgrade of
`llvm.nvvm.atomic.load.add.f32` attaches); a seq_cst cluster-scope load
lowered with CTA scope is for sm_70, which has no clusters; a cast between
two specific address spaces has no meaning (the backend makes its result
undefined, as a test checks).

The false alarms (`classify.py`):

* `subbyte-store`: an integer of fewer bits than its bytes in memory (`i1`,
  `i20`): alive2 does not take a byte written whole as refining one written
  with fewer bits, and a byte written with fewer bits loads as poison.
* `subword-atomic-widened`: 8- and 16-bit atomics as a CAS loop on the
  aligned 32-bit word that holds them, which reaches bytes beyond the
  object in alive2's memory (the GPU's allocations are granular).
* `inline-asm`: opaque in the IR, its text in the PTX.
* `byval-escape-copy`: a byval argument whose address escapes is copied to
  the stack, and the copy's address escapes (both dangle at the return).
* `nan-payload`: bf16 arithmetic emulated in f32 on sm_70 gives NaNs whose
  payloads the IR's operation does not (PTX has one canonical NaN).
* `undef-store-dropped`: SelectionDAG drops a store of undef, which over
  memory that was poison is not a refinement (the GPU's memory has bits).
* `ignore-denormal-mode-metadata`: the metadata that allows atom.add.f32's
  flush is dropped for alive2, which does not translate it.

What finding them took, in ptx-tv and alive2's GPU semantics: memory holds
bits (loads read bytes, each frozen; atomics read-modify-write frozen
bytes), acquire and release fenced on their own sides, atomics on private
memory without fences, dynamic shared memory one object, casts between
specific address spaces poison, and the source's datalayout and denormal
mode in the lifted function.
