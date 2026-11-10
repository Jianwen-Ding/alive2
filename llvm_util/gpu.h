#pragma once

// Copyright (c) 2018-present The Alive2 Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

// The semantics of code for Apple GPUs, in three forms:
//
//  * upstream GPU IR: what a front end hands the AIR target (LLVM IR with
//    OpenCL/SPIR-V work-item builtins, AIR system-value calls, atomicrmw,
//    LLVM intrinsics, ...);
//  * AIR: the module after the AIR target's portable passes
//    (llc -mtriple=air -air-portable): calls of AIR's functions (air.*) and
//    kernels whose system values are parameters described by !air.kernel;
//  * AGX IR: the module after the AGX backend's agx-lower-air (triple agx):
//    AGX's address spaces, agx_kernel kernels and llvm.agx.* intrinsics.
//
// Each form is translated into one canonical LLVM IR form, which alive2's
// translation from LLVM IR then gives a meaning, so that any two of the forms
// can be compared (upstream -> AIR, AIR -> AGX, upstream -> AGX):
//
//  * there are no address spaces (alive2 has one): pointers of AIR's (thread
//    0, device 1, constant 2, threadgroup 3) and AGX's (private 5, constant
//    4) are pointers of address space 0, so that memories of different
//    spaces may alias, which only keeps alive2 from proving what depends on
//    their being disjoint;
//  * system values are computed from twelve uninterpreted, readnone values,
//    @alive.gpu.sr.<value>.<x|y|z>() (thread position in the threadgroup,
//    threadgroup position in the grid, threads per threadgroup, threadgroups
//    per grid), under the assumptions every launch satisfies; kernel
//    parameters that !air.kernel describes as system values are dropped;
//  * threadgroup buffers (an addrspace(3) pointer parameter, or AGX's
//    zero-sized threadgroup variable that stands for the launch's dynamic
//    threadgroup memory) are trailing parameters;
//  * AIR's functions are LLVM instructions and intrinsics;
//  * the memory model is sequential, as alive2's is: atomics are plain
//    accesses, and an ordered atomic or fence is a call of @alive.gpu.fence,
//    an unknown function that may change any memory other threads can see;
//  * SIMD-group functions and barriers are calls that read and write
//    inaccessible memory, so that they keep their number and order.
//
// See README.md, "Apple GPUs (AIR and AGX)", for what is and is not modelled.

#include <string>
#include <utility>
#include <vector>

namespace llvm {
class Function;
class GlobalValue;
class Module;
}

namespace llvm_util {

/// Whether M is code for Apple GPUs in one of the three forms above.
bool is_gpu_module(const llvm::Module &M);

/// The canonical form of F, as a function of F's module (with F's name, while
/// it lives). F is not changed, but is renamed meanwhile.
class GPUCanonicalFunction {
public:
  explicit GPUCanonicalFunction(llvm::Function &F);
  ~GPUCanonicalFunction();
  GPUCanonicalFunction(const GPUCanonicalFunction &) = delete;
  GPUCanonicalFunction &operator=(const GPUCanonicalFunction &) = delete;

  llvm::Function &function() const { return *canon; }

private:
  llvm::Function *orig;
  llvm::Function *canon = nullptr;
  std::string orig_name;
  // Global values renamed meanwhile, and their names.
  std::vector<std::pair<llvm::GlobalValue *, std::string>> renamed;
  // Global values created for the canonical form.
  std::vector<llvm::GlobalValue *> created;
  friend struct GPUCanonicalizer;
};

}
