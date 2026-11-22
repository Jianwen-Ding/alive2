#pragma once

// Copyright (c) 2018-present The Alive2 Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

// The PTX formalization (scripts/ptx-tv/ptx.formal), built into alive2: the
// meaning of PTX's arithmetic, logic, comparison and conversion instructions
// after the PTX ISA, in a declarative format. ptx-tv's lifter reads the same
// file (scripts/ptx-tv/ptx_formal.py), so that a PTX instruction and the NVVM
// intrinsic that stands for it (llvm.nvvm.f2i.rz for cvt.rzi.s32.f32, ...)
// mean the same.

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/IRBuilder.h"
#include <memory>
#include <string>

namespace llvm_util::gpu {

class PTXFormal {
public:
  /// The formalization alive2 was built with.
  static const PTXFormal &builtin();

  /// Parse a formalization's text (Err: what is wrong with it, if anything).
  static std::unique_ptr<PTXFormal> parse(llvm::StringRef Text,
                                          std::string &Err);
  ~PTXFormal();

  /// Emit the meaning of the PTX instruction Opcode ("cvt.rzi.s32.f32") with
  /// source operands Args (integers: the operands' bits) before B's insertion
  /// point: the destination's bits, or null if no entry is for the
  /// instruction (Err says why, if it is an error in the entry).
  llvm::Value *lift(llvm::IRBuilder<> &B, llvm::StringRef Opcode,
                    llvm::ArrayRef<llvm::Value *> Args,
                    std::string &Err) const;

  struct Impl;

private:
  PTXFormal();
  std::unique_ptr<Impl> P;
};

/// The PTX instruction an NVVM intrinsic stands for, by its name without
/// "llvm.nvvm." ("f2i.rz" -> "cvt.rzi.s32.f32"), or "".
std::string ptx_instruction_of_nvvm(llvm::StringRef Name);

} // namespace llvm_util::gpu
