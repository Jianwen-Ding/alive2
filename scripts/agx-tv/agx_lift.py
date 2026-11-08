#!/usr/bin/env python3
# Copyright (c) 2018-present The Alive2 Authors.
# Distributed under the MIT license that can be found in the LICENSE file.
"""Lift AGX machine code to LLVM IR, for translation validation of the AGX
backend (an arm-tv for AGX).

The input is a kernel in LLVM IR for the AGX target and the assembly the AGX
backend prints for it (llc -mtriple=agx). The output is an LLVM IR function
with the kernel's signature whose body is the per-thread meaning of the
machine code, so that alive-tv can check that the machine code refines the
IR (see README.md, "AGX machine code").

The per-thread model (llvm/docs/AGXUsage.rst, "Execution model"): a thread is
active while its 16-bit nesting counter r0l is 0. Every instruction other than
control flow acts only for an active thread; if/else/while/push/pop update the
counter, and the jumps jmp_exec_none / jmp_exec_any skip code that no active
thread runs. For one thread, a jump taken exactly when the thread is
inactive (jmp_exec_none) or active (jmp_exec_any) is one of the executions:
the others differ only in the masked instructions they also run, which do
nothing for it, and leave its counter as they found it. Cross-lane
operations (shuffles, ballots) and barriers are calls of the same functions
alive2's GPU semantics gives the IR's, so they compare as such.

The meaning of each instruction is that of Dougall Johnson's Apple G13 GPU
architecture reference (https://dougallj.github.io/applegpu/docs.html), not
the backend's own patterns, so that the lifted code is an independent account
of what the hardware does.
"""

import argparse
import re
import struct
import sys

# ---------------------------------------------------------------------------
# The input IR: the kernel's signature.


def split_top(s, sep=','):
  """Split s at sep outside of parentheses, brackets and braces."""
  out, depth, cur = [], 0, ''
  for c in s:
    if c in '([{<':
      depth += 1
    elif c in ')]}>':
      depth -= 1
    if c == sep and depth == 0:
      out.append(cur)
      cur = ''
    else:
      cur += c
  if cur.strip():
    out.append(cur)
  return [x.strip() for x in out]


class Param:
  def __init__(self, text):
    self.text = text
    m = re.match(r'(ptr(?: addrspace\(\d+\))?|<[^>]*>|\[[^\]]*\]|\S+)', text)
    self.type = m.group(1)
    self.name = text.split()[-1]
    if not self.name.startswith('%'):
      raise LiftError('parameter without a name: ' + text)
    b = re.search(r'byref\(([^)]*)\)', text)
    self.byref = b.group(1) if b else None


# The launch contract of the lifted kernel: device buffer arguments are
# aligned to 16 bytes (Metal's and the CUDA drivers' allocations are; the
# lifted function says so with align 16, so that an argument that is not is
# poison there and such launches are not checked). The GPU rounds an access's
# address down to its item size; with aligned buffers and offsets in items,
# every access is aligned, and its alignment spares alive2 modeling bytes and
# addresses (guarded.ll: 14 s -> 2 s). --exact-alignment drops the contract
# (then accesses are byte-aligned, and buffers' addresses rounded).
BUFFER_ALIGN = 16


def lifted_param(p, align):
  """A parameter's text in the lifted function: without capture attributes
  (the lifted code takes pointer arguments' addresses apart, which they
  would make poison; dropping them only weakens the target's assumptions),
  and a device pointer with the contract's alignment."""
  t = re.sub(r'\s*\b(captures\([^)]*\)|nocapture)', '', p.text)
  if (not align or p.type != 'ptr addrspace(1)' or p.byref or
      re.search(r'\balign \d+', t)):
    return t
  return t.replace(p.type, '%s align %d' % (p.type, BUFFER_ALIGN), 1)


def one_line_headers(ll):
  """The module with each function definition's header on one line."""
  out, lines, i = [], ll.split('\n'), 0
  while i < len(lines):
    l = lines[i]
    if l.startswith('define ') and '{' not in l:
      while i + 1 < len(lines) and '{' not in l:
        i += 1
        l = l.rstrip() + ' ' + lines[i].strip()
    out.append(l)
    i += 1
  return '\n'.join(out)


def parse_signature(ll, name):
  m = re.search(r'^define\s+(.*?)@' + re.escape(name) + r'\(', ll, re.M)
  if not m:
    raise LiftError('no definition of @' + name)
  i, depth = m.end(), 1
  while depth:
    depth += {'(': 1, ')': -1}.get(ll[i], 0)
    i += 1
  params_text = ll[m.end():i - 1]
  suffix = ll[i:ll.index('{', i)]
  params = [Param(p) for p in split_top(params_text)] if params_text.strip() else []
  return m.group(1), params, suffix


def float_flushes(env):
  """A denormal_fpenv(...) body with float's denormals flushed (sign kept),
  and the other types' as they were."""
  parts = [p.strip() for p in env.split(',')] if env else []
  default = [p for p in parts if not p.startswith('float:')]
  return ', '.join(default + ['float: preservesign'])


def kernel_fpenv(ll, name):
  """The denormal_fpenv body of @name, or ''."""
  m = re.search(r'^define\s+[^\n]*@' + re.escape(name) + r'\(.*$', ll, re.M)
  line = m.group(0)
  e = re.search(r'denormal_fpenv\(([^)]*)\)', line[line.rindex(')'):])
  if e:
    return e.group(1)
  for g in re.findall(r'#(\d+)', line[line.rindex(')'):]):
    a = re.search(r'^attributes #%s = \{(.*)\}' % g, ll, re.M)
    if a:
      e = re.search(r'denormal_fpenv\(([^)]*)\)', a.group(1))
      if e:
        return e.group(1)
  return ''


def flushing_source(ll):
  """The module with the float denormals of its agx_kernels flushed: the
  contract the AGX backend compiles them for without ieee-fp32."""
  kernels = re.findall(r'^define\s+[^\n]*agx_kernel[^\n]*@([\w.$]+)\(', ll, re.M)
  out = ll
  groups = set()
  for k in kernels:
    m = re.search(r'^define\s+[^\n]*@' + re.escape(k) + r'\(.*$', out, re.M)
    line = m.group(0)
    tail = line[line.rindex(')'):]
    if 'denormal_fpenv(' in tail:
      new = line[:line.rindex(')')] + re.sub(
          r'denormal_fpenv\(([^)]*)\)',
          lambda e: 'denormal_fpenv(%s)' % float_flushes(e.group(1)), tail)
    else:
      gs = [g for g in re.findall(r'#(\d+)', tail)
            if re.search(r'^attributes #%s = \{[^}]*denormal_fpenv' % g, out, re.M)]
      if gs:
        groups.update(gs)
        continue
      i = line.rindex('{')
      new = line[:i] + 'denormal_fpenv(float: preservesign) ' + line[i:]
    out = out.replace(line, new, 1)
  for g in groups:
    out = re.sub(r'^(attributes #%s = \{.*?)denormal_fpenv\(([^)]*)\)' % g,
                 lambda e: e.group(1) + 'denormal_fpenv(%s)' % float_flushes(e.group(2)),
                 out, flags=re.M)
  return out


# ---------------------------------------------------------------------------
# The assembly.


class LiftError(Exception):
  pass


def kernel_body(asm, name):
  lines = asm.split('\n')
  try:
    start = lines.index(name + ':') + 1
  except ValueError:
    for i, l in enumerate(lines):
      if l.startswith(name + ':'):
        start = i + 1
        break
    else:
      raise LiftError('no code for ' + name)
  body = []
  for l in lines[start:]:
    if l.startswith('.Lfunc_end'):
      break
    body.append(l)
  return body


def kernel_record(asm, name):
  """The kernel's record of the .agx.kernels section, as 32-bit words."""
  data = bytearray()
  insec = False
  for l in asm.split('\n'):
    s = l.strip()
    if s.startswith('.section'):
      insec = '.agx.kernels' in s
      continue
    if not insec:
      continue
    m = re.match(r'\.(long|byte|short|zero|ascii|asciz)\s+(.*)', s)
    if not m:
      continue
    kind, arg = m.group(1), m.group(2)
    if kind == 'long':
      for v in arg.split(','):
        data += struct.pack('<I', int(v, 0) & 0xffffffff)
    elif kind == 'short':
      for v in arg.split(','):
        data += struct.pack('<H', int(v, 0) & 0xffff)
    elif kind == 'byte':
      for v in arg.split(','):
        data += bytes([int(v, 0) & 0xff])
    elif kind == 'zero':
      data += bytes(int(arg.split(',')[0], 0))
    else:
      data += bytes(arg.strip('"'), 'latin1').decode('unicode_escape').encode(
          'latin1') + (b'\0' if kind == 'asciz' else b'')
  off = 0
  while off + 12 <= len(data):
    size, version, nlen = struct.unpack_from('<III', data, off)
    if size == 0:
      break
    nm = data[off + 12:off + 12 + nlen].decode()
    if nm == name:
      words = list(struct.unpack_from('<%dI' % (size // 4), data, off))
      return words, (12 + ((nlen + 3) & ~3)) // 4
    off += size
  raise LiftError('no .agx.kernels record for ' + name)


# ---------------------------------------------------------------------------
# Registers.

REG = re.compile(r'^(r|u)(\d+)(l|h)?$')


def reg_halves(tok):
  """The 16-bit halves of a register operand (rN, rNl, rNh, rA_rB, uN ...):
  (file, [half indices]) or None."""
  parts = tok.split('_')
  file, halves = None, []
  for p in parts:
    m = REG.match(p)
    if not m:
      return None
    f, n, lh = m.group(1), int(m.group(2)), m.group(3)
    if file and f != file:
      return None
    file = f
    if lh == 'l':
      halves.append(2 * n)
    elif lh == 'h':
      halves.append(2 * n + 1)
    else:
      halves += [2 * n, 2 * n + 1]
  return file, halves


class Operand:
  """A source or destination: a register (with modifiers) or an immediate."""

  def __init__(self, tok):
    self.tok = tok
    self.mods = set()
    base = tok
    while True:
      m = re.match(r'^(.*)\.(abs|neg|sx|cache|discard)$', base)
      if not m:
        break
      base = m.group(1)
      self.mods.add(m.group(2))
    self.base = base
    rh = reg_halves(base)
    if rh:
      self.file, self.halves = rh
      self.imm = None
    else:
      self.file, self.halves = None, None
      try:
        self.imm = int(base, 0)
        self.fimm = None
      except ValueError:
        try:
          self.fimm = float(base)
          self.imm = None
        except ValueError:
          raise LiftError('operand ' + tok)

  @property
  def is_reg(self):
    return self.halves is not None

  @property
  def bits(self):
    return 16 * len(self.halves) if self.halves else None


# ---------------------------------------------------------------------------
# The lifter.


def fp_imm_bits(x, bits):
  if bits == 32:
    return '0x%016X' % struct.unpack('<Q', struct.pack('<d', x))[0]
  return '0x%016X' % struct.unpack('<Q', struct.pack('<d', x))[0]


ICMP = {'ueq': 'eq', 'nueq': 'ne', 'seq': 'eq', 'nseq': 'ne',
        'ult': 'ult', 'ugt': 'ugt', 'ugte': 'uge', 'ulte': 'ule',
        'slt': 'slt', 'sgt': 'sgt', 'sgte': 'sge', 'slte': 'sle'}
# The float conditions; the "n" forms are their negations (true on NaN).
FCMP = {'eq': 'oeq', 'lt': 'olt', 'gt': 'ogt', 'gte': 'oge', 'lte': 'ole',
        'neq': 'une', 'nlt': 'uge', 'ngt': 'ule', 'ngte': 'ult',
        'nlte': 'ugt'}

SR = {80: ('thread.position.in.grid.x', 'llvm.agx.thread.position.in.grid.x'),
      81: ('thread.position.in.grid.y', 'llvm.agx.thread.position.in.grid.y'),
      82: ('thread.position.in.grid.z', 'llvm.agx.thread.position.in.grid.z'),
      0: ('', 'llvm.agx.threadgroup.position.in.grid.x'),
      1: ('', 'llvm.agx.threadgroup.position.in.grid.y'),
      2: ('', 'llvm.agx.threadgroup.position.in.grid.z'),
      4: ('', 'llvm.agx.threads.per.threadgroup.x'),
      5: ('', 'llvm.agx.threads.per.threadgroup.y'),
      6: ('', 'llvm.agx.threads.per.threadgroup.z'),
      48: ('', 'llvm.agx.thread.position.in.threadgroup.x'),
      49: ('', 'llvm.agx.thread.position.in.threadgroup.y'),
      50: ('', 'llvm.agx.thread.position.in.threadgroup.z'),
      51: ('', 'llvm.agx.thread.index.in.threadgroup'),
      52: ('', 'llvm.agx.thread.index.in.simdgroup'),
      53: ('', 'llvm.agx.simdgroup.index.in.threadgroup')}

MEM_BYTES = {'i8': 1, 'i16': 2, 'i32': 4}


class Lifter:
  def __init__(self, params, record, record_start, body):
    self.params = params
    self.words, self.wstart = record, record_start
    self.body = body
    self.lines = []         # the IR of the body
    self.n = 0              # value numbering
    self.gpr = set()        # the GPR halves used
    self.decls = set()
    self.uniform = {}       # uniform half -> IR i16 value
    self.uprov = {}         # uniform half -> (pointer param, byte offset)
    self.rprov = {}         # GPR half -> pointer param, if one pointer
    self.uwhole = {}        # uniform halves -> the IR value they hold whole
    self.uptr = {}          # uniform halves -> the pointer parameter in them
    self.entry = []
    self.raw = None         # a RawCase: lift a recorded program
    self.pending = []       # (slot, register operand, value slot, flag slot)
    self.last_fence = False  # the last instruction but waits was a barrier
    self.nest = 0           # the nesting counter's value where known
    self.lazy = {}          # uniform half -> (byref param, byte, size)
    self.tgvar = None       # the threadgroup variable, '@name'
    self.exact_align = False  # no buffer alignment contract (BUFFER_ALIGN)
    self.shadows = []       # the allocas of pending load results

  # -- values ---------------------------------------------------------------

  def tmp(self):
    self.n += 1
    return '%%v%d' % self.n

  def emit(self, s):
    self.lines.append('  ' + s)

  def op(self, s):
    t = self.tmp()
    self.emit('%s = %s' % (t, s))
    return t

  def decl(self, d):
    self.decls.add(d)

  # -- the kernel's inputs --------------------------------------------------

  def init_uniforms(self):
    w, i = self.words, self.wstart
    halves, tpg, gdata, gprs, tgbytes, stack = w[i:i + 6]
    nargs = w[i + 6]
    args = [w[i + 7 + 3 * k:i + 10 + 3 * k] for k in range(nargs)]
    ext = w[i + 7 + 3 * nargs:]
    if gdata != 0xffffffff:
      raise LiftError('a global data buffer')
    if stack:
      raise LiftError('stack memory (%d bytes per thread)' % stack)
    if nargs != len(self.params):
      raise LiftError('the record has %d arguments, the IR %d'
                      % (nargs, len(self.params)))
    e = self.entry
    for p, (kind, size, half) in zip(self.params, args):
      if half == 0xffffffff:
        continue
      if kind == 1:
        # A pointer to the bytes: the parameter itself.
        self.set_uniform_value(half, p.name, 'ptr', 64, p)
      elif p.byref:
        # The bytes of a by-reference argument, which the driver puts in the
        # uniforms: loaded where the code reads them (an argument it does
        # not read may point nowhere).
        for k in range(0, size, 2):
          self.lazy[half + k // 2] = (p, k, size)
      else:
        bits = size * 8
        self.set_uniform_value(half, p.name, p.type, bits, p)
    # threadgroups_per_grid.
    for k, c in enumerate('xyz'):
      v = '%%u.tpg.%s' % c
      fn = 'llvm.agx.threadgroups.per.grid.' + c
      self.decl('declare i32 @%s()' % fn)
      e.append('%s = call i32 @%s()' % (v, fn))
      self.split32(v, tpg + 2 * k)
    # Extensions.
    j = 0
    while j < len(ext):
      tag = ext[j]
      if tag == 6:
        n = ext[j + 1]
        for q in range(n):
          half, value = ext[j + 2 + 2 * q], ext[j + 3 + 2 * q]
          self.uniform[half] = str(value & 0xffff)
          self.uniform[half + 1] = str(value >> 16)
        j += 2 + 2 * n
      elif tag == 0:
        break
      else:
        raise LiftError('kernel record extension %d' % tag)
    self.nuniform = halves

  def set_uniform_value(self, half, name, ty, bits, p):
    e = self.entry
    base = '%%u.arg.%d' % half
    # The driver puts bits in the uniforms: an argument that is poison in
    # the IR is some value there.
    if not ty.startswith('ptr'):
      e.append('%s.fr = freeze %s %s' % (base, ty, name))
      name = base + '.fr'
    if ty.startswith('ptr'):
      e.append('%s = ptrtoint %s %s to i64' % (base, ty, name))
      for k in range(4):
        self.uprov[half + k] = (p, k)
      self.uptr[tuple(range(half, half + 4))] = p
      ty, bits = 'i64', 64
    elif ty in ('float', 'half', 'bfloat', 'double'):
      ity = 'i%d' % bits
      e.append('%s = bitcast %s %s to %s' % (base, ty, name, ity))
      ty = ity
    else:
      e.append('%s = add %s %s, 0' % (base, ty, name))
    if bits >= 16:
      self.uwhole[tuple(range(half, half + bits // 16))] = (base, bits)
    if bits < 16:
      e.append('%s.w = zext %s %s to i16' % (base, ty, base))
      self.uniform[half] = base + '.w'
      return
    for k in range(bits // 16):
      h = '%s.%d' % (base, k)
      e.append('%s.s = lshr %s %s, %d' % (h, ty, base, 16 * k))
      e.append('%s = trunc %s %s.s to i16' % (h, ty, h))
      self.uniform[half + k] = h

  def split32(self, v, half):
    e = self.entry
    e.append('%s.lo = trunc i32 %s to i16' % (v, v))
    e.append('%s.s = lshr i32 %s, 16' % (v, v))
    e.append('%s.hi = trunc i32 %s.s to i16' % (v, v))
    self.uniform[half] = v + '.lo'
    self.uniform[half + 1] = v + '.hi'

  # -- registers ------------------------------------------------------------

  def read_half(self, file, h):
    if file == 'u' and h not in self.uniform and h in self.lazy:
      p, k, size = self.lazy[h]
      g = '%%u.byref.%d' % h
      e = self.entry
      e.append('%s.p = getelementptr i8, %s %s, i64 %d' % (g, p.type, p.name, k))
      if k + 1 < size:
        e.append('%s.v = load i16, %s %s.p, align 1' % (g, p.type, g))
      else:
        e.append('%s.b = load i8, %s %s.p, align 1' % (g, p.type, g))
        e.append('%s.v = zext i8 %s.b to i16' % (g, g))
      e.append('%s = freeze i16 %s.v' % (g, g))
      self.uniform[h] = g
    if file == 'u':
      if h not in self.uniform:
        raise LiftError('uniform half %d, which the record does not set' % h)
      return self.uniform[h]
    if h == 0:
      return self.op('load i16, ptr %nest')
    r = self.read_reg32(h // 2)
    if h % 2:
      r = self.op('lshr i32 %s, 16' % r)
    return self.op('trunc i32 %s to i16' % r)

  def read_reg32(self, n):
    """GPR rN (32 bits); r0 holds the nesting counter only."""
    if n == 0:
      raise LiftError('r0 other than the nesting counter r0l')
    self.gpr.add(n)
    return self.op('load i32, ptr %%r%d' % n)

  def read(self, o, bits, signed=None):
    """Operand o as an i<bits> integer: registers zero-extended (sign with
    .sx) to bits, or their low bits; immediates as they are."""
    if not o.is_reg:
      if o.imm is None:
        raise LiftError('a float immediate in an integer operation: ' + o.tok)
      return str(o.imm & ((1 << bits) - 1))
    whole = self.uwhole.get(tuple(o.halves)) if o.file == 'u' else None
    hs = o.halves
    full = (o.file == 'r' and len(hs) % 2 == 0 and
            all(hs[k] % 2 == 0 and hs[k + 1] == hs[k] + 1
                for k in range(0, len(hs), 2)))
    if whole:
      v, w = whole
      hv = []
    elif full:
      # Whole 32-bit registers.
      regs = [self.read_reg32(hs[k] // 2) for k in range(0, len(hs), 2)]
      w = 32 * len(regs)
      v = regs[0]
      if len(regs) > 1:
        v = self.op('zext i32 %s to i%d' % (v, w))
        for k, x in enumerate(regs[1:], 1):
          z = self.op('zext i32 %s to i%d' % (x, w))
          z = self.op('shl i%d %s, %d' % (w, z, 32 * k))
          v = self.op('or i%d %s, %s' % (w, v, z))
      hv = []
    else:
      hv = [self.read_half(o.file, h) for h in o.halves]
      w = 16 * len(hv)
      v = None
    for k, x in enumerate(hv):
      z = self.op('zext i16 %s to i%d' % (x, w)) if w > 16 else x
      if k:
        z = self.op('shl i%d %s, %d' % (w, z, 16 * k))
        v = self.op('or i%d %s, %s' % (w, v, z))
      else:
        v = z
    sx = 'sx' in o.mods if signed is None else signed
    if w < bits:
      v = self.op('%s i%d %s to i%d' % ('sext' if sx else 'zext', w, v, bits))
    elif w > bits:
      v = self.op('trunc i%d %s to i%d' % (w, v, bits))
    return v

  def readf(self, o, bits=32):
    """Operand o as a float, with its .abs and .neg modifiers."""
    ty = {32: 'float', 16: 'half'}[bits]
    if not o.is_reg:
      if o.fimm is not None:
        v = fp_imm_bits(o.fimm, bits)
      else:
        v = fp_imm_bits(float(o.imm), bits)
      return v
    if o.bits == 16 and bits == 32:
      # A half source of a float32 operation: its value, as a float.
      i = self.read(o, 16, signed=False)
      h = self.op('bitcast i16 %s to half' % i)
      v = self.op('fpext half %s to float' % h)
    else:
      i = self.read(o, bits, signed=False)
      v = self.op('bitcast i%d %s to %s' % (bits, i, ty))
    if 'abs' in o.mods:
      self.decl('declare %s @llvm.fabs.%s(%s)' % (ty, 'f%d' % bits, ty))
      v = self.op('call %s @llvm.fabs.f%d(%s %s)' % (ty, bits, ty, v))
    if 'neg' in o.mods:
      v = self.op('fneg %s %s' % (ty, v))
    if self.raw and bits == 32:
      v = self.flush(v)
    return v

  def flush(self, v):
    """A float32 denormal as a zero of its sign (the f32 ALU: applegpu's
    FloatSrc / FloatDst). In alive2's mode the function's denormal-fp-math
    says it instead."""
    self.decl('declare float @llvm.fabs.f32(float)')
    self.decl('declare float @llvm.copysign.f32(float, float)')
    a = self.op('call float @llvm.fabs.f32(float %s)' % v)
    tiny = self.op('fcmp olt float %s, 0x3810000000000000' % a)
    z = self.op('call float @llvm.copysign.f32(float 0.0, float %s)' % v)
    return self.op('select i1 %s, float %s, float %s' % (tiny, z, v))

  def active(self):
    if self.nest == 0:
      return 'true'
    c = self.read_half('r', 0)
    return self.op('icmp eq i16 %s, 0' % c)

  def write(self, d, val, bits, act):
    """Write i<bits> val to the halves of d where the thread is active."""
    if not d.is_reg or d.file != 'r':
      raise LiftError('destination ' + d.tok)
    w = d.bits
    if bits < w:
      val = self.op('zext i%d %s to i%d' % (bits, val, w))
    elif bits > w:
      val = self.op('trunc i%d %s to i%d' % (bits, val, w))
    hs = d.halves
    k = 0
    while k < len(hs):
      h = hs[k]
      if h == 0:
        raise LiftError('a write of r0l by ' + d.tok)
      if h % 2 == 0 and k + 1 < len(hs) and hs[k + 1] == h + 1:
        # A whole 32-bit register.
        x = self.op('lshr i%d %s, %d' % (w, val, 16 * k)) if k else val
        x = self.op('trunc i%d %s to i32' % (w, x)) if w > 32 else x
        n = h // 2
        if act == 'true':
          self.gpr.add(n)
          self.emit('store i32 %s, ptr %%r%d' % (x, n))
          k += 2
          continue
        old = self.read_reg32(n)
        new = self.op('select i1 %s, i32 %s, i32 %s' % (act, x, old))
        self.emit('store i32 %s, ptr %%r%d' % (new, n))
        k += 2
        continue
      x = self.op('lshr i%d %s, %d' % (w, val, 16 * k)) if k else val
      x = self.op('trunc i%d %s to i16' % (w, x)) if w > 16 else x
      n = h // 2
      old = self.read_reg32(n)
      # The registers hold bits, not poison: a poison half (bytes memory
      # never initialized) is some value, so that it does not poison the
      # other half of the register.
      x = self.op('freeze i16 %s' % x)
      x32 = self.op('zext i16 %s to i32' % x)
      if h % 2:
        x32 = self.op('shl i32 %s, 16' % x32)
        keep = self.op('and i32 %s, 65535' % old)
      else:
        keep = self.op('and i32 %s, -65536' % old)
      ins = self.op('or i32 %s, %s' % (keep, x32))
      new = ins if act == 'true' else self.op(
          'select i1 %s, i32 %s, i32 %s' % (act, ins, old))
      self.emit('store i32 %s, ptr %%r%d' % (new, n))
      k += 1

  def writef(self, d, fval, bits, act):
    ty = {32: 'float', 16: 'half'}[bits]
    if bits == 32 and d.bits == 16:
      # A float32 result in a 16-bit register: as a half.
      if self.raw:
        fval = self.flush(fval)
      fval = self.op('fptrunc float %s to half' % fval)
      bits, ty = 16, 'half'
    if self.raw:
      # The GPU's NaN results are the canonical quiet NaN.
      if bits == 32:
        fval = self.flush(fval)
      nan = self.op('fcmp uno %s %s, 0.0' % (ty, fval))
      fval = self.op('select i1 %s, %s %s, %s %s' % (
          nan, ty, '0x7FF8000000000000', ty, fval))
    i = self.op('bitcast %s %s to i%d' % (ty, fval, bits))
    self.write(d, i, bits, act)

  # -- addresses ------------------------------------------------------------

  def provenance(self, o):
    """The pointer parameter a 64-bit base register holds (plus an offset),
    if one does: its four halves those of the parameter, in order, or a
    64-bit sum with one."""
    table = self.uprov if o.file == 'u' else self.rprov
    got = [table.get(h) for h in o.halves]
    if any(g is None for g in got):
      return None
    ps = {g[0] for g in got}
    if len(ps) != 1 or [g[1] for g in got] != list(range(len(got))):
      return None
    return ps.pop()

  def address(self, base, index, sgn, scale, esz=1):
    p = self.uptr.get(tuple(base.halves)) if base.file == 'u' else None
    if index.is_reg:
      x = self.read(index, 32, signed=False)
      x = self.op('%s i32 %s to i64' % ('sext' if sgn == 'signed' else 'zext',
                                        x))
    else:
      x = str(index.imm)
    off = self.op('shl i64 %s, %d' % (x, scale)) if scale else x
    if p is not None:
      # The pointer argument itself: an offset from it.
      # Device buffers are aligned to BUFFER_ALIGN (the lifted function's
      # contract, see buffer_align), so the GPU's rounding of the address
      # down to the item size changes nothing; without the contract, the
      # offset less the argument's address mod the item size.
      assert esz <= BUFFER_ALIGN
      if self.exact_align and esz > 1:
        pi = self.op('ptrtoint %s %s to i64' % (p.type, p.name))
        m = self.op('and i64 %s, %d' % (pi, esz - 1))
        off = self.op('sub i64 %s, %s' % (off, m))
      return self.op('getelementptr i8, %s %s, i64 %s' % (p.type, p.name, off))
    b = self.read(base, 64, signed=False)
    if esz > 1:
      # The base rounds down to the item size.
      b = self.op('and i64 %s, %d' % (b, -esz))
    a = self.op('add i64 %s, %s' % (b, off))
    if self.raw:
      return self.raw.address(self, a, esz)
    p = self.provenance(base)
    if p is None:
      return self.op('inttoptr i64 %s to ptr addrspace(1)' % a)
    pi = self.op('ptrtoint %s %s to i64' % (p.type, p.name))
    d = self.op('sub i64 %s, %s' % (a, pi))
    return self.op('getelementptr i8, %s %s, i64 %s' % (p.type, p.name, d))

  def masked(self, act, fn):
    """Run fn in a block that only an active thread executes."""
    if act == 'true':
      fn()
      return
    self.n += 1
    k = self.n
    self.emit('br i1 %s, label %%m%d, label %%m%d.end' % (act, k, k))
    self.lines.append('m%d:' % k)
    fn()
    self.emit('br label %%m%d.end' % k)
    self.lines.append('m%d.end:' % k)

  # -- instructions ---------------------------------------------------------

  def lift(self):
    self.init_uniforms()
    self.lines.append('body:')
    labels = {}
    insts = []
    for l in self.body:
      s = l.split('//')[0].rstrip()
      if not s.strip():
        continue
      m = re.match(r'^(\.LBB\w+|\S+):$', s.strip())
      if m:
        insts.append(('label', m.group(1)))
        continue
      if s.strip().startswith('.'):
        continue
      insts.append(('inst', s.strip()))
    self.track_provenance(insts)
    terminated = False
    for kind, x in insts:
      if kind == 'label':
        self.last_fence = False
        self.nest = None
        if not terminated:
          self.emit('br label %%"%s"' % x)
        self.lines.append('"%s":' % x)
        terminated = False
        continue
      if terminated:
        self.n += 1
        self.lines.append('dead%d:' % self.n)
        terminated = False
      terminated = self.inst(x)
    if not terminated:
      self.emit('unreachable')

  def track_provenance(self, insts):
    """Which GPR halves hold the halves of a pointer parameter: copies of
    its uniforms (a move: bitop 0101 or 1010-free mov forms, iadd of 0), and
    64-bit sums with a register that holds it. One answer per register
    half over the whole kernel, or none."""
    seen = {}

    def assign(h, v):
      if h in seen and seen[h] != v:
        v = None
      seen[h] = v
      self.rprov[h] = v

    def src_halves(o):
      t = self.uprov if o.file == 'u' else self.rprov
      return [t.get(h) for h in o.halves]

    for kind, x in insts:
      if kind != 'inst':
        continue
      mn, ops = self.split_inst(x)
      ops = [t for t in ops if not t.startswith('lsl ')]
      try:
        if mn == 'bitop' and ops[0] == '0101':
          mn, ops = 'mov', [ops[1], ops[2]]
        elif mn == 'iadd' and len(ops) == 3 and ops[2] == '0':
          mn, ops = 'mov', ops[:2]
        d = Operand(ops[0]) if ops else None
      except LiftError:
        continue
      if d is None or not d.is_reg or d.file != 'r' or mn == 'device_store':
        continue
      if mn == 'mov':
        sop = Operand(ops[1])
        if sop.is_reg and sop.bits == d.bits:
          for h, v in zip(d.halves, src_halves(sop)):
            assign(h, v)
          continue
      elif mn in ('iadd', 'isub', 'imadd') and d.bits == 64:
        srcs = [Operand(t) for t in ops[1:]]
        ps = {self.provenance(o) for o in srcs if o.is_reg and o.bits == 64}
        ps.discard(None)
        if len(ps) == 1 and mn != 'imadd':
          p = ps.pop()
          for k, h in enumerate(d.halves):
            assign(h, (p, k))
          continue
      for h in d.halves:
        assign(h, None)

  @staticmethod
  def split_inst(x):
    m = re.match(r'^(\S+)\s*(.*)$', x)
    mn, rest = m.group(1), m.group(2)
    rest = re.sub(r'\s*\([^)]*\)', '', rest)  # get_sr's name
    return mn, [t.strip() for t in rest.split(',')] if rest else []

  def inst(self, x):
    """Lift one instruction; True for a terminator."""
    mn, ops = self.split_inst(x)
    shift = 0
    for k, t in enumerate(ops):
      if t.startswith('lsl '):
        shift = int(t[4:])
        del ops[k]
        break
    if mn not in ('wait', 'memory_barrier'):
      self.last_fence = False
    mask = None
    if ops and ops[-1].startswith('mask '):
      mask = int(ops.pop()[5:], 0)
    O = lambda i: Operand(ops[i])

    if mn == 'trap':
      # The GPU faults the dispatch: llvm.trap.
      self.decl('declare void @llvm.trap()')
      self.emit('call void @llvm.trap()')
      self.emit('unreachable')
      return True
    if mn == 'wait':
      # The loads of the slot land (in issue order), for the threads that
      # issued them, active now or not.
      slot = int(ops[0]) if ops else 0
      keep = []
      for (g, reg, val, flag, bits) in self.pending:
        if g != slot:
          keep.append((g, reg, val, flag, bits))
          continue
        v = self.op('load i%d, ptr %s' % (bits, val))
        f = self.op('load i1, ptr %s' % flag)
        self.write(reg, v, bits, f)
        self.emit('store i1 false, ptr %s' % flag)
      self.pending = keep
      return False
    if mn == 'stop':
      self.emit('ret void')
      return True

    # Control flow.
    if mn in ('jmp_exec_none', 'jmp_exec_any'):
      a = self.active()
      self.n += 1
      nxt = 'fall%d' % self.n
      t, f = ('"%s"' % ops[0], nxt) if mn == 'jmp_exec_any' else (nxt, '"%s"' % ops[0])
      self.emit('br i1 %s, label %%%s, label %%%s' % (a, t.strip('"') if t == nxt else t, f if f != nxt else nxt))
      self.lines.append('%s:' % nxt)
      if mn == 'jmp_exec_none':
        # Falling through, the thread is active: the counter is 0 here
        # (stored as such, so that the masked instructions up to the next
        # change of it are unconditional, without a test per instruction).
        self.store_r0l('0')
      return False
    if mn in ('if_icmp', 'if_fcmp', 'else_icmp', 'else_fcmp', 'while_icmp',
              'while_fcmp'):
      self.cf(mn, ops)
      return False
    if mn in ('pop_exec', 'push_exec'):
      n = int(ops[1])
      c = self.read_half('r', 0)
      if mn == 'pop_exec':
        # r0l = max(r0l - n, 0)
        lt = self.op('icmp ult i16 %s, %d' % (c, n))
        s = self.op('sub i16 %s, %d' % (c, n))
        v = self.op('select i1 %s, i16 0, i16 %s' % (lt, s))
      else:
        z = self.op('icmp eq i16 %s, 0' % c)
        s = self.op('add i16 %s, %d' % (c, n))
        v = self.op('select i1 %s, i16 0, i16 %s' % (z, s))
      self.store_r0l(v)
      return False

    if mn == 'mov_imm' and len(ops) == 3:
      ops = ops[:2]  # applegpu's disassembly adds a field
    if mn == 'mov_imm' and ops[0] == 'r0l':
      self.store_r0l(str(int(ops[1], 0) & 0xffff))
      return False
    act = self.active()
    if mn == 'mov_imm':
      d = O(0)
      self.write(d, str(int(ops[1], 0) & ((1 << d.bits) - 1)), d.bits, act)
      return False
    if mn == 'get_sr':
      n = int(ops[1][2:])
      if n not in SR and not self.raw:
        raise LiftError('special register sr%d' % n)
      if self.raw:
        v = self.op('call i32 @sr(i32 %d, i32 %%lane)' % n)
      else:
        fn = SR[n][1]
        self.decl('declare i32 @%s()' % fn)
        v = self.op('call i32 @%s()' % fn)
      self.write(O(0), v, 32, act)
      return False
    m = re.match(r'^(iadd|isub)(\.sat)?$', mn)
    if m and m.group(2):
      # Saturating to D's width, signed when a source is .sx; only without
      # a shift and up to 32 bits.
      d, A, B = O(0), O(1), O(2)
      w = d.bits
      if shift or w > 32:
        raise LiftError('saturating 64-bit or shifted ' + mn)
      sx = 'sx' in A.mods or 'sx' in B.mods
      W = 64
      a, b = self.read(A, W), self.read(B, W)
      v = self.op('%s i64 %s, %s' % ('add' if m.group(1) == 'iadd' else 'sub',
                                     a, b))
      lo, hi = ((-(1 << (w - 1)), (1 << (w - 1)) - 1) if sx else (0, (1 << w) - 1))
      v = self.op('call i64 @llvm.smax.i64(i64 %s, i64 %d)' % (v, lo))
      v = self.op('call i64 @llvm.smin.i64(i64 %s, i64 %d)' % (v, hi))
      self.decl('declare i64 @llvm.smax.i64(i64, i64)')
      self.decl('declare i64 @llvm.smin.i64(i64, i64)')
      self.write(d, self.op('trunc i64 %s to i%d' % (v, w)), w, act)
      return False
    if mn in ('iadd', 'isub'):
      d = O(0)
      w = d.bits
      a, b = self.read(O(1), w), self.read(O(2), w)
      # B << shift, or 0 for shifts of 5 and more (applegpu)
      if shift >= 5:
        b = '0'
      elif shift:
        b = self.op('shl i%d %s, %d' % (w, b, shift))
      v = self.op('%s i%d %s, %s' % ('add' if mn == 'iadd' else 'sub', w, a, b))
      self.write(d, v, w, act)
      return False
    if mn in ('imadd', 'imsub'):
      d = O(0)
      w = d.bits
      a, b, c = self.read(O(1), w), self.read(O(2), w), self.read(O(3), w)
      if shift >= 5:
        c = '0'
      elif shift:
        c = self.op('shl i%d %s, %d' % (w, c, shift))
      m = self.op('mul i%d %s, %s' % (w, a, b))
      self.write(d, self.op('%s i%d %s, %s' % ('add' if mn == 'imadd' else 'sub',
                                               w, m, c)), w, act)
      return False
    if mn in ('and', 'or', 'xor', 'nand', 'nor', 'xnor', 'bitop'):
      if mn == 'bitop':
        tt = ops.pop(0)
        table = [int(c) for c in tt]
      else:
        table = {'and': [0, 0, 0, 1], 'or': [0, 1, 1, 1],
                 'xor': [0, 1, 1, 0], 'nand': [1, 1, 1, 0],
                 'nor': [1, 0, 0, 0], 'xnor': [1, 0, 0, 1]}[mn]
      d = O(0)
      w = d.bits
      a, b = self.read(O(1), w), self.read(O(2), w)
      if table in ([0, 0, 1, 1], [1, 1, 0, 0]):
        # Undefined in applegpu; A on the GPU (the simulator's recordings).
        self.write(d, a, w, act)
        return False
      self.write(d, self.truth_table(table, a, b, w), w, act)
      return False
    if mn == 'asrh':
      # (sign_extend(a, A's bits) << 32) >> (b & 0x7f)
      d, A = O(0), O(1)
      w = d.bits
      aw = A.bits if A.is_reg else w
      a = self.op('sext i%d %s to i128' % (aw, self.read(A, aw)))
      a = self.op('shl i128 %s, 32' % a)
      sv = self.op('zext i%d %s to i128' % (w, self.read(O(2), w)))
      sv = self.op('and i128 %s, 127' % sv)
      v = self.op('ashr i128 %s, %s' % (a, sv))
      self.write(d, self.op('trunc i128 %s to i%d' % (v, w)), w, act)
      return False
    if mn == 'asr':
      d = O(0)
      w = d.bits
      # sign_extend(a, A's bits) >> (b & 0x7f), in 128 bits
      A = O(1)
      aw = A.bits if A.is_reg else w
      a = self.op('sext i%d %s to i128' % (aw, self.read(A, aw)))
      s = self.op('zext i%d %s to i128' % (w, self.read(O(2), w)))
      s = self.op('and i128 %s, 127' % s)
      v = self.op('ashr i128 %s, %s' % (a, s))
      self.write(d, self.op('trunc i128 %s to i%d' % (v, w)), w, act)
      return False
    if mn in ('bfi', 'extr', 'bfeil', 'shlhi', 'shrhi'):
      self.bitfield(mn, ops, mask, act)
      return False
    if mn in ('bitrev', 'popcount', 'ffs'):
      # On the 32 bits of the source (zero-extended), then the destination's.
      d = O(0)
      w = 32
      a = self.read(O(1), w)
      intr = {'bitrev': 'bitreverse', 'popcount': 'ctpop'}.get(mn)
      if intr:
        self.decl('declare i%d @llvm.%s.i%d(i%d)' % (w, intr, w, w))
        v = self.op('call i%d @llvm.%s.i%d(i%d %s)' % (w, intr, w, w, a))
      else:
        # ffs: the index of the highest set bit, -1 for none (applegpu).
        self.decl('declare i%d @llvm.ctlz.i%d(i%d, i1)' % (w, w, w))
        z = self.op('call i%d @llvm.ctlz.i%d(i%d %s, i1 false)' % (w, w, w, a))
        v = self.op('sub i%d %d, %s' % (w, w - 1, z))
      self.write(d, v, w, act)
      return False
      return False
    if mn in ('icmpsel', 'fcmpsel'):
      cc = ops.pop(0)
      d = O(0)
      w = d.bits
      if mn == 'icmpsel':
        sg = True if cc.lstrip('n').startswith('s') else None
        a, b = self.read(O(1), w, sg), self.read(O(2), w, sg)
        c = self.op('icmp %s i%d %s, %s' % (ICMP[cc], w, a, b))
      else:
        a, b = self.readf(O(1), w), self.readf(O(2), w)
        c = self.fcmp(cc, a, b, w)
      x, y = self.read(O(3), w), self.read(O(4), w)
      self.write(d, self.op('select i1 %s, i%d %s, i%d %s' % (c, w, x, w, y)),
                 w, act)
      return False
    m = re.match(r'^(fadd|fmul|fmadd)(32|16)(\.sat)?$', mn)
    if m:
      if m.group(3):
        raise LiftError('saturating ' + mn)
      w = int(m.group(2))
      ty = {32: 'float', 16: 'half'}[w]
      d = O(0)
      a, b = self.readf(O(1), w), self.readf(O(2), w)
      if m.group(1) == 'fmadd':
        c = self.readf(O(3), w)
        self.decl('declare %s @llvm.fma.f%d(%s, %s, %s)' % (ty, w, ty, ty, ty))
        v = self.op('call %s @llvm.fma.f%d(%s %s, %s %s, %s %s)'
                    % (ty, w, ty, a, ty, b, ty, c))
      else:
        v = self.op('%s %s %s, %s' % ('fadd' if m.group(1) == 'fadd' else 'fmul',
                                      ty, a, b))
      self.writef(d, v, w, act)
      return False
    m = re.match(r'^(floor|ceil|trunc|rint)(\.sat)?$', mn)
    if m and len(ops) == 2:
      if m.group(2):
        raise LiftError('saturating ' + mn)
      d = O(0)
      w = d.bits
      ty = {32: 'float', 16: 'half'}[w]
      a = self.readf(O(1), w)
      fn = {'floor': 'floor', 'ceil': 'ceil', 'trunc': 'trunc',
            'rint': 'roundeven'}[m.group(1)]
      self.decl('declare %s @llvm.%s.f%d(%s)' % (ty, fn, w, ty))
      self.writef(d, self.op('call %s @llvm.%s.f%d(%s %s)' % (ty, fn, w, ty, a)),
                  w, act)
      return False
    if mn == 'convert':
      self.convert(ops, act)
      return False
    if mn in ('device_load', 'device_store'):
      self.device_mem(mn, ops, shift, act)
      return False
    if mn in ('threadgroup_load', 'threadgroup_store'):
      self.threadgroup_mem(mn, ops, act)
      return False
    if mn == 'memory_barrier':
      # Orders the thread's memory accesses for others (alive2's GPU
      # semantics: a fence). Barriers with nothing but waits between them
      # are one fence (an acq_rel fence is a release and an acquire
      # barrier).
      if not self.last_fence:
        self.emit('fence seq_cst')
      self.last_fence = True
      return False
    if mn in ('threadgroup_barrier',):
      self.decl('declare void @llvm.agx.threadgroup.barrier()')
      self.emit('call void @llvm.agx.threadgroup.barrier()')
      return False
    raise LiftError('instruction ' + x)

  def store_r0l(self, v):
    self.emit('store i16 %s, ptr %%nest' % v)
    self.nest = int(v) if re.match(r'^-?\d+$', v) else None

  def truth_table(self, table, a, b, w):
    """bitop: bit i of the result is table[a_i + 2 * b_i] (applegpu)."""
    terms = []
    na = self.op('xor i%d %s, -1' % (w, a))
    nb = self.op('xor i%d %s, -1' % (w, b))
    for idx, bit in enumerate(table):
      if not bit:
        continue
      x = a if idx & 1 else na
      y = b if idx & 2 else nb
      terms.append(self.op('and i%d %s, %s' % (w, x, y)))
    if not terms:
      return '0'
    v = terms[0]
    for t in terms[1:]:
      v = self.op('or i%d %s, %s' % (w, v, t))
    return v

  def fcmp(self, cc, a, b, w):
    ty = {32: 'float', 16: 'half'}[w]
    if cc in FCMP:
      return self.op('fcmp %s %s %s, %s' % (FCMP[cc], ty, a, b))
    inv = cc.startswith('n') and cc[1:] in ('ltn', 'gtn')
    base = cc[1:] if inv else cc
    if base in ('ltn', 'gtn'):
      # NaN loses: (a < b, both numbers) or (b NaN, a not) (the simulator)
      c = self.op('fcmp %s %s %s, %s' % ('olt' if base == 'ltn' else 'ogt',
                                        ty, a, b))
      bn = self.op('fcmp uno %s %s, 0.0' % (ty, b))
      ao = self.op('fcmp ord %s %s, 0.0' % (ty, a))
      x = self.op('and i1 %s, %s' % (bn, ao))
      c = self.op('or i1 %s, %s' % (c, x))
      return self.op('xor i1 %s, true' % c) if inv else c
    raise LiftError('float condition ' + cc)

  def cond(self, mn, cc, a, b):
    if 'icmp' in mn:
      sg = True if cc.lstrip('n').startswith('s') else None
      x, y = self.read(Operand(a), 32, sg), self.read(Operand(b), 32, sg)
      return self.op('icmp %s i32 %s, %s' % (ICMP[cc], x, y))
    x, y = self.readf(Operand(a)), self.readf(Operand(b))
    return self.fcmp(cc, x, y, 32)

  def cf(self, mn, ops):
    """if / else / while on the nesting counter r0l (applegpu): if: an
    inactive thread goes n deeper, an active one whose test fails is off
    until the else (1); else: an active thread goes off (n), one off since
    the if (1) comes on where the test holds; while: a thread below n stays
    on where the test holds, else leaves the loop (n)."""
    cc, a, b, n = ops[1], ops[2], ops[3], int(ops[4])
    c = self.cond(mn, cc, a, b)
    r = self.read_half('r', 0)
    z = self.op('icmp eq i16 %s, 0' % r)
    if mn.startswith('if_'):
      off = self.op('select i1 %s, i16 0, i16 1' % c)
      inc = self.op('add i16 %s, %d' % (r, n))
      v = self.op('select i1 %s, i16 %s, i16 %s' % (z, off, inc))
    elif mn.startswith('else_'):
      eqn = self.op('icmp eq i16 %s, 1' % r)
      on = self.op('select i1 %s, i16 0, i16 1' % c)
      v1 = self.op('select i1 %s, i16 %s, i16 %s' % (eqn, on, r))
      v = self.op('select i1 %s, i16 %d, i16 %s' % (z, n, v1))
    else:
      below = self.op('icmp ult i16 %s, %d' % (r, n))
      off = self.op('select i1 %s, i16 0, i16 %d' % (c, n))
      v = self.op('select i1 %s, i16 %s, i16 %s' % (below, off, r))
    self.store_r0l(v)

  def bitfield(self, mn, ops, mask, act):
    """applegpu's bfi, bfeil and extr, exactly (in 128 bits: the shift is
    c & 0x7f, and the mask (1 << m) - 1, or all ones for m = 0):
      bfi:   (a & ~(mask << s)) | ((b & mask) << s)
      bfeil: (a & ~mask) | ((b >> s) & mask)
      extr:  (((b << 32) | a) >> s) & mask"""
    d = Operand(ops[0])
    w = d.bits
    W = 128
    ext = lambda o: self.op('zext i%d %s to i%d' % (w, self.read(o, w), W))
    a, b = ext(Operand(ops[1])), ext(Operand(ops[2]))
    s = self.op('and i%d %s, 127' % (W, ext(Operand(ops[3]))))
    m = str(mask if mask is not None else (1 << 32) - 1)
    if mn == 'bfi':
      bm = self.op('and i%d %s, %s' % (W, b, m))
      ms = self.op('shl i%d %s, %s' % (W, m, s))
      keep = self.op('xor i%d %s, -1' % (W, ms))
      lo = self.op('and i%d %s, %s' % (W, a, keep))
      ins = self.op('shl i%d %s, %s' % (W, bm, s))
      v = self.op('or i%d %s, %s' % (W, lo, ins))
    elif mn in ('shlhi', 'shrhi'):
      # shlhi: (((b << s) >> 32) & sm) | (a & ~sm), sm = mask << max(s - 32, 0)
      # shrhi: (((b << 32) >> s) & sm) | (a & ~sm), sm = (mask << 32) >> min(s, 32)
      if mn == 'shlhi':
        s32 = self.op('call i128 @llvm.usub.sat.i128(i128 %s, i128 32)' % s)
        self.decl('declare i128 @llvm.usub.sat.i128(i128, i128)')
        sm = self.op('shl i128 %s, %s' % (m, s32))
        hi = self.op('shl i128 %s, %s' % (b, s))
        hi = self.op('lshr i128 %s, 32' % hi)
      else:
        s32 = self.op('call i128 @llvm.umin.i128(i128 %s, i128 32)' % s)
        self.decl('declare i128 @llvm.umin.i128(i128, i128)')
        m32 = self.op('shl i128 %s, 32' % m)
        sm = self.op('lshr i128 %s, %s' % (m32, s32))
        b32 = self.op('shl i128 %s, 32' % b)
        hi = self.op('lshr i128 %s, %s' % (b32, s))
      x = self.op('and i128 %s, %s' % (hi, sm))
      nsm = self.op('xor i128 %s, -1' % sm)
      y = self.op('and i128 %s, %s' % (a, nsm))
      v = self.op('or i128 %s, %s' % (x, y))
    elif mn == 'extr':
      b2 = self.op('shl i%d %s, 32' % (W, b))
      ab = self.op('or i%d %s, %s' % (W, a, b2))
      r = self.op('lshr i%d %s, %s' % (W, ab, s))
      v = self.op('and i%d %s, %s' % (W, r, m))
    else:
      r = self.op('lshr i%d %s, %s' % (W, b, s))
      r = self.op('and i%d %s, %s' % (W, r, m))
      keep = self.op('and i%d %s, %d' % (W, a, ((1 << W) - 1) ^ int(m)))
      v = self.op('or i%d %s, %s' % (W, keep, r))
    self.write(d, self.op('trunc i%d %s to i%d' % (W, v, w)), w, act)

  def convert(self, ops, act):
    """convert MODE, D, S, ROUND: an integer source has the mode's width,
    read from S zero-extended (sign-extended with .sx) or truncated; a float
    source and destination are as wide as their register."""
    mode, d, s, rnd = ops[0], Operand(ops[1]), Operand(ops[2]), ops[3]
    m = re.match(r'^(f|s|u)(\d*)_to_(f|s|u)(\d*)$', mode)
    if not m:
      raise LiftError('convert ' + mode)
    fk, fb, tk, tb = m.group(1), m.group(2), m.group(3), m.group(4)
    if tk == 'f':
      if rnd != 'rte':
        raise LiftError('convert to float rounding ' + rnd)
      bits = int(fb)
      x = self.read(s, bits)
      fw = d.bits
      fty = {32: 'float', 16: 'half'}[fw]
      v = self.op('%s i%d %s to %s' % ('sitofp' if fk == 's' else 'uitofp',
                                       bits, x, fty))
      self.writef(d, v, fw, act)
    else:
      if rnd != 'rtz':
        raise LiftError('convert to integer rounding ' + rnd)
      bits = int(tb)
      fw = s.bits if s.is_reg else 32
      fty = {32: 'float', 16: 'half'}[fw]
      x = self.readf(s, fw)
      fn = 'llvm.fpto%si.sat.i%d.f%d' % (tk, bits, fw)
      self.decl('declare i%d @%s(%s)' % (bits, fn, fty))
      v = self.op('call i%d @%s(%s %s)' % (bits, fn, fty, x))
      self.write(d, v, bits, act)

  def threadgroup_mem(self, mn, ops, act):
    """threadgroup_load / threadgroup_store FMT, MASK, R, BASE, INDEX: the
    bytes BASE + 4 * size * INDEX + size * component of threadgroup memory
    (written to the registers at once, unlike a device load)
    (the AGX Performance Simulator's tg_words, a recorded finding), BASE a
    16-bit register, a uniform or 0 and INDEX a 16-bit register or an
    immediate. Threadgroup memory is the kernel's threadgroup variable (the
    backend places it at 0); a load that would read beyond it reads 0 and
    a store there is dropped on the GPU, which here is beyond the variable."""
    if self.tgvar is None:
      raise LiftError('threadgroup memory other than one threadgroup variable')
    fmt, mask, R, base, idx = ops[0], ops[1], ops[2], ops[3], ops[4]
    if fmt not in MEM_BYTES:
      raise LiftError('threadgroup format ' + fmt)
    esz = MEM_BYTES[fmt]
    ty = 'i%d' % (8 * esz)
    positions = ['xyzw'.index(c) for c in mask]
    parts = R.split('_')
    Bo, Io = Operand(base), Operand(idx)

    def body():
      b = self.read(Bo, 32, signed=False) if (Bo.is_reg or Bo.imm) else '0'
      i = self.read(Io, 32, signed=False)
      off = self.op('mul i32 %s, %d' % (i, 4 * esz))
      a = self.op('add i32 %s, %s' % (b, off))
      a = self.op('zext i32 %s to i64' % a)
      p = self.op('getelementptr i8, ptr addrspace(3) %s, i64 %s' % (self.tgvar, a))
      for k, pos in enumerate(positions):
        q = self.op('getelementptr i8, ptr addrspace(3) %s, i64 %d'
                    % (p, pos * esz)) if pos else p
        reg = Operand(parts[k])
        if mn == 'threadgroup_load':
          v = self.op('load %s, ptr addrspace(3) %s, align 1' % (ty, q))
          if 8 * esz < reg.bits:
            v = self.op('zext %s %s to i%d' % (ty, v, reg.bits))
          # Threadgroup loads write their registers at once (the simulator's
          # tg_mem; the backend waits for none).
          self.write(reg, v, reg.bits, 'true')
        else:
          v = self.read(reg, 8 * esz)
          self.emit('store %s %s, ptr addrspace(3) %s, align 1' % (ty, v, q))
    self.masked(act, body)

  def device_mem(self, mn, ops, shift, act):
    g, fmt, mask = ops[0], ops[1], ops[2]
    if fmt not in MEM_BYTES:
      raise LiftError('memory format ' + fmt)
    if len(ops) > 7 and ops[7] != '0':
      raise LiftError('a coherent access (' + ops[7] + ')')
    esz = MEM_BYTES[fmt]
    # The components of the mask: memory positions, read into consecutive
    # registers (applegpu).
    positions = ['xyzw'.index(c) for c in mask]
    count = len(positions)
    R, A, Oi, sgn = Operand(ops[3]), Operand(ops[4]), Operand(ops[5]), ops[6]
    # A shift field of 3 is 2 (the AGX Performance Simulator, as the GPU).
    scale = {1: 0, 2: 1, 4: 2}[esz] + (2 if shift == 3 else shift)
    ty = 'i%d' % (8 * esz)
    # The register tuple names one register per component, 16- or 32-bit
    # (applegpu's MemoryReg); a narrower memory value is zero-extended.
    parts = R.base.split('_')
    if len(parts) < count:
      raise LiftError('memory register ' + R.tok)
    if esz == 4 and any(Operand(x).bits != 32 for x in parts[:count]):
      raise LiftError('32-bit memory values in 16-bit registers: ' + R.tok)

    def body():
      p = self.address(A, Oi, sgn, scale, esz)
      for k, pos in enumerate(positions):
        q = self.op('getelementptr %s, ptr addrspace(1) %s, i64 %d'
                    % (ty, p, pos)) if pos else p
        reg = Operand(parts[k])
        rbits = reg.bits
        if mn == 'device_load':
          # Aligned to the item: the base rounds down to it and the offsets
          # are multiples of it (which spares alive2 modeling single bytes).
          v = self.op('load %s, ptr addrspace(1) %s, align %d' % (ty, q, esz))
          if 8 * esz < rbits:
            v = self.op('zext %s %s to i%d' % (ty, v, rbits))
          # Its registers are written at the wait on its slot.
          self.n += 1
          val, flag = '%%pend%d' % self.n, '%%pend%d.on' % self.n
          self.shadows.append((val, flag, rbits))
          self.emit('store i%d %s, ptr %s' % (rbits, v, val))
          self.emit('store i1 true, ptr %s' % flag)
          self.pending.append((int(g), reg, val, flag, rbits))
        else:
          v = self.read(reg, 8 * esz)
          self.emit('store %s %s, ptr addrspace(1) %s, align %d'
                    % (ty, v, q, esz))
    self.masked(act, body)

  # -- the function ---------------------------------------------------------

  def function(self, name, prefix, params_text, suffix):
    out = ['define %s@%s(%s)%s{' % (prefix, name, params_text, suffix)]
    out.append('entry:')
    # r0l, the nesting counter, starts at 0: the thread is active. The other
    # registers hold nothing the kernel may read.
    out.append('  %nest = alloca i16')
    out.append('  store i16 0, ptr %nest')
    for val, flag, bits in self.shadows:
      out.append('  %s = alloca i%d' % (val, bits))
      out.append('  %s = alloca i1' % flag)
      out.append('  store i1 false, ptr %s' % flag)
    for n in sorted(self.gpr):
      # Whatever the register held: some value, unknown (freeze poison), so
      # that writing half of it leaves the other half defined.
      out.append('  %%r%d = alloca i32' % n)
      out.append('  %%r%d.init = freeze i32 poison' % n)
      out.append('  store i32 %%r%d.init, ptr %%r%d' % (n, n))
    out += ['  ' + e for e in self.entry]
    out.append('  br label %body')
    out += self.lines
    out.append('}')
    return '\n'.join(out)


class RawCase:
  """A program recorded on the GPU (the AGX Performance Simulator's coverage
  suite): lifted as @prog(i32 lane), run for every lane by @main against the
  recorded buffers, which it prints."""

  def __init__(self, case):
    import base64
    self.case = case
    L = case['layout']
    self.table_va, self.out_va = L['table_va'], L['out_va']
    self.table = base64.b64decode(case['input']['table'])
    words = L['out_words']
    out = bytearray()
    for i in range(words):
      v = 0xcdcdcdcd if i < words - 16 else 0xa5a50000 + (i - (words - 16))
      out += struct.pack('<I', v)
    self.out = bytes(out)
    self.launch = case['launch']

  def uniforms(self, lifter):
    for name, v in self.case['layout']['uniforms'].items():
      if v == 'out_va':
        v = self.out_va
      elif v == 'table_va':
        v = self.table_va
      halves = reg_halves(name)[1]
      for k, h in enumerate(halves):
        lifter.uniform[h] = str((v >> (16 * k)) & 0xffff)

  def address(self, lifter, a, esz):
    return lifter.op('call ptr addrspace(1) @xlate(i64 %s)' % a)

  def module(self, fn):
    g, wg = self.launch['grid'], self.launch['wg']
    if g[1] != 1 or g[2] != 1 or wg[1] != 1 or wg[2] != 1:
      raise LiftError('a launch of more than one dimension')
    count = self.launch['count']
    wx = wg[0]
    srs = {80: '%lane', 0: '%tg', 48: '%tid', 4: str(wx), 5: '1', 6: '1',
           1: '0', 2: '0', 49: '0', 50: '0', 81: '0', 82: '0', 51: '%tid',
           52: '%simdlane', 53: '%simd'}
    cases = '\n'.join('    i32 %d, label %%s%d' % (k, k) for k in srs)
    blocks = '\n'.join('s%d:\n  ret i32 %s' % (k, v) for k, v in srs.items())
    def cstr(b):
      return 'c"' + ''.join('\\%02X' % x for x in b) + '"'
    return """
@table = global [%(nt)d x i8] %(table)s, align 16
@out = global [%(no)d x i8] %(out)s, align 16

declare i64 @write(i32, ptr, i64)

define ptr addrspace(1) @xlate(i64 %%va) {
  %%t = sub i64 %%va, %(tva)d
  %%int = icmp ult i64 %%t, %(nt)d
  br i1 %%int, label %%table, label %%out
table:
  %%pt = getelementptr i8, ptr @table, i64 %%t
  %%qt = addrspacecast ptr %%pt to ptr addrspace(1)
  ret ptr addrspace(1) %%qt
out:
  %%o = sub i64 %%va, %(ova)d
  %%ino = icmp ult i64 %%o, %(no)d
  br i1 %%ino, label %%outok, label %%bad
outok:
  %%po = getelementptr i8, ptr @out, i64 %%o
  %%qo = addrspacecast ptr %%po to ptr addrspace(1)
  ret ptr addrspace(1) %%qo
bad:
  call void @llvm.trap()
  unreachable
}

declare void @llvm.trap()

define i32 @sr(i32 %%n, i32 %%lane) {
  %%tg = udiv i32 %%lane, %(wx)d
  %%tid = urem i32 %%lane, %(wx)d
  %%simdlane = urem i32 %%lane, 32
  %%simd = udiv i32 %%tid, 32
  switch i32 %%n, label %%bad [
%(cases)s
  ]
%(blocks)s
bad:
  call void @llvm.trap()
  unreachable
}

%(fn)s

define i32 @main() {
entry:
  br label %%loop
loop:
  %%l = phi i32 [ 0, %%entry ], [ %%n, %%loop ]
  call void @prog(i32 %%l)
  %%n = add i32 %%l, 1
  %%done = icmp eq i32 %%n, %(count)d
  br i1 %%done, label %%exit, label %%loop
exit:
  call i64 @write(i32 1, ptr @out, i64 %(no)d)
  call i64 @write(i32 1, ptr @table, i64 %(nt)d)
  ret i32 0
}
""" % dict(nt=len(self.table), no=len(self.out), table=cstr(self.table),
           out=cstr(self.out), tva=self.table_va, ova=self.out_va, wx=wx,
           cases=cases, blocks=blocks, fn=fn, count=count)


def main_case(args):
  import gzip, json
  case = json.load(gzip.open(args.case) if args.case.endswith('.gz')
                   else open(args.case))
  # agx_asm's disassembly: name, size and text of each instruction. A jump's
  # target is pc+N or pc-N from its own address: a label there.
  insts = []
  pc = 0
  for part in open(args.asm).read().split('|'):
    t = part.strip().split(None, 2)
    if len(t) == 3:
      insts.append((pc, t[2]))
      pc += int(t[1])
  targets = set()
  for at, text in insts:
    m = re.search(r'\bpc([+-]\d+)', text)
    if m:
      targets.add(at + int(m.group(1)))
  body = []
  for at, text in insts:
    if at in targets:
      body.append('Lpc%d:' % at)
    m = re.search(r'\bpc([+-]\d+)', text)
    if m:
      text = text[:m.start()] + 'Lpc%d' % (at + int(m.group(1))) + text[m.end():]
    body.append('\t' + text)
  if pc in targets:
    body.append('Lpc%d:' % pc)
  rc = RawCase(case)
  lf = Lifter([], None, None, body)
  lf.raw = rc
  try:
    lf.init_uniforms = lambda: rc.uniforms(lf)
    lf.lift()
    fn = lf.function('prog', 'void ', 'i32 %lane', ' ')
    text = rc.module(fn)
  except LiftError as e:
    print('agx-lift: cannot lift %s: %s' % (case['name'], e), file=sys.stderr)
    return 2
  text = 'target triple = "x86_64-unknown-linux-gnu"\n' + \
      '\n'.join(sorted(lf.decls)) + text
  (open(args.output, 'w') if args.output != '-' else sys.stdout).write(text)
  return 0


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
  ap.add_argument('ir', help='the kernel in AGX LLVM IR (.ll)')
  ap.add_argument('asm', help='the assembly llc printed for it (.s)')
  ap.add_argument('function', help='the kernel')
  ap.add_argument('-o', '--output', default='-')
  ap.add_argument('--exact-alignment', action='store_true',
                  help='check every buffer alignment, not only %d-byte '
                  'aligned buffers (slower)' % BUFFER_ALIGN)
  ap.add_argument('--source-out', help='also write the module as checked: '
                  'its kernels\' float denormals flushed (without ieee-fp32)')
  ap.add_argument('--case', help='lift a recorded coverage case instead: '
                  'its JSON, with ASM the disassembly (agx_asm !HEX) of its '
                  'code, and FUNCTION ignored')
  args = ap.parse_args()
  if args.case:
    return main_case(args)
  ll = one_line_headers(open(args.ir).read())
  asm = open(args.asm).read()
  if args.source_out:
    open(args.source_out, 'w').write(flushing_source(ll))
  try:
    prefix, params, suffix = parse_signature(ll, args.function)
    words, start = kernel_record(asm, args.function)
    lf = Lifter(params, words, start, kernel_body(asm, args.function))
    lf.exact_align = args.exact_alignment
    tgs = re.findall(r'^(@[\w.$"]+) = [^\n]*addrspace\(3\) global', ll, re.M)
    if len(tgs) == 1:
      lf.tgvar = tgs[0]
    lf.lift()
    # The f32 ALU flushes denormal sources and results to zero (applegpu's
    # FloatSrc / FloatDst), except under AGX's ieee-fp32 feature; the other
    # types' denormals as the kernel has them (its code flushes those).
    env = kernel_fpenv(ll, args.function)
    suffix = re.sub(r'#\d+', '', suffix)
    suffix = re.sub(r'denormal_fpenv\([^)]*\)', '', suffix)
    if not re.search(r'"target-features"="[^"]*\+ieee-fp32', ll):
      env = float_flushes(env)
    if env:
      # after (local_)unnamed_addr, before the other function attributes
      m = re.match(r'\s*((?:local_)?unnamed_addr\b)?', suffix)
      suffix = (' ' + (m.group(1) or '') + ' denormal_fpenv(%s)' % env +
                suffix[m.end():])
    fn = lf.function(args.function, prefix,
                     ', '.join(lifted_param(p, not lf.exact_align)
                               for p in params), suffix)
  except LiftError as e:
    print('agx-lift: cannot lift %s: %s' % (args.function, e), file=sys.stderr)
    return 2
  head = []
  for l in ll.split('\n'):
    if l.startswith('target ') or l.startswith('source_filename'):
      head.append(l)
  # The module's variables (alive2 pairs them by name with the IR's).
  head.append('')
  head += [l for l in ll.split('\n') if re.match(r'^@[\w.$"]+ = ', l)]
  text = '\n'.join(head + [''] + sorted(lf.decls) + ['', fn, ''])
  (open(args.output, 'w') if args.output != '-' else sys.stdout).write(text)
  return 0


if __name__ == '__main__':
  sys.exit(main())
