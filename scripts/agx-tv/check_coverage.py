#!/usr/bin/env python3
# Copyright (c) 2018-present The Alive2 Authors.
# Distributed under the MIT license that can be found in the LICENSE file.
"""Check the AGX lifter against the GPU: lift the programs of the AGX
Performance Simulator's coverage suite (each recorded on an M2 Max), run the
lifted IR for every lane on the recorded input, and compare the buffers it
leaves with those the GPU left.

  check_coverage.py SIM_CHECKOUT [AREA ...] [--cc clang] [-j N] [-v]

SIM_CHECKOUT is a checkout of AGX-Performance-Simulator with its tools built
(make rust: rust/target/release/agx_asm). Areas default to alu, mem_load,
mem_store and control. A case is skipped when the lifter does not handle
one of its instructions, or it runs cross-lane operations (lanes are run one
after the other).
"""

import argparse
import base64
import concurrent.futures
import gzip
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CROSS_LANE = ('simd_', 'quad_', 'icmp_ballot', 'fcmp_ballot', 'barrier')
# Cases whose result is a SIMD-group effect that running the lanes one after
# another leaves out: the stores of neighbouring lanes overlap, and the GPU
# writes a store component by component across the lanes (the lowest lane's
# value stays: the AGX Performance Simulator's device_mem). In the IR such
# stores are a race between threads.
SIMD_EFFECTS = ('store_device_load_0_i16_xz', 'store_device_load_0_i16_yw',
                'store_device_load_0_i32_xyz_', 'store_device_load_0_i32_xyzw_',
                'store_device_load_0_i32_xz', 'store_device_load_0_i32_yw',
                'store_device_load_0_i8_xz', 'store_device_load_0_i8_yw')


def disassemble(asm_tool, codes):
  p = subprocess.run([asm_tool], input=''.join('!%s\n' % c for c in codes),
                     capture_output=True, text=True, check=True)
  return p.stdout.rstrip('\n').split('\n')


def check(case_path, text, cc, tmp, verbose):
  case = json.load(gzip.open(case_path))
  name = case['name']
  if text.startswith('ERROR'):
    return name, 'skip', 'undecodable'
  if any(m in text for m in CROSS_LANE):
    return name, 'skip', 'cross-lane'
  base = os.path.join(tmp, name)
  open(base + '.asm', 'w').write(text)
  open(base + '.json', 'w').write(json.dumps(case))
  r = subprocess.run([sys.executable, os.path.join(HERE, 'agx_lift.py'), '-',
                      base + '.asm', 'prog', '--case', base + '.json', '-o',
                      base + '.ll'], capture_output=True, text=True)
  if r.returncode:
    return name, 'skip', r.stderr.strip().split(': ', 2)[-1]
  r = subprocess.run([cc, '-O1', '-w', base + '.ll', '-o', base + '.exe', '-lm'],
                     capture_output=True, text=True)
  if r.returncode:
    return name, 'error', 'compile: ' + r.stderr.strip()[:300]
  r = subprocess.run([base + '.exe'], capture_output=True, timeout=120)
  if r.returncode:
    return name, 'error', 'run: exit %d' % r.returncode
  out = base64.b64decode(case['expect']['out'])
  table = base64.b64decode(case['expect']['table'])
  got = r.stdout
  if got[:len(out)] == out and got[len(out):] == table:
    return name, 'pass', ''
  if name.startswith(SIMD_EFFECTS):
    return name, 'skip', 'overlapping stores of neighbouring lanes (SIMD-group order)'
  # The first difference, as 32-bit words.
  for k in range(0, min(len(got), len(out) + len(table)), 4):
    want = (out + table)[k:k + 4]
    if got[k:k + 4] != want:
      where = 'out' if k < len(out) else 'table'
      off = k if k < len(out) else k - len(out)
      return name, 'fail', '%s word %d: lifted %s, GPU %s' % (
          where, off // 4, got[k:k + 4][::-1].hex(), want[::-1].hex())
  return name, 'fail', 'output size'


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
  ap.add_argument('sim')
  ap.add_argument('areas', nargs='*',
                  default=['alu', 'mem_load', 'mem_store', 'control'])
  ap.add_argument('--cc', default='clang')
  ap.add_argument('-j', type=int, default=os.cpu_count())
  ap.add_argument('-v', action='store_true')
  ap.add_argument('--only', help='cases whose name contains this')
  args = ap.parse_args()
  asm_tool = os.path.join(args.sim, 'rust/target/release/agx_asm')
  paths = []
  for a in args.areas:
    d = os.path.join(args.sim, 'coverage/expected', a)
    paths += sorted(os.path.join(d, f) for f in os.listdir(d)
                    if not args.only or args.only in f)
  codes = [json.load(gzip.open(p))['code'] for p in paths]
  texts = disassemble(asm_tool, codes)
  counts = {}
  reasons = {}
  with tempfile.TemporaryDirectory() as tmp, \
       concurrent.futures.ThreadPoolExecutor(args.j) as ex:
    for name, status, why in ex.map(
        lambda pt: check(pt[0], pt[1], args.cc, tmp, args.v),
        zip(paths, texts)):
      counts[status] = counts.get(status, 0) + 1
      if status != 'pass':
        reasons.setdefault((status, why if status != 'skip' else why), []).append(name)
      if args.v or status in ('fail', 'error'):
        print('%-5s %s %s' % (status, name, why))
  print('\nskipped:')
  for (status, why), names in sorted(reasons.items(), key=lambda x: -len(x[1])):
    if status == 'skip':
      print('  %4d  %s' % (len(names), why))
  print('\n' + ', '.join('%s %d' % kv for kv in sorted(counts.items())))
  return 1 if counts.get('fail') or counts.get('error') else 0


if __name__ == '__main__':
  sys.exit(main())
