#!/usr/bin/env python3
# Copyright (c) 2018-present The Alive2 Authors.
# Distributed under the MIT license that can be found in the LICENSE file.
"""Sort batch.py's mismatches by the false-alarm patterns found by hand
(see README.md); print the others, which are for triage.

  classify.py WORK results.jsonl
"""
import collections, json, re, sys

work, res = sys.argv[1], sys.argv[2]
rs = [json.loads(l) for l in open(res)]


def func_text(text, name):
  m = re.search(r'^define[^\n]*@"?%s"?\(.*?^}' % re.escape(name), text,
                re.M | re.S)
  return m.group(0) if m else ''


def ptx_text(text, name):
  m = re.search(r'\b%s\(.*?^}' % re.escape(name), text, re.M | re.S)
  return m.group(0) if m else ''


def classify(r):
  d = '%s/%s' % (work, r['file'])
  src = func_text(open(d + '/src.ll').read(), r['func'])
  src0 = func_text(open(d + '/src0.ll').read(), r['func'])
  # The finding (findings/README.md): an <N x i1> parameter, return value
  # or argument declared as one byte and accessed past it.
  if re.search(r'<\d+ x i1>', src.split('\n')[0]) and \
     re.search(r'\.b8\s+[\w$]+\[1\]', open(d + '/out.ptx').read()):
    return 'FINDING-vNi1-param-out-of-bounds'
  # Inline asm: opaque to alive2 in the IR, its text in the PTX.
  if re.search(r'\basm\b', src):
    return 'inline-asm'
  # !atomic.ignore.denormal.mode (dropped for alive2) allows atom.add.f32's
  # flush.
  if 'ignore.denormal.mode' in src0:
    return 'ignore-denormal-mode-metadata'
  # A store of undef dropped over memory that was poison.
  if re.search(r'store [^,]* undef,', src) and 'Mismatch in memory' in r.get('why', ''):
    return 'undef-store-dropped'
  ptx = ptx_text(open(d + '/out.ptx').read(), r['func'])
  try:
    out = open('%s/%s.alive.txt' % (d, r['func'])).read()
  except OSError:
    out = ''
  # Stores of integers that are not whole bytes: alive2 does not take a
  # byte written whole as refining one written with fewer bits.
  widths = [int(w) for w in re.findall(
      r'(?:store|load) (?:atomic )?(?:volatile )?(?:<\d+ x )?i(\d+)\b', src)]
  if any(w % 8 for w in widths) and re.search(r'written with \d+ bits', out):
    return 'subbyte-store'
  # Sub-word atomics emulated by a CAS loop on the aligned 32-bit word.
  if re.search(r'(atomicrmw|cmpxchg)[^\n]*\bi(8|16)\b|(atomicrmw|cmpxchg)[^\n]*\b(half|bfloat)\b', src) and \
     re.search(r'and\.b64\s+%rd\d+, %rd\d+, -4', ptx):
    return 'subword-atomic-widened'
  # An ordered atomicrmw the backend lowers to a CAS loop between fences:
  # the GPU's memory has bits where the IR's had poison, and alive2's fence
  # calls then differ (unproven, not refuted).
  if re.search(r'atomicrmw[^\n]*(acq_rel|seq_cst|acquire|release)', src) and \
     re.search(r'atom\.[\w.:]*cas', ptx) and \
     re.search(r'alive\.gpu\.fence triggered UB', out):
    return 'ordered-cas-loop-unproven'
  # A byval argument whose address escapes: NVPTX copies it to the stack
  # and escapes the copy's (both dangle after the return).
  if 'byval(' in src and (re.search(r'Target value: pointer\(local', out) or
                          '__local_depot' in ptx):
    return 'byval-escape-copy'
  # NaN payloads.
  if re.search(r'(Source|Target) value: .*NaN', out):
    return 'nan-payload'
  return None


cls = collections.Counter()
rest = []
for r in rs:
  if r['status'] != 'INCORRECT':
    continue
  c = classify(r)
  if c:
    cls[c] += 1
  else:
    rest.append(r)
print('statuses:', dict(collections.Counter(r['status'] for r in rs)))
print('known false alarms:', dict(cls))
print('to triage: %d' % len(rest))
for r in rest:
  print('  %s %s: %s' % (r['file'], r['func'], r.get('why', '')))
