#!/usr/bin/env python3
# Copyright (c) 2018-present The Alive2 Authors.
# Distributed under the MIT license that can be found in the LICENSE file.
"""Random NVPTX functions over memory: loads and stores of mixed widths at
offsets and alignments (under-aligned ones the backend splits), vector
accesses, extract/insert/shuffle, and some arithmetic between."""
import random, sys

SC = ['i8', 'i16', 'i32', 'i64']
VEC = ['<2 x i8>', '<4 x i8>', '<2 x i16>', '<4 x i16>', '<2 x i32>',
       '<4 x i32>', '<8 x i8>', '<3 x i16>', '<3 x i8>']


def size(t):
  if t.startswith('<'):
    n, e = t[1:-1].split(' x ')
    return int(n) * int(e[1:]) // 8
  return int(t[1:]) // 8


def gen_func(name, rnd):
  space = rnd.choice(['ptr addrspace(1)', 'ptr', 'ptr addrspace(3)'])
  lines = []
  cnt = [0]
  vals = {}

  def new():
    cnt[0] += 1
    return '%%v%d' % cnt[0]

  def addr(off):
    if off == 0:
      return '%p'
    a = new()
    lines.append('%s = getelementptr i8, %s %%p, i64 %d' % (a, space, off))
    return a

  for _ in range(rnd.randint(3, 7)):
    t = rnd.choice(SC + VEC)
    sz = size(t)
    nat = 1
    while nat * 2 <= sz and nat < 16:
      nat *= 2
    al = rnd.choice([nat, nat, 1, 2, 4]) if sz > 1 else 1
    al = min(al, nat)
    off = rnd.choice([0, 1, 2, 4, 8, 12, 16]) // al * al
    k = rnd.random()
    if k < 0.45:
      v = new()
      lines.append('%s = load %s, %s %s, align %d' % (v, t, space, addr(off), al))
      vals.setdefault(t, []).append(v)
    elif k < 0.6 and t.startswith('<') and vals.get(t):
      src = rnd.choice(vals[t])
      n, e = t[1:-1].split(' x ')
      v = new()
      if rnd.random() < 0.5:
        lines.append('%s = extractelement %s %s, i32 %d' % (v, t, src, rnd.randrange(int(n))))
        vals.setdefault(e, []).append(v)
      else:
        mask = ', '.join('i32 %d' % rnd.randrange(2 * int(n)) for _ in range(int(n)))
        other = rnd.choice(vals[t])
        lines.append('%s = shufflevector %s %s, %s %s, <%s x i32> <%s>' % (v, t, src, t, other, n, mask))
        vals[t].append(v)
    elif k < 0.75 and vals.get(t):
      a, b = rnd.choice(vals[t]), rnd.choice(vals[t])
      v = new()
      op = rnd.choice(['add', 'xor', 'mul', 'lshr', 'shl', 'and'])
      if op in ('lshr', 'shl'):
        e = t[1:-1].split(' x ')[1] if t.startswith('<') else t
        s = rnd.randrange(int(e[1:]))
        b = ('<' + ', '.join('%s %d' % (e, s) for _ in range(int(t[1:-1].split(' x ')[0]))) + '>') if t.startswith('<') else str(s)
      lines.append('%s = %s %s %s, %s' % (v, op, t, a, b))
      vals[t].append(v)
    else:
      if vals.get(t):
        v = rnd.choice(vals[t])
      elif not t.startswith('<'):
        v = str(rnd.randrange(1 << min(31, 8 * sz)))
      else:
        continue
      lines.append('store %s %s, %s %s, align %d' % (t, v, space, addr(off), al))
  # Something observable at the end: store every loaded value somewhere.
  i = 0
  for t, vs in sorted(vals.items()):
    lines.append('store %s %s, ptr addrspace(1) %%out, align 1' % (t, vs[-1]))
    i += 1
    break
  head = 'define void @%s(%s %%p, ptr addrspace(1) %%out) {' % (name, space)
  return '\n'.join([head] + ['  ' + l for l in lines] + ['  ret void', '}'])


def main():
  seed, count = int(sys.argv[1]), int(sys.argv[2])
  rnd = random.Random(seed)
  print('; RUN: llc -mtriple=nvptx64 -mcpu=%s' % ['sm_70', 'sm_80', 'sm_90'][seed % 3])
  print('target triple = "nvptx64-nvidia-cuda"\n')
  for i in range(count):
    print(gen_func('m%d_%d' % (seed, i), rnd))
    print()


main()
