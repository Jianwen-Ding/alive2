#!/usr/bin/env python3
# Copyright (c) 2018-present The Alive2 Authors.
# Distributed under the MIT license that can be found in the LICENSE file.
"""Run ptx-tv's pipeline over a directory of LLVM IR files (llvm/test/
CodeGen/NVPTX, or a fuzzer's): each function compiled with its file's first
llc RUN line's -mcpu/-mattr, lifted and (--verify) checked by alive-tv.
Results as JSON lines; classify.py sorts the mismatches.

  batch.py DIR WORK [--verify] [-j N] [--per-file N] [--skip REGEX]
           [--only FILE] [--smt-to MS] [--wall S] > results.jsonl
"""
import argparse, concurrent.futures as cf, json, os, re, shlex, subprocess, sys

LIFT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                    'ptx_lift.py')
LLC = os.environ.get('LLC', 'llc')
OPT = os.environ.get('OPT', 'opt')
ALIVE = os.environ.get('ALIVE_TV', 'alive-tv')


def run(cmd, timeout, **kw):
  try:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          **kw)
  except subprocess.TimeoutExpired:
    return None


def llc_flags(path):
  """The first RUN line's llc flags for nvptx64, or None."""
  text = open(path, errors='replace').read()
  for line in text.split('\n'):
    m = re.search(r'RUN:.*?\bllc\b(.*?)(\||$)', line)
    if not m or 'not llc' in line:
      continue
    args = m.group(1)
    if 'nvptx64' not in args and 'nvptx64' not in text[:2000]:
      continue
    if '-march=nvptx ' in args + ' ' or 'nvptx-' in args and 'nvptx64' not in args:
      continue
    flags = re.findall(r'-m(?:cpu|attr)=[\w.,+-]+', args)
    return flags
  return None


def prepare(path, work):
  name = os.path.basename(path)[:-3]
  d = os.path.join(work, name)
  os.makedirs(d, exist_ok=True)
  flags = llc_flags(path)
  if flags is None:
    return name, d, None, 'no nvptx64 llc RUN line'
  r = run([OPT, '-mtriple=nvptx64-nvidia-cuda', '-passes=verify', '-S', path,
           '-o', d + '/src0.ll'], 60)
  if not r or r.returncode:
    return name, d, None, 'opt rejects the file'
  src = open(d + '/src0.ll').read()
  src = re.sub(r', !(tbaa|tbaa\.struct|invariant\.load|alias\.scope|noalias|'
               r'nontemporal|mem\.cache_hint|atomic\.[a-z.]*|[a-z.]*nvvm[a-z.]*)'
               r' ![0-9]+', '', src)
  open(d + '/src.ll', 'w').write(src)
  r = run([LLC, '-mtriple=nvptx64-nvidia-cuda'] + flags +
          [d + '/src.ll', '-o', d + '/out.ptx'], 120)
  if not r or r.returncode:
    return name, d, None, 'llc fails'
  funcs = re.findall(r'^define\s+[^@]*@"?([\w.$]+)"?\(', src, re.M)
  return name, d, funcs, ' '.join(flags)


def one(job, verify, smt_to, wall):
  name, d, f = job
  res = {'file': name, 'func': f}
  raw = '%s/%s.raw.ll' % (d, f)
  r = run([sys.executable, LIFT, d + '/src.ll', d + '/out.ptx', f, '-o', raw],
          60)
  if r is None:
    return dict(res, status='lift-timeout')
  if r.returncode:
    msg = r.stderr.strip().split('\n')[-1]
    return dict(res, status='cannot-lift', why=re.sub(r'^ptx_lift: cannot lift \S+: ', '', msg)[:200])
  r = run([OPT, '-passes=sroa,mem2reg', '-S', raw, '-o', '%s/%s.ll' % (d, f)],
          60)
  if not r or r.returncode:
    return dict(res, status='invalid-lifted-ir',
                why=(r.stderr if r else 'timeout')[:300])
  if not verify:
    return dict(res, status='lifted')
  r = run([ALIVE, d + '/src.ll', '%s/%s.ll' % (d, f), '-func=' + f,
           '-disable-undef-input', '-smt-to=%d' % smt_to], wall)
  if r is None:
    return dict(res, status='wall-timeout')
  out = r.stdout + r.stderr
  open('%s/%s.alive.txt' % (d, f), 'w').write(out)
  if "doesn't verify" in out:
    err = re.search(r'ERROR: (.*)', out)
    return dict(res, status='INCORRECT', why=err.group(1) if err else '')
  if 'seems to be correct' in out:
    return dict(res, status='correct')
  if 'ERROR: Timeout' in out:
    return dict(res, status='smt-timeout')
  err = re.search(r'ERROR: (.*)', out)
  return dict(res, status='alive-error', why=(err.group(1) if err else out[-200:]))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('dir')
  ap.add_argument('work')
  ap.add_argument('--verify', action='store_true')
  ap.add_argument('--only', help='file of "file func" lines to run')
  ap.add_argument('-j', type=int, default=4)
  ap.add_argument('--smt-to', type=int, default=20000)
  ap.add_argument('--wall', type=int, default=90)
  ap.add_argument('--per-file', type=int, default=0)
  ap.add_argument('--skip', help='regex of test file names to skip')
  a = ap.parse_args()
  files = sorted(os.path.join(a.dir, f) for f in os.listdir(a.dir)
                 if f.endswith('.ll'))
  if a.skip:
    files = [f for f in files if not re.search(a.skip, os.path.basename(f))]
  only = None
  if a.only:
    only = set(tuple(l.split()) for l in open(a.only) if l.strip())
    files = [f for f in files if os.path.basename(f)[:-3] in {o[0] for o in only}]
  jobs = []
  with cf.ThreadPoolExecutor(a.j) as ex:
    for name, d, funcs, info in ex.map(lambda p: prepare(p, a.work), files):
      if funcs is None:
        print(json.dumps({'file': name, 'status': 'skipped', 'why': info}),
              flush=True)
        continue
      if a.per_file:
        funcs = funcs[:a.per_file]
      for f in funcs:
        if only is None or (name, f) in only:
          jobs.append((name, d, f))
  with cf.ThreadPoolExecutor(a.j) as ex:
    for res in ex.map(lambda j: one(j, a.verify, a.smt_to, a.wall), jobs):
      print(json.dumps(res), flush=True)


main()
