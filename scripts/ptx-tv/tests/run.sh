#!/usr/bin/env bash
# ptx-tv's tests: kernels whose PTX verifies, and PTX with a seeded mistake
# that alive2 must reject. LLC, OPT and ALIVE_TV as ptx-tv; JOBS checks run
# at once (default: half the processors); each query may take 120 s unless a
# case says otherwise.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
tv=$here/../ptx-tv
jobs=${JOBS:-$(( ($(nproc) + 1) / 2 ))}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
n=0

check() {  # check correct|incorrect DESCRIPTION ptx-tv-args...
  local want=$1 what=$2 out r start=$SECONDS
  shift 2
  [[ " $* " == *" -smt-to="* ]] || set -- "$@" -smt-to=120000
  out=$("$tv" "$@" 2>&1)
  if [[ $want == correct ]]; then
    local funcs ok
    funcs=$(grep -c '^== ' <<<"$out")
    ok=$(grep -c 'Transformation seems to be correct' <<<"$out")
    [[ $funcs -gt 0 && $funcs == "$ok" ]] &&
      ! grep -qE "doesn't verify|cannot lift|not valid|ERROR" <<<"$out" &&
      r=ok || r=FAIL
  else
    grep -q "doesn't verify" <<<"$out" && r=ok || r=FAIL
  fi
  [[ $r == ok ]] || echo "$out" | tail -20
  echo "$r: $what ($((SECONDS - start)) s)"
}
expect() {  # in the background, at most $jobs at a time
  while (( $(jobs -rp | wc -l) >= jobs )); do wait -n; done
  n=$((n + 1))
  check "$@" > "$tmp/result.$n" &
}

expect correct "guarded verifies" --keep "$tmp/guarded" "$here/guarded.ll"
expect correct "shared verifies" --keep "$tmp/shared" "$here/shared.ll"
expect correct "vec verifies" --keep "$tmp/vec" "$here/vec.ll"
expect correct "bits verifies" --keep "$tmp/bits" "$here/bits.ll"
expect correct "float verifies" --keep "$tmp/float" "$here/float.ll"
expect correct "intrinsics verify" --keep "$tmp/intrinsics" \
  "$here/intrinsics.ll"
# alive2 checks loops unrolled to a bound.
unroll=(-src-unroll=2 -tgt-unroll=2 -smt-to=300000)
expect correct "loop verifies (two iterations)" --keep "$tmp/loop" \
  "$here/loop.ll" "${unroll[@]}"
wait  # the mutations edit those kernels' PTX

mutate() {  # mutate NAME KERNEL FUNCTION SED [alive-tv options]
  local name=$1 kernel=$2 func=$3 edit=$4; shift 4
  sed "$edit" "$tmp/$kernel/out.ptx" > "$tmp/$name.ptx"
  if cmp -s "$tmp/$name.ptx" "$tmp/$kernel/out.ptx"; then
    n=$((n + 1))
    echo "FAIL: $name: the mutation did not apply" > "$tmp/result.$n"
    return
  fi
  expect incorrect "$name is rejected" --ptx "$tmp/$name.ptx" --func "$func" \
    "$here/$kernel.ll" "$@"
}
mutate condition guarded guarded 's/setp.ge.s32/setp.gt.s32/'
mutate immediate guarded guarded 's/\(xor.b32[ \t]*%r[0-9]*, %r[0-9]*\), 255/\1, 254/'
mutate scale guarded guarded 's/\(mul.wide.s32[ \t]*%rd[0-9]*, %r[0-9]*\), 4/\1, 8/'
mutate index-sign guarded guarded 's/mul.wide.s32/mul.wide.u32/'
mutate buffer guarded guarded 's/\[guarded_param_1\]/[guarded_param_0]/'
mutate barrier shared shared '/bar.sync/d'
mutate mirror shared shared 's/\(sub.s32[ \t]*%r[0-9]*\), 255,/\1, 254,/'
mutate shift-sign bits shifts 's/shr.u32/shr.s32/'
mutate min-sign bits counts 's/min.s32/min.u32/'
mutate param-sign bits counts 's/ld.param.s16/ld.param.b16/'
mutate count bits wide 's/clz.b64/popc.b64/'
mutate narrow-store bits narrow 's/st.global.b16/st.global.b32/'
mutate select-sign bits dev 's/setp.gt.u32/setp.gt.s32/'
mutate fadd float fma_like 's/add.rn.f32/sub.rn.f32/'
mutate fmul float scale 's/mul.rn.f32/add.rn.f32/'
mutate cvt-rounding intrinsics cvt 's/cvt.rzi.s32.f32/cvt.rni.s32.f32/'
mutate prmt-selector intrinsics prmt 's/0x1076U/0x1067U/'
mutate loop-step loop loop 's/\(add.s32[ \t]*\(%r[0-9]*\), \2\), 1;/\1, 2;/' "${unroll[@]}"
wait

fail=0
for i in $(seq 1 $n); do
  cat "$tmp/result.$i"
  grep -q '^FAIL' "$tmp/result.$i" && fail=1
done
exit $fail
