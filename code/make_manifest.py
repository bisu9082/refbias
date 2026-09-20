#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refbias — MANIFEST.md 생성
===========================
저장소의 모든 파일에 대해 바이트 크기와 SHA-256 앞 16자를 기록한다.
.git, __pycache__, *.pyc, 그리고 MANIFEST.md 자신은 제외한다.

  python3 code/make_manifest.py          MANIFEST.md 재생성
  python3 code/make_manifest.py --check  재생성하지 않고 현재 MANIFEST 와 대조만

--check 는 불일치가 있으면 종료코드 1 을 반환하므로 CI 에 걸 수 있다.
"""
import os, sys, hashlib, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {'.git', '__pycache__'}
HEADER = ("# MANIFEST\n\n저장소 파일 목록. SHA-256 앞 16자와 바이트 크기. "
          "`python3 code/make_manifest.py` 로 재생성한다.\n\n")
FOOTER = "\n`refbias_overleaf_SafetyScience.zip` 은 Overleaf 업로드용 LaTeX 패키지다.\n"

def walk():
    out = []
    for dp, dn, fn in os.walk(ROOT):
        dn[:] = [d for d in dn if d not in SKIP_DIRS]
        for f in fn:
            p = os.path.relpath(os.path.join(dp, f), ROOT).replace('\\', '/')
            if p.startswith('.git/') or p == 'MANIFEST.md' or p.endswith('.pyc'):
                continue
            out.append(p)
    return sorted(out, key=lambda p: (p.count('/'), p))

def digest(p):
    b = open(os.path.join(ROOT, p), 'rb').read()
    return len(b), hashlib.sha256(b).hexdigest()[:16]

def build():
    body = '| path | bytes | sha256[:16] |\n|---|---:|---|\n'
    for p in walk():
        n, h = digest(p)
        body += f'| `{p}` | {n:,} | `{h}` |\n'
    return HEADER + body + FOOTER

def check():
    path = os.path.join(ROOT, 'MANIFEST.md')
    if not os.path.exists(path):
        print('MANIFEST.md 없음'); return 1
    rows = re.findall(r'\| `([^`]+)` \| ([\d,]+) \| `([0-9a-f]{16})` \|',
                      open(path, encoding='utf-8').read())
    listed = {p for p, _, _ in rows}
    actual = set(walk())
    bad = []
    for p, b, h in rows:
        if p not in actual:
            bad.append((p, '파일 없음')); continue
        n, d = digest(p)
        if n != int(b.replace(',', '')): bad.append((p, f'크기 {n} != {b}'))
        elif d != h: bad.append((p, f'해시 {d} != {h}'))
    for p in sorted(actual - listed):
        bad.append((p, 'MANIFEST 미등재'))
    print(f'MANIFEST {len(rows)}건 대조 · 불일치 {len(bad)}건')
    for p, why in bad[:20]:
        print(f'  {p} — {why}')
    return 1 if bad else 0

if __name__ == '__main__':
    if '--check' in sys.argv:
        sys.exit(check())
    open(os.path.join(ROOT, 'MANIFEST.md'), 'w', encoding='utf-8').write(build())
    print(f'MANIFEST.md 재생성 — {len(walk())}개 파일')
    sys.exit(check())
