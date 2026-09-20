#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refbias — 원자료 → 원고 직결 검증
==================================
integrity_audit.py 는 '원자료 → 산출물 JSON → 본문' 사슬을 본다. 이 스크립트는
산출물 JSON 을 **전혀 참조하지 않고** labels/ 와 data/ 만으로 원고의 핵심 수치를
처음부터 다시 계산해 paper/main.tex 의 서술과 직접 대조한다. 산출물이 통째로
잘못되었더라도 여기서는 걸린다.

의존: pandas, numpy, scipy, scikit-learn
"""
import os, re, sys
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX  = open(os.path.join(ROOT, 'paper', 'main.tex'), encoding='utf-8').read()
_ok = _bad = 0
_fails = []

def chk(name, exp, got, tol=0.0):
    global _ok, _bad
    if got is None:
        good = False
    elif isinstance(exp, (int, float, np.floating)):
        good = abs(float(exp) - float(got)) <= tol
    else:
        good = str(exp) == str(got)
    print(f"{'OK  ' if good else 'FAIL'} {name:<34} 재계산 {exp!s:>9}   본문 {got!s:>9}")
    _ok += good; _bad += (not good)
    if not good: _fails.append(name)

def T(pat):
    m = re.search(pat, TEX); return m.group(1) if m else None
def F(pat):
    v = T(pat); return float(v.replace(',', '')) if v else None
def I(pat):
    v = T(pat); return int(v.replace(',', '')) if v else None
def L(name, enc='utf-8-sig'):
    return pd.read_csv(os.path.join(ROOT, name), encoding=enc)

A  = L('labels/goldset_labels_coderA.csv')
H1 = L('labels/human_labels_H1.csv')
H2 = L('labels/human_labels_H2.csv')
DS = L('labels/human_sampling_design.csv')
CA = L('data/caris_incidents_raw.csv', 'cp949')
CM = L('data/caris_match.csv')

print("=" * 72); print("1. 골드셋 — 코더A 라벨에서 직접"); print("=" * 72)
chk("골드셋 n", len(A), I(r'gold set of ([\d,]+) articles'))
chk("산업 p_E", round(100 * (A[A.block == '산업'].label == 'EVENT').mean(), 1),
    F(r'In the industrial block ([\d.]+)'), 0.05)
chk("안보 p_E", round(100 * (A[A.block == '안보'].label == 'EVENT').mean(), 1),
    F(r'In the CBRN security block ([\d.]+)'), 0.05)
chk("재래 p_E", round(100 * (A[A.block == '비CBRN'].label == 'EVENT').mean(), 1),
    F(r'In the conventional-terrorism block ([\d.]+)'), 0.05)
chk("전체 p_E", round(100 * (A.label == 'EVENT').mean(), 1),
    F(r'Across the gold set, ([\d.]+)'), 0.05)

print("\n" + "=" * 72); print("2. 인간 검증 — 사후층화 독립 구현"); print("=" * 72)
h = H1[['sid', 'label']].merge(H2[['sid', 'label']], on='sid', suffixes=('_1', '_2'))
chk("인간-인간 kappa (5분류)", round(cohen_kappa_score(h.label_1, h.label_2), 2),
    F(r'\$\\kappa = (0\.85)\$'), 0.005)
cons = h[h.label_1 == h.label_2].merge(
    A[['sid', 'block', 'label']].rename(columns={'label': 'A'}), on='sid')
cons['cell'] = cons.block + '|' + np.where(cons.A == 'EVENT', 'EVENT', 'non-EVENT')
N = dict(zip(DS.cell, DS.N_gold))
def ps(b):
    num = den = 0.0
    for s in ('EVENT', 'non-EVENT'):
        g = cons[cons.cell == f'{b}|{s}']
        if len(g) == 0: continue
        num += N[f'{b}|{s}'] * (g.label_1 == 'EVENT').mean(); den += N[f'{b}|{s}']
    return 100 * num / den
chk("산업 인간 p_E", round(ps('산업'), 1), F(r'([\d.]+)\\% \(22\.9--39\.9\) in the industrial block'), 0.05)
chk("안보 인간 p_E", round(ps('안보'), 1), F(r'([\d.]+)\\% \(0\.6--10\.9\) in the security block'), 0.05)
chk("재래 인간 p_E", round(ps('비CBRN'), 1), F(r'([\d.]+)\\% \(95\\% CI 30\.2--40\.9\)'), 0.05)
chk("안보 합의 n", int((cons.block == '안보').sum()), I(r'built on the ([\d,]+) articles of this block'))

print("\n" + "=" * 72); print("3. 등록부 — 원 CSV에서"); print("=" * 72)
CA['dt'] = pd.to_datetime(CA['사고일자'], errors='coerce')
w = CA[(CA.dt >= '2021-09-09') & (CA.dt <= '2025-04-28')].copy()
w['m'] = (w['제1사고물질'].fillna('') + '|' + w['제2사고물질'].fillna('') + '|' + w['제3사고물질'].fillna(''))
R = {'불산': r'불산|불화수소|플루오르화수소', '암모니아': r'암모니아', '염소': r'^염소|차아염소',
     '황산': r'황산', '톨루엔': r'톨루엔', '질산': r'질산'}
cnt = {k: int(w.m.str.contains(v, regex=True, na=False).sum()) for k, v in R.items()}
art = {r['qid']: int(r['보도기사수']) for _, r in CM.iterrows()}
qm = {'Q01': '불산', 'Q02': '암모니아', 'Q03': '염소', 'Q04': '황산', 'Q07': '톨루엔', 'Q08': '질산'}
chk("대조창 사고수", len(w), I(r'the registry records (\d+) chemical accidents'))
chk("6물질 별개사고", int(w.m.str.contains('|'.join(R.values()), regex=True, na=False).sum()),
    I(r'of which (\d+) name at least one'))
chk("물질별 할당합", sum(cnt.values()), I(r'counting nine mixed-acid incidents twice, the figure is (\d+)'))
A6 = [art[q] for q in qm]; I6 = [cnt[qm[q]] for q in qm]
r = stats.linregress(np.log10(I6), np.log10(A6))
chk("log--log 기울기", round(-r.slope, 2), F(r'the log--log slope of articles on incidents is \$-([\d.]+)\$'), 0.005)
chk("Spearman rho", round(stats.spearmanr(I6, A6).statistic, 2), F(r'Spearman \$\\rho = ([\d.]+)\$, \$n = 6\$'), 0.005)
chk("기울기=1 검정 t(4)", round(abs((r.slope - 1) / r.stderr), 1), F(r'is rejected \(\$t\(4\) = ([\d.]+)\$'), 0.05)
for q, k, nm in [('Q01', '불산', 'hydrogen fluoride'), ('Q03', '염소', 'chlorine')]:
    chk(f"{k} 기사/사고", round(art[q] / cnt[k], 1), F(rf'([\d.]+) \({nm}, {art[q]} for {cnt[k]}\)'), 0.05)

print("\n" + "=" * 72)
print(f"원자료→본문 직결 검증 {_ok + _bad}건 · 일치 {_ok} · 불일치 {_bad}")
if _fails: print("불일치:", ", ".join(_fails))
print("=" * 72)
sys.exit(1 if _bad else 0)
