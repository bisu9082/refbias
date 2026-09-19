#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refbias — 사건 0건 셀의 구간 처리
==================================
5차 패널 리뷰어 B [MAJOR] 대응.

Table 6의 블록별 인간 p_E 는 (블록 × 코더A EVENT/비EVENT) 6개 셀 위의
사후층화 추정치다. 대조 블록의 비EVENT 셀에서 인간 사건이 0건 관측되어,
그 셀의 재표집 분산이 정확히 0이 된다. 결과적으로 블록 share의 상한이
코더A 값에 구조적으로 고정되고, 구간이 실제 불확실성보다 좁아진다.

처리: 0건 셀의 셀 내부 비율에 Jeffreys 사전(Beta(1/2,1/2))의 95% 상한을
적용해 블록 share의 상한을 다시 계산한다. 동일한 이유로 상한이 없는
점추정으로 보고되던 TPR = 1.00 에도 Jeffreys 구간을 준다.

출력 outputs/Step5v10/zero_cell_ci.json
"""
import os, sys, json
import pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths as P

BLK = {'산업': 'industrial', '안보': 'security', '비CBRN': 'conventional'}

def jeffreys(k, n, a=0.05):
    """Beta(1/2,1/2) 사전 기반 신뢰구간. k=0 이면 하한 0, k=n 이면 상한 1."""
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k + 0.5, n - k + 0.5)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 0.5, n - k + 0.5)
    return float(lo), float(hi)

def main():
    des = pd.read_csv(os.path.join(P.ROOT, 'labels', 'human_sampling_design.csv'),
                      encoding='utf-8-sig')
    A = pd.read_csv(P.goldset_A(), encoding='utf-8-sig')
    H1 = pd.read_csv(os.path.join(P.ROOT, 'labels', 'human_labels_H1.csv'), encoding='utf-8-sig')
    H2 = pd.read_csv(os.path.join(P.ROOT, 'labels', 'human_labels_H2.csv'), encoding='utf-8-sig')

    h = H1[['sid', 'label']].merge(H2[['sid', 'label']], on='sid', suffixes=('_1', '_2'))
    h = h[h.label_1 == h.label_2].copy()                       # 두 코더 합의분
    h['human_event'] = (h.label_1 == 'EVENT').astype(int)
    m = h.merge(A[['sid', 'block', 'label']], on='sid').rename(columns={'label': 'A_label'})
    m['cell'] = m['block'] + '|' + m['A_label'].eq('EVENT').map({True: 'EVENT', False: 'non-EVENT'})
    N = dict(zip(des['cell'], des['N_gold']))

    out = {'cells': {}, 'blocks': {}, 'tpr': {}}
    for kr, en in BLK.items():
        cells, num_lo, num_hi, num_pt, denom = {}, 0.0, 0.0, 0.0, 0.0
        for suffix in ['EVENT', 'non-EVENT']:
            c = f'{kr}|{suffix}'
            sub = m[m.cell == c]
            k, n = int(sub.human_event.sum()), int(len(sub))
            lo, hi = jeffreys(k, n)
            cells[c] = {'k_human_event': k, 'n_consensus': n, 'N_gold': int(N[c]),
                        'p_hat': (k / n if n else None), 'jeffreys95': [lo, hi]}
            num_pt += N[c] * (k / n if n else 0); num_lo += N[c] * lo; num_hi += N[c] * hi
            denom += N[c]
        out['cells'].update(cells)
        out['blocks'][en] = {'pE_point': num_pt / denom,
                             'pE_jeffreys95': [num_lo / denom, num_hi / denom]}
        # TPR: 코더A EVENT 셀에서 인간 사건 / 전체 인간 사건 (가중)
        e, ne = m[m.cell == f'{kr}|EVENT'], m[m.cell == f'{kr}|non-EVENT']
        tp = N[f'{kr}|EVENT'] * (e.human_event.sum() / len(e))
        fn = N[f'{kr}|non-EVENT'] * (ne.human_event.sum() / len(ne)) if len(ne) else 0
        k_ne, n_ne = int(ne.human_event.sum()), int(len(ne))
        lo_ne, hi_ne = jeffreys(k_ne, n_ne)
        fn_hi = N[f'{kr}|non-EVENT'] * hi_ne
        out['tpr'][en] = {'point': tp / (tp + fn) if (tp + fn) else None,
                          'lower_from_jeffreys_fn_upper': tp / (tp + fn_hi) if (tp + fn_hi) else None,
                          'nonEVENT_cell_human_events': k_ne, 'nonEVENT_cell_n': n_ne}

    json.dump(out, open(P.v10('zero_cell_ci.json'), 'w'), ensure_ascii=False, indent=1)
    print(f"{'셀':22s}{'k':>4s}{'n':>5s}{'N_gold':>8s}{'p':>8s}{'Jeffreys 95%':>22s}")
    for c, v in out['cells'].items():
        j = v['jeffreys95']
        print(f"{c:22s}{v['k_human_event']:4d}{v['n_consensus']:5d}{v['N_gold']:8d}"
              f"{(v['p_hat'] or 0):8.3f}   [{j[0]:.3f}, {j[1]:.3f}]")
    print(f"\n{'블록':14s}{'p_E':>8s}{'Jeffreys 95%':>24s}")
    for b, v in out['blocks'].items():
        j = v['pE_jeffreys95']
        print(f"{b:14s}{100*v['pE_point']:8.1f}   [{100*j[0]:5.1f}, {100*j[1]:5.1f}]")
    print(f"\n{'블록':14s}{'TPR':>7s}{'하한(FN 상한 기준)':>22s}{'비EVENT 셀 사건':>18s}")
    for b, v in out['tpr'].items():
        print(f"{b:14s}{v['point']:7.2f}{v['lower_from_jeffreys_fn_upper']:22.2f}"
              f"{v['nonEVENT_cell_human_events']:10d}/{v['nonEVENT_cell_n']}")
    print('\n→', P.v10('zero_cell_ci.json'))

if __name__ == '__main__':
    main()
