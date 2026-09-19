#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refbias — CARIS 물질 매칭의 재현율(recall) 감사
================================================
5차 패널 리뷰어 A [MAJOR] 대응.

analyze_21_registry_recode.py 는 오탐(false positive)만 제거했다. CARIS의
물질 필드는 통제어휘가 아닌 자유기술이므로, substring 규칙이 이명·염 형태·
혼합물 기재로 진짜 사례를 놓칠 수 있다. 이 스크립트는 그 미탐을 측정한다.

절차
  1) 엄격 재코딩 집합(registry_recode.json의 strict)을 출발점으로 삼는다.
  2) 확장 사전으로 물질 필드를 재매칭해 추가 후보를 뽑는다.
  3) 물질 필드에는 없으나 '사고내용'/'사고원인' 본문에 물질명이 나오는 건을
     전수 열거한다 — 자유기술 필드 누락을 잡는 직접 검사.
  4) 2)·3)의 후보를 한 건씩 판정한다. 판정 근거를 JSON에 남긴다.
  5) 판정 반영 후 회귀를 재적합해 null의 안정성을 확인한다.

출력 outputs/Step5v10/registry_recall.json
"""
import os, sys, json
import pandas as pd
from scipy import stats
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths as P

W_FROM, W_TO = '2021-09-09', '2025-04-28'
SUBS = ['불산', '암모니아', '염소', '황산', '톨루엔', '질산']
EXPAND = {
 '불산':   r'불산|불화수소|플루오르화수소|무수불산|폐불산|불화수소산',
 '암모니아': r'암모니아|NH3|무수암모니아',
 '염소':   r'염소|Cl2|염소가스',
 '황산':   r'황산|발연황산|폐황산|H2SO4',
 '톨루엔':  r'톨루엔|toluene|메틸벤젠',
 '질산':   r'질산|폐질산|HNO3',
}
# 확장 규칙이 끌어오는 비대상 화합물
EXCLUDE = {
 '불산':   r'붕불산|불소\(',
 '암모니아': r'암모늄',
 '염소':   r'차아염소|아염소|염화|염소산',
 '황산':   r'황산알루미늄|황산칼슘|황산구리|황산나트륨|황산마그네슘',
 '톨루엔':  r'디이소시아네이|비닐톨루엔',
 '질산':   r'질산암모늄|질산칼륨|질산나트륨|질산염|질산은',
}
# 4) 후보 전건 판정. key = (물질, 사고일자)
ADJUDICATION = {
 ('불산', '2024-08-16'): (True,
   '물질 필드는 이플루오르화암모늄(염 형태)이나 사고내용이 "HF 배관 교체 작업 중 '
   '배관 내 잔존된 물질"로 누출물을 HF로 명시. 진짜 미탐으로 판정하여 계수에 가산.'),
 ('암모니아', '2025-03-10'): (False,
   '누출물은 질산(60%) 40L. 암모니아수는 운반 목적으로만 언급되어 누출과 무관.'),
 ('황산', '2022-12-14'): (False,
   '누출물은 수산화나트륨. 황산은 폭발로 파손된 설비로만 언급되어 누출물이 아님.'),
 ('톨루엔', '2023-01-26'): (False,
   'RAW C9+C의 구성은 솔벤트 나프타·인덴·비닐톨루엔·스티렌. 비닐톨루엔은 '
   '메틸스티렌으로 톨루엔과 별개 화합물.'),
 ('질산', '2025-02-27'): (False,
   '질산암모늄(염) 포대 낙하. 엄격 재코딩이 질산염을 제외한 기준과 일치.'),
}

def load():
    df = pd.read_csv(os.path.join(P.ROOT, 'data', 'caris_incidents_raw.csv'), encoding='cp949')
    df['dt'] = pd.to_datetime(df['사고일자'], errors='coerce')
    df = df[(df['dt'] >= W_FROM) & (df['dt'] <= W_TO)].copy()
    df['mats'] = (df['제1사고물질'].fillna('') + '|' +
                  df['제2사고물질'].fillna('') + '|' + df['제3사고물질'].fillna(''))
    df['desc'] = df['사고내용'].fillna('') + ' ' + df['사고원인'].fillna('')
    return df.reset_index(drop=True)

def fit(counts, articles):
    x = np.log10([counts[s] for s in SUBS]); y = np.log10([articles[s] for s in SUBS])
    r = stats.linregress(x, y)
    t1 = (r.slope - 1) / r.stderr
    return {'slope': r.slope, 'p': r.pvalue, 'r': r.rvalue, 'stderr': r.stderr,
            't_slope_eq_1': t1, 'p_slope_eq_1': 2 * stats.t.sf(abs(t1), len(SUBS) - 2),
            'spearman_rho': stats.spearmanr(
                [counts[s] for s in SUBS], [articles[s] for s in SUBS]).statistic}

def main():
    df = load()
    rec = json.load(open(P.v10('registry_recode.json')))
    strict = dict(rec['strict']['per_substance'])
    cm = pd.read_csv(os.path.join(P.ROOT, 'data', 'caris_match.csv'), encoding='utf-8-sig')
    qmap = {'Q01': '불산', 'Q02': '암모니아', 'Q03': '염소',
            'Q04': '황산', 'Q07': '톨루엔', 'Q08': '질산'}
    articles = {qmap[r['qid']]: int(r['보도기사수'])
                for _, r in cm.iterrows() if r['qid'] in qmap}

    cand, corrected = {}, dict(strict)
    for s in SUBS:
        m_ref = (df['mats'].str.contains(EXPAND[s], regex=True, na=False)
                 & ~df['mats'].str.contains(EXCLUDE[s], regex=True, na=False))
        dsc_pat = EXPAND[s] + (r'|플루오르화암모늄' if s == '불산' else '')
        m_dsc = (df['desc'].str.contains(dsc_pat, regex=True, na=False)
                 & ~df['desc'].str.contains(EXCLUDE[s], regex=True, na=False))
        rows = []
        m_dsc = m_dsc | (df['mats'].str.contains(r'플루오르화암모늄', na=False) if s == '불산' else False)
        for _, r in df[m_dsc & ~m_ref].iterrows():
            key = (s, str(r['dt'].date()))
            acc, why = ADJUDICATION.get(key, (None, '[판정 필요]'))
            rows.append({'date': str(r['dt'].date()), 'mats': r['mats'].strip('|'),
                         'desc': str(r['사고내용'])[:160], 'accepted': acc, 'reason': why})
            if acc: corrected[s] = corrected[s] + 1
        cand[s] = rows
    fits = {'strict': fit(strict, articles), 'recall_corrected': fit(corrected, articles)}
    out = {'window': [W_FROM, W_TO], 'n_incidents': int(len(df)),
           'article_counts': articles, 'strict_counts': strict,
           'recall_corrected_counts': corrected,
           'candidates_from_description': cand, 'regression': fits,
           'recall_of_strict_rule': {s: round(strict[s] / corrected[s], 4) for s in SUBS}}
    json.dump(out, open(P.v10('registry_recall.json'), 'w'), ensure_ascii=False, indent=1)

    print(f'대조창 {W_FROM}~{W_TO}, 사고 {len(df)}건\n')
    print(f"{'물질':8s}{'엄격':>6s}{'재현율보정':>10s}{'recall':>9s}")
    for s in SUBS:
        print(f'{s:8s}{strict[s]:6d}{corrected[s]:10d}{strict[s]/corrected[s]:9.3f}')
    print(f'\n본문 교차검색 후보 {sum(len(v) for v in cand.values())}건 중 '
          f'채택 {sum(1 for v in cand.values() for r in v if r["accepted"])}건')
    for k, f in fits.items():
        print(f"\n[{k}] slope {f['slope']:+.4f}  p {f['p']:.3f}  r {f['r']:+.3f}  "
              f"| H0 slope=1: t({len(SUBS)-2}) = {f['t_slope_eq_1']:.2f}, p = {f['p_slope_eq_1']:.4f}")
    print('\n→', P.v10('registry_recall.json'))
    return out

if __name__ == '__main__':
    main()
