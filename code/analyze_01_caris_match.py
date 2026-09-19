#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refbias — 산업 화학사고 질의군의 E(실사건 수) 절대값 대조
========================================================
리뷰어 예상지적 5번 대응.
보도량(빅카인즈 기사 수)을 공식 사건 집계(화학물질안전원 화학사고정보)와 1:1 대조한다.

대조 창: 2021-09-09 ~ 2025-04-28
  코퍼스 시작일과 CARIS 최종 수록일의 교집합. 두 자료원이 동시에 덮는 유일한 구간이다.

출력 data/caris_match.csv
"""
import pandas as pd, os, re, gzip, warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W_FROM, W_TO = '2021-09-09', '2025-04-28'

# 질의 → CARIS 대조 규칙
RULES = {
 'Q01': ('불산 누출',       'mat',  r'불산|불화수소|플루오르화수소'),
 'Q02': ('암모니아 누출 사고', 'mat',  r'암모니아'),
 'Q03': ('염소 가스 누출',    'mat',  r'^염소|차아염소'),
 'Q04': ('황산 유출 사고',    'mat',  r'황산'),
 'Q05': ('화학공장 폭발',     'type', r'폭발'),
 'Q06': ('유해화학물질 유출',  'type', r'누출'),
 'Q07': ('톨루엔 누출',      'mat',  r'톨루엔'),
 'Q08': ('질산 누출',       'mat',  r'질산'),
 'Q09': ('유독가스 누출 대피', 'type', r'누출'),
 'Q10': ('화학사고 대응',     'all',  r'.'),
}

def load_caris():
    p = os.path.join(BASE, 'data', 'caris_incidents_raw.csv')
    df = pd.read_csv(p, encoding='cp949')
    df['dt'] = pd.to_datetime(df['사고일자'], errors='coerce')
    df = df[(df['dt'] >= W_FROM) & (df['dt'] <= W_TO)].copy()
    df['mats'] = (df['제1사고물질'].fillna('') + '|' +
                  df['제2사고물질'].fillna('') + '|' + df['제3사고물질'].fillna(''))
    return df

def load_corpus_counts():
    """산업 블록 질의의 대조창 내 기사 수"""
    p = os.path.join(BASE, 'data', 'corpus_bigkinds.csv.gz')
    cnt, first = {}, True
    for ch in pd.read_csv(p, compression='gzip', dtype=str, chunksize=20000,
                          usecols=['qid', 'block', '일자']):
        ch = ch[ch['block'] == '산업']
        if ch.empty: continue
        d = pd.to_datetime(ch['일자'], format='%Y%m%d', errors='coerce')
        ch = ch[(d >= W_FROM) & (d <= W_TO)]
        for q, n in ch.groupby('qid').size().items():
            cnt[q] = cnt.get(q, 0) + int(n)
    return cnt

def main():
    caris = load_caris()
    news  = load_corpus_counts()
    rows = []
    for qid, (q, kind, pat) in RULES.items():
        if kind == 'mat':
            sub = caris[caris['mats'].str.contains(pat, regex=True, na=False)]
        elif kind == 'type':
            sub = caris[caris['사고유형'].astype(str).str.contains(pat, na=False)]
        else:
            sub = caris
        E = len(sub); N = news.get(qid, 0)
        rows.append(dict(qid=qid, query=q, 대조기준=f'{kind}:{pat}',
                         보도기사수=N, 공식사건수=E,
                         기사당사건비=round(N / E, 2) if E else None,
                         사망=int(sub['사망_직접'].fillna(0).sum()) if E else 0,
                         부상=int(sub['부상_직접'].fillna(0).sum()) if E else 0))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(BASE, 'data', 'caris_match.csv'),
               index=False, encoding='utf-8-sig')
    print(f'대조 창 {W_FROM} ~ {W_TO}  (CARIS {len(caris)}건)')
    print(out.to_string(index=False))
    print()
    tot_n = out['보도기사수'].sum()
    print(f'산업 블록 대조창 내 기사 {tot_n:,}건 / 공식 사건 {len(caris)}건')

if __name__ == '__main__':
    main()
