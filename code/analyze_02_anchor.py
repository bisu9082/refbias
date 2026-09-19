#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refbias — 앵커 사건 재인용률 측정
=================================
참조 편향의 직접 증거. 특정 과거 사건(앵커)이 현재 보도에 반복 인용되는 비율을
동일 블록의 대조 질의와 비교한다.

앵커 후보는 질의별로 지정한다. 대조군 대비 재인용률의 차이가 참조 성분의 크기다.
출력 data/anchor_rates.csv
"""
import pandas as pd, os, warnings
warnings.filterwarnings('ignore')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ANCHORS = {
    'Q01': ('불산 누출',    ['구미', '2012']),          # 2012 구미 불산 누출
    'Q04': ('황산 유출',    ['구미', '2012']),          # 대조군
    'Q08': ('질산 누출',    ['구미', '2012']),          # 대조군
    'Q03': ('염소 가스 누출', ['구미', '2012']),          # 대조군
    'Q12': ('사린가스 공격', ['옴진리교', '도쿄', '1995', '구타']),
    'Q14': ('탄저균 테러',   ['2001', '미국', '우편']),
    'Q19': ('노비촉',      ['스크리팔', '솔즈베리', '나발니', '2018']),
    'Q16': ('방사능 유출',   ['후쿠시마', '체르노빌']),
}

def main():
    p = os.path.join(BASE, 'data', 'corpus_bigkinds.csv.gz')
    want = set(ANCHORS)
    buf = {k: [] for k in want}
    for ch in pd.read_csv(p, compression='gzip', dtype=str, chunksize=20000,
                          usecols=['qid', '일자', '제목', '본문', '키워드']):
        for k in want:
            s = ch[ch['qid'] == k]
            if len(s): buf[k].append(s)
    rows = []
    for qid, (name, terms) in ANCHORS.items():
        if not buf[qid]: continue
        d = pd.concat(buf[qid])
        t = d['제목'].fillna('') + ' ' + d['본문'].fillna('') + ' ' + d['키워드'].fillna('')
        r = dict(qid=qid, query=name, n=len(d), 앵커어=' / '.join(terms))
        hit = pd.Series(False, index=d.index)
        for w in terms:
            h = t.str.contains(w, na=False)
            r[f'비율_{w}'] = round(float(h.mean()), 4)
            hit |= h
        r['앵커언급률'] = round(float(hit.mean()), 4)
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(BASE, 'data', 'anchor_rates.csv'),
               index=False, encoding='utf-8-sig')
    print(out.to_string(index=False))

if __name__ == '__main__':
    main()
