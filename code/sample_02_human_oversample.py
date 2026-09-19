#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refbias — 인간 검증용 400건 표본 재설계 (리뷰어 C 요구 반영)
==============================================================
설계: 안보 블록 EVENT 전수(95) + 안보 비EVENT 150 + 산업 78 + 재래식 77 = 400
      (층 = 블록 x 코더A 라벨의 EVENT/비EVENT 이분)
각 층의 포함확률 pi = n_sampled / n_gold 를 시트와 별도 CSV에 기록한다.
→ analyze_12_human.py 가 1/pi 가중(IPW)으로 설계기반 추정을 수행한다.

기존 human_subset_400.csv 를 쓰고 싶으면 이 스크립트를 실행하지 않으면 된다
(analyze_12_human.py 는 실제 표집된 시트에서 포함확률을 역산하므로 둘 다 처리한다).

출력: Step4/human_labeling/{human_labels_H1.xlsx, human_labels_H2.xlsx,
      human_subset_400.csv, human_design.csv}   ※ 기존 파일은 .bak 로 백업
"""
import os, shutil
import numpy as np, pandas as pd
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HD=BASE+'/Step4/human_labeling'; os.makedirs(HD,exist_ok=True)
RNG=np.random.default_rng(20260908)
PLAN={('안보',True):95, ('안보',False):150, ('산업',None):78, ('비CBRN',None):77}

g=pd.read_csv(BASE+'/data/goldset_labeled_A.csv',dtype=str)
g['isE']=(g.label=='EVENT')
pick=[]; design=[]
for (blk,ev),n in PLAN.items():
    pool=g[g.block==blk] if ev is None else g[(g.block==blk)&(g.isE==ev)]
    n=min(n,len(pool)); idx=RNG.choice(pool.index.values,n,replace=False)
    pick.append(g.loc[idx])
    design.append(dict(block=blk,event_stratum=('EVENT' if ev else 'non-EVENT') if ev is not None else 'all',
                       N_gold=len(pool),n_sampled=n,pi=n/len(pool)))
sub=pd.concat(pick).sample(frac=1,random_state=1).reset_index(drop=True)
pd.DataFrame(design).to_csv(HD+'/human_design.csv',index=False,encoding='utf-8-sig')
sub[['sid']].to_csv(HD+'/human_subset_400.csv',index=False,encoding='utf-8-sig')

sheet=sub[['sid','query','일자','언론사','제목','본문']].copy()
sheet.columns=['sid','질의','일자','언론사','제목','본문(앞 200자)']
sheet['본문(앞 200자)']=sheet['본문(앞 200자)'].fillna('').str.replace(r'\s+',' ',regex=True).str[:200]
for c in ['label','ev_date','ev_place','ev_object','memo']: sheet[c]=''
for f in ['human_labels_H1.xlsx','human_labels_H2.xlsx']:
    p=HD+'/'+f
    if os.path.exists(p): shutil.copy(p,p+'.bak')
    sheet.to_excel(p,index=False)
print(pd.DataFrame(design).to_string(index=False))
print('n =',len(sub),'| 시트 2개와 human_design.csv 생성 (기존 파일은 .bak 백업)')
