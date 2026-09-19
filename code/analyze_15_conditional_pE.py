#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refbias — 관련성 조건부 사건 비율 (analyze_15_conditional_pE.py)
================================================================
심사 시뮬레이션에서 제기된 핵심 반론: 세 블록의 질의가 서로 다른 도구로 생성되어
검색 정밀도가 다르므로(IRRELEVANT 비율 9.2% / 25.6% / 14.5%), 블록 대비가
사건 발생률 차이가 아니라 검색 정밀도 차이의 산물일 수 있다.

이를 검정하기 위해 무관 기사를 제외한 조건부 사건 비율
    p_E^cond = EVENT / (V - IRRELEVANT)
를 모델 라벨과 인간 기준(사후층화) 양쪽에서 산출하고, 질의 수준 검정을 재실행한다.
출력: Step5r/conditional_pE.json
"""
import os, json, itertools
import numpy as np, pandas as pd
from scipy import stats as st
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT=BASE+'/Step5r'; RNG=np.random.default_rng(20260909); NB=10000
BL={'산업':'industrial','안보':'security','비CBRN':'conventional'}
CATS=['EVENT','REF','INST','OTHER','IRRELEVANT']
AL={'event':'EVENT','ref':'REF','inst':'INST','other':'OTHER','irrelevant':'IRRELEVANT',
    '사건':'EVENT','참조':'REF','제도':'INST','기타':'OTHER','무관':'IRRELEVANT'}
def nm(x):
    if pd.isna(x): return None
    s=str(x).strip(); return AL.get(s.lower(), s.upper() if s.upper() in CATS else None)
g=pd.read_csv(BASE+'/data/goldset_labeled_A.csv',dtype=str); g['isE']=(g.label=='EVENT')
res={'note':'p_E^cond = EVENT/(V-IRRELEVANT). 블록별 검색 정밀도 차이를 통제한 사건 비율.'}

# ---- 모델 라벨 ----
res['model']={}
for kb,b in BL.items():
    s=g[g.block==kb]; V=len(s); E=int((s.label=='EVENT').sum()); IR=int((s.label=='IRRELEVANT').sum())
    res['model'][b]=dict(V=V,EVENT=E,IRRELEVANT=IR,pE=E/V,precision_of_query=(V-IR)/V,pE_cond=E/(V-IR))
    print('%-13s V=%4d p_E=%.3f 무관=%.3f → 조건부 %.3f'%(b,V,E/V,IR/V,E/(V-IR)))

q=(g.groupby(['qid','block'])
   .apply(lambda s: pd.Series({'pE':(s.label=='EVENT').mean(),
                               'pE_cond':(s.label=='EVENT').sum()/max((s.label!='IRRELEVANT').sum(),1)}),
          include_groups=False).reset_index())
gr={BL[kb]:q[q.block==kb].pE_cond.values for kb in BL}
H=st.kruskal(*gr.values()); n=sum(len(v) for v in gr.values())
eta=(H.statistic-len(gr)+1)/(n-len(gr))
pw={}
for a,b2 in itertools.combinations(gr,2):
    u=st.mannwhitneyu(gr[a],gr[b2],alternative='two-sided')
    pw['%s_vs_%s'%(a,b2)]=dict(U=float(u.statistic),p=float(u.pvalue),
        r=float(abs(st.norm.ppf(u.pvalue/2))/np.sqrt(len(gr[a])+len(gr[b2]))))
items=sorted(pw.items(),key=lambda kv:kv[1]['p']); m=len(items); prev=0
for i,(k,v) in enumerate(items):
    hp=min(1.0,max(prev,(m-i)*v['p'])); pw[k]['p_holm']=hp; prev=hp
res['model_query_level']=dict(kruskal_H=float(H.statistic),p=float(H.pvalue),eta2=float(eta),
    medians={k:float(np.median(v)) for k,v in gr.items()},pairwise=pw)
print('\n질의수준 조건부 KW H=%.3f p=%.4f eta2=%.3f'%(H.statistic,H.pvalue,eta))
for k,v in pw.items(): print('  %-32s U=%.1f p=%.4f Holm=%.4f r=%.2f'%(k,v['U'],v['p'],v['p_holm'],v['r']))

# ---- 인간 기준 (사후층화) ----
Hm={t:pd.read_excel(f'{BASE}/Step4/human_labeling/human_labels_{t}.xlsx',dtype=str)[['sid','label']]
      .assign(label=lambda x:x.label.map(nm)) for t in ['H1','H2']}
d=(Hm['H1'].rename(columns={'label':'h1'}).merge(Hm['H2'].rename(columns={'label':'h2'}),on='sid')
   .merge(g[['sid','block','label']].rename(columns={'label':'A'}),on='sid'))
d['cell']=d.block+'|'+np.where(d.A=='EVENT','EVENT','non-EVENT')
N=g.assign(cell=g.block+'|'+np.where(g.isE,'EVENT','non-EVENT')).groupby('cell').size()
c=d[d.h1==d.h2].copy(); c['y']=c.h1
res['human']={}
print('\n인간 기준 (사후층화)')
for kb,b in BL.items():
    s=c[c.block==kb]
    num=sum(N[cl]*(gp.y=='EVENT').mean() for cl,gp in s.groupby('cell'))
    den=sum(N[cl]*(gp.y!='IRRELEVANT').mean() for cl,gp in s.groupby('cell'))
    tot=sum(N[cl] for cl in s.cell.unique())
    bn=np.zeros(NB); bd=np.zeros(NB)
    for cl,gp in s.groupby('cell'):
        y=(gp.y=='EVENT').values.astype(float); r=(gp.y!='IRRELEVANT').values.astype(float)
        k=len(y); idx=RNG.integers(0,k,size=(NB,k))
        bn+=N[cl]*y[idx].mean(1); bd+=N[cl]*r[idx].mean(1)
    o=bn/np.maximum(bd,1e-9)
    ci=[float(np.percentile(o,2.5)),float(np.percentile(o,97.5))]
    res['human'][b]=dict(pE=float(num/tot),relevant_share=float(den/tot),pE_cond=float(num/den),ci95=ci)
    print('  %-13s p_E=%.4f 관련성=%.3f → 조건부 %.4f [%.4f, %.4f]'%(b,num/tot,den/tot,num/den,ci[0],ci[1]))
sec=res['human']['security']['ci95']; ind=res['human']['industrial']['ci95']; con=res['human']['conventional']['ci95']
res['human']['security_ci_disjoint_from_controls']=bool(sec[1]<ind[0] and sec[1]<con[0])
print('\n안보 CI 상한 %.4f < 산업 하한 %.4f, 비CBRN 하한 %.4f → 겹침 없음: %s'%(
    sec[1],ind[0],con[0],res['human']['security_ci_disjoint_from_controls']))
json.dump(res,open(OUT+'/conditional_pE.json','w'),indent=1,ensure_ascii=False,default=float)
print('\nsaved Step5r/conditional_pE.json')
