#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""refbias v8 최종 수치 확정 — 사후층화 인간 추정 + 증폭계수 + 코퍼스 RG 보정 (일관 가중)"""
import os,json
import numpy as np, pandas as pd
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT=BASE+'/Step5r'
RNG=np.random.default_rng(20260909); NB=10000
BL={'산업':'industrial','안보':'security','비CBRN':'conventional'}
ps=json.load(open(OUT+'/human_ps.json')); A_J=json.load(open(OUT+'/revision_A.json'))
g=pd.read_csv(BASE+'/data/goldset_labeled_A.csv',dtype=str)
out={}
# 블록별 m (Table 1 과 동일 정의: 블록 내 사건 병합)
import re,unicodedata
def norm(s): return re.sub(r'\s+','',unicodedata.normalize('NFKC',str(s or '')).strip().lower())
def cluster(ev,w=3):
    ev=ev.copy(); ev['d']=pd.to_datetime(ev['ev_date'],format='%Y%m%d',errors='coerce')
    ev['p']=ev['ev_place'].map(norm); ev['o']=ev['ev_object'].map(norm)
    reps=[]; ids={}; nxt=0
    for i,r in ev.sort_values('d').iterrows():
        hit=None
        for k,dd,pp,oo in reps:
            if pp==r['p'] and oo==r['o'] and pd.notna(dd) and pd.notna(r['d']) and abs((r['d']-dd).days)<=w: hit=k;break
        if hit is None: hit=nxt; nxt+=1; reps.append((hit,r['d'],r['p'],r['o']))
        ids[i]=hit
    return pd.Series(ids)
ev=g[g.label=='EVENT']
M={}
for kb,b in BL.items():
    s=ev[ev.block==kb]; c=cluster(s); M[b]=len(s)/c.nunique()
out['m_block']=M
out['blocks']={}
for kb,b in BL.items():
    v=ps['pE_human_ps'][b]; p=v['pE']; ci=v['ci95']; m=M[b]
    out['blocks'][b]=dict(n_drawn=v['n_drawn'],n_consensus=v['n_consensus'],pE_coderA=v['pE_coderA'],
        pE_human=p,ci95=ci,m=m,A=m/p,A_ci=[m/ci[1],m/ci[0]],
        TPR=ps['rg_identity'][b]['TPR'],FPR=ps['rg_identity'][b]['FPR'])
    print('%-13s p_E %.4f [%.4f,%.4f]  m %.5f  A %.2f [%.2f,%.2f]  TPR %.3f FPR %.3f'%(
        b,p,ci[0],ci[1],m,m/p,m/ci[1],m/ci[0],ps['rg_identity'][b]['TPR'],ps['rg_identity'][b]['FPR']))
# 코퍼스 RG 보정 (일관 가중 TPR/FPR, p_obs 는 사후층화 코퍼스 점유율)
CORP={b:A_J['poststrat_corpus_pE'][b] for b in BL.values()}
# TPR/FPR 도 층 내 복원 재표집으로 함께 전파한다 (본문 3.7 절 기술과 일치)
import pandas as _pd
_h=_pd.read_json(OUT+'/human_ps.json') if False else None
hp=json.load(open(OUT+'/human_ps.json'))
CATS=['EVENT','REF','INST','OTHER','IRRELEVANT']
AL={'event':'EVENT','ref':'REF','inst':'INST','other':'OTHER','irrelevant':'IRRELEVANT','사건':'EVENT','참조':'REF','제도':'INST','기타':'OTHER','무관':'IRRELEVANT'}
def _nm(x):
    if pd.isna(x): return None
    t=str(x).strip(); return AL.get(t.lower(), t.upper() if t.upper() in CATS else None)
_H={t:pd.read_excel(f'{BASE}/Step4/human_labeling/human_labels_{t}.xlsx',dtype=str)[['sid','label']].assign(label=lambda x:x.label.map(_nm)) for t in ['H1','H2']}
_d=(_H['H1'].rename(columns={'label':'h1'}).merge(_H['H2'].rename(columns={'label':'h2'}),on='sid')
    .merge(g[['sid','block','label']].rename(columns={'label':'A'}),on='sid'))
_d['cell']=_d.block+'|'+np.where(_d.A=='EVENT','EVENT','non-EVENT')
_N=g.assign(isE=(g.label=='EVENT')).assign(cell=lambda x:x.block+'|'+np.where(x.isE,'EVENT','non-EVENT')).groupby('cell').size()
_c=_d[_d.h1==_d.h2].copy(); _c['y']=_c.h1
def rate_boot(kb,nb=NB):
    sub=_c[_c.block==kb]
    tn=np.zeros(nb); td=np.zeros(nb); fn=np.zeros(nb); fd=np.zeros(nb)
    for cell,gp in sub.groupby('cell'):
        n=len(gp); Nc=_N[cell]
        hu=(gp.y=='EVENT').values; ae=(gp.A=='EVENT').values
        idx=RNG.integers(0,n,size=(nb,n))
        HU=hu[idx]; AE=ae[idx]
        tn+=Nc*(AE&HU).mean(1); td+=Nc*HU.mean(1)
        fn+=Nc*(AE&~HU).mean(1); fd+=Nc*(~HU).mean(1)
    return tn/np.maximum(td,1e-12), fn/np.maximum(fd,1e-12)
INV={v:k for k,v in BL.items()}
for b in BL.values():
    tpr=out['blocks'][b]['TPR']; fpr=out['blocks'][b]['FPR']
    tb,fb=rate_boot(INV[b])
    po=CORP[b]['pE']; lo,hi=CORP[b]['ci95']; sd=(hi-lo)/3.92
    pb=np.clip(RNG.normal(po,sd,NB),0,1)
    den=tb-fb; ok=den>0.05
    o=np.clip((pb[ok]-fb[ok])/den[ok],0,1)
    pt=float(np.clip((po-fpr)/(tpr-fpr),0,1))
    out['blocks'][b]['rate_ci']=dict(TPR=[float(np.percentile(tb,2.5)),float(np.percentile(tb,97.5))],
                                     FPR=[float(np.percentile(fb,2.5)),float(np.percentile(fb,97.5))],
                                     boot_valid_frac=float(ok.mean()))
    out['blocks'][b]['corpus']=dict(V=CORP[b]['V'],pE_obs=po,ci_obs=CORP[b]['ci95'],pE_rg=pt,
        ci_rg=[float(np.percentile(o,2.5)),float(np.percentile(o,97.5))])
    print('%-13s corpus %.3f → RG %.3f [%.3f,%.3f]'%(b,po,pt,*out['blocks'][b]['corpus']['ci_rg']))
out['domestic']=ps['domestic_security']; out['disagreements']=ps['disagreements']
out['consensus_sensitivity']=ps['consensus_sensitivity']; out['rg_identity_max_abs_diff']=max(abs(v['diff']) for v in ps['rg_identity'].values())
json.dump(out,open(OUT+'/v8_final_numbers.json','w'),indent=1,ensure_ascii=False,default=float)
print('\nRG-직접추정 최대 절대차 %.2e  → 동일 추정량'%out['rg_identity_max_abs_diff'])
print('saved Step5r/v8_final_numbers.json')
