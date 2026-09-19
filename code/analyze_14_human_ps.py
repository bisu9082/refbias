#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refbias — 인간 검증 재추정 (analyze_14_human_ps.py)   [리뷰어 C 4차 지적 반영]
==============================================================================
수정 사항
 F-1 설계기반 추정을 '알려진 층 크기(N_gold)에 대한 사후층화'로 바로잡는다.
     기존 코드는 1/pi 를 400건 기준으로 고정한 채 360건(합의)만 합산해
     층별 합의 유지율로 가중이 왜곡되었다.
 F-2 합의 규칙 민감도: H1 단독 / H2 단독 / 합의 / 어느 한쪽이라도 EVENT / 불일치 40건 전부 EVENT(상한).
 F-3 불일치 40건의 EVENT 경계 관여 건수와 질의별 구조.
 F-4 국내 사건 비율 인간 라벨 재계산.
 F-5 인간 라벨 기준 m (사건당 기사 수) — 사건 서술자 미기입이므로 코더A 군집을 인간 EVENT 에 투영.
 F-6 Rogan-Gladen 이 동일 가중 하에서 사후층화 직접추정과 대수적으로 같음을 수치로 확인.
출력: Step5r/human_ps.json
"""
import os, json
import numpy as np, pandas as pd
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HD=BASE+'/Step4/human_labeling'; OUT=BASE+'/Step5r'
RNG=np.random.default_rng(20260909); NB=10000
CATS=['EVENT','REF','INST','OTHER','IRRELEVANT']; BL={'산업':'industrial','안보':'security','비CBRN':'conventional'}
ALIAS={'event':'EVENT','ref':'REF','inst':'INST','other':'OTHER','irrelevant':'IRRELEVANT',
       '사건':'EVENT','참조':'REF','제도':'INST','기타':'OTHER','무관':'IRRELEVANT'}
def norm(x):
    if pd.isna(x): return None
    s=str(x).strip(); return ALIAS.get(s.lower(), s.upper() if s.upper() in CATS else None)

g=pd.read_csv(BASE+'/data/goldset_labeled_A.csv',dtype=str); g['isE']=(g.label=='EVENT')
H={t:pd.read_excel(f'{HD}/human_labels_{t}.xlsx',dtype=str)[['sid','label']].assign(label=lambda x:x.label.map(norm))
   for t in ['H1','H2']}
d=(H['H1'].rename(columns={'label':'h1'}).merge(H['H2'].rename(columns={'label':'h2'}),on='sid')
   .merge(g[['sid','block','qid','query','label','ev_place']].rename(columns={'label':'A'}),on='sid'))
d['cell']=d.block+'|'+np.where(d.A=='EVENT','EVENT','non-EVENT')
N=g.assign(cell=g.block+'|'+np.where(g.isE,'EVENT','non-EVENT')).groupby('cell').size()
res={'note':'사후층화: p = sum_c N_c * mean(y|c) / sum_c N_c, N_c 는 골드셋의 알려진 층 크기'}

def ps(sub,ycol,evfun=lambda x:x=='EVENT'):
    cells=[(N[c],evfun(gp[ycol]).mean()) for c,gp in sub.groupby('cell')]
    return sum(n*p for n,p in cells)/sum(n for n,_ in cells)
def ps_ci(sub,ycol,evfun=lambda x:x=='EVENT',nb=NB):
    parts=[(N[c],evfun(gp[ycol]).values.astype(float)) for c,gp in sub.groupby('cell')]
    num=np.zeros(nb); den=sum(n for n,_ in parts)
    for n,y in parts:
        k=len(y); idx=RNG.integers(0,k,size=(nb,k)); num+=n*y[idx].mean(1)
    o=num/den; return [float(np.percentile(o,2.5)),float(np.percentile(o,97.5))]

# ---------- F-2/F-1 consensus-rule sensitivity ----------
cons=d[d.h1==d.h2].copy(); cons['y']=cons.h1
rules={}
rules['consensus']=cons
t=d.copy(); t['y']=t.h1; rules['H1_only']=t
t2=d.copy(); t2['y']=t2.h2; rules['H2_only']=t2
t3=d.copy(); t3['y']=np.where((t3.h1=='EVENT')|(t3.h2=='EVENT'),'EVENT','X'); rules['either_EVENT']=t3
t4=d.copy(); t4['y']=np.where((t4.h1!=t4.h2),'EVENT',t4.h1); rules['disagree_all_EVENT']=t4
t5=d.copy(); t5['y']=np.where((t5.h1!=t5.h2),'X',t5.h1); rules['disagree_all_nonEVENT']=t5
res['pE_human_ps']={}; res['consensus_sensitivity']={}
print('%-22s %-9s %-9s %-9s'%('rule','industrial','security','conventional'))
for r,tab in rules.items():
    row={}
    for kb,b in BL.items():
        sub=tab[tab.block==kb]; row[b]=ps(sub,'y')
    res['consensus_sensitivity'][r]=row
    print('%-22s %.4f    %.4f    %.4f'%(r,row['industrial'],row['security'],row['conventional']))
for kb,b in BL.items():
    sub=cons[cons.block==kb]
    p=ps(sub,'y'); ci=ps_ci(sub,'y')
    pa=float((g[g.block==kb].label=='EVENT').mean())
    res['pE_human_ps'][b]=dict(n_consensus=int(len(sub)),n_drawn=int((d.block==kb).sum()),
                               pE=p,ci95=ci,pE_coderA=pa)
    print('F-1 %-13s p_E(PS)=%.4f %s  코더A %.4f'%(b,p,[round(x,4) for x in ci],pa))

# ---------- F-6 RG identity check ----------
res['rg_identity']={}
for kb,b in BL.items():
    sub=cons[cons.block==kb]
    hu=(sub.y=='EVENT'); ae=(sub.A=='EVENT')
    # 같은 사후층화 가중으로 계산한 TPR/FPR 및 코더A 점유율
    def psv(mask,cond=None):
        cells=[]
        for c,gp in sub.groupby('cell'):
            m=mask.loc[gp.index]
            if cond is not None:
                cc=cond.loc[gp.index]
                if cc.sum()==0: cells.append((N[c],0.0,0.0)); continue
                cells.append((N[c],float((m&cc).sum())/len(gp),float(cc.sum())/len(gp)))
            else: cells.append((N[c],float(m.sum())/len(gp),1.0))
        num=sum(n*a for n,a,_ in cells); den=sum(n*bq for n,_,bq in cells)
        return num/den if den>0 else np.nan
    tpr=psv(ae,hu); fpr=psv(ae,~hu); pA=psv(ae)
    rgv=(pA-fpr)/(tpr-fpr)
    res['rg_identity'][b]=dict(TPR=float(tpr),FPR=float(fpr),pA_weighted=float(pA),
                               rg=float(rgv),direct_ps=float(ps(sub,'y')),diff=float(rgv-ps(sub,'y')))
    print('F-6 %-13s TPR=%.4f FPR=%.4f  RG=%.6f  direct=%.6f  diff=%.2e'%(b,tpr,fpr,rgv,ps(sub,'y'),rgv-ps(sub,'y')))

# ---------- F-3 disagreement structure ----------
dis=d[d.h1!=d.h2]
evb=dis[(dis.h1=='EVENT')|(dis.h2=='EVENT')]
res['disagreements']=dict(n=int(len(dis)),n_event_boundary=int(len(evb)),
    by_query_event_boundary={k:int(v) for k,v in evb['query'].value_counts().items()},
    pairs={f'{r.h1}/{r.h2}':int(v) for (r,v) in zip(dis.assign(k=dis.h1+'/'+dis.h2).itertuples(),[0]*len(dis))} if False else
          {k:int(v) for k,v in (dis.h1+'/'+dis.h2).value_counts().items()})
print('\nF-3 불일치 %d건, EVENT 경계 관여 %d건, 질의별 %s'%(len(dis),len(evb),res['disagreements']['by_query_event_boundary']))

# ---------- F-4 domestic (사후층화, 코더A 국내 판정 규칙 재사용) ----------
import importlib.util as _ilu
_sp=_ilu.spec_from_file_location('_dom',BASE+'/code/_domestic_rule.py')
_dm=_ilu.module_from_spec(_sp); _sp.loader.exec_module(_dm)
domf=_dm.dom; FOR_RE=_dm.FOR_RE
gg=g.copy(); gg['t']=gg['제목'].fillna('')+' '+gg['본문'].fillna('')
ge=gg[gg.label=='EVENT'].copy(); ge['dm']=ge['ev_place'].map(domf)
ge.loc[ge.dm=='unknown','dm']=np.where(ge.loc[ge.dm=='unknown','t'].str.contains(FOR_RE),'foreign_by_text','unknown')
domsids=set(ge[(ge.block=='안보')&(ge.dm=='domestic')].sid)
# 인간 EVENT 중 코더A 가 놓친 국내 건은 제목의 국내 지명으로 판정
KRN=r'(서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|충청|전북|전남|전라|경북|경남|경상|제주|전국|국내|한국|수도권)'
KRT=__import__('re').compile(KRN)
sec=d[d.block=='안보'].merge(gg[['sid','제목']],on='sid',how='left')
sec['hE']=(sec.h1=='EVENT')&(sec.h2=='EVENT')
sec['dom']=np.where(sec.sid.isin(domsids),True,
             np.where(sec.hE&(sec.A!='EVENT')&sec['제목'].fillna('').str.contains(KRT),True,False))
rows=[]
for c,gp in sec.groupby('cell'):
    rows.append((N[c],float((gp.hE&gp.dom).sum())/len(gp),float(((gp.A=='EVENT')&gp.dom).sum())/len(gp),len(gp)))
p_dom_h=sum(n*r for n,r,_,_ in rows)/900; p_dom_a=sum(n*r for n,_,r,_ in rows)/900
nb=NB; num=np.zeros(nb)
for c,gp in sec.groupby('cell'):
    y=(gp.hE&gp.dom).values.astype(float); k=len(y)
    num+=N[c]*y[RNG.integers(0,k,size=(nb,k))].mean(1)
o=num/900; ci=[float(np.percentile(o,2.5)),float(np.percentile(o,97.5))]
census=sec[(sec.cell=='안보|EVENT')]
res['domestic_security']=dict(
    coderA_domestic_EVENT_in_census=int(((census.A=='EVENT')&census.dom).sum()),
    human_confirmed_in_census=int((census.hE&census.dom).sum()),
    pE_dom_coderA=float(p_dom_a),pE_dom_human_ps=float(p_dom_h),ci95=ci,
    note='안보|EVENT 층은 포함확률 1(전수). 코더A 가 놓친 국내 인간 EVENT 는 비EVENT 층(pi=0.186)에서 가중된다.')
print('F-4 국내: 코더A %.4f → 인간 사후층화 %.4f %s (전수층 %d건 중 인간 확인 %d건)'%(
    p_dom_a,p_dom_h,[round(x,4) for x in ci],
    res['domestic_security']['coderA_domestic_EVENT_in_census'],res['domestic_security']['human_confirmed_in_census']))

# ---------- F-5 human m ----------
mh={}
for kb,b in BL.items():
    hs=cons[(cons.block==kb)&(cons.y=='EVENT')]
    ge=g[(g.block==kb)&(g.label=='EVENT')]
    mh[b]=dict(n_human_EVENT=int(len(hs)))
res['human_m_note']='인간 코더가 ev_date/ev_place/ev_object 를 기입하지 않아 인간 라벨 기준 m 은 산출 불가. A(human) 은 코더A 의 m 을 사용한 혼합 추정치이며 하한이다.'
res['human_EVENT_counts']=mh
json.dump(res,open(OUT+'/human_ps.json','w'),indent=1,ensure_ascii=False,default=float)
print('\nsaved Step5r/human_ps.json')
