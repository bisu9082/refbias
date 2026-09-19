#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refbias — Step 8 2차 코멘트 대응 재분석 (analyze_11_round2.py)
D1 A의 CI를 층화 부트스트랩 p_E CI의 단조변환으로 통일
D2 Cramér's V: 3×5 표와 EVENT-vs-rest 3×2 표
D3 국내/국외 분류 집계 세부 (재분류 건수)
D4 Mann–Whitney U를 첫 번째 집단 기준으로 통일
D5 앵커 ρ: 제목+발췌 기준 vs 키워드 포함 기준; 골드셋에 키워드 필드 결합해 교차표 재계산; 키워드 단독 히트의 EVENT 비율
D6 사후층화 CI 민감도: 블록 내 질의 단위 클러스터 부트스트랩
D7 Leave-one-query-out 분류기 평가 (미학습 질의 일반화)
D8 질의 어휘(발생 명사 포함 여부)와 p_E; Q30을 안보 블록으로 재배치한 검정
D9 N_events 추정치 vs 등록부 사고수 (6개 물질)
D10 표본크기별 p_E CI 폭 (Wilson) — DRR 지침용
D11 블록별 TPR/FPR (Rogan–Gladen 입력) 재보고
"""
import os, re, json, itertools
import numpy as np, pandas as pd
from scipy import stats as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT=BASE+'/Step5r'
RNG=np.random.default_rng(20260908); res={}
BL={'산업':'industrial','안보':'security','비CBRN':'conventional'}
A=json.load(open(OUT+'/revision_A.json')); B1=json.load(open(OUT+'/revision_B1.json')); B2=json.load(open(OUT+'/revision_B2.json'))
g=pd.read_csv(BASE+'/data/goldset_labeled_A.csv',dtype=str); g['isE']=(g.label=='EVENT').astype(int)
g['text']=g['제목'].fillna('')+' ▁ '+g['본문'].fillna('').str.replace(r'\s+',' ',regex=True).str[:200]
Q=pd.read_csv(BASE+'/data/queries_v2.csv',dtype=str)

# D1
res['A_ci_from_pE']={}
for b,name in list(BL.items())+[('all','all')]:
    m=B2['cluster_sensitivity'][0][name]['m']; p=A['pE_block'][name]; lo,hi=p['ci95_stratified']
    res['A_ci_from_pE'][name]=dict(m=m,pE=p['pE'],A=m/p['pE'],ci95=[m/hi,m/lo])
    print('D1 %-13s A=%.2f [%.2f, %.2f]'%(name,m/p['pE'],m/hi,m/lo))

# D2
ct=pd.crosstab(g.block,g.label); chi=st.chi2_contingency(ct)[0]; n=len(g); V5=np.sqrt(chi/(n*(min(ct.shape)-1)))
ct2=pd.crosstab(g.block,g.isE); chi2=st.chi2_contingency(ct2)[0]; V2=np.sqrt(chi2/n)
res['cramers_V']=dict(five_component=V5,event_vs_rest=V2); print('D2 V5=%.3f V2=%.3f'%(V5,V2))

# D3
e=pd.read_csv(OUT+'/event_domestic_coding.csv',dtype=str)
res['domestic_detail']=dict(no_place=int(e.ev_place.isna().sum()),by_dom=e.dom.value_counts().to_dict())
print('D3',res['domestic_detail'])

# D4 (U for first-named group)
qpe=g.groupby(['qid','block']).isE.mean().reset_index()
def U_first(a,b):
    u=st.mannwhitneyu(a,b,alternative='two-sided'); return u.statistic,u.pvalue
res['U_first']={}
for a,b in [('안보','산업'),('안보','비CBRN'),('산업','비CBRN')]:
    ua,pa=U_first(qpe[qpe.block==a].isE,qpe[qpe.block==b].isE); res['U_first'][f'{BL[a]} vs {BL[b]}']=dict(U=ua,p=pa)
print('D4',res['U_first'])

# D5 anchor: title+excerpt vs +keywords; gold cross-tab with keywords
ANCH={'Q01':{'Gumi':['구미']},'Q04':{'Gumi':['구미']},'Q08':{'Gumi':['구미']},'Q03':{'Gumi':['구미']},
 'Q12':{'Aum':['옴진리교','오움진리교'],'Tokyo':['도쿄','동경'],'Ghouta':['구타','고우타']},
 'Q14':{'US':['미국'],'mail':['우편','편지']},
 'Q19':{'Skripal':['스크리팔','스크립팔'],'Salisbury':['솔즈베리','솔즈버리'],'Navalny':['나발니','나발리']},
 'Q16':{'Fukushima':['후쿠시마'],'Chernobyl':['체르노빌']}}
c=pd.read_csv(BASE+'/data/corpus_bigkinds.csv.gz',dtype=str,usecols=['qid','뉴스 식별자','제목','본문','키워드'])
c=c[c.qid.isin(ANCH)].copy()
def hit(t,groups):
    pat='|'.join(re.escape(v) for vs in groups.values() for v in vs); return t.str.contains(pat,na=False)
res['anchor_bases']={}
for q,groups in ANCH.items():
    d=c[c.qid==q]; te=d['제목'].fillna('')+' '+d['본문'].fillna(''); kw=d['키워드'].fillna('')
    h_te=hit(te,groups); h_all=h_te|hit(kw,groups)
    gg=g[g.qid==q].merge(d[['뉴스 식별자','키워드']].drop_duplicates('뉴스 식별자'),on='뉴스 식별자',how='left')
    gte=hit(gg['제목'].fillna('')+' '+gg['본문'].fillna(''),groups); gkw=hit(gg['키워드'].fillna(''),groups); gall=gte|gkw; gkwonly=gkw&~gte
    def share(mask,lab):
        return float((gg.label[mask]==lab).mean()) if mask.sum() else None
    res['anchor_bases'][q]=dict(V=len(d),rho_title_excerpt=float(h_te.mean()),rho_with_keywords=float(h_all.mean()),
        gold_n=len(gg),gold_mention_te=float(gte.mean()),gold_mention_all=float(gall.mean()),gold_mention_te_n=int(gte.sum()),gold_mention_all_n=int(gall.sum()),
        EVENT_share_te=share(gte,'EVENT'),EVENT_share_all=share(gall,'EVENT'),REF_share_all=share(gall,'REF'),
        keyword_only_n=int(gkwonly.sum()),EVENT_share_keyword_only=share(gkwonly,'EVENT'),
        nonevent_anchor_share_all=float((gall&(gg.label!='EVENT')).mean()))
    r=res['anchor_bases'][q]; print('D5 %s V=%d rho_te=%.3f rho_all=%.3f | gold te %.2f all %.2f (n=%d) EVENT|all=%s kw-only n=%d EVENT=%s nonevent-anchor=%.2f'%(q,r['V'],r['rho_title_excerpt'],r['rho_with_keywords'],r['gold_mention_te'],r['gold_mention_all'],r['gold_mention_all_n'],r['EVENT_share_all'],r['keyword_only_n'],r['EVENT_share_keyword_only'],r['nonevent_anchor_share_all']))

# D6 post-strat: query-level cluster bootstrap
corpus=pd.read_csv(BASE+'/data/corpus_bigkinds.csv.gz',dtype=str,usecols=['qid','일자']); corpus['year']=corpus['일자'].str[:4]
Vqy=corpus.groupby(['qid','year']).size().rename('Vqy').reset_index()
gy=g.groupby(['qid','year']).agg(n=('isE','size'),e=('isE','sum')).reset_index().merge(Vqy,on=['qid','year']).merge(Q[['qid','block']],on='qid')
res['poststrat_query_boot']={}
for b in list(BL)+['all']:
    sub=gy if b=='all' else gy[gy.block==b]; qs=sub.qid.unique(); outs=[]
    for _ in range(4000):
        pick=RNG.choice(qs,len(qs),replace=True); s=pd.concat([sub[sub.qid==q] for q in pick])
        p=np.where(s.n>0,s.e/s.n.replace(0,np.nan),0); outs.append((p*s.Vqy).sum()/s.Vqy.sum())
    # Wilson-type stratum-level: use Jeffreys posterior per stratum
    outs2=[]
    for _ in range(4000):
        p=RNG.beta(sub.e+0.5,sub.n-sub.e+0.5); outs2.append((p*sub.Vqy).sum()/sub.Vqy.sum())
    res['poststrat_query_boot'][BL.get(b,'all')]=dict(query_cluster_ci=list(np.percentile(outs,[2.5,97.5])),jeffreys_ci=list(np.percentile(outs2,[2.5,97.5])))
    print('D6 %-13s cluster [%.3f, %.3f] jeffreys [%.3f, %.3f]'%(BL.get(b,'all'),*res['poststrat_query_boot'][BL.get(b,'all')]['query_cluster_ci'],*res['poststrat_query_boot'][BL.get(b,'all')]['jeffreys_ci']))

# D7 leave-one-query-out
CATS=['EVENT','REF','INST','OTHER','IRRELEVANT']; rows=[]
for q in sorted(g.qid.unique()):
    tr=g.qid!=q; te=~tr
    v=TfidfVectorizer(analyzer='char_wb',ngram_range=(2,4),min_df=2,sublinear_tf=True,max_features=120000)
    Xtr=v.fit_transform(g.text[tr]); Xte=v.transform(g.text[te])
    clf=LogisticRegression(C=4.0,class_weight='balanced',max_iter=300,tol=1e-3,random_state=0).fit(Xtr,g.label[tr]); pred=clf.predict(Xte)
    yt=g.label[te].values
    rows.append(dict(qid=q,block=BL[g.block[te].iloc[0]],n=int(te.sum()),EVENT=int((yt=='EVENT').sum()),macro_f1=f1_score(yt,pred,average='macro',labels=CATS,zero_division=0),
                     event_f1=f1_score((yt=='EVENT').astype(int),(pred=='EVENT').astype(int),zero_division=0),pE_true=float((yt=='EVENT').mean()),pE_pred=float((pred=='EVENT').mean())))
loqo=pd.DataFrame(rows); loqo.to_csv(OUT+'/loqo.csv',index=False)
# pooled EVENT-F1 over all held-out predictions by block
res['loqo']={'by_block':{},'overall':{}}
for b in list(BL.values())+['all']:
    s=loqo if b=='all' else loqo[loqo.block==b]
    res['loqo']['by_block'][b]=dict(macro_f1_mean=float(s.macro_f1.mean()),event_f1_mean=float(s.event_f1.mean()),
        event_f1_weighted=float((s.event_f1*s.n).sum()/s.n.sum()),pE_true=float((s.pE_true*s.n).sum()/s.n.sum()),pE_pred=float((s.pE_pred*s.n).sum()/s.n.sum()))
    print('D7 %-13s macroF1=%.3f eventF1=%.3f (weighted %.3f) pE true %.3f pred %.3f'%(b,*[res['loqo']['by_block'][b][k] for k in ['macro_f1_mean','event_f1_mean','event_f1_weighted','pE_true','pE_pred']]))

# D8 query wording: occurrence noun
OCC=['누출','유출','폭발','공격','난사','돌진','협박','신고','예고','피폭','사용','실험']
Q['occ']=Q['query'].apply(lambda s:any(o in s for o in OCC))
qq=qpe.merge(Q[['qid','query','occ']],on='qid')
res['wording']={'queries_with_occurrence_noun':qq.groupby('block').occ.sum().to_dict()}
for b in BL:
    s=qq[qq.block==b]; res['wording'][BL[b]]={'occ_true_pE':float(s[s.occ].isE.mean()) if s.occ.any() else None,'occ_false_pE':float(s[~s.occ].isE.mean()) if (~s.occ).any() else None,'n_occ':int(s.occ.sum())}
print('D8',res['wording'])
# security queries with occurrence noun vs industrial/conventional
sec_occ=qq[(qq.block=='안보')&qq.occ].isE; ind=qq[qq.block=='산업'].isE; conv=qq[qq.block=='비CBRN'].isE
res['wording']['security_occ_vs_industrial']=dict(U=st.mannwhitneyu(sec_occ,ind).statistic,p=st.mannwhitneyu(sec_occ,ind).pvalue,n=len(sec_occ),median_sec_occ=float(sec_occ.median()))
res['wording']['security_occ_vs_conventional']=dict(p=st.mannwhitneyu(sec_occ,conv).pvalue)
print('D8 security occ-noun queries median pE %.3f vs industrial p=%.4f conv p=%.4f'%(sec_occ.median(),res['wording']['security_occ_vs_industrial']['p'],res['wording']['security_occ_vs_conventional']['p']))
# Q30 reassigned to security
q2=qpe.copy(); q2.loc[q2.qid=='Q30','block']='안보'
grp={b:q2[q2.block==b].isE.values for b in BL}; kw=st.kruskal(*grp.values())
res['wording']['Q30_to_security']=dict(kruskal_p=kw.pvalue,sec_vs_ind_p=st.mannwhitneyu(grp['안보'],grp['산업']).pvalue,sec_vs_conv_p=st.mannwhitneyu(grp['安보' if False else '안보'],grp['비CBRN']).pvalue,ind_vs_conv_p=st.mannwhitneyu(grp['산업'],grp['비CBRN']).pvalue)
print('D8 Q30->security',res['wording']['Q30_to_security'])

# D9 N_events vs registry
cm=pd.read_csv(BASE+'/data/caris_match.csv',dtype=str); cm=cm[cm['대조기준'].str.startswith('mat:')]
Vq=corpus.groupby('qid').size();
# window-limited V (to 2025-04-28) per query
corpus_w=pd.read_csv(BASE+'/data/corpus_bigkinds.csv.gz',dtype=str,usecols=['qid','일자']); corpus_w=corpus_w[corpus_w['일자']<='20250428']
Vw=corpus_w.groupby('qid').size()
ps_q={}
for q in cm.qid:
    s=gy[gy.qid==q]; ps_q[q]=float((np.where(s.n>0,s.e/s.n.replace(0,np.nan),0)*s.Vqy).sum()/s.Vqy.sum())
qm={r['qid']:r for r in B2['m_vs_f']}
rows=[]
for _,r in cm.iterrows():
    q=r.qid; Vwin=int(Vw.get(q,0)); pe=ps_q[q]; m=qm[q]['m']; dom=B1['domestic']['by_query'].get(q,{});
    ev=g[(g.qid==q)&(g.isE==1)]; domshare=(dom.get('domestic',0)/max(dom.get('EVENT',1),1)) if dom else 1
    N=Vwin*pe/m; Ndom=N*domshare
    rows.append(dict(qid=q,query=r['query'],V_window=Vwin,pE_poststrat=pe,m_sample=m,N_events=N,domestic_share=domshare,N_events_domestic=Ndom,registry=int(r['공식사건수']),ratio=Ndom/int(r['공식사건수'])))
nev=pd.DataFrame(rows); res['N_events_vs_registry']=nev.to_dict('records')
sp=st.spearmanr(nev.N_events_domestic,nev.registry); lr=st.linregress(np.log(nev.registry),np.log(nev.N_events_domestic))
res['N_events_vs_registry_stats']=dict(spearman=sp.statistic,p=sp.pvalue,loglog_slope=lr.slope,r=lr.rvalue,ratio_min=float(nev.ratio.min()),ratio_max=float(nev.ratio.max()))
print(nev[['query','V_window','pE_poststrat','m_sample','N_events_domestic','registry','ratio']].round(2).to_string()); print('D9',res['N_events_vs_registry_stats'])

# D10 Wilson half-widths
def wilson(p,n,z=1.96):
    d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h,c+h
res['wilson']={f'p={p} n={n}':[round(x*100,1) for x in wilson(p,n)] for p in [0.1,0.3] for n in [100,200,400,800]}
print('D10',res['wilson'])

# D11 TPR/FPR per block
rg=json.load(open(BASE+'/Step4/outputs/rogan_gladen.json')); res['tpr_fpr']=rg; print('D11',rg)

json.dump(res,open(OUT+'/revision_D.json','w'),indent=1,ensure_ascii=False,default=float); print('saved')
