#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refbias — Step 4 분류기 실험 (analysis_main.py)
===============================================
과제  : 기사 5범주 분류 (EVENT/REF/INST/OTHER/IRRELEVANT). 핵심 지표는 EVENT-F1.
제안  : 문자 n-gram(2-4) TF-IDF + 클래스가중 로지스틱 회귀  [HO-2 사양]
베이스라인: majority / random / 키워드규칙(질의 토큰 전부 포함→EVENT) / 단어 TF-IDF+LR
절제  : 제목만 / 제목+본문 / +빅카인즈 분류 원핫 / 문자 vs 단어
검증  : 시드 10 × 층화 5겹.  통계: 정규성(Shapiro) → 대응 t 또는 Wilcoxon, Cohen's d, Bootstrap 10,000
Sanity: ① 단조성(학습 비율↑ → F1↑) ② random≈1/K ③ 블록 간 교차조건 일관성
"""
import os, sys, json, time, warnings, re
import numpy as np, pandas as pd
from scipy import stats as st
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
warnings.filterwarnings('ignore')

W=os.path.expanduser('~/rbwork'); OUT=W+'/out'; os.makedirs(OUT,exist_ok=True)
DATA=os.path.expanduser('~/mnt/claude_research/refbias/data/goldset_labeled_A.csv')
CATS=['EVENT','REF','INST','OTHER','IRRELEVANT']; SEEDS=list(range(10)); NBOOT=10000
log=open(OUT+'/run.log','a'); 
def P(*a):
    s=' '.join(str(x) for x in a); print(s,flush=True); log.write(s+'\n'); log.flush()

d=pd.read_csv(DATA,dtype=str)
d['title']=d['제목'].fillna(''); d['body']=d['본문'].fillna('').str.replace(r'\s+',' ',regex=True).str[:200]
d['text']=d['title']+' ▁ '+d['body']; d['bk']=d['사건/사고 분류1'].fillna('없음')
y=d['label'].values; yE=(y=='EVENT').astype(int)
P(f'n={len(d)}  분포={dict(pd.Series(y).value_counts())}')

def ef1(yt,pred):
    return f1_score((yt=='EVENT').astype(int),(pred=='EVENT').astype(int),zero_division=0)

def kw_rule(df):
    """키워드 규칙: 질의 토큰이 제목+본문에 전부 등장하면 EVENT, 아니면 OTHER"""
    out=[]
    for q,t in zip(df['query'],df['text']):
        out.append('EVENT' if all(tok in t for tok in q.split()) else 'OTHER')
    return np.array(out)

def make_X(tr,te,cfg):
    parts_tr,parts_te=[],[]
    if cfg['feat'] in ('title','title+body','title+body+bk'):
        col='title' if cfg['feat']=='title' else 'text'
        if cfg['ngram']=='char':
            v=TfidfVectorizer(analyzer='char_wb',ngram_range=(2,4),min_df=2,sublinear_tf=True,max_features=120000)
        else:
            v=TfidfVectorizer(analyzer='word',token_pattern=r'[가-힣A-Za-z0-9]{2,}',ngram_range=(1,2),min_df=2,sublinear_tf=True)
        parts_tr.append(v.fit_transform(d.loc[tr,col])); parts_te.append(v.transform(d.loc[te,col]))
    if cfg['feat']=='title+body+bk':
        oh=OneHotEncoder(handle_unknown='ignore'); parts_tr.append(oh.fit_transform(d.loc[tr,['bk']])); parts_te.append(oh.transform(d.loc[te,['bk']]))
    return hstack(parts_tr).tocsr(), hstack(parts_te).tocsr()

CONFIGS={
 'proposed_char_tb'  : dict(feat='title+body',    ngram='char'),
 'abl_char_title'    : dict(feat='title',         ngram='char'),
 'abl_word_tb'       : dict(feat='title+body',    ngram='word'),
 'abl_char_tb_bk'    : dict(feat='title+body+bk', ngram='char'),
}
MODE=sys.argv[1] if len(sys.argv)>1 else 'cv'
SEED_ARG=[int(sys.argv[2])] if len(sys.argv)>2 else SEEDS
FOLD_ARG=int(sys.argv[3]) if len(sys.argv)>3 else None
RAW=OUT+'/raw_results.csv'
rows=[]
t0=time.time()
for seed in (SEED_ARG if MODE=='cv' else []):
    skf=StratifiedKFold(5,shuffle=True,random_state=seed)
    for fold,(tr,te) in enumerate(skf.split(d,y)):
        if FOLD_ARG is not None and fold!=FOLD_ARG: continue
        yt=y[te]
        # baselines
        maj=np.array([pd.Series(y[tr]).mode()[0]]*len(te))
        rng=np.random.default_rng(seed*100+fold); rnd=rng.choice(CATS,len(te))
        kw=kw_rule(d.iloc[te])
        for name,pred in [('base_majority',maj),('base_random',rnd),('base_keyword',kw)]:
            rows.append(dict(seed=seed,fold=fold,model=name,acc=accuracy_score(yt,pred),
                             macro_f1=f1_score(yt,pred,average='macro',labels=CATS,zero_division=0),
                             event_f1=ef1(yt,pred),
                             event_prec=((pred=='EVENT')&(yt=='EVENT')).sum()/max((pred=='EVENT').sum(),1),
                             event_rec=((pred=='EVENT')&(yt=='EVENT')).sum()/max((yt=='EVENT').sum(),1)))
        for name,cfg in CONFIGS.items():
            Xtr,Xte=make_X(tr,te,cfg)
            clf=LogisticRegression(C=4.0,class_weight='balanced',max_iter=300,tol=1e-3,random_state=seed)
            clf.fit(Xtr,y[tr]); pred=clf.predict(Xte)
            rows.append(dict(seed=seed,fold=fold,model=name,acc=accuracy_score(yt,pred),
                             macro_f1=f1_score(yt,pred,average='macro',labels=CATS,zero_division=0),
                             event_f1=ef1(yt,pred),
                             event_prec=((pred=='EVENT')&(yt=='EVENT')).sum()/max((pred=='EVENT').sum(),1),
                             event_rec=((pred=='EVENT')&(yt=='EVENT')).sum()/max((yt=='EVENT').sum(),1)))
            # 교차조건(블록별) — 제안 모델만
            if name=='proposed_char_tb':
                for b in ['산업','안보','비CBRN']:
                    m=(d.iloc[te]['block'].values==b)
                    if m.sum()>20:
                        rows.append(dict(seed=seed,fold=fold,model=f'proposed@{b}',acc=accuracy_score(yt[m],pred[m]),
                                         macro_f1=f1_score(yt[m],pred[m],average='macro',labels=CATS,zero_division=0),
                                         event_f1=ef1(yt[m],pred[m]),
                                         event_prec=np.nan,event_rec=np.nan))
    P(f'seed {seed} 완료  {time.time()-t0:.0f}s')
if MODE=='cv':
    pd.DataFrame(rows).to_csv(RAW,index=False,encoding='utf-8-sig',mode='a',header=not os.path.exists(RAW)); sys.exit(0)
raw=pd.read_csv(RAW)

# ── 단조성: 학습 비율 25/50/75/100% (시드 10, 5겹 중 1겹)
MONO=OUT+'/monotonicity.csv'
mono=[]
for seed in (SEED_ARG if MODE=='mono' else []):
    skf=StratifiedKFold(5,shuffle=True,random_state=seed); tr,te=next(iter(skf.split(d,y)))
    rng=np.random.default_rng(seed)
    for frac in [0.25,0.5,0.75,1.0]:
        sub=rng.choice(tr,int(len(tr)*frac),replace=False)
        Xtr,Xte=make_X(sub,te,CONFIGS['proposed_char_tb'])
        clf=LogisticRegression(C=4.0,class_weight='balanced',max_iter=300,tol=1e-3,random_state=seed).fit(Xtr,y[sub])
        pred=clf.predict(Xte)
        mono.append(dict(seed=seed,frac=frac,macro_f1=f1_score(y[te],pred,average='macro',labels=CATS,zero_division=0)))
if MODE=='mono':
    pd.DataFrame(mono).to_csv(MONO,index=False,mode='a',header=not os.path.exists(MONO)); sys.exit(0)
mono=pd.read_csv(MONO)
mm=mono.groupby('frac')['macro_f1'].mean(); mono_ok=bool(all(np.diff(mm.values)>=-0.005))
P('단조성 macro-F1 by frac:',mm.round(3).to_dict(),'→',mono_ok)

# ── 요약 통계
def agg(m):
    s=raw[raw.model==m]; return {k:(round(s[k].mean(),4),round(s[k].std(ddof=1),4)) for k in ['acc','macro_f1','event_f1','event_prec','event_rec']}
summary={m:agg(m) for m in raw.model.unique()}
for m,v in summary.items(): P(f'{m:22s} acc={v["acc"][0]:.3f}±{v["acc"][1]:.3f}  macroF1={v["macro_f1"][0]:.3f}±{v["macro_f1"][1]:.3f}  EVENT-F1={v["event_f1"][0]:.3f}±{v["event_f1"][1]:.3f}')

# 제안 vs 최강 베이스라인(키워드) 및 vs 단어절제 — 대응 검정
def paired(a_name,b_name,metric):
    A=raw[raw.model==a_name].sort_values(['seed','fold'])[metric].values
    B=raw[raw.model==b_name].sort_values(['seed','fold'])[metric].values
    diff=A-B; W_,pw=st.shapiro(diff)
    if pw>=0.05: t,p=st.ttest_rel(A,B); test='paired t'
    else: t,p=st.wilcoxon(A,B); test='Wilcoxon'
    dz=diff.mean()/diff.std(ddof=1)
    rng=np.random.default_rng(20260908); bs=[rng.choice(diff,len(diff)).mean() for _ in range(NBOOT)]
    return dict(test=test,shapiro_p=float(pw),stat=float(t),p=float(p),cohen_dz=float(dz),
                mean_diff=float(diff.mean()),ci95=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))])
tests={'proposed_vs_keyword_eventF1':paired('proposed_char_tb','base_keyword','event_f1'),
       'proposed_vs_keyword_macroF1':paired('proposed_char_tb','base_keyword','macro_f1'),
       'proposed_vs_word_macroF1':paired('proposed_char_tb','abl_word_tb','macro_f1'),
       'proposed_vs_titleonly_macroF1':paired('proposed_char_tb','abl_char_title','macro_f1'),
       'bk_feature_gain_macroF1':paired('abl_char_tb_bk','proposed_char_tb','macro_f1')}
for k,v in tests.items(): P(f'{k:32s} {v["test"]:9s} p={v["p"]:.2e} dz={v["cohen_dz"]:+.2f} Δ={v["mean_diff"]:+.4f} CI{v["ci95"]}')

# Sanity
rnd_acc=summary['base_random']['acc'][0]; base_ok=bool(abs(rnd_acc-0.20)<0.03)
blk={b:float(summary[f'proposed@{b}']['macro_f1'][0]) for b in ['산업','안보','비CBRN']}
cross_ok=bool((max(blk.values())/min(blk.values()))<1.5 and min(blk.values())>summary['base_keyword']['macro_f1'][0])  # 패턴 일관성: 최대/최소 <1.5 이고 전 블록이 키워드 베이스라인 상회
P(f'Sanity ② random acc={rnd_acc:.3f} (이론 0.200) → {base_ok}')
P(f'Sanity ③ 블록별 macro-F1 {blk} 범위 {max(blk.values())-min(blk.values()):.3f} → {cross_ok}')
decision='PROCEED' if (mono_ok and base_ok and cross_ok) else 'REFINE'
json.dump(dict(n=len(d),seeds=len(SEEDS),folds=5,bootstrap=NBOOT,summary=summary,tests=tests,
               monotonicity={str(k):float(v) for k,v in mm.round(4).items()},sanity=dict(monotonicity=mono_ok,baseline=base_ok,cross_condition=cross_ok,block_f1=blk),
               decision=decision),open(OUT+'/experiment_summary.json','w'),ensure_ascii=False,indent=1)
P('DECISION',decision,'  총',f'{time.time()-t0:.0f}s')
