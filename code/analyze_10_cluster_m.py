#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refbias — Step 5 롤백 재분석 B2 (analyze_10_cluster_m.py)
==========================================================
C1 골드셋 사건 군집 규칙 민감도: 날짜 창 ±1/±3/±7일, 장소 무시, 대상 무시 → 블록별 m, A
C2 표집비율 f=n/V 와 표본 m 의 관계 (m 편향 진단)
C3 코퍼스 수준 m 재추정: 분류기(전체 골드셋 학습)로 코퍼스 EVENT 예측 → 질의 내 (게재일 ±3일, 제목+발췌 TF-IDF 코사인 ≥ θ,
   빅카인즈 위치 필드 공유) 규칙으로 군집 → m_corpus, A_corpus = V / N_events.
   규칙 검증: 골드셋 EVENT 기사에 같은 규칙을 적용해 LLM 추출 (ev_date, place, object) 군집과 비교 (ARI, m 비교)
"""
import os, re, json, unicodedata, itertools
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT=BASE+'/Step5r'; os.makedirs(OUT,exist_ok=True)
res={}; BL={'산업':'industrial','안보':'security','비CBRN':'conventional'}
def norm(s): return re.sub(r'\s+','',unicodedata.normalize('NFKC',str(s or '')).strip().lower())

g=pd.read_csv(BASE+'/data/goldset_labeled_A.csv',dtype=str)
g['title']=g['제목'].fillna(''); g['body']=g['본문'].fillna('').str.replace(r'\s+',' ',regex=True).str[:200]
g['text']=g['title']+' ▁ '+g['body']

# ---------- C1 clustering sensitivity on gold ----------
def cluster_gold(ev,window=3,use_place=True,use_object=True):
    ev=ev.copy(); ev['d']=pd.to_datetime(ev['ev_date'],format='%Y%m%d',errors='coerce')
    ev['p']=ev['ev_place'].map(norm) if use_place else ''; ev['o']=ev['ev_object'].map(norm) if use_object else ''
    ids={}; reps=[]; nxt=0
    for i,r in ev.sort_values('d').iterrows():
        hit=None
        for k,dd,pp,oo in reps:
            if pp==r['p'] and oo==r['o'] and pd.notna(dd) and pd.notna(r['d']) and abs((r['d']-dd).days)<=window: hit=k; break
        if hit is None: hit=nxt; nxt+=1; reps.append((hit,r['d'],r['p'],r['o']))
        ids[i]=hit
    return pd.Series(ids)
ev=g[g.label=='EVENT']
res['cluster_sensitivity']=[]
for window,up,uo,name in [(3,True,True,'±3d place+object (main)'),(1,True,True,'±1d place+object'),(7,True,True,'±7d place+object'),
                          (3,False,True,'±3d object only'),(3,True,False,'±3d place only'),(3,False,False,'±3d date only')]:
    cid=cluster_gold(ev,window,up,uo); tmp=ev.assign(cid=cid)
    row=dict(rule=name)
    for b in list(BL)+['all']:
        s=tmp if b=='all' else tmp[tmp.block==b]; n_all=len(g) if b=='all' else (g.block==b).sum()
        u=s.cid.nunique(); row[BL.get(b,'all')]=dict(unique_events=int(u),m=len(s)/u,A=n_all/u)
    res['cluster_sensitivity'].append(row)
    print('C1 %-26s'%name,{k:(v['unique_events'],round(v['m'],2),round(v['A'],2)) for k,v in row.items() if k!='rule'})

# ---------- C2 sampling fraction vs m ----------
corpus=pd.read_csv(BASE+'/data/corpus_bigkinds.csv.gz',dtype=str)
V=corpus.groupby('qid').size()
qm=[]
main=cluster_gold(ev,3,True,True); tmp=ev.assign(cid=main)
for q,s in tmp.groupby('qid'):
    n=(g.qid==q).sum(); qm.append(dict(qid=q,n=int(n),V=int(V[q]),f=n/V[q],EVENT=len(s),unique=int(s.cid.nunique()),m=len(s)/s.cid.nunique()))
qm=pd.DataFrame(qm); res['m_vs_f']=qm.to_dict('records')
from scipy import stats as st
sp=st.spearmanr(qm.f,qm.m); print('C2 spearman(f, m)= %.2f p=%.3f ; m at f>=0.2: %.2f ; m at f<0.05: %.2f'%(sp.statistic,sp.pvalue,qm[qm.f>=0.2].m.mean(),qm[qm.f<0.05].m.mean()))
res['m_vs_f_spearman']=dict(rho=sp.statistic,p=sp.pvalue,m_mean_f_ge_0_2=qm[qm.f>=0.2].m.mean(),m_mean_f_lt_0_05=qm[qm.f<0.05].m.mean())

# ---------- C3 corpus-level clustering ----------
corpus['title']=corpus['제목'].fillna(''); corpus['body']=corpus['본문'].fillna('').str.replace(r'\s+',' ',regex=True).str[:200]
corpus['text']=corpus['title']+' ▁ '+corpus['body']; corpus['d']=pd.to_datetime(corpus['일자'],format='%Y%m%d',errors='coerce')
corpus['loc']=corpus['위치'].fillna('')
# classifier trained on full gold set (same config as proposed)
vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(2,4),min_df=2,sublinear_tf=True,max_features=120000)
Xg=vec.fit_transform(g['text']); clf=LogisticRegression(C=4.0,class_weight='balanced',max_iter=300,tol=1e-3,random_state=0).fit(Xg,g['label'])
corpus['pred']=clf.predict(vec.transform(corpus['text']))
print('C3 corpus predicted EVENT share by block',corpus.groupby('block').pred.apply(lambda s:(s=='EVENT').mean()).round(3).to_dict())

def cluster_text(df,window=3,theta=0.35,use_loc=True):
    """greedy same-event grouping within a query: date within ±window, cosine(title+excerpt)>=theta, and (if use_loc) share a location token"""
    if len(df)==0: return pd.Series(dtype=int)
    df=df.sort_values('d'); v=TfidfVectorizer(analyzer='char_wb',ngram_range=(2,4),sublinear_tf=True); X=v.fit_transform(df['text'])
    locs=[set(x.split(',')) - {''} for x in df['loc']]
    ids=np.full(len(df),-1); reps=[]  # (cluster_id, idx_of_rep, date)
    dates=df['d'].values
    for i in range(len(df)):
        hit=-1
        for cid,j,dj in reps:
            if pd.isna(dates[i]) or pd.isna(dj): continue
            dd=abs((dates[i]-dj)/np.timedelta64(1,'D'))
            if dd>window: continue
            if use_loc and locs[i] and locs[j] and not (locs[i]&locs[j]): continue
            if cosine_similarity(X[i],X[j])[0,0]>=theta: hit=cid; break
        if hit<0: hit=len(reps); reps.append((hit,i,dates[i]))
        ids[i]=hit
    return pd.Series(ids,index=df.index)

# validation on gold EVENT articles: text rule vs LLM-field rule
gold_ev=g[g.label=='EVENT'].copy(); gold_ev['d']=pd.to_datetime(gold_ev['일자'],format='%Y%m%d',errors='coerce')
gold_ev=gold_ev.merge(corpus[['뉴스 식별자','loc']].drop_duplicates('뉴스 식별자'),on='뉴스 식별자',how='left'); gold_ev['loc']=gold_ev['loc'].fillna('')
assert len(gold_ev)==(g.label=='EVENT').sum()
ref=cluster_gold(gold_ev.set_index(gold_ev.index),3,True,True)
res['text_rule_validation']=[]
for theta in [0.25,0.35,0.45]:
    for use_loc in [True,False]:
        cid=pd.concat([cluster_text(s,3,theta,use_loc) for _,s in gold_ev.groupby('qid')]).reindex(gold_ev.index)
        # make ids unique across queries
        cid=cid.astype(str)+'_'+gold_ev['qid']
        ari=adjusted_rand_score(ref.reindex(gold_ev.index).astype(str)+'_'+gold_ev['qid'],cid)
        m_text=len(gold_ev)/cid.nunique(); m_ref=len(gold_ev)/(ref.astype(str)+'_'+gold_ev['qid']).nunique()
        res['text_rule_validation'].append(dict(theta=theta,use_loc=use_loc,ARI=ari,m_text=m_text,m_llm=m_ref))
        print('C3 val theta=%.2f loc=%s ARI=%.3f m_text=%.3f m_llm=%.3f'%(theta,use_loc,ari,m_text,m_ref))
best=max(res['text_rule_validation'],key=lambda r:r['ARI']); res['text_rule_chosen']=best
theta,use_loc=best['theta'],best['use_loc']

# apply to corpus predicted EVENT articles
pe=corpus[corpus.pred=='EVENT']
res['corpus_m']={}
rows=[]
for q,s in pe.groupby('qid'):
    cid=cluster_text(s,3,theta,use_loc); u=cid.nunique()
    rows.append(dict(qid=q,block=BL[s.block.iloc[0]],V=int(V[q]),pred_EVENT=len(s),unique_events=int(u),m_corpus=len(s)/u,A_corpus=V[q]/u))
qc=pd.DataFrame(rows); res['corpus_m']['by_query']=qc.to_dict('records')
for b in list(BL.values())+['all']:
    s=qc if b=='all' else qc[qc.block==b]
    res['corpus_m'][b]=dict(V=int(s.V.sum()),pred_EVENT=int(s.pred_EVENT.sum()),unique_events=int(s.unique_events.sum()),
                            m_corpus=s.pred_EVENT.sum()/s.unique_events.sum(),A_corpus_raw=s.V.sum()/s.unique_events.sum())
    print('C3 %-13s V=%6d predEVENT=%5d events=%5d m=%.2f A_raw=%.2f'%(b,*[res['corpus_m'][b][k] for k in ['V','pred_EVENT','unique_events','m_corpus','A_corpus_raw']]))
qc.to_csv(OUT+'/corpus_events_by_query.csv',index=False,encoding='utf-8-sig')
pe[['qid','block','뉴스 식별자','일자','제목']].to_csv(OUT+'/corpus_pred_event_articles.csv.gz',index=False,encoding='utf-8-sig',compression='gzip')
json.dump(res,open(OUT+'/revision_B2.json','w'),indent=1,ensure_ascii=False,default=float)
print('saved')
