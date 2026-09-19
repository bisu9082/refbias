#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""refbias — 전체 코퍼스 109,610건 추론 및 질의별 성분 분해
제안 모델(문자 n-gram TF-IDF + 가중 LR)을 정답셋 2,867건 전량으로 학습해 코퍼스에 적용한다.
증폭계수 추정: A_full = m_block / p_E_full  (m = 정답셋에서 관측한 중복보도 계수, 블록별)"""
import os, sys, json, pickle, numpy as np, pandas as pd, warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore')
W=os.path.expanduser('~/rbwork'); OUT=W+'/out'; R=os.path.expanduser('~/mnt/claude_research/refbias')
CATS=['EVENT','REF','INST','OTHER','IRRELEVANT']
g=pd.read_csv(R+'/data/goldset_labeled_A.csv',dtype=str)
txt=lambda df:(df['제목'].fillna('')+' ▁ '+df['본문'].fillna('').str.replace(r'\s+',' ',regex=True).str[:200])
MODE=sys.argv[1]; PK=OUT+'/model.pkl'
if MODE=='fit':
    v=TfidfVectorizer(analyzer='char_wb',ngram_range=(2,4),min_df=2,sublinear_tf=True,max_features=120000)
    X=v.fit_transform(txt(g)); clf=LogisticRegression(C=4.0,class_weight='balanced',max_iter=300,tol=1e-3,random_state=0).fit(X,g['label'])
    pickle.dump((v,clf),open(PK,'wb')); print('학습 완료 vocab',len(v.vocabulary_)); sys.exit(0)
v,clf=pickle.load(open(PK,'rb'))
LO,HI=int(sys.argv[2]),int(sys.argv[3])   # 처리할 청크 번호 범위 [LO,HI)
PART=OUT+f'/pred_part_{LO:02d}.csv'
# 정답셋에서 블록별 중복보도 계수 m
m_blk={}
for b,s in g.groupby('block'):
    e=(s['label']=='EVENT').sum(); u=s.loc[s['label']=='EVENT','event_id'].nunique(); m_blk[b]=e/u
agg={}
if MODE=='run':
  for i,ch in enumerate(pd.read_csv(R+'/data/corpus_bigkinds.csv.gz',compression='gzip',dtype=str,chunksize=10000,
                                  usecols=['qid','block','query','일자','제목','본문'])):
    if i<LO: continue
    if i>=HI: break
    pred=clf.predict(v.transform(txt(ch)))
    ch['pred']=pred; ch['year']=ch['일자'].str[:4]
    for (q,b,qq,yr,p),n in ch.groupby(['qid','block','query','year','pred']).size().items():
        agg[(q,b,qq,yr,p)]=agg.get((q,b,qq,yr,p),0)+int(n)
    print(f'chunk {i} 처리 {len(ch)}',flush=True)
  pd.DataFrame([dict(qid=k[0],block=k[1],query=k[2],year=k[3],pred=k[4],n=vv) for k,vv in agg.items()]).to_csv(PART,index=False,encoding='utf-8-sig'); sys.exit(0)
import glob
t=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(OUT+'/pred_part_*.csv'))]).groupby(['qid','block','query','year','pred'],as_index=False)['n'].sum()
t.to_csv(OUT+'/corpus_pred_by_query_year.csv',index=False,encoding='utf-8-sig')
# 질의별 성분 비율 + 증폭계수
piv=t.groupby(['qid','block','query','pred'])['n'].sum().unstack('pred').reindex(columns=CATS).fillna(0)
piv['V']=piv.sum(axis=1)
for c in CATS: piv[c+'_pct']=(piv[c]/piv['V']*100).round(2)
piv=piv.reset_index()
piv['m']=piv['block'].map(m_blk); piv['pE']=piv['EVENT']/piv['V']
piv['A_full']=(piv['m']/piv['pE']).round(2)
piv['E_hat']=(piv['V']/piv['A_full']).round(0)   # 추정 고유 사건 수
piv.sort_values('A_full',ascending=False).to_csv(OUT+'/corpus_components_by_query.csv',index=False,encoding='utf-8-sig')
blk=t.groupby(['block','pred'])['n'].sum().unstack('pred').reindex(columns=CATS).fillna(0); blk['V']=blk.sum(axis=1)
for c in CATS: blk[c+'_pct']=(blk[c]/blk['V']*100).round(2)
blk['m']=blk.index.map(m_blk); blk['A_full']=(blk['m']/(blk['EVENT']/blk['V'])).round(2)
blk.to_csv(OUT+'/corpus_components_by_block.csv',encoding='utf-8-sig')
print(blk[['V','EVENT_pct','REF_pct','INST_pct','OTHER_pct','IRRELEVANT_pct','m','A_full']].to_string())
print('\n전체 V=%d  EVENT%%=%.1f'%(blk['V'].sum(), blk['EVENT'].sum()/blk['V'].sum()*100))
