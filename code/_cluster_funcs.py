#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""refbias — 사건 군집 규칙 공용 모듈 (analyze_10_cluster_m.py 에서 분리)"""
import re, unicodedata
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def norm(s): return re.sub(r'\s+','',unicodedata.normalize('NFKC',str(s or '')).strip().lower())

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

