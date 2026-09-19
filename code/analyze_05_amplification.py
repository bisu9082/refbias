#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refbias — 사건 클러스터링 및 증폭계수 산출
==========================================
EVENT 기사를 (ev_date ± 3일, ev_place, ev_object) 기준으로 병합해 고유 사건 수 u를 센다.
표본 n건 안에 고유 사건이 u개면, 보도량 1건당 사건은 u/n건이다.

  사건성 비율      p_E = EVENT / n
  중복보도 계수    m   = EVENT / u        (사건 1건당 EVENT 기사 수)
  증폭계수         A   = n / u = 1 / (p_E / m)
  참조 성분        R   = REF / n
  제도 성분        I   = INST / n
  질의 정밀도      P   = (n - IRRELEVANT) / n

Bootstrap 10,000회로 95% CI를 낸다.
"""
import pandas as pd, numpy as np, os, re, unicodedata
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RNG=np.random.default_rng(20260908); NBOOT=10000

def norm(s):
    s=unicodedata.normalize('NFKC',str(s or '')).strip().lower()
    return re.sub(r'\s+','',s)

def cluster(df):
    """EVENT 행에 사건 id 부여. 날짜 ±3일 + 장소 + 대상 일치 시 동일 사건."""
    ev=df[df['label']=='EVENT'].copy()
    ev['d']=pd.to_datetime(ev['ev_date'],format='%Y%m%d',errors='coerce')
    ev['p']=ev['ev_place'].map(norm); ev['o']=ev['ev_object'].map(norm)
    cid={}; nxt=0
    reps=[]   # (id, date, place, object)
    for i,r in ev.sort_values('d').iterrows():
        hit=None
        for j,(k,dd,pp,oo) in enumerate(reps):
            if pp==r['p'] and oo==r['o'] and pd.notna(dd) and pd.notna(r['d']) \
               and abs((r['d']-dd).days)<=3:
                hit=k; break
        if hit is None:
            hit=nxt; nxt+=1; reps.append((hit,r['d'],r['p'],r['o']))
        cid[i]=hit
    df['event_id']=pd.Series(cid)
    return df

def stats(sub):
    n=len(sub)
    if n==0: return None
    e=(sub['label']=='EVENT').sum()
    u=sub.loc[sub['label']=='EVENT','event_id'].nunique()
    return dict(n=n, EVENT=int(e), 고유사건=int(u),
                pE=e/n, 중복계수=e/u if u else np.nan, 증폭계수=n/u if u else np.nan,
                REF=(sub['label']=='REF').mean(), INST=(sub['label']=='INST').mean(),
                OTHER=(sub['label']=='OTHER').mean(), IRR=(sub['label']=='IRRELEVANT').mean())

def boot_ci_A(sub):
    """증폭계수 A = m / pE 의 95% CI.
    재표본에서 nunique(event_id)는 구조적으로 과소추정되므로 u는 재계산하지 않는다.
    중복보도 계수 m은 표본 상수로 고정하고, 비율 pE 만 부트스트랩한다."""
    st=stats(sub)
    if st is None or not np.isfinite(st['중복계수']): return (np.nan,np.nan)
    m=st['중복계수']; y=(sub['label']=='EVENT').values.astype(float); n=len(y)
    ps=np.array([y[RNG.integers(0,n,n)].mean() for _ in range(NBOOT)])
    ps=ps[ps>0]
    return tuple(np.percentile(m/ps,[2.5,97.5]))

def main():
    d=pd.read_csv(BASE+'/data/goldset_labeled_A.csv',dtype=str)
    d=cluster(d)
    d.to_csv(BASE+'/data/goldset_labeled_A.csv',index=False,encoding='utf-8-sig')

    print('=== 블록별 ===')
    rows=[]
    for b,g in d.groupby('block'):
        st=stats(g); lo,hi=boot_ci_A(g)
        st.update(block=b, A_lo=lo, A_hi=hi); rows.append(st)
    tot=stats(d); lo,hi=boot_ci_A(d); tot.update(block='전체',A_lo=lo,A_hi=hi); rows.append(tot)
    t=pd.DataFrame(rows)[['block','n','EVENT','고유사건','pE','중복계수','증폭계수','A_lo','A_hi','REF','INST','OTHER','IRR']]
    for c in ['pE','REF','INST','OTHER','IRR']: t[c]=(t[c]*100).round(1)
    for c in ['중복계수','증폭계수','A_lo','A_hi']: t[c]=t[c].round(2)
    print(t.to_string(index=False))
    t.to_csv(BASE+'/data/amplification_by_block.csv',index=False,encoding='utf-8-sig')

    print('\n=== 질의별 (증폭계수 상위 12) ===')
    rows=[]
    for q,g in d.groupby(['qid','query','block']):
        st=stats(g)
        if st: st.update(qid=q[0],query=q[1],block=q[2]); rows.append(st)
    q=pd.DataFrame(rows)[['qid','block','query','n','EVENT','고유사건','pE','중복계수','증폭계수','REF','INST','IRR']]
    for c in ['pE','REF','INST','IRR']: q[c]=(q[c]*100).round(1)
    for c in ['중복계수','증폭계수']: q[c]=q[c].round(2)
    q=q.sort_values('증폭계수',ascending=False)
    print(q.head(12).to_string(index=False))
    q.to_csv(BASE+'/data/amplification_by_query.csv',index=False,encoding='utf-8-sig')
    print('\n=== 질의별 (증폭계수 하위 6) ===')
    print(q.tail(6).to_string(index=False))

if __name__=='__main__': main()
