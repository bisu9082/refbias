#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refbias — v9 개정용 신규 분석 (analyze_16_v9_additions.py)
=========================================================
심사 시뮬레이션에서 제기된 반론에 실측으로 답한다.

V-A 짝 맞춘 신뢰도 표: 5범주 대 5범주, 이진 대 이진, 블록별
    (기존 서술은 5범주 모델-모델 0.79 와 이진 모델-인간 0.24 를 나란히 놓아 지표가 어긋났다)
V-B 안보 블록 코더A 2x2 원 카운트, 정밀도, 위음성 4건의 잭나이프
V-C 앵커 인용률을 비사건 기사에 조건부로 재계산
    (rho 와 1-p_E 가 독립이 아니므로, 사건 희소성과 참조 부하를 분리한다)
V-D 산업 대 비CBRN 차이의 신뢰구간과 등가성 (검출 가능 효과크기)
V-E 등록부 회귀의 x 측정오차 감쇠: 기울기 1 과 화해시키는 오차 크기
출력: Step5r/v9_additions.json
"""
import os, json, itertools, re
import numpy as np, pandas as pd
from scipy import stats as st
from sklearn.metrics import cohen_kappa_score, confusion_matrix
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT=BASE+'/Step5r'; RNG=np.random.default_rng(20260909); NB=10000
BL={'산업':'industrial','안보':'security','비CBRN':'conventional'}
CATS=['EVENT','REF','INST','OTHER','IRRELEVANT']
AL={'event':'EVENT','ref':'REF','inst':'INST','other':'OTHER','irrelevant':'IRRELEVANT',
    '사건':'EVENT','참조':'REF','제도':'INST','기타':'OTHER','무관':'IRRELEVANT'}
def nm(x):
    if pd.isna(x): return None
    s=str(x).strip(); return AL.get(s.lower(), s.upper() if s.upper() in CATS else None)
def kci(a,b,labs):
    a=np.asarray(a); b=np.asarray(b); n=len(a)
    k=cohen_kappa_score(a,b,labels=labs); cm=confusion_matrix(a,b,labels=labs)/n
    pi_=cm.sum(1); pj=cm.sum(0); pe=(pi_*pj).sum()
    A=sum(cm[i,i]*(1-(pi_[i]+pj[i])*(1-k))**2 for i in range(len(labs)))
    B=(1-k)**2*sum(cm[i,j]*(pj[i]+pi_[j])**2 for i in range(len(labs)) for j in range(len(labs)) if i!=j)
    C=(k-pe*(1-k))**2
    se=np.sqrt(max((A+B-C)/((1-pe)**2*n),0))
    return dict(kappa=float(k),ci95=[float(k-1.96*se),float(k+1.96*se)],n=int(n))
res={}

g=pd.read_csv(BASE+'/data/goldset_labeled_A.csv',dtype=str); g['isE']=(g.label=='EVENT')
B=pd.concat([pd.read_csv(f'{BASE}/Step4/labels_B/{f}',dtype=str)
             for f in sorted(os.listdir(BASE+'/Step4/labels_B')) if f.endswith('.csv')])
H={t:pd.read_excel(f'{BASE}/Step4/human_labeling/human_labels_{t}.xlsx',dtype=str)[['sid','label']]
     .assign(label=lambda x:x.label.map(nm)) for t in ['H1','H2']}
d=(H['H1'].rename(columns={'label':'h1'}).merge(H['H2'].rename(columns={'label':'h2'}),on='sid')
   .merge(g[['sid','block','qid','query','label']].rename(columns={'label':'A'}),on='sid')
   .merge(B[['sid','label']].rename(columns={'label':'Bc'}),on='sid',how='left'))
cons=d[d.h1==d.h2].copy(); cons['y']=cons.h1

# ---------- V-A 짝 맞춘 신뢰도 ----------
AB=g[['sid','block','label']].rename(columns={'label':'A'}).merge(B[['sid','label']].rename(columns={'label':'Bc'}),on='sid')
def pair(x,y,sub,tag):
    o={'n':int(len(sub)),
       'five_class':kci(sub[x],sub[y],CATS),
       'binary':kci(sub[x]=='EVENT',sub[y]=='EVENT',[False,True])}
    for kb,b in BL.items():
        s=sub[sub.block==kb]
        if len(s)>=10:
            o[b]={'five_class':kci(s[x],s[y],CATS),'binary':kci(s[x]=='EVENT',s[y]=='EVENT',[False,True])}
    return o
res['matched_kappa']={
 'A_vs_B':      pair('A','Bc',AB,'A-B'),
 'human_vs_A':  pair('y','A',cons,'H-A'),
 'human_vs_B':  pair('y','Bc',cons[cons.Bc.notna()],'H-B'),
 'H1_vs_H2':    pair('h1','h2',d,'H1-H2'),
}
print('== V-A 짝 맞춘 kappa ==')
for k,v in res['matched_kappa'].items():
    sec=v.get('security',{})
    print('%-12s n=%4d 5범주 %.3f 이진 %.3f | 안보 5범주 %s 이진 %s'%(
        k,v['n'],v['five_class']['kappa'],v['binary']['kappa'],
        ('%.3f'%sec['five_class']['kappa']) if sec else '-',
        ('%.3f'%sec['binary']['kappa']) if sec else '-'))

# ---------- V-B 안보 2x2 ----------
s=cons[cons.block=='안보']
hu=(s.y=='EVENT').values; ae=(s.A=='EVENT').values
tp=int((hu&ae).sum()); fp=int((~hu&ae).sum()); fn=int((hu&~ae).sum()); tn=int((~hu&~ae).sum())
res['security_2x2']=dict(TP=tp,FP=fp,FN=fn,TN=tn,n=int(len(s)),
    precision=tp/(tp+fp),recall_unweighted=tp/(tp+fn),
    cell_counts={c:dict(n=int(len(gp)),human_EVENT=int((gp.y=='EVENT').sum()))
                 for c,gp in s.assign(cell=np.where(s.A=='EVENT','A-EVENT','A-nonEVENT')).groupby('cell')})
# 위음성 잭나이프: FN 을 0~2배로 두었을 때 사후층화 p_E
N=g.assign(cell=g.block+'|'+np.where(g.isE,'EVENT','non-EVENT')).groupby('cell').size()
def ps_sec(mod_fn):
    tot=0; num=0
    for cell,gp in s.assign(cell='안보|'+np.where(s.A=='EVENT','EVENT','non-EVENT')).groupby('cell'):
        y=(gp.y=='EVENT').astype(float).values.copy()
        if cell.endswith('non-EVENT') and mod_fn is not None:
            idx=np.where(y==1)[0]
            if mod_fn=='drop_one' and len(idx): y[idx[0]]=0
            if mod_fn=='add_one':
                z=np.where(y==0)[0]
                if len(z): y[z[0]]=1
        num+=N[cell]*y.mean(); tot+=N[cell]
    return num/tot
res['security_2x2']['jackknife_pE']=dict(base=float(ps_sec(None)),
    drop_one_FN=float(ps_sec('drop_one')),add_one_FN=float(ps_sec('add_one')))
print('\n== V-B 안보 2x2 ==')
print('TP=%d FP=%d FN=%d TN=%d | 정밀도 %.3f | p_E 잭나이프 base %.4f / FN-1 %.4f / FN+1 %.4f'%(
    tp,fp,fn,tn,tp/(tp+fp),*[res['security_2x2']['jackknife_pE'][k] for k in ['base','drop_one_FN','add_one_FN']]))

# ---------- V-C 비사건 조건부 앵커 인용률 ----------
ANCH={'Q01':['구미'],'Q04':['구미'],'Q08':['구미'],'Q03':['구미'],
      'Q12':['옴진리교','오움진리교','도쿄','동경','구타','고우타'],
      'Q14':['미국','우편','편지'],
      'Q19':['스크리팔','스크립팔','솔즈베리','솔즈버리','나발니','나발리'],
      'Q16':['후쿠시마','체르노빌']}
NAME={'Q01':'hydrogen fluoride','Q04':'sulfuric acid','Q08':'nitric acid','Q03':'chlorine',
      'Q12':'sarin','Q14':'anthrax','Q19':'Novichok','Q16':'radiation leak'}
c=pd.read_csv(BASE+'/data/corpus_bigkinds.csv.gz',dtype=str,usecols=['qid','뉴스 식별자','제목','본문','키워드'])
c=c[c.qid.isin(ANCH)].copy()
c['t']=c['제목'].fillna('')+' '+c['본문'].fillna('')+' '+c['키워드'].fillna('')
gg=g.copy(); gg['t']=gg['제목'].fillna('')+' '+gg['본문'].fillna('')
res['anchor_conditional']={}
print('\n== V-C 비사건 조건부 앵커 인용률 (골드셋) ==')
for q,toks in ANCH.items():
    pat='|'.join(map(re.escape,toks))
    sub=gg[gg.qid==q]
    if len(sub)==0: continue
    hit=sub['t'].str.contains(pat,na=False)
    nonev=(sub.label!='EVENT')
    corp=c[c.qid==q]; rho_all=float(corp['t'].str.contains(pat,na=False).mean())
    r=dict(query=NAME[q],n_gold=int(len(sub)),rho_corpus_all=rho_all,
           rho_gold_all=float(hit.mean()),
           rho_given_nonEVENT=float(hit[nonev].mean()) if nonev.sum() else None,
           rho_given_EVENT=float(hit[~nonev].mean()) if (~nonev).sum() else None,
           n_nonEVENT=int(nonev.sum()))
    res['anchor_conditional'][q]=r
    print('  %-20s rho(전체)=%.3f  rho(비사건 조건부)=%s  n_비사건=%d'%(
        NAME[q],r['rho_gold_all'],('%.3f'%r['rho_given_nonEVENT']) if r['rho_given_nonEVENT'] is not None else '-',r['n_nonEVENT']))
tgt={NAME[q]:res['anchor_conditional'][q]['rho_given_nonEVENT'] for q in ['Q19','Q16','Q12','Q14','Q01'] if q in res['anchor_conditional']}
ctl={NAME[q]:res['anchor_conditional'][q]['rho_given_nonEVENT'] for q in ['Q03','Q04','Q08'] if q in res['anchor_conditional']}
n_ctl=sum(res['anchor_conditional'][q]['n_nonEVENT'] for q in ['Q03','Q04','Q08'] if q in res['anchor_conditional'])
hits_ctl=sum(round(res['anchor_conditional'][q]['rho_given_nonEVENT']*res['anchor_conditional'][q]['n_nonEVENT'])
             for q in ['Q03','Q04','Q08'] if q in res['anchor_conditional'])
# 대조군이 0/n 이면 rule of three 로 단측 95% 상한을 준다
upper=3.0/n_ctl if hits_ctl==0 else None
res['anchor_conditional']['_contrast_nonEVENT']=dict(
    targets=tgt,controls=ctl,control_hits=int(hits_ctl),control_n=int(n_ctl),
    control_upper95_rule_of_three=(float(upper) if upper else None),
    min_ratio_vs_control_upper=(float(min(tgt.values())/upper) if upper else None))
print('  비사건 조건부 — 표적 %s'%{k:round(v,3) for k,v in tgt.items()})
print('  비사건 조건부 — 대조 %d/%d 건, 단측 95%% 상한 %.4f → 최소 배율 %.1f'%(
    hits_ctl,n_ctl,upper if upper else float("nan"),
    (min(tgt.values())/upper) if upper else float("nan")))

# ---------- V-D 산업 대 비CBRN 등가성 ----------
q_pE=(g.groupby(['qid','block']).apply(lambda s:(s.label=='EVENT').mean(),include_groups=False)
      .rename('pE').reset_index())
a=q_pE[q_pE.block=='산업'].pE.values; b=q_pE[q_pE.block=='비CBRN'].pE.values
diff=a.mean()-b.mean()
bs=np.array([RNG.choice(a,len(a)).mean()-RNG.choice(b,len(b)).mean() for _ in range(NB)])
ci=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]
# 검출 가능 효과크기 (양측 alpha .05, power .80, n=10+10, Mann-Whitney 근사)
from math import sqrt
za,zb=st.norm.ppf(0.975),st.norm.ppf(0.80)
sd=np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2)
mde=(za+zb)*sd*sqrt(1/len(a)+1/len(b))
res['control_equivalence']=dict(mean_industrial=float(a.mean()),mean_conventional=float(b.mean()),
    difference=float(diff),ci95=ci,sd_pooled=float(sd),mde_80power=float(mde),
    interpretation='n=10+10 에서 검출 가능한 최소 차이')
print('\n== V-D 대조군 등가성 ==\n  산업 %.3f vs 비CBRN %.3f  차이 %.3f [%.3f, %.3f]  검출가능 최소차이(80%%) %.3f'%(
    a.mean(),b.mean(),diff,ci[0],ci[1],mde))

# ---------- V-E 등록부 감쇠 ----------
rA=json.load(open(OUT+'/revision_A.json'))['registry']
tab=pd.DataFrame(rA['table'])
x=np.log(tab.incidents.astype(float).values); y=np.log(tab.articles.astype(float).values)
b_obs=float(np.polyfit(x,y,1)[0]); var_x=float(x.var(ddof=1))
ci=rA['direct']['ci95']
# 고전적 감쇠: b_obs = lambda * b_true,  lambda = var_x/(var_x+var_u) in (0,1]
lam_point=b_obs/1.0
lam_upper=ci[1]/1.0
def var_u(lam): return var_x*(1-lam)/lam if 0<lam<=1 else None
res['registry_attenuation']=dict(
  slope_obs=b_obs, ci95=ci, var_log_incidents=var_x,
  lambda_needed_point=lam_point,
  lambda_needed_at_ci_upper=(lam_upper if 0<lam_upper<=1 else None),
  var_u_at_ci_upper=(var_u(lam_upper) if 0<lam_upper<=1 else None),
  error_share_at_ci_upper=((1-lam_upper) if 0<lam_upper<=1 else None),
  note=('점추정치는 음수이므로 고전적 감쇠(참 기울기 1 을 0 쪽으로만 축소)로는 재현되지 않는다. '
        '다만 신뢰구간 상한 %.2f 는 신뢰도 lambda=%.2f, 즉 log 사고건수 분산의 %.0f%% 가 '
        '측정오차일 때의 참 기울기 1 과 양립한다.')%(ci[1],lam_upper,100*(1-lam_upper)) if 0<lam_upper<=1 else 'n/e')
res['registry_severity_adjusted']=rA.get('severity_adjusted')
print('\n== V-E 등록부 감쇠 ==')
print('  관측 기울기 %.4f (CI %.2f~%.2f), log 사고건수 분산 %.3f'%(b_obs,ci[0],ci[1],var_x))
print('  %s'%res['registry_attenuation']['note'])
print('  심각도 보정(기존 산출): %s'%res['registry_severity_adjusted'])

json.dump(res,open(OUT+'/v9_additions.json','w'),indent=1,ensure_ascii=False,default=float)
print('\nsaved Step5r/v9_additions.json')
