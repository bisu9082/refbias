#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refbias — 인간 검증 분석 (analyze_12_human.py)   [Step 8 3차 리뷰어 3인 공통 요구]
====================================================================================
입력: Step4/human_labeling/human_labels_H1.xlsx, _H2.xlsx  (label 열 기입 필요)
      (선택) human_design.csv — 없으면 실제 표집 결과에서 층별 포함확률을 역산

수행
 H1 인간-인간 신뢰도: kappa(5범주/이진), 전체·블록별, 95% CI
 H2 인간-모델 타당도: 인간 vs 코더A, 인간 vs 코더B (kappa, 전체·블록별)
 H3 설계기반(IPW) 재추정: 층 = 블록 x 코더A EVENT 여부, 가중 1/pi
     → 인간 라벨 기준 블록별 p_E 와 95% CI (층 내 부트스트랩)
 H4 코더A 오분류 보정: 인간 기준 민감도/위양성률(블록별, IPW) → Rogan-Gladen 보정 p_E
 H5 증폭계수 A 재계산 (m 은 코더A 값 고정; 하한 성질 유지)
출력: Step5r/human_validation.json, Step5r/human_validation_section.tex (4.7절 붙여넣기용)
"""
import os, json, sys
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HD=BASE+'/Step4/human_labeling'; OUT=BASE+'/Step5r'; os.makedirs(OUT,exist_ok=True)
RNG=np.random.default_rng(20260908); NBOOT=10000
CATS=['EVENT','REF','INST','OTHER','IRRELEVANT']
BL={'산업':'industrial','안보':'security','비CBRN':'conventional'}
ALIAS={'event':'EVENT','ref':'REF','inst':'INST','other':'OTHER','irrelevant':'IRRELEVANT','irrel':'IRRELEVANT',
       '사건':'EVENT','참조':'REF','제도':'INST','기타':'OTHER','무관':'IRRELEVANT'}
res={}

def norm_label(x):
    if pd.isna(x): return None
    s=str(x).strip()
    return ALIAS.get(s.lower(), s.upper() if s.upper() in CATS else None)

def kappa_ci(a,b,labels=None):
    """Cohen's kappa + 점근 95% CI (Fleiss 분산)"""
    a=np.asarray(a); b=np.asarray(b); n=len(a)
    if n==0: return None
    labs=labels or sorted(set(a)|set(b))
    k=cohen_kappa_score(a,b,labels=labs)
    cm=confusion_matrix(a,b,labels=labs)/n
    pi_=cm.sum(1); pj=cm.sum(0); po=np.trace(cm); pe=(pi_*pj).sum()
    A=sum(cm[i,i]*(1-(pi_[i]+pj[i])*(1-k))**2 for i in range(len(labs)))
    B=(1-k)**2*sum(cm[i,j]*(pj[i]+pi_[j])**2 for i in range(len(labs)) for j in range(len(labs)) if i!=j)
    C=(k-pe*(1-k))**2
    var=(A+B-C)/((1-pe)**2*n)
    se=np.sqrt(max(var,0))
    return dict(kappa=float(k),ci95=[float(k-1.96*se),float(k+1.96*se)],n=int(n))

# ---------- 입력 ----------
g=pd.read_csv(BASE+'/data/goldset_labeled_A.csv',dtype=str); g['isE']=(g.label=='EVENT')
try:
    B=pd.concat([pd.read_csv(f'{BASE}/Step4/labels_B/{f}',dtype=str) for f in sorted(os.listdir(BASE+'/Step4/labels_B')) if f.endswith('.csv')])
except Exception: B=pd.DataFrame(columns=['sid','label'])
H={}
for tag,f in [('H1','human_labels_H1.xlsx'),('H2','human_labels_H2.xlsx')]:
    p=HD+'/'+f
    if not os.path.exists(p): sys.exit(f'[중단] {p} 없음')
    d=pd.read_excel(p,dtype=str)[['sid','label']]
    d['label']=d['label'].map(norm_label)
    bad=d.label.isna().sum()
    print(f'{tag}: {len(d)}행, 미기입/인식불가 {bad}행')
    if bad: print(f'   → {tag} 의 label 열을 {CATS} 중 하나로 채우세요 (대소문자·한글 별칭 허용)')
    H[tag]=d.dropna(subset=['label'])
if min(len(H['H1']),len(H['H2']))==0: sys.exit('[중단] 라벨이 하나도 기입되지 않았습니다')

d=(H['H1'].rename(columns={'label':'h1'})
   .merge(H['H2'].rename(columns={'label':'h2'}),on='sid')
   .merge(g[['sid','block','qid','label']].rename(columns={'label':'A'}),on='sid')
   .merge(B[['sid','label']].rename(columns={'label':'Bc'}),on='sid',how='left'))
print('양쪽 모두 기입된 기사:',len(d))
res['n_coded']=int(len(d)); res['by_block']={BL[b]:int(n) for b,n in d.block.value_counts().items()}

# ---------- H1 인간-인간 ----------
def block_kappas(x,y,sub):
    o={'overall_5class':kappa_ci(sub[x],sub[y],CATS),
       'overall_binary':kappa_ci(sub[x]=='EVENT',sub[y]=='EVENT',[False,True])}
    for b,s in sub.groupby('block'):
        if len(s)>=10:
            o[BL[b]+'_5class']=kappa_ci(s[x],s[y],CATS)
            o[BL[b]+'_binary']=kappa_ci(s[x]=='EVENT',s[y]=='EVENT',[False,True])
    return o
res['human_human']=block_kappas('h1','h2',d)
agree=(d.h1==d.h2); res['human_human']['percent_agreement']=float(agree.mean())
print('H1 인간-인간 kappa 5범주 %.3f / 이진 %.3f / 일치율 %.3f'%(
    res['human_human']['overall_5class']['kappa'],res['human_human']['overall_binary']['kappa'],agree.mean()))

# 합의 라벨: 일치 시 그 값, 불일치 시 제외(주 분석) — 민감도로 H1 단독도 보고
d['cons']=np.where(agree,d.h1,None)
res['n_consensus']=int(agree.sum())

# ---------- H2 인간-모델 ----------
res['human_vs_A']=block_kappas('cons','A',d[agree]); res['human_vs_A_H1only']=block_kappas('h1','A',d)
if d.Bc.notna().any():
    sb=d[agree&d.Bc.notna()]
    if len(sb)>=10: res['human_vs_B']=block_kappas('cons','Bc',sb)
print('H2 인간(합의) vs 코더A: 5범주 %.3f / 이진 %.3f'%(
    res['human_vs_A']['overall_5class']['kappa'],res['human_vs_A']['overall_binary']['kappa']))

# ---------- 포함확률(층 = 블록 x 코더A EVENT 여부) ----------
dp=HD+'/human_design.csv'
strata=g.assign(cell=g.block+'|'+np.where(g.isE,'EVENT','non-EVENT')).groupby('cell').size().rename('N_gold')
n_s=d.assign(cell=d.block+'|'+np.where(d.A=='EVENT','EVENT','non-EVENT')).groupby('cell').size().rename('n_samp')
pi=(pd.concat([strata,n_s],axis=1).fillna(0))
pi['pi']=(pi.n_samp/pi.N_gold).clip(upper=1.0)
res['inclusion_prob']=pi.reset_index().to_dict('records')
print('층별 포함확률'); print(pi.to_string())
d['cell']=d.block+'|'+np.where(d.A=='EVENT','EVENT','non-EVENT')
d['w']=d.cell.map((1/pi['pi'].replace(0,np.nan)).to_dict())

# ---------- H3 IPW p_E ----------
def ipw_pE(sub):
    w=sub.w.values; y=(sub.cons=='EVENT').values.astype(float)
    return float((w*y).sum()/w.sum())
def ipw_ci(sub,nboot=NBOOT):
    """층(cell) 내 복원 재표집 부트스트랩 — 벡터화"""
    cells=[(gp.w.values, (gp.cons=='EVENT').values.astype(float)) for _,gp in sub.groupby('cell')]
    num=np.zeros(nboot); den=np.zeros(nboot)
    for w,y in cells:
        n=len(w)
        if n==0: continue
        idx=RNG.integers(0,n,size=(nboot,n))
        num+=(w[idx]*y[idx]).sum(1); den+=w[idx].sum(1)
    outs=num/den
    return [float(np.percentile(outs,2.5)),float(np.percentile(outs,97.5))]
cons=d[agree]
res['pE_human']={}
for b,s in cons.groupby('block'):
    p=ipw_pE(s); ci=ipw_ci(s) if len(s)>=30 else [None,None]
    pa=float((g[g.block==b].label=='EVENT').mean())
    res['pE_human'][BL[b]]=dict(n=int(len(s)),pE_human_ipw=p,ci95=ci,pE_coderA=pa,diff=p-pa)
    print('H3 %-13s n=%3d  p_E(인간,IPW)=%.3f %s  vs 코더A %.3f'%(BL[b],len(s),p,ci,pa))

# ---------- H4 코더A 오분류 보정 ----------
res['coderA_vs_human']={}
for b,s in cons.groupby('block'):
    hu=(s.cons=='EVENT'); a=(s.A=='EVENT'); w=s.w.values
    tpr=float((w*(a&hu)).sum()/max((w*hu).sum(),1e-9)); fpr=float((w*(a&~hu)).sum()/max((w*~hu).sum(),1e-9))
    pobs=float((g[g.block==b].label=='EVENT').mean())
    rg=(pobs-fpr)/(tpr-fpr) if (tpr-fpr)>0.05 else None
    res['coderA_vs_human'][BL[b]]=dict(n_human_EVENT=int(hu.sum()),TPR=tpr,FPR=fpr,pE_obs_coderA=pobs,
                                       pE_rogan_gladen=(None if rg is None else float(np.clip(rg,0,1))))
    print('H4 %-13s TPR=%.2f FPR=%.2f → RG 보정 p_E=%s'%(BL[b],tpr,fpr,'n/e' if rg is None else '%.3f'%np.clip(rg,0,1)))

# ---------- H5 A 재계산 ----------
mvals={'산업':1.75,'안보':1.12,'비CBRN':1.19}
res['A_human']={}
for b,m in mvals.items():
    v=res['pE_human'].get(BL[b])
    if not v or not v['pE_human_ipw']: continue
    res['A_human'][BL[b]]=dict(m_coderA=m,A=m/v['pE_human_ipw'],
        ci95=[m/v['ci95'][1],m/v['ci95'][0]] if v['ci95'][0] else None)
for k,v in res['A_human'].items(): print('H5 %-13s A(인간)=%.2f  %s'%(k,v['A'],('[%.2f, %.2f]'%tuple(v['ci95'])) if v['ci95'] else ''))

json.dump(res,open(OUT+'/human_validation.json','w'),indent=1,ensure_ascii=False,default=float)

# ---------- 4.7절 LaTeX ----------
hh=res['human_human']; ha=res['human_vs_A']
def f(x): return '%.3f'%x
tex=[r"\subsection{Human validation}",r"\label{sec:human}",""]
tex.append("Two authors coded the %d-article subset independently under the same codebook, blind to the model labels. "
 "Agreement between the two human coders is $\\kappa = %s$ (95\\%% CI %s--%s) over the five components and $%s$ for the "
 "EVENT-versus-rest decision, with %.1f\\%% raw agreement; the %d articles on which they agreed form the reference standard below."
 %(res['n_coded'],f(hh['overall_5class']['kappa']),f(hh['overall_5class']['ci95'][0]),f(hh['overall_5class']['ci95'][1]),
   f(hh['overall_binary']['kappa']),100*hh['percent_agreement'],res['n_consensus']))
tex.append("")
tex.append("Against that standard, coder A reaches $\\kappa = %s$ over five components and $%s$ on the EVENT boundary"
 %(f(ha['overall_5class']['kappa']),f(ha['overall_binary']['kappa']))
 +(", by block %s"%', '.join('%s %s'%(k.split('_')[0],f(v['kappa'])) for k,v in ha.items() if k.endswith('_binary') and not k.startswith('overall')) if any(k.endswith('_binary') and not k.startswith('overall') for k in ha) else "")+".")
tex.append("")
rows=[]
for b,v in res['pE_human'].items():
    rg=res['coderA_vs_human'][b]
    rows.append("%s & %d & %.1f & %s & %.1f & %.2f & %.2f & %s \\\\"%(
        b.capitalize(),v['n'],100*v['pE_human_ipw'],
        ('[%.1f, %.1f]'%(100*v['ci95'][0],100*v['ci95'][1])) if v['ci95'][0] else '--',
        100*v['pE_coderA'],rg['TPR'],rg['FPR'],
        ('%.1f'%(100*rg['pE_rogan_gladen'])) if rg['pE_rogan_gladen'] is not None else 'n/e'))
tex += [r"\begin{table}[!htb]",r"\centering\small",
 r"\caption{Human validation of the model labels. $p_E$ (human) is the inverse-probability-weighted event share on the human reference standard; TPR and FPR are coder A's sensitivity and false-positive rate against it; the last column applies the Rogan--Gladen correction to coder A's block event share.}",
 r"\label{tab:human}",r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrlrrrr@{}}",r"\toprule",
 r"Block & $n$ & $p_E$ human (\%) & CI & $p_E$ coder A (\%) & TPR & FPR & $p_E$ corrected (\%) \\",r"\midrule"]+rows+[r"\bottomrule",r"\end{tabular*}",r"\end{table}"]
open(OUT+'/human_validation_section.tex','w').write('\n'.join(tex)+'\n')
print('\nsaved:',OUT+'/human_validation.json,',OUT+'/human_validation_section.tex')
print('→ main.tex 의 [HUMAN-400] 자리에 human_validation_section.tex 내용을 넣으면 됩니다.')
