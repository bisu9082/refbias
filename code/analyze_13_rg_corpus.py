#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refbias — 인간 검증 반영 재추정 (analyze_13_rg_corpus.py)
=========================================================
인간 400건 검증에서 얻은 블록별 TPR/FPR 로 Rogan-Gladen 오분류 보정을 적용한다.
 R1 골드셋 수준 p_E 보정
 R2 코퍼스 수준(사후층화) p_E 보정
 R3 증폭계수 A = m / p_E 재계산 (m 은 코더A 값, 하한 성질 유지)
 R4 불일치 구조 집계 (안보 블록 코더A EVENT 위양성의 질의별 분포)
CI: TPR/FPR 은 층 내 복원 재표집, p_obs 는 기존 CI 를 정규근사한 난수로
    동시 부트스트랩(10,000회)하여 전파한다.
출력: Step5r/rg_corrected.json, Step5r/rg_corrected_table.tex
"""
import os, json
import numpy as np, pandas as pd
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HD=BASE+'/Step4/human_labeling'; OUT=BASE+'/Step5r'
RNG=np.random.default_rng(20260909); NBOOT=10000
CATS=['EVENT','REF','INST','OTHER','IRRELEVANT']
BL={'산업':'industrial','안보':'security','비CBRN':'conventional'}
ALIAS={'event':'EVENT','ref':'REF','inst':'INST','other':'OTHER','irrelevant':'IRRELEVANT',
       '사건':'EVENT','참조':'REF','제도':'INST','기타':'OTHER','무관':'IRRELEVANT'}
def norm(x):
    if pd.isna(x): return None
    s=str(x).strip(); return ALIAS.get(s.lower(), s.upper() if s.upper() in CATS else None)

g=pd.read_csv(BASE+'/data/goldset_labeled_A.csv',dtype=str); g['isE']=(g.label=='EVENT')
H={t:pd.read_excel(f'{HD}/human_labels_{t}.xlsx',dtype=str)[['sid','label']].assign(
     label=lambda x:x.label.map(norm)) for t in ['H1','H2']}
d=(H['H1'].rename(columns={'label':'h1'}).merge(H['H2'].rename(columns={'label':'h2'}),on='sid')
   .merge(g[['sid','block','qid','query','label']].rename(columns={'label':'A'}),on='sid'))
# 포함확률은 실제 표집(400건 전체) 기준으로 계산한다. 합의 여부로 재계산하면 설계가중이 왜곡된다.
strata=g.assign(cell=g.block+'|'+np.where(g.isE,'EVENT','non-EVENT')).groupby('cell').size()
nsamp=d.assign(cell=d.block+'|'+np.where(d.A=='EVENT','EVENT','non-EVENT')).groupby('cell').size()
d=d[d.h1==d.h2].copy(); d['cons']=d.h1
pi=(nsamp/strata).clip(upper=1.0)
d['cell']=d.block+'|'+np.where(d.A=='EVENT','EVENT','non-EVENT'); d['w']=d.cell.map(1/pi)

A_JSON=json.load(open(OUT+'/revision_A.json'))
GOLD={b:dict(p=v['pE'],ci=v['ci95_stratified']) for b,v in A_JSON['pE_block'].items() if b!='all'}
CORP={b:dict(p=v['pE'],ci=v['ci95']) for b,v in A_JSON['poststrat_corpus_pE'].items() if b in BL.values()}
M_CODER={'industrial':1.75,'security':1.12,'conventional':1.19}   # 코더A 기준 기사/사건 (하한)

def boot_rates(sub,nboot=NBOOT):
    """층 내 복원 재표집으로 TPR/FPR 부트스트랩 표본 생성"""
    cells=[(gp.w.values,(gp.cons=='EVENT').values,(gp.A=='EVENT').values) for _,gp in sub.groupby('cell')]
    tn=np.zeros(nboot); td=np.zeros(nboot); fn=np.zeros(nboot); fd=np.zeros(nboot)
    for w,hu,a in cells:
        n=len(w)
        if n==0: continue
        idx=RNG.integers(0,n,size=(nboot,n)); W=w[idx]; HU=hu[idx]; AA=a[idx]
        tn+=(W*(AA&HU)).sum(1); td+=(W*HU).sum(1)
        fn+=(W*(AA&~HU)).sum(1); fd+=(W*~HU).sum(1)
    return np.divide(tn,np.maximum(td,1e-9)), np.divide(fn,np.maximum(fd,1e-9))

def rg(p_obs,ci,tpr_b,fpr_b):
    sd=max((ci[1]-ci[0])/3.92,1e-6)
    p_b=np.clip(RNG.normal(p_obs,sd,len(tpr_b)),0,1)
    den=tpr_b-fpr_b; ok=den>0.05
    out=np.clip((p_b[ok]-fpr_b[ok])/den[ok],0,1)
    return None,[float(np.percentile(out,2.5)),float(np.percentile(out,97.5))],float(ok.mean())

res={'note_A':'증폭계수 A 는 인간 IPW 골드셋 p_E 를 분모로 한다(코퍼스 RG 보정은 분모 불안정으로 A 산출에 쓰지 않음).',
     'note':'TPR/FPR = 인간 합의 라벨(360건) 기준 코더A 성능, IPW. RG 보정 = (p_obs-FPR)/(TPR-FPR).',
     'n_consensus':int(len(d)),'blocks':{}}
print('%-13s %-28s %-28s %-22s'%('block','p_E gold obs → RG','p_E corpus obs → RG','A corpus (RG)'))
for kb,b in BL.items():
    sub=d[d.block==kb]
    tpr_b,fpr_b=boot_rates(sub)
    # 점추정치는 표본 자체에서(부트스트랩 평균이 아니라) 계산한다
    _w=sub.w.values; _hu=(sub.cons=='EVENT').values; _a=(sub.A=='EVENT').values
    tpr=float((_w*(_a&_hu)).sum()/max((_w*_hu).sum(),1e-9))
    fpr=float((_w*(_a&~_hu)).sum()/max((_w*~_hu).sum(),1e-9))
    _,gci,gok=rg(GOLD[b]['p'],GOLD[b]['ci'],tpr_b,fpr_b)
    _,cci,cok=rg(CORP[b]['p'],CORP[b]['ci'],tpr_b,fpr_b)
    pointrg=lambda po: float(np.clip((po-fpr)/max(tpr-fpr,1e-9),0,1))
    gp=pointrg(GOLD[b]['p']); cp=pointrg(CORP[b]['p'])
    m=M_CODER[b]
    # 증폭계수는 인간 직접추정(골드셋 IPW)을 주 추정치로 쓴다.
    # 코퍼스 RG 보정은 p_obs 가 FPR 에 근접하면 분모가 0 에 붙어 A 가 발산하므로 보고하지 않는다.
    hv=json.load(open(OUT+'/human_validation.json'))['pE_human'][b]
    ph=hv['pE_human_ipw']; pci=hv['ci95']
    A=m/ph if ph>0 else None
    Aci=[m/pci[1],m/pci[0]] if pci[0] and pci[0]>0 else None
    corpus_A_stable = (cci[0] > 0.02)
    res['blocks'][b]=dict(
        n_human=int(len(sub)),TPR=tpr,FPR=fpr,
        TPR_ci=[float(np.percentile(tpr_b,2.5)),float(np.percentile(tpr_b,97.5))],
        FPR_ci=[float(np.percentile(fpr_b,2.5)),float(np.percentile(fpr_b,97.5))],
        gold=dict(pE_obs=GOLD[b]['p'],ci_obs=GOLD[b]['ci'],pE_rg=gp,ci_rg=gci,boot_valid_frac=gok),
        corpus=dict(V=A_JSON['poststrat_corpus_pE'][b]['V'],pE_obs=CORP[b]['p'],ci_obs=CORP[b]['ci'],
                    pE_rg=cp,ci_rg=cci,boot_valid_frac=cok),
        amplification=dict(basis='human IPW gold-set p_E',m_coderA=m,pE_human=ph,ci95_pE_human=pci,
                           A=A,ci95=Aci,corpus_rg_usable=bool(corpus_A_stable)))
    print('%-13s %.3f → %.3f [%.3f,%.3f]   %.3f → %.3f [%.3f,%.3f]   A(human)=%.1f [%.1f, %.1f]  corpusRG usable=%s'%(
        b,GOLD[b]['p'],gp,gci[0],gci[1],CORP[b]['p'],cp,cci[0],cci[1],A,Aci[0],Aci[1],corpus_A_stable))

# R4 안보 위양성 구조
sec=d[d.block=='안보']; fp=sec[(sec.A=='EVENT')&(sec.cons!='EVENT')]
res['security_false_positive_structure']=dict(
    n_A_EVENT=int((sec.A=='EVENT').sum()),n_false_positive=int(len(fp)),
    by_human_label={k:int(v) for k,v in fp.cons.value_counts().items()},
    by_query={k:int(v) for k,v in fp['query'].value_counts().items()})
fn=sec[(sec.A!='EVENT')&(sec.cons=='EVENT')]
res['security_false_negative']=dict(n=int(len(fn)),by_A_label={k:int(v) for k,v in fn.A.value_counts().items()})
print('\n안보 위양성 %d/%d (인간 라벨: %s)'%(len(fp),(sec.A=='EVENT').sum(),res['security_false_positive_structure']['by_human_label']))
print('질의별:',res['security_false_positive_structure']['by_query'])

json.dump(res,open(OUT+'/rg_corrected.json','w'),indent=1,ensure_ascii=False,default=float)

rows=[]
for b in ['industrial','security','conventional']:
    v=res['blocks'][b]
    rows.append('%s & %.2f & %.2f & %.1f & %.1f [%.1f, %.1f] & %.1f & %.1f [%.1f, %.1f] \\\\'%(
        b.capitalize(),v['TPR'],v['FPR'],
        100*v['gold']['pE_obs'],100*v['gold']['pE_rg'],100*v['gold']['ci_rg'][0],100*v['gold']['ci_rg'][1],
        100*v['corpus']['pE_obs'],100*v['corpus']['pE_rg'],100*v['corpus']['ci_rg'][0],100*v['corpus']['ci_rg'][1]))
tex=[r'\begin{table}[!htb]',r'\centering\small',
 r'\caption{Misclassification-corrected event shares. TPR and FPR are coder A''s sensitivity and false-positive rate against the human reference standard; corrected shares apply the Rogan--Gladen estimator with uncertainty in both the rates and the observed share propagated by bootstrap (10,000 resamples).}',
 r'\label{tab:rg}',r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrlrl@{}}',r'\toprule',
 r'Block & TPR & FPR & \multicolumn{2}{c}{Gold set $p_E$ (\%)} & \multicolumn{2}{c}{Corpus $p_E$ (\%)} \\',
 r'\cmidrule(lr){4-5}\cmidrule(lr){6-7}',
 r' & & & observed & corrected & observed & corrected \\',r'\midrule']+rows+[r'\bottomrule',r'\end{tabular*}',r'\end{table}']
open(OUT+'/rg_corrected_table.tex','w').write('\n'.join(tex)+'\n')
print('\nsaved: Step5r/rg_corrected.json, Step5r/rg_corrected_table.tex')
