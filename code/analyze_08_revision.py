#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refbias — Step 5 롤백 재분석 A (analyze_08_revision.py)
=========================================================
R1 등록부 대조: log(articles) ~ log(incidents) 직접 회귀 + 기울기 95% CI + Spearman + 심각도 공변량
R2 블록 비교를 p_E(사건성 비율)로: 질의수준 Kruskal-Wallis / Mann-Whitney + Holm + rank-biserial r
R3 분류기 검정: 시드 단위(n=10) 대응 비교 + Nadeau-Bengio 보정 t (50겹)
R4 코퍼스 사건 비율 사후층화(post-stratification) 추정: p_E(corpus) = Σ V_q p̂_q / Σ V_q  (층화 부트스트랩 CI)
R5 층화(질의×연도) 부트스트랩 p_E CI
R6 질의별 표집비율 n/V (m 편향 진단)
R7 Fig.1c 연율화 (drone attack)
R8 코더 A-B 불일치 대칭 계수
R9 질의 제외 민감도 (Q15 DPRK 핵실험 / Q10 화학사고 대응 / Q30 원전 방사선 피폭 / Q23 드론 공격)
"""
import os, json, itertools
import numpy as np, pandas as pd
from scipy import stats as st
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT=BASE+'/Step5r'; os.makedirs(OUT,exist_ok=True)
RNG=np.random.default_rng(20260908); NBOOT=10000
res={}

g=pd.read_csv(BASE+'/data/goldset_labeled_A.csv',dtype=str)
g['isE']=(g['label']=='EVENT').astype(int)
Q=pd.read_csv(BASE+'/data/queries_v2.csv',dtype=str)
corpus=pd.read_csv(BASE+'/data/corpus_bigkinds.csv.gz',dtype=str,usecols=['qid','block','일자','뉴스 식별자'])
V=corpus.groupby('qid').size().rename('V')
BL={'산업':'industrial','안보':'security','비CBRN':'conventional'}

# ---------------- R1 registry ----------------
cm=pd.read_csv(BASE+'/data/caris_match.csv',dtype=str)
cm=cm[cm['대조기준'].str.startswith('mat:')].copy()
cm['a']=cm['보도기사수'].astype(float); cm['n']=cm['공식사건수'].astype(float)
cm['cas']=cm['사망'].astype(float)+cm['부상'].astype(float)
x=np.log(cm['n']); y=np.log(cm['a'])
lr=st.linregress(x,y); nn=len(x); tcrit=st.t.ppf(0.975,nn-2)
ratio=st.linregress(x,np.log(cm['a']/cm['n']))
sp=st.spearmanr(cm['n'],cm['a']); ke=st.kendalltau(cm['n'],cm['a'])
# severity-adjusted: log a ~ log n + log(1+casualties)
X=np.column_stack([np.ones(nn),x,np.log1p(cm['cas'])])
beta,_,_,_=np.linalg.lstsq(X,y,rcond=None)
res['registry']=dict(n_substances=nn,
    direct=dict(slope=lr.slope,ci95=[lr.slope-tcrit*lr.stderr,lr.slope+tcrit*lr.stderr],r=lr.rvalue,p=lr.pvalue),
    ratio_form=dict(slope=ratio.slope,r=ratio.rvalue,note='spurious: slope = direct slope - 1'),
    spearman=dict(rho=sp.statistic,p=sp.pvalue), kendall=dict(tau=ke.statistic,p=ke.pvalue),
    severity_adjusted=dict(slope_incidents=beta[1],slope_log1p_casualties=beta[2]),
    table=cm[['qid','query','a','n','cas']].rename(columns={'a':'articles','n':'incidents','cas':'casualties'}).to_dict('records'))
print('R1 direct slope %.3f [%.2f, %.2f] r=%.3f p=%.3f | ratio slope %.3f r=%.3f | spearman %.3f p=%.2f'%(
    lr.slope,*res['registry']['direct']['ci95'],lr.rvalue,lr.pvalue,ratio.slope,ratio.rvalue,sp.statistic,sp.pvalue))

# ---------------- R2 p_E-based block tests ----------------
def holm(ps):
    idx=np.argsort(ps); m=len(ps); adj=np.empty(m); run=0
    for k,i in enumerate(idx):
        run=max(run,ps[i]*(m-k)); adj[i]=min(1,run)
    return adj
def query_level(df):
    q=df.groupby(['qid','block']).agg(n=('isE','size'),pE=('isE','mean')).reset_index()
    q['invpE']=1/q['pE'].replace(0,np.nan)
    return q
def block_tests(q,col='pE'):
    grp={b:q.loc[q.block==b,col].dropna().values for b in BL}
    kw=st.kruskal(*grp.values()); k=3; n=sum(len(v) for v in grp.values())
    eta2=(kw.statistic-k+1)/(n-k)
    pairs=[]; ps=[]
    for a,b in itertools.combinations(BL,2):
        u=st.mannwhitneyu(grp[a],grp[b],alternative='two-sided')
        r=1-2*u.statistic/(len(grp[a])*len(grp[b]))
        pairs.append(dict(pair=f'{BL[a]} vs {BL[b]}',U=u.statistic,p=u.pvalue,rank_biserial_r=r,
                          median_a=float(np.median(grp[a])),median_b=float(np.median(grp[b]))))
        ps.append(u.pvalue)
    adj=holm(np.array(ps))
    for pr,pa in zip(pairs,adj): pr['p_holm']=pa
    sw={BL[b]:st.shapiro(v).pvalue for b,v in grp.items() if len(v)>=3}
    return dict(kruskal_H=kw.statistic,kruskal_p=kw.pvalue,eta2=eta2,pairs=pairs,shapiro_p=sw,
                medians={BL[b]:float(np.median(v)) for b,v in grp.items()})
q_all=query_level(g)
res['block_tests_pE']=block_tests(q_all,'pE')
res['block_tests_invpE']=block_tests(q_all,'invpE')
print('R2 p_E KW H=%.2f p=%.4f eta2=%.2f'%(res['block_tests_pE']['kruskal_H'],res['block_tests_pE']['kruskal_p'],res['block_tests_pE']['eta2']))
for pr in res['block_tests_pE']['pairs']: print('   ',pr['pair'],'U=%.1f p=%.4f holm=%.4f r=%.2f'%(pr['U'],pr['p'],pr['p_holm'],pr['rank_biserial_r']))

# ---------------- R5 stratified bootstrap of block p_E ----------------
def strat_boot_pE(df):
    strata=[grp['isE'].values for _,grp in df.groupby(['qid','year'])]
    n=len(df); out=np.empty(NBOOT)
    for b in range(NBOOT):
        s=0
        for arr in strata:
            s+=arr[RNG.integers(0,len(arr),len(arr))].sum()
        out[b]=s/n
    return np.percentile(out,[2.5,97.5])
res['pE_block']={}
for b in list(BL)+['all']:
    sub=g if b=='all' else g[g.block==b]
    lo,hi=strat_boot_pE(sub)
    res['pE_block'][BL.get(b,'all')]=dict(n=len(sub),EVENT=int(sub.isE.sum()),pE=sub.isE.mean(),ci95_stratified=[lo,hi])
print('R5',{k:(round(v['pE'],3),[round(x,3) for x in v['ci95_stratified']]) for k,v in res['pE_block'].items()})

# ---------------- R6 sampling fraction ----------------
sf=q_all.merge(V,left_on='qid',right_index=True)
sf['f']=sf['n']/sf['V']
sf=sf.merge(Q[['qid','query']],on='qid')
res['sampling_fraction']=sf[['qid','query','block','n','V','f','pE']].sort_values('f').to_dict('records')
print('R6 f range %.4f – %.2f; queries with f<0.05: %d'%(sf.f.min(),sf.f.max(),(sf.f<0.05).sum()))

# ---------------- R4 post-stratified corpus p_E ----------------
gy=g.groupby(['qid','year']).agg(n=('isE','size'),e=('isE','sum')).reset_index()
corpus['year']=corpus['일자'].str[:4]
Vqy=corpus.groupby(['qid','year']).size().rename('Vqy').reset_index()
ps_=gy.merge(Vqy,on=['qid','year'],how='right')
ps_['n']=ps_['n'].fillna(0); ps_['e']=ps_['e'].fillna(0)
# strata with no gold sample (should be none/few): use query-level pE
qpE=g.groupby('qid').isE.mean()
ps_['p']=np.where(ps_['n']>0,ps_['e']/ps_['n'].replace(0,np.nan),ps_['qid'].map(qpE))
ps_=ps_.merge(Q[['qid','block']],on='qid')
def poststrat(df,boot=True):
    p=(df['p']*df['Vqy']).sum()/df['Vqy'].sum()
    if not boot: return p
    outs=np.empty(2000)
    for b in range(2000):
        pb=np.where(df['n']>0,RNG.binomial(df['n'].astype(int),df['p'].clip(0,1))/df['n'].replace(0,np.nan),df['p'])
        outs[b]=(pb*df['Vqy']).sum()/df['Vqy'].sum()
    return p,np.percentile(outs,[2.5,97.5])
res['poststrat_corpus_pE']={}
for b in list(BL)+['all']:
    sub=ps_ if b=='all' else ps_[ps_.block==b]
    p,ci=poststrat(sub)
    res['poststrat_corpus_pE'][BL.get(b,'all')]=dict(V=int(sub.Vqy.sum()),pE=p,ci95=list(ci),
        strata=int(len(sub)),strata_without_gold=int((sub.n==0).sum()))
    print('R4 %-13s V=%7d pE=%.4f [%.4f, %.4f]'%(BL.get(b,'all'),sub.Vqy.sum(),p,*ci))
# excluding DPRK (Q15) for security block
sub=ps_[(ps_.block=='안보')&(ps_.qid!='Q15')]; p,ci=poststrat(sub)
res['poststrat_corpus_pE']['security_excl_Q15']=dict(V=int(sub.Vqy.sum()),pE=p,ci95=list(ci))
sub=ps_[(ps_.block=='비CBRN')&(ps_.qid!='Q23')]; p,ci=poststrat(sub)
res['poststrat_corpus_pE']['conventional_excl_Q23']=dict(V=int(sub.Vqy.sum()),pE=p,ci95=list(ci))

# ---------------- R3 classifier tests ----------------
raw=pd.read_csv(BASE+'/Step4/outputs/raw_results.csv')
def seed_level(m1,m2,metric):
    a=raw[raw.model==m1].groupby('seed')[metric].mean(); b=raw[raw.model==m2].groupby('seed')[metric].mean()
    d=(a-b).reindex(a.index); sw=st.shapiro(d).pvalue
    tt=st.ttest_rel(a,b); wx=st.wilcoxon(a,b) if len(d)>=6 else None
    return dict(n_seeds=len(d),delta_mean=d.mean(),delta_ci95=[d.mean()-st.t.ppf(0.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d)),
                d.mean()+st.t.ppf(0.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))],shapiro_p=sw,t=tt.statistic,p_t=tt.pvalue,
                p_wilcoxon=(wx.pvalue if wx else None),dz=d.mean()/d.std(ddof=1))
def nadeau_bengio(m1,m2,metric,k=5,r=10):
    a=raw[raw.model==m1].set_index(['seed','fold'])[metric]; b=raw[raw.model==m2].set_index(['seed','fold'])[metric]
    d=(a-b).dropna().values; n=len(d); n2=1/k; n1=1-n2
    var=d.var(ddof=1); t=d.mean()/np.sqrt((1/n+n2/n1)*var); p=2*st.t.sf(abs(t),n-1)
    return dict(n_folds=n,delta_mean=d.mean(),t_corrected=t,p_corrected=p)
comps={'proposed_vs_keyword_eventF1':('proposed_char_tb','base_keyword','event_f1'),
       'proposed_vs_keyword_macroF1':('proposed_char_tb','base_keyword','macro_f1'),
       'proposed_vs_word_macroF1':('proposed_char_tb','abl_word_tb','macro_f1'),
       'proposed_vs_title_macroF1':('proposed_char_tb','abl_char_title','macro_f1'),
       'proposed_vs_bktag_macroF1':('proposed_char_tb','abl_char_tb_bk','macro_f1'),
       'proposed_vs_bktag_eventF1':('proposed_char_tb','abl_char_tb_bk','event_f1')}
res['classifier_tests']={}
for k,(m1,m2,met) in comps.items():
    res['classifier_tests'][k]=dict(seed_level=seed_level(m1,m2,met),nadeau_bengio=nadeau_bengio(m1,m2,met))
    s=res['classifier_tests'][k]['seed_level']; nb=res['classifier_tests'][k]['nadeau_bengio']
    print('R3 %-28s Δ=%+.4f [%+.4f,%+.4f] seed-t p=%.2e wilcoxon p=%s | NB t=%.2f p=%.2e'%(k,s['delta_mean'],*s['delta_ci95'],s['p_t'],
          ('%.4f'%s['p_wilcoxon']) if s['p_wilcoxon'] else '-',nb['t_corrected'],nb['p_corrected']))
# per-block rows for Table 2
res['per_block_model']={m:dict(macro_f1=float(raw[raw.model==m].macro_f1.mean()),macro_f1_sd=float(raw[raw.model==m].macro_f1.std()),
                               event_f1=float(raw[raw.model==m].event_f1.mean()),event_f1_sd=float(raw[raw.model==m].event_f1.std()))
                        for m in ['proposed@산업','proposed@안보','proposed@비CBRN']}

# ---------------- R7 drone attack annualised ----------------
dr=corpus[corpus.qid=='Q23'].groupby('year').size()
months={'2021':(12-9)+1-(8/30),'2022':12,'2023':12,'2024':12,'2025':12,'2026':8+8/30}  # 9 Sep 2021 .. 8 Sep 2026
res['drone_yearly']={y:dict(articles=int(dr.get(y,0)),months=round(months[y],2),per_month=float(dr.get(y,0))/months[y]) for y in months}
print('R7',{y:(v['articles'],round(v['per_month'])) for y,v in res['drone_yearly'].items()})
# all queries yearly for annualised Fig1c
res['corpus_yearly_by_query']={q:{y:int(v) for y,v in grp.groupby('year').size().items()} for q,grp in corpus.groupby('qid')}

# ---------------- R8 A-B disagreement symmetric ----------------
B=pd.concat([pd.read_csv(BASE+f'/Step4/labels_B/B_0{i}.csv',dtype=str) for i in (1,2,3)])
ab=g[['sid','label']].merge(B[['sid','label']],on='sid',suffixes=('_A','_B'))
dis=ab[ab.label_A!=ab.label_B]
pairs=dis.apply(lambda r:' / '.join(sorted([r.label_A,r.label_B])),axis=1).value_counts()
res['disagreement_pairs_symmetric']={'n_double_coded':len(ab),'n_disagree':len(dis),'pairs':pairs.to_dict()}
print('R8',len(ab),len(dis),pairs.head(4).to_dict())

# ---------------- R9 query-exclusion sensitivity ----------------
res['exclusion_sensitivity']={}
for name,excl in {'none':[],'excl_Q15_DPRK':['Q15'],'excl_Q10_response':['Q10'],'excl_Q30_reactor':['Q30'],'excl_Q23_drone':['Q23'],
                  'excl_Q15_Q10_Q30_Q23':['Q15','Q10','Q30','Q23']}.items():
    sub=g[~g.qid.isin(excl)]
    bt=block_tests(query_level(sub),'pE')
    res['exclusion_sensitivity'][name]=dict(n=len(sub),pE_by_block={BL[b]:float(sub[sub.block==b].isE.mean()) for b in BL},
        kruskal_p=bt['kruskal_p'],eta2=bt['eta2'],pairs={p['pair']:dict(p_holm=p['p_holm'],r=p['rank_biserial_r']) for p in bt['pairs']})
    print('R9 %-22s'%name,{k:round(v,3) for k,v in res['exclusion_sensitivity'][name]['pE_by_block'].items()},'KW p=%.4f'%bt['kruskal_p'])

def conv(o):
    if isinstance(o,(np.floating,)): return float(o)
    if isinstance(o,(np.integer,)): return int(o)
    if isinstance(o,np.ndarray): return o.tolist()
    raise TypeError(str(type(o)))
json.dump(res,open(OUT+'/revision_A.json','w'),indent=1,ensure_ascii=False,default=conv)
print('saved',OUT+'/revision_A.json')
