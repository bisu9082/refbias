import os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch, Rectangle
from matplotlib.ticker import ScalarFormatter, NullFormatter
W='/home/claude/rb'; OUT=W+'/Step4/outputs'; R=W+'/data'; F=W+'/figure/set_v2'; RV=json.load(open(W+'/Step5r/revision_A.json')); RB1=json.load(open(W+'/Step5r/revision_B1.json')); os.makedirs(F,exist_ok=True)
# ── style
CAT={'EVENT':'#B5352A','REF':'#D97B1E','INST':'#1A9484','OTHER':'#3F5FAE','IRRELEVANT':'#9E9E9E'}
BLK={'산업':'#B5352A','안보':'#D97B1E','비CBRN':'#1A9484'}; BEN={'산업':'Industrial','안보':'Security (CBRN)','비CBRN':'Conventional terror'}; BORD=['산업','안보','비CBRN']
INK,INK2,GRID='#1F1F1F','#5A5A5A','#E4E4E4'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':17,'axes.labelsize':19,'axes.titlesize':19,'xtick.labelsize':16,'ytick.labelsize':16,'legend.fontsize':16,
 'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':INK2,'axes.linewidth':1.0,'xtick.color':INK2,'ytick.color':INK2,'axes.labelcolor':INK,'text.color':INK,
 'axes.grid':True,'axes.grid.axis':'y','grid.color':GRID,'grid.linewidth':0.8,'axes.axisbelow':True,'legend.frameon':False,'savefig.facecolor':'white','figure.facecolor':'white'})
def P(ax,l,x=None,y=None,off=None,dy=0.012):
    # 패널 라벨: y축(축선) 바로 왼쪽, 축 상단 위 — 전 패널 동일 규칙
    f=ax.figure; f.canvas.draw(); p=ax.get_position()
    f.text(p.x0-0.010,p.y1+dy,l,fontsize=30,fontweight='bold',ha='right',va='bottom',color=INK)
def save(fig,name): fig.savefig(f'{F}/{name}.png',dpi=300,bbox_inches='tight'); fig.savefig(f'{F}/{name}.pdf',bbox_inches='tight'); plt.close(fig); print('saved',name)
QEN={'Q01':'HF leak','Q02':'NH3 leak','Q03':'Cl2 gas leak','Q04':'H2SO4 spill','Q05':'Chem. plant explosion','Q06':'Hazmat release','Q07':'Toluene leak','Q08':'HNO3 leak','Q09':'Toxic-gas evacuation','Q10':'Chem. accident response',
'Q11':'Chemical weapon use','Q12':'Sarin attack','Q13':'Bioterrorism','Q14':'Anthrax terror','Q15':'DPRK nuclear test','Q16':'Radiation leak','Q17':'Dirty bomb','Q18':'White-powder report','Q19':'Novichok','Q20':'DPRK bio-chem weapons',
'Q21':'Explosive terror','Q22':'IED','Q23':'Drone attack','Q24':'Mass shooting','Q25':'Vehicle ramming','Q26':'Bomb threat','Q27':'Terror-threat post','Q28':'Public-facility terror','Q29':'Airport bomb report','Q30':'Reactor radiation exposure'}
qmeta=pd.read_csv(R+'/queries_v2.csv'); ab=pd.read_csv(R+'/amplification_by_block.csv').set_index('block'); aq=pd.read_csv(R+'/amplification_by_query.csv')
S=json.load(open(OUT+'/experiment_summary.json')); cp=pd.read_csv(OUT+'/corpus_pred_by_query_year.csv')

# ═══════════ FIG 1 · Framework & data ═══════════
fig=plt.figure(figsize=(20,15),dpi=100); gs=fig.add_gridspec(2,2,width_ratios=[1.15,1],height_ratios=[0.78,1.5],hspace=0.30,wspace=0.30)
# (b) corpus by query — 먼저 만들어 좌측 기준선을 얻는다
axb=fig.add_subplot(gs[1,0]); cnt=cp.groupby(['qid','block'])['n'].sum().reset_index().merge(qmeta[['qid']],on='qid')
cnt['block']=pd.Categorical(cnt['block'],BORD); cnt=cnt.sort_values(['block','n'],ascending=[True,True]).reset_index(drop=True)
axb.barh(range(len(cnt)),cnt['n'],color=[BLK[b] for b in cnt['block']],height=0.72)
axb.set_yticks(range(len(cnt))); axb.set_yticklabels([QEN[q] for q in cnt['qid']],fontsize=14); axb.set_xscale('log'); axb.grid(axis='x'); axb.grid(False,axis='y')
axb.set_xlabel('Articles per query, 2021-09-09 to 2026-09-08  (BigKinds, 54 outlets; log scale)',fontsize=18); axb.tick_params(axis='x',labelsize=16)
for i,v in enumerate(cnt['n']): axb.text(v*1.15,i,f'{v:,}',va='center',fontsize=13,color=INK2)
axb.set_xlim(30,3e5); axb.legend(handles=[Patch(color=BLK[b],label=BEN[b]) for b in BORD],loc='lower right',fontsize=16)
# (c) drone growth, annualised (articles per month; 2021 and 2026 are partial years)
axc=fig.add_subplot(gs[1,1]); dy_=RV['drone_yearly']; yrs=list(dy_.keys()); pm=[dy_[y]['per_month'] for y in yrs]
axc.bar(yrs,pm,color=BLK['비CBRN'],width=0.66,edgecolor='white')
for i,(y,v) in enumerate(zip(yrs,pm)): axc.text(i,v+45,f"{v:,.0f}\n({dy_[y]['articles']:,} in {dy_[y]['months']:.0f} mo)" if y in ('2021','2026') else f"{v:,.0f}",ha='center',fontsize=13.5,color=INK)
axc.text(5,pm[-1]+330,f'×{pm[-1]/pm[1]:.1f} vs 2022',ha='center',fontsize=17,fontweight='bold',color=INK)
axc.set_ylabel('Articles per month: "drone attack"',fontsize=18); axc.set_xlabel('Year  (2021: from Sep 9;  2026: to Sep 8)',fontsize=18); axc.set_ylim(0,2900); axc.tick_params(labelsize=16)
fig.canvas.draw(); pb=axb.get_position(); pc=axc.get_position()
LX=pb.x0-0.010                                  # 패널 라벨: y축 바로 왼쪽
fig.text(LX,pb.y1+0.012,'b',fontsize=30,fontweight='bold',ha='right',va='bottom',color=INK)
fig.text(pc.x0-0.010,pc.y1+0.012,'c',fontsize=30,fontweight='bold',ha='right',va='bottom',color=INK)
# (a) schematic — 좌측선을 b와 맞추고 우측 끝까지
top=fig.add_subplot(gs[0,:]); pt=top.get_position(); top.set_position([pb.x0,pt.y0,pc.x1-pb.x0,pt.height]); ax=top
ax.set_axis_off(); ax.set_xlim(0,100); ax.set_ylim(0,10)
fig.text(LX,pt.y1+0.005,'a',fontsize=30,fontweight='bold',ha='right',va='bottom',color=INK)
segs=[('NEW',12,CAT['EVENT']),('FOLLOW',8,'#D9776E'),('REF',5,CAT['REF']),('INST',28,CAT['INST']),('OTHER',24,CAT['OTHER']),('IRRELEVANT',23,CAT['IRRELEVANT'])]
x=0
for n,wd,c in segs:
    ax.add_patch(Rectangle((x,5.6),wd,2.4,facecolor=c,edgecolor='white',lw=2.5,hatch='///' if n=='IRRELEVANT' else None)); ax.text(x+wd/2,6.8,n,ha='center',va='center',color='white',fontsize=17,fontweight='bold'); x+=wd
ax.annotate('',xy=(0,8.75),xytext=(100,8.75),arrowprops=dict(arrowstyle='<->',color=INK2,lw=1.8)); ax.text(50,9.45,'V  =  observed news volume for a query',ha='center',fontsize=19,color=INK)
ax.annotate('',xy=(0,5.1),xytext=(20,5.1),arrowprops=dict(arrowstyle='<->',color=CAT['EVENT'],lw=1.8)); ax.text(10,4.25,'EVENT = NEW + FOLLOW',ha='center',fontsize=16,color=CAT['EVENT'])
ax.text(0,2.5,r'$N_{events}=\dfrac{\mathrm{EVENT}}{m}$    ( $m$ = articles per unique event )',fontsize=20,color=INK,va='center')
ax.text(52,2.5,r'Amplification    $A=\dfrac{V}{N_{events}}=\dfrac{m}{p_E}$',fontsize=20,color=INK,va='center')
ax.text(0,0.5,r'Reference bias    $\rho$  =  share of articles citing a canonical anchor event  (measured independently of the labels)',fontsize=17,color=INK2,va='center')
save(fig,'Fig1_framework_data')

# ═══════════ FIG 2 · Decomposition & amplification ═══════════
fig=plt.figure(figsize=(20,15),dpi=100); gs=fig.add_gridspec(2,2,width_ratios=[1,1],height_ratios=[1,1.35],hspace=0.28,wspace=0.36)
ax=fig.add_subplot(gs[0,0]); b=ab.loc[BORD]; bottom=np.zeros(3); comp={'EVENT':'pE','REF':'REF','INST':'INST','OTHER':'OTHER','IRRELEVANT':'IRR'}
for k,col in comp.items():
    v=b[col].values; bars=ax.bar([BEN[x] for x in BORD],v,bottom=bottom,color=CAT[k],width=0.62,edgecolor='white',lw=2,hatch='///' if k=='IRRELEVANT' else None,label=k)
    for i,(vv,bb) in enumerate(zip(v,bottom)):
        if vv>=6: ax.text(i,bb+vv/2,f'{vv:.0f}',ha='center',va='center',color='white',fontsize=16,fontweight='bold')
    bottom+=v
ax.set_ylabel('Share of articles (%)'); ax.set_ylim(0,100); ax.legend(ncol=5,loc='upper center',bbox_to_anchor=(0.5,1.10),handlelength=1.2,columnspacing=1.0,fontsize=15); P(ax,'a',dy=0.040)
ax=fig.add_subplot(gs[0,1]); A=b['증폭계수'].values
ax.bar([BEN[x] for x in BORD],A,color=[BLK[x] for x in BORD],width=0.62,edgecolor='white'); ax.errorbar(range(3),A,yerr=[A-b['A_lo'],b['A_hi']-A],fmt='none',ecolor=INK,elinewidth=1.6,capsize=7)
for i,(v,hi) in enumerate(zip(A,b['A_hi'])): ax.text(i,hi+0.35,f'{v:.2f}',ha='center',fontsize=17,color=INK)
ax.axhline(1,ls='--',c=INK2,lw=1); ax.text(2.48,1.25,'A = 1 (no amplification)',fontsize=13,color=INK2,ha='right',va='bottom',bbox=dict(fc='white',ec='none',pad=1.5)); ax.set_ylabel('Amplification factor A'); ax.set_ylim(0,15); ax.set_xlim(-0.5,2.5); ax.text(0.02,0.97,'error bars: 95% CI of $p_E$\n(stratified bootstrap; $m$ fixed)',transform=ax.transAxes,ha='left',va='top',fontsize=13,color=INK2); P(ax,'b',dy=0.040)
# (c) strip by block: query-level EVENT share p_E (primary contrast; m depends on sampling fraction)
ax=fig.add_subplot(gs[1,0]); rng=np.random.default_rng(1); g_=pd.read_csv(R+'/goldset_labeled_A.csv',dtype=str); g_['isE']=(g_.label=='EVENT').astype(float)
qpe=g_.groupby(['qid','block']).isE.mean().reset_index()
for i,bk in enumerate(BORD):
    v=qpe.loc[qpe.block==bk,'isE'].values*100; xj=i+rng.uniform(-0.16,0.16,len(v))
    ax.scatter(xj,v,s=110,color=BLK[bk],alpha=0.85,edgecolor='white',lw=1.2,zorder=3); ax.hlines(np.median(v),i-0.3,i+0.3,color=INK,lw=2.5,zorder=4)
ax.set_xticks(range(3)); ax.set_xticklabels([f"{BEN[x].replace(' terror',chr(10)+'terror')}\nmedian {np.median(qpe.loc[qpe.block==x,'isE'])*100:.0f}%" for x in BORD]); ax.set_ylabel('Query-level EVENT share $p_E$ (%)'); ax.set_xlim(-0.6,2.6); ax.set_ylim(-3,132); ax.set_yticks([0,20,40,60,80,100])
T=RV['block_tests_pE']; pr={p['pair']:p for p in T['pairs']}
ax.text(0.98,0.96,f"Kruskal–Wallis H(2) = {T['kruskal_H']:.2f}, p = {T['kruskal_p']:.4f}, η² = {T['eta2']:.2f}\nSecurity vs Industrial  p = {pr['industrial vs security']['p_holm']:.3f}, r = {abs(pr['industrial vs security']['rank_biserial_r']):.2f}\nSecurity vs Conventional  p = {pr['security vs conventional']['p_holm']:.3f}, r = {abs(pr['security vs conventional']['rank_biserial_r']):.2f}\nIndustrial vs Conventional  p = {pr['industrial vs conventional']['p_holm']:.2f}, r = {abs(pr['industrial vs conventional']['rank_biserial_r']):.2f}\n(Holm-adjusted; n = 10 queries per block)",transform=ax.transAxes,va='top',ha='right',fontsize=13.5,color=INK,bbox=dict(boxstyle='round,pad=0.4',fc='white',ec=GRID))
P(ax,'c')
# (d) query-level A sorted
ax=fig.add_subplot(gs[1,1]); q=aq.sort_values('증폭계수').reset_index(drop=True)
ax.barh(range(len(q)),q['증폭계수'],color=[BLK[x] for x in q['block']],height=0.72)
ax.set_yticks(range(len(q))); ax.set_yticklabels([QEN[x] for x in q['qid']],fontsize=13.5); ax.set_xscale('log'); ax.grid(axis='x'); ax.grid(False,axis='y')
for i,(v,e,n) in enumerate(zip(q['증폭계수'],q['EVENT'],q['n'])): ax.text(v*1.12,i,f'{v:.1f}   ({e}/{n} EVENT)',va='center',fontsize=12.5,color=INK2)
ax.set_xlim(1,400); ax.set_xlabel('Amplification A = n / unique events (log)'); ax.axvline(1,color=INK2,lw=1,ls='--'); P(ax,'d')
save(fig,'Fig2_decomposition_amplification')

# ═══════════ FIG 3 · Reference bias & official ground truth ═══════════
fig=plt.figure(figsize=(20,10.5),dpi=100); gs=fig.add_gridspec(1,3,width_ratios=[1.15,1.1,0.95],wspace=0.46)
ax=fig.add_subplot(gs[0,0]); AN=RB1['anchor']
rows=[('Q19','Novichok\n(Salisbury 2018 / Navalny 2020)'),('Q12','Sarin attack\n(Aum / Tokyo 1995 / Ghouta)'),('Q16','Radiation leak\n(Fukushima / Chernobyl)'),('Q14','Anthrax terror\n(2001 US letters)'),('Q01','HF leak\n(2012 Gumi)'),('Q03','Cl2 gas leak'),('Q04','H2SO4 spill'),('Q08','HNO3 leak')]
rows=sorted(rows,key=lambda r:AN[r[0]]['rho_no_year'])
for i,(q,en) in enumerate(rows):
    r=AN[q]; ctrl=q in ('Q03','Q04','Q08'); col=CAT['IRRELEVANT'] if ctrl else CAT['EVENT']
    if 'per_event' in r:  # stacked by event, overlap removed approximately by drawing larger first
        ev=sorted(r['per_event'].items(),key=lambda kv:-kv[1]); tot=r['rho_no_year']
        ax.barh(i,tot*100,color=col,height=0.7,edgecolor='white')
        ax.barh(i,ev[0][1]*100,color=col,height=0.7,edgecolor='white',alpha=1.0)
        ax.barh(i,(tot-ev[0][1])*100,left=ev[0][1]*100,color='#D9776E',height=0.7,edgecolor='white',hatch='\\\\')
        ax.text(ev[0][1]*100/2,i,f"{ev[0][0].split()[0]}\n{ev[0][1]*100:.0f}%",ha='center',va='center',color='white',fontsize=12,fontweight='bold')
    else:
        ax.barh(i,r['rho_no_year']*100,color=col,height=0.7,edgecolor='white',hatch='///' if ctrl else None)
    ax.text(r['rho_no_year']*100+1.5,i,f"{r['rho_no_year']*100:.1f}%  (n={r['V']})",va='center',fontsize=14,color=INK)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([en for _,en in rows],fontsize=14.5); ax.grid(axis='x'); ax.grid(False,axis='y')
ax.set_xlim(0,105); ax.set_xlabel('Articles citing a past anchor (%)')
ax.legend(handles=[Patch(color=CAT['EVENT'],label='CBRN / rare-hazard queries (dominant anchor)'),Patch(facecolor='#D9776E',hatch='\\\\',edgecolor='white',label='other anchor of the same hazard (non-exclusive)'),Patch(facecolor=CAT['IRRELEVANT'],hatch='///',edgecolor='white',label='Same-block controls (Gumi 2012 as anchor)')],loc='lower center',bbox_to_anchor=(0.55,1.0),fontsize=12.5,ncol=1); P(ax,'a',off=0.20,dy=0.075)
# (b) CARIS
ax=fig.add_subplot(gs[0,1]); cm=pd.read_csv(R+'/caris_match.csv'); cm=cm[cm['대조기준'].str.startswith('mat')]
lab={'불산 누출':'HF','암모니아 누출 사고':'NH$_3$','염소 가스 누출':'Cl$_2$','황산 유출 사고':'H$_2$SO$_4$','톨루엔 누출':'Toluene','질산 누출':'HNO$_3$'}
RG=RV['registry']['direct']; xs,ys=np.log10(cm['공식사건수']),np.log10(cm['보도기사수']); sl,ic=np.polyfit(xs,ys,1)
xx=np.linspace(0.9,1.8,50); ax.plot(10**xx,10**(sl*xx+ic),'--',c=INK2,lw=1.6,zorder=1)
ax.scatter(cm['공식사건수'],cm['보도기사수'],s=260,color=CAT['EVENT'],edgecolor='white',lw=1.5,zorder=3)
for _,r in cm.iterrows(): ax.annotate(lab[r['query']],(r['공식사건수'],r['보도기사수']),xytext={'질산 누출':(-16,-22),'황산 유출 사고':(9,-20),'톨루엔 누출':(9,-20)}.get(r['query'],(9,7)),textcoords='offset points',fontsize=17)
ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(True,axis='both')
for a_ in (ax.xaxis,ax.yaxis): a_.set_major_formatter(ScalarFormatter()); a_.set_minor_formatter(NullFormatter())
ax.set_xticks([10,20,30,50]); ax.set_yticks([100,200,300]); ax.set_xlim(8,70); ax.set_ylim(70,400)
ax.set_xlabel('Official incidents (CARIS, 2021-09 to 2025-04)',fontsize=17); ax.set_ylabel('Articles (BigKinds, same window)')
ax.text(0.03,0.05,f"log–log slope = {RG['slope']:+.2f}  [95% CI {RG['ci95'][0]:.2f}, {RG['ci95'][1]:+.2f}]\nr = {RG['r']:.2f}, Spearman ρ = {RV['registry']['spearman']['rho']:.2f}, n = {len(cm)}",transform=ax.transAxes,fontsize=14,color=INK,ha='left',va='bottom',bbox=dict(boxstyle='round,pad=0.4',fc='white',ec=GRID)); P(ax,'b',off=0.075,dy=0.075)
# (c) articles vs incidents per substance (two rows of bars, one axis each? -> use single axis: articles per incident with incident count annotated)
ax=fig.add_subplot(gs[0,2]); c2=cm.sort_values('기사당사건비',ascending=True)
ax.barh(range(len(c2)),c2['기사당사건비'],color=CAT['EVENT'],height=0.66,edgecolor='white')
ax.set_yticks(range(len(c2))); ax.set_yticklabels([lab[q] for q in c2['query']],fontsize=17); ax.grid(axis='x'); ax.grid(False,axis='y')
for i,(v,e,n) in enumerate(zip(c2['기사당사건비'],c2['공식사건수'],c2['보도기사수'])): ax.text(v+0.5,i,f'{v:.1f}\n{n} articles / {e} incidents',va='center',fontsize=12.5,color=INK2)
ax.set_xlim(0,31); ax.set_xlabel('Articles per official incident'); P(ax,'c',off=0.075,dy=0.075)
save(fig,'Fig3_reference_bias_ground_truth')

# ═══════════ FIG 4 · Classifier & validation ═══════════
fig=plt.figure(figsize=(20,13),dpi=100); gs=fig.add_gridspec(2,2,hspace=0.40,wspace=0.32)
ax=fig.add_subplot(gs[0,0]); s=S['summary']
models=[('base_majority','Majority'),('base_random','Random'),('base_keyword','Keyword\nrule'),('abl_char_title','Char\n(title)'),('abl_word_tb','Word\nn-gram'),('abl_char_tb_bk','Char\n+ tag'),('proposed_char_tb','Char\n(proposed)')]
x=np.arange(len(models)); w=0.38
ax.bar(x-w/2,[s[m]['macro_f1'][0] for m,_ in models],w,yerr=[s[m]['macro_f1'][1] for m,_ in models],color=CAT['OTHER'],label='Macro-F1',capsize=3,error_kw=dict(lw=1))
ax.bar(x+w/2,[s[m]['event_f1'][0] for m,_ in models],w,yerr=[s[m]['event_f1'][1] for m,_ in models],color=CAT['EVENT'],label='EVENT-F1',capsize=3,error_kw=dict(lw=1))
for i,(m,_) in enumerate(models): ax.text(i+w/2,s[m]['event_f1'][0]+0.03,f"{s[m]['event_f1'][0]:.2f}",ha='center',fontsize=13.5,color=INK)
ax.set_xticks(x); ax.set_xticklabels([n for _,n in models],fontsize=14); ax.set_ylabel('Score (mean ± SD, 50 folds)'); ax.set_ylim(0,1.05); ax.legend(loc='upper left'); P(ax,'a',x=-0.14)
ax=fig.add_subplot(gs[0,1]); mono=pd.read_csv(OUT+'/monotonicity.csv'); mm=mono.groupby('frac')['macro_f1'].agg(['mean','std'])
ax.fill_between(mm.index*100,mm['mean']-mm['std'],mm['mean']+mm['std'],color=CAT['INST'],alpha=0.18,lw=0)
ax.plot(mm.index*100,mm['mean'],marker='o',ms=10,lw=2.4,color=CAT['INST'],mec='white',mew=1.5)
for xv,yv in zip(mm.index*100,mm['mean']): ax.text(xv,yv+0.012,f'{yv:.3f}',ha='center',fontsize=14.5,color=INK)
ax.set_xticks([25,50,75,100]); ax.set_xlabel('Training data used (%)'); ax.set_ylabel('Macro-F1 (mean ± SD)'); ax.set_ylim(0.5,0.78); P(ax,'b',x=-0.14)
# (c) confusion A vs B
ax=fig.add_subplot(gs[1,0]); A_=pd.concat([pd.read_csv(f) for f in sorted(__import__('glob').glob(W+'/Step4/labels_A/*.csv'))],ignore_index=True)
B_=pd.concat([pd.read_csv(f) for f in sorted(__import__('glob').glob(W+'/Step4/labels_B/*.csv'))],ignore_index=True)
m=A_.merge(B_,on='sid',suffixes=('_A','_B')); C=['EVENT','REF','INST','OTHER','IRRELEVANT']
ct=pd.crosstab(m['label_A'].str.strip(),m['label_B'].str.strip()).reindex(index=C,columns=C).fillna(0); norm=ct.div(ct.sum(axis=1),axis=0)
im=ax.imshow(norm.values,cmap=matplotlib.colors.LinearSegmentedColormap.from_list('t',['#F4FAF9','#1A9484','#0B4F47']),vmin=0,vmax=1,aspect='auto'); ax.grid(False)
for i in range(5):
    for j in range(5): ax.text(j,i,f'{int(ct.values[i,j])}\n{norm.values[i,j]*100:.0f}%',ha='center',va='center',fontsize=13.5,color='white' if norm.values[i,j]>0.5 else INK)
ax.set_xticks(range(5)); ax.set_xticklabels(C,fontsize=14.5,rotation=0); ax.set_yticks(range(5)); ax.set_yticklabels(C,fontsize=13.5); ax.set_xlabel('Coder B (n = 600)\nκ = 0.790 [0.751, 0.829]   ·   EVENT vs rest κ = 0.875',fontsize=16); ax.set_ylabel('Coder A'); ax.tick_params(length=0)
for sp in ax.spines.values(): sp.set_visible(False)
P(ax,'c')
# (d) per-block
ax=fig.add_subplot(gs[1,1]); raw=pd.read_csv(OUT+'/raw_results.csv')
pb=raw[raw.model.str.startswith('proposed@')].groupby('model')[['macro_f1','event_f1']].agg(['mean','std']); pb.index=[i.split('@')[1] for i in pb.index]; pb=pb.loc[BORD]
x=np.arange(3); w=0.36
ax.bar(x-w/2,pb[('macro_f1','mean')],w,yerr=pb[('macro_f1','std')],color=CAT['OTHER'],label='Macro-F1',capsize=4)
ax.bar(x+w/2,pb[('event_f1','mean')],w,yerr=pb[('event_f1','std')],color=CAT['EVENT'],label='EVENT-F1',capsize=4)
for i in range(3): ax.text(i+w/2,pb[('event_f1','mean')].iloc[i]+pb[('event_f1','std')].iloc[i]+0.02,f"{pb[('event_f1','mean')].iloc[i]:.2f}",ha='center',fontsize=14.5,color=INK)
ax.set_xticks(x); ax.set_xticklabels([BEN[b] for b in BORD]); ax.set_ylim(0,1.18); ax.set_ylabel('Score by block'); ax.legend(loc='upper left',ncol=2)
ax.annotate('only 95 EVENT articles\nin the gold set',xy=(1+w/2,0.51),xytext=(1.0,0.80),fontsize=13.5,color=INK2,arrowprops=dict(arrowstyle='->',color=INK2)); P(ax,'d',x=-0.14)
save(fig,'Fig4_classifier_validation')
