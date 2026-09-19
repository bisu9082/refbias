#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""refbias Fig5 — 인간 검증 (a) 블록별 p_E 세 경로 비교  (b) 안보 블록 코더A EVENT 판정의 인간 라벨 분해"""
import os, json
import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
mpl.use('Agg')
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fin=json.load(open(BASE+'/Step5r/v8_final_numbers.json')); rg=fin
C={'A':'#3F5FAE','H':'#B5352A','R':'#1A9484','N':'#8A8A8A','O':'#D97B1E'}
AX,TK,LG=19,16,16
mpl.rcParams.update({'font.family':'DejaVu Sans','axes.linewidth':1.0,'pdf.fonttype':42,'ps.fonttype':42})
BLK=['industrial','security','conventional']; LAB=['Industrial','Security\n(CBRN)','Conventional']
fig,axes=plt.subplots(1,2,figsize=(13.2,5.6),gridspec_kw={'width_ratios':[1.15,1]})

# ---- (a) p_E three routes ----
ax=axes[0]; x=np.arange(3); w=0.26
def get(b):
    v=fin['blocks'][b]; rng=[fin['consensus_sensitivity'][r][b] for r in fin['consensus_sensitivity']]
    return (100*v['pE_coderA'],(100*v['pE_human'],[100*v['ci95'][0],100*v['ci95'][1]]),
            [100*min(rng),100*max(rng)])
for i,b in enumerate(BLK):
    a,(hp,hci),rr=get(b)
    ax.bar(x[i]-w/2,a,w,color=C['A'],edgecolor='none',label='Coder A (model)' if i==0 else None)
    ax.bar(x[i]+w/2,hp,w,color=C['H'],edgecolor='none',label='Human standard' if i==0 else None)
    ax.errorbar(x[i]+w/2,hp,yerr=[[hp-hci[0]],[hci[1]-hp]],fmt='none',ecolor='#3A3A3A',elinewidth=1.4,capsize=4)
    ax.plot([x[i]+w/2+0.13]*2,rr,color=C['N'],lw=6,solid_capstyle='butt',alpha=.55,
            label='Range over disagreement rules' if i==0 else None)
ax.set_xticks(x); ax.set_xticklabels(LAB,fontsize=TK)
ax.set_ylabel('Event share $p_E$ (%)',fontsize=AX); ax.set_ylim(0,58)
ax.tick_params(labelsize=TK); 
for sp in ['top','right']: ax.spines[sp].set_visible(False)
ax.annotate(r'$\kappa$ = 0.69',(0,49),ha='center',fontsize=TK-1,color='#3A3A3A')
ax.annotate(r'$\kappa$ = 0.24',(1,16.5),ha='center',fontsize=TK-1,color=C['H'])
ax.annotate(r'$\kappa$ = 0.94',(2,52),ha='center',fontsize=TK-1,color='#3A3A3A')
ax.text(-0.13,1.01,'a',transform=ax.transAxes,fontsize=AX+3,fontweight='bold',va='bottom',ha='right')

# ---- (b) security false-positive decomposition ----
ax=axes[1]
fp=json.load(open(BASE+'/Step5r/rg_corrected.json'))['security_false_positive_structure']; q=fp['by_query']
order=sorted(q.items(),key=lambda kv:-kv[1])
names={'노비촉':'Novichok','방사능 유출':'Radiation leak','북한 핵실험':'DPRK nuclear test','탄저균 테러':'Anthrax',
       '화학무기 사용':'Chemical weapons','더티밤':'Dirty bomb','사린가스 공격':'Sarin','북한 생화학무기':'DPRK CB weapons',
       '백색가루 신고':'White powder'}
lab=[names.get(k,k) for k,_ in order]; val=[v for _,v in order]
y=np.arange(len(val))[::-1]
ax.barh(y,val,color=C['O'],edgecolor='none')
ax.set_yticks(y); ax.set_yticklabels(lab,fontsize=TK-2)
ax.set_xlabel('Rejected coder A EVENT calls',fontsize=AX-1)
ax.tick_params(labelsize=TK); ax.set_xlim(0,29)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
ax.text(0.97,0.06,'61 of 79 coder A EVENT calls\n56 of them coded IRRELEVANT',transform=ax.transAxes,
        ha='right',va='bottom',fontsize=TK-2,color='#3A3A3A')
ax.text(-0.34,1.01,'b',transform=ax.transAxes,fontsize=AX+3,fontweight='bold',va='bottom',ha='right')

plt.tight_layout(rect=[0,0,1,0.90])
h,l=axes[0].get_legend_handles_labels()
fig.legend(h,l,fontsize=LG,frameon=False,loc='upper center',bbox_to_anchor=(0.5,1.0),ncol=3,columnspacing=1.6,handlelength=1.3)
out=BASE+'/figure/Fig5_human_validation.pdf'; os.makedirs(BASE+'/figure',exist_ok=True)
plt.savefig(out,bbox_inches='tight'); plt.savefig(out.replace('.pdf','.png'),dpi=200,bbox_inches='tight')
print('saved',out)
