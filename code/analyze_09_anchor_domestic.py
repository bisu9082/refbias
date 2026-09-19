#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refbias — Step 5 롤백 재분석 B1 (analyze_09_anchor_domestic.py)
================================================================
B1 앵커 사전 재검토: 토큰별 기여(철자 변형 포함), 연도토큰 제외 ρ, 골드셋 앵커언급 × EVENT 라벨 교차표,
   Novichok 앵커를 사건별(솔즈베리 2018 / 나발니 2020)로 분해, Fukushima 연도별 ρ(2023 처리수 방류 분리)
B2 국내/국외 사건 코딩: 골드셋 EVENT 기사 ev_place 기준 → 블록·질의별 국내 사건 비율, 국내 한정 p_E
"""
import os, re, json
import numpy as np, pandas as pd
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT=BASE+'/Step5r'; os.makedirs(OUT,exist_ok=True)
res={}

ANCH={ # qid: (name, {token_group: [variants]})
 'Q01':('hydrogen fluoride',{'Gumi':['구미'],'2012':['2012']}),
 'Q04':('sulfuric acid (control)',{'Gumi':['구미'],'2012':['2012']}),
 'Q08':('nitric acid (control)',{'Gumi':['구미'],'2012':['2012']}),
 'Q03':('chlorine (control)',{'Gumi':['구미'],'2012':['2012']}),
 'Q12':('sarin',{'Aum':['옴진리교','오움진리교'],'Tokyo':['도쿄','동경'],'1995':['1995'],'Ghouta':['구타','고우타']}),
 'Q14':('anthrax',{'2001':['2001'],'US':['미국'],'mail':['우편','편지']}),
 'Q19':('Novichok',{'Skripal':['스크리팔','스크립팔'],'Salisbury':['솔즈베리','솔즈버리'],'Navalny':['나발니','나발리'],'2018':['2018'],'2020':['2020']}),
 'Q16':('radiation leak',{'Fukushima':['후쿠시마'],'Chernobyl':['체르노빌']}),
}
EVENT_GROUPS={'Q19':{'Salisbury 2018':['Skripal','Salisbury'],'Navalny 2020':['Navalny']},
              'Q16':{'Fukushima':['Fukushima'],'Chernobyl':['Chernobyl']}}
YEAR_TOKENS={'2012','1995','2001','2018','2020'}

cols=['qid','뉴스 식별자','일자','제목','본문','키워드']
c=pd.read_csv(BASE+'/data/corpus_bigkinds.csv.gz',dtype=str,usecols=cols)
c=c[c.qid.isin(ANCH)].copy()
c['t']=c['제목'].fillna('')+' '+c['본문'].fillna('')+' '+c['키워드'].fillna('')
c['year']=c['일자'].str[:4]
g=pd.read_csv(BASE+'/data/goldset_labeled_A.csv',dtype=str)
g['t']=g['제목'].fillna('')+' '+g['본문'].fillna('')

def hits(t,groups):
    h={}
    for k,vs in groups.items():
        h[k]=t.str.contains('|'.join(map(re.escape,vs)),na=False)
    return pd.DataFrame(h)

rows=[]; res['anchor']={}
for qid,(name,groups) in ANCH.items():
    d=c[c.qid==qid]; H=hits(d['t'],groups)
    any_all=H.any(axis=1); any_noyear=H[[k for k in groups if k not in YEAR_TOKENS]].any(axis=1)
    r=dict(qid=qid,query=name,V=len(d),rho_all=any_all.mean(),rho_no_year=any_noyear.mean(),
           per_token={k:float(H[k].mean()) for k in groups})
    if qid in EVENT_GROUPS:
        r['per_event']={ev:float(H[toks].any(axis=1).mean()) for ev,toks in EVENT_GROUPS[qid].items()}
        r['per_event_by_year']={ev:{y:float(H.loc[d.year==y,toks].any(axis=1).mean()) for y in sorted(d.year.unique())} for ev,toks in EVENT_GROUPS[qid].items()}
    r['rho_by_year']={y:float(any_noyear[d.year==y].mean()) for y in sorted(d.year.unique())}
    r['V_by_year']={y:int((d.year==y).sum()) for y in sorted(d.year.unique())}
    # gold-set cross-tab: anchor mention (no-year lexicon) x label
    gg=g[g.qid==qid]; GH=hits(gg['t'],{k:v for k,v in groups.items() if k not in YEAR_TOKENS}).any(axis=1)
    ct=pd.crosstab(GH,gg['label'])
    r['gold_crosstab']={'n':len(gg),'anchor_mention_rate':float(GH.mean()),
        'by_label':{lab:dict(n=int((gg.label==lab).sum()),anchor_share=float(GH[gg.label==lab].mean()) if (gg.label==lab).sum() else None) for lab in ['EVENT','REF','INST','OTHER','IRRELEVANT']},
        'mentions_that_are_EVENT':float((gg.label[GH]=='EVENT').mean()) if GH.sum() else None,
        'mentions_that_are_REF':float((gg.label[GH]=='REF').mean()) if GH.sum() else None}
    res['anchor'][qid]=r
    print('%-24s V=%5d rho=%.3f no-year=%.3f | %s | gold: mention %.2f of which EVENT %.2f REF %.2f'%(name,len(d),r['rho_all'],r['rho_no_year'],
          {k:round(v,3) for k,v in r['per_token'].items()},r['gold_crosstab']['anchor_mention_rate'],
          r['gold_crosstab']['mentions_that_are_EVENT'] or 0,r['gold_crosstab']['mentions_that_are_REF'] or 0))
    if 'per_event' in r: print('     per_event',{k:round(v,3) for k,v in r['per_event'].items()},'| by year',{k:{y:round(x,2) for y,x in v.items()} for k,v in r['per_event_by_year'].items()})

# ---------- B2 domestic / foreign ----------
KR=['서울','부산','대구','인천','광주','대전','울산','세종','경기','강원','충북','충남','충청','전북','전남','전라','경북','경남','경상','제주','전국','국내','한국','수도권',
    '시','군','구','읍','면','동','도']
KR_RE=re.compile(r'(서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|충청|전북|전남|전라|경북|경남|경상|제주|전국|국내|한국|수도권|[가-힣]{1,4}(시|군|구|읍|면|동)(\s|$))')
FOREIGN=['미국','러시아','우크라이나','일본','중국','영국','프랑스','독일','시리아','이란','이라크','이스라엘','북한','평양','인도','파키스탄','아프가니스탄','튀르키예','터키','사우디','예멘','레바논','가자','팔레스타인','호주','캐나다','멕시코','브라질','유럽','아프리카','뉴올리언스','뉴욕','런던','파리','모스크바','키이우','벨기에','네덜란드','스페인','이탈리아','폴란드','대만','필리핀','태국','베트남','인도네시아','말레이시아','뉴질랜드','홍콩','마카오','남아공','나이지리아','소말리아','수단','리비아','이집트','요르단','카타르','UAE','두바이','쿠웨이트','조지아','아르메니아','아제르바이잔','벨라루스','체첸','다게스탄','크림','돈바스','헤르손','자포리자','오데사','하르키우','모스크바','텔아비브','예루살렘','다마스쿠스','바그다드','테헤란','카불','이슬라마바드','나토','NATO','이집트','오사카','도쿄','후쿠시마']
FOR_RE=re.compile('|'.join(map(re.escape,FOREIGN)))
def dom(place):
    if pd.isna(place) or not str(place).strip(): return 'unknown'
    p=str(place)
    KR_NAME=re.compile(r'(서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|충청|전북|전남|전라|경북|경남|경상|제주|전국|국내|한국|수도권)')
    if FOR_RE.search(p) and not KR_NAME.search(p): return 'foreign'
    if KR_RE.search(p): return 'domestic'
    return 'unknown'
e=g[g.label=='EVENT'].copy(); e['dom']=e['ev_place'].map(dom)
# resolve unknown with title/body foreign mention heuristic
e.loc[e.dom=='unknown','dom']=np.where(e.loc[e.dom=='unknown','t'].str.contains(FOR_RE),'foreign_by_text','unknown')
BL={'산업':'industrial','안보':'security','비CBRN':'conventional'}
res['domestic']={'by_block':{},'by_query':{}}
for b,grp in e.groupby('block'):
    vc=grp.dom.value_counts(); n_all=(g.block==b).sum()
    domn=int(vc.get('domestic',0)); forn=int(vc.get('foreign',0)+vc.get('foreign_by_text',0)); unk=int(vc.get('unknown',0))
    res['domestic']['by_block'][BL[b]]=dict(EVENT=len(grp),domestic=domn,foreign=forn,unknown=unk,
        domestic_share_of_EVENT=domn/len(grp),pE_domestic_only=domn/n_all,pE_all=len(grp)/n_all)
    print('B2 %-13s EVENT=%4d domestic=%4d foreign=%4d unknown=%3d | pE_dom=%.3f pE_all=%.3f'%(BL[b],len(grp),domn,forn,unk,domn/n_all,len(grp)/n_all))
for (q,qn),grp in e.groupby(['qid','query']):
    vc=grp.dom.value_counts(); n_all=(g.qid==q).sum()
    res['domestic']['by_query'][q]=dict(query=qn,EVENT=len(grp),domestic=int(vc.get('domestic',0)),foreign=int(vc.get('foreign',0)+vc.get('foreign_by_text',0)),unknown=int(vc.get('unknown',0)),n=int(n_all))
for q in ['Q23','Q24','Q15','Q11','Q16','Q21']:
    print('   ',q,res['domestic']['by_query'].get(q))
e[['sid','qid','block','query','ev_place','dom']].to_csv(OUT+'/event_domestic_coding.csv',index=False,encoding='utf-8-sig')
json.dump(res,open(OUT+'/revision_B1.json','w'),indent=1,ensure_ascii=False,default=float)
print('saved')
