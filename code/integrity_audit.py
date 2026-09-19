#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refbias — 데이터 정합성 전수 감사 (integrity_audit.py)   [C-3]
==============================================================
3단 대조를 수행한다.
 L1 원자료 → 산출물: 라벨 파일에서 직접 재계산한 값이 배포 JSON 과 일치하는가
 L2 산출물 → 본문: 배포 JSON 값이 main.tex 에 문맥 정규식으로 그대로 실려 있는가
 L3 교차 파일: 같은 양이 여러 산출물에 있을 때 서로 일치하는가
추가로 자리표시자, 미정의 인용, 코드 보안 스캔을 확인한다.
"""
import os,re,json,sys,subprocess
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import _paths as P
BASE=P.ROOT
TEX=P.tex()
OUT=os.path.dirname(P.out('v8_final_numbers.json'))
CATS=['EVENT','REF','INST','OTHER','IRRELEVANT']
AL={'event':'EVENT','ref':'REF','inst':'INST','other':'OTHER','irrelevant':'IRRELEVANT',
    '사건':'EVENT','참조':'REF','제도':'INST','기타':'OTHER','무관':'IRRELEVANT'}
BL={'산업':'industrial','안보':'security','비CBRN':'conventional'}
def nm(x):
    if pd.isna(x): return None
    s=str(x).strip(); return AL.get(s.lower(), s.upper() if s.upper() in CATS else None)
rows=[]
def chk(layer,name,expected,got,tol=None):
    if tol is None: ok = (expected==got)
    else:
        try: ok = abs(float(expected)-float(got))<=tol
        except Exception: ok=False
    rows.append((layer,name,expected,got,ok))

tex=open(TEX,encoding='utf-8').read()
_sup=os.path.join(os.path.dirname(TEX),'supplement.tex')
sup=open(_sup,encoding='utf-8').read() if os.path.exists(_sup) else ''
F=json.load(open(OUT+'/v8_final_numbers.json'))
HV=json.load(open(OUT+'/human_validation.json'))
CD=json.load(open(OUT+'/conditional_pE.json'))
V9=json.load(open(OUT+'/v9_additions.json'))
RA=json.load(open(OUT+'/revision_A.json'))

# ================= L1 원자료 → 산출물 =================
g=pd.read_csv(P.goldset_A(),dtype=str); g['isE']=(g.label=='EVENT')
def _readh(t):
    f=P.human(t)
    d=pd.read_excel(f,dtype=str) if f.endswith('.xlsx') else pd.read_csv(f,dtype=str)
    return d[['sid','label']].assign(label=lambda x:x.label.map(nm))
H={t:_readh(t) for t in ['H1','H2']}
d=(H['H1'].rename(columns={'label':'h1'}).merge(H['H2'].rename(columns={'label':'h2'}),on='sid')
   .merge(g[['sid','block','label']].rename(columns={'label':'A'}),on='sid'))
d['cell']=d.block+'|'+np.where(d.A=='EVENT','EVENT','non-EVENT')
N=g.assign(cell=g.block+'|'+np.where(g.isE,'EVENT','non-EVENT')).groupby('cell').size()
c=d[d.h1==d.h2].copy(); c['y']=c.h1
for kb,b in BL.items():
    s=c[c.block==kb]
    ps=sum(N[cl]*(gp.y=='EVENT').mean() for cl,gp in s.groupby('cell'))/sum(N[cl] for cl in s.cell.unique())
    chk('L1','pE_human %s'%b,round(ps,6),round(F['blocks'][b]['pE_human'],6),1e-6)
    num=sum(N[cl]*(gp.y=='EVENT').mean() for cl,gp in s.groupby('cell'))
    den=sum(N[cl]*(gp.y!='IRRELEVANT').mean() for cl,gp in s.groupby('cell'))
    chk('L1','pE_cond_human %s'%b,round(num/den,6),round(CD['human'][b]['pE_cond'],6),1e-6)
    w=s.cell.map({cl:N[cl]/len(gp) for cl,gp in s.groupby('cell')}).values
    hu=(s.y=='EVENT').values; ae=(s.A=='EVENT').values
    tpr=(w*(ae&hu)).sum()/max((w*hu).sum(),1e-9); fpr=(w*(ae&~hu)).sum()/max((w*~hu).sum(),1e-9)
    chk('L1','TPR %s'%b,round(float(tpr),4),round(F['blocks'][b]['TPR'],4),1e-4)
    chk('L1','FPR %s'%b,round(float(fpr),4),round(F['blocks'][b]['FPR'],4),1e-4)
chk('L1','kappa HH 5class',round(cohen_kappa_score(d.h1,d.h2,labels=CATS),6),
    round(HV['human_human']['overall_5class']['kappa'],6),1e-6)
chk('L1','kappa HA binary',round(cohen_kappa_score(c.y=='EVENT',c.A=='EVENT'),6),
    round(HV['human_vs_A']['overall_binary']['kappa'],6),1e-6)
s2=c[c.block=='안보']; hu=(s2.y=='EVENT').values; ae=(s2.A=='EVENT').values
chk('L1','security TP',int((hu&ae).sum()),V9['security_2x2']['TP'])
chk('L1','security FP',int((~hu&ae).sum()),V9['security_2x2']['FP'])
chk('L1','security FN',int((hu&~ae).sum()),V9['security_2x2']['FN'])
chk('L1','security precision',round(float((hu&ae).sum()/max((ae).sum(),1)),4),
    round(V9['security_2x2']['precision'],4),1e-4)
for kb,b in BL.items():
    s=g[g.block==kb]
    chk('L1','pE_cond_model %s'%b,round(float((s.label=='EVENT').sum()/(s.label!='IRRELEVANT').sum()),6),
        round(CD['model'][b]['pE_cond'],6),1e-6)

# ================= L2 산출물 → 본문 =================
def find(pat,name,expected):
    m=re.search(pat,tex); chk('L2',name,expected,m.group(1) if m else None)
def finds(pat,name,expected):
    m=re.search(pat,sup); chk('L2',name,expected,m.group(1) if m else None)
find(r'Agreement between them is \$\\kappa = ([0-9.]+)\$','kappa HH 5class in tex','%.3f'%HV['human_human']['overall_5class']['kappa'])
find(r'the human event share is ([0-9.]+)\\% \(95','pE_human conventional in tex','%.1f'%(100*F['blocks']['conventional']['pE_human']))
find(r'and ([0-9.]+)\\% \([0-9.]+--[0-9.]+\) in the security block, against','pE_human security in tex','%.1f'%(100*F['blocks']['security']['pE_human']))
find(r'gives \$A = ([0-9.]+)\$','A security in tex','%.1f'%F['blocks']['security']['A'])
find(r'the event share is ([0-9.]+)\\% in the industrial block, ([0-9.]+)\\% in the conventional','pE_cond industrial in tex','%.1f'%(100*CD['model']['industrial']['pE_cond']))
find(r'and ([0-9.]+)\\% in the security block, and the query-level test','pE_cond security model in tex','%.1f'%(100*CD['model']['security']['pE_cond']))
find(r'gives \$H\(2\) = ([0-9.]+)\$','cond KW H in tex','%.2f'%CD['model_query_level']['kruskal_H'])
find(r'human event shares of ([0-9.]+)\\% \(42\.0--63\.0\) industrial','pE_cond human industrial in tex','%.1f'%(100*CD['human']['industrial']['pE_cond']))
find(r'and ([0-9.]+)\\% \(7\.3--21\.8\) security','pE_cond human security in tex','%.1f'%(100*CD['human']['security']['pE_cond']))
find(r'The full table is (\d+) true positives','security TP in tex',str(V9['security_2x2']['TP']))
find(r'(\d+) false positives, ','security FP in tex',str(V9['security_2x2']['FP']))
find(r"coder A's precision on the EVENT class in this block is ([0-9.]+)",'security precision in tex','%.2f'%V9['security_2x2']['precision'])
find(r'the two human coders agree with each other there at ([0-9.]+)','HH security binary in tex','%.3f'%V9['matched_kappa']['H1_vs_H2']['security']['binary']['kappa'])
find(r'the anchor citation rate is ([0-9.]+)\\% for Novichok','anchor nonEVENT Novichok in tex','%.1f'%(100*V9['anchor_conditional']['Q19']['rho_given_nonEVENT']))
find(r'return 0 of (\d+) such articles','anchor control n in tex',str(V9['anchor_conditional']['_contrast_nonEVENT']['control_n']))
find(r'an upper bound of ([0-9.]+)\\% by the rule of three','anchor control upper in tex','%.1f'%(100*V9['anchor_conditional']['_contrast_nonEVENT']['control_upper95_rule_of_three']))
find(r'the difference is ([0-9.]+) percentage points','control diff in tex','%.1f'%(100*V9['control_equivalence']['difference']))
find(r'the smallest difference detectable with 80\\% power is (\d+) points','MDE in tex','%d'%round(100*V9['control_equivalence']['mde_80power']))
# 짝맞춤 kappa 표
for label,pk in [('Model A vs model B','A_vs_B'),('Human H1 vs human H2','H1_vs_H2'),
                 ('Human standard vs model A','human_vs_A'),('Human standard vs model B','human_vs_B')]:
    m=re.search(re.escape(label)+r' & (\d+) & ([0-9.]+) & ([0-9.]+) & ([0-9.]+) & ([0-9.]+) \\\\',tex)
    got=m.groups() if m else (None,)*5
    v=V9['matched_kappa'][pk]
    exp=('%d'%v['n'],'%.3f'%v['five_class']['kappa'],'%.3f'%v['binary']['kappa'],
         '%.3f'%v['security']['five_class']['kappa'],'%.3f'%v['security']['binary']['kappa'])
    for i,f in enumerate(['n','5class','binary','sec5','secbin']):
        chk('L2','tab:kappa %s %s'%(label,f),exp[i],got[i])

# ---- v10 추가분 (A1 인간 m / A2 앵커 비의존) ----
HM=json.load(open(P.v10('human_m_security.json'))); AF=json.load(open(P.v10('anchor_free_check.json')))
finds(r'They resolve into (\d+) events','v10 n_events in SI',str(HM['variants']['all22']['n_events']))
finds(r'giving \$m = ([0-9.]+)\$ against 1\.12','v10 m_human in SI','%.2f'%HM['variants']['all22']['m'])
finds(r'human events it is ([0-9.]+);','v10 m_census in SI','%.2f'%HM['variants']['census18']['m'])
finds(r'removing it gives ([0-9.]+), while removing any other cluster','v10 LOO min in SI','%.2f'%HM['leave_one_cluster_out']['range'][0])
find(r'duplication factor is between ([0-9.]+) and 2\.00','v10 m lower in tex','%.2f'%HM['leave_one_cluster_out']['range'][0])
finds(r'same design as \$p_E\$, gives ([0-9.]+)\.','v10 weighted m in SI','%.2f'%HM['variants']['all22_design_weighted']['m'])
chk('L3','v10 rule applied == key clusters',True,HM['rule_applied']['match'])
find(r'cites a past occurrence in ([0-9.]+)\\% of articles','v10 anchorfree HF in tex','%.1f'%(100*AF['hf']['rate']))
find(r'against ([0-9.]+)\\% \(0\.4--3\.3\) for chlorine','v10 anchorfree Cl in tex','%.1f'%(100*AF['controls']['염소 가스 누출']['rate']))
chk('L3','v10 A = m/pE security',round(HM['variants']['all22']['A'],3),
    round(HM['variants']['all22']['m']/HM['pE_human_security']['pE'],3),1e-3)
chk('L3','v10 coderA m matches v8',round(HM['coderA_m_security'],6),round(F['m_block']['security'],6),1e-6)

HQ=json.load(open(P.v10('human_query_tests.json')))
find(r'event share \(\$H\(2\) = ([0-9.]+)\$, \$p = 0\.041','O1 human KW H in tex','%.2f'%HQ['human_pE']['kruskal_H'])
find(r'the rank test does not reject at all \(\$H\(2\) = ([0-9.]+)\$','O1 human cond KW H in tex','%.2f'%HQ['human_pE_conditional']['kruskal_H'])
find(r'gives 5\.0\\% \(0\.6--([0-9.]+)\)','O1 sec clustered hi in tex','%.1f'%(100*HQ['block_level_query_clustered']['pE']['security']['ci95_query_cluster'][1]))
find(r'conditioning on topical relevance gives 13\.9\\% \(1\.9--([0-9.]+)\)','O1 sec cond clustered hi in tex','%.1f'%(100*HQ['block_level_query_clustered']['pE_cond']['security']['ci95_query_cluster'][1]))
chk('L3','O1 안보 조건부 구간이 두 대조군과 비중첩',True,
    HQ['block_level_query_clustered']['pE_cond']['no_overlap_with_industrial'] and HQ['block_level_query_clustered']['pE_cond']['no_overlap_with_conventional'])
RR=json.load(open(P.v10('registry_recode.json')))
find(r'The six matched sets contain (\d+) distinct incidents','v12 registry unique in tex',str(RR['original']['unique_incidents']))
find(r'and (\d+) substance assignments','v12 registry sum in tex',str(RR['original']['sum_over_substances']))
find(r'and the slope from \$-0\.07\$ to \$-0\.0(\d)\$','v12 strict slope in tex','%d'%round(abs(RR['strict']['regression']['slope'])*100))
chk('L3','v12 strict recode null holds',True, RR['strict']['regression']['p']>0.05 and RR['original']['regression']['p']>0.05)

# ================= L3 교차 파일 =================
chk('L3','m_block security',round(F['m_block']['security'],6),round(95/85,6),1e-6)
chk('L3','A = m/pE security',round(F['blocks']['security']['A'],4),
    round(F['m_block']['security']/F['blocks']['security']['pE_human'],4),1e-4)
chk('L3','registry slope',round(RA['registry']['direct']['slope'],6),round(V9['registry_attenuation']['slope_obs'],6),1e-6)
chk('L3','human_validation superseded flag',True,'_SUPERSEDED' in HV)
chk('L3','conditional security < controls',True,bool(CD['human']['security']['ci95'][1]<CD['human']['industrial']['ci95'][0]))

# ================= 부가 점검 =================
ph=re.findall(r'\[[A-Z][A-Z0-9-]{3,}[^\]]*\]',tex)+(['GHUSER'] if 'GHUSER' in tex else [])
chk('X','자리표시자 0건',[],ph)  # GHUSER 는 미결 항목이므로 통과시키지 않는다
log=open(P.texlog(),encoding='utf-8',errors='ignore').read()
import re as _re
_undef=len(_re.findall(r'Warning: (?:Reference|Citation) [^\n]*undefined',log))+log.count('There were undefined references')
chk('X','LaTeX undefined refs/citations',0,_undef)  # 폰트 shape 경고는 제외
sec=subprocess.run("grep -rnE '\\b(eval|exec|os\\.system)\\(' %s/code/*.py | grep -v exec_module | wc -l"%BASE,
                   shell=True,capture_output=True,text=True).stdout.strip()
chk('X','코드 보안 스캔 (eval/exec/os.system)','0',sec)

bad=[r for r in rows if not r[4]]
for lay,n,e,gt,ok in rows:
    print('%s %-3s %-42s expect %-22s got %s'%(('OK  ' if ok else 'FAIL'),lay,n,str(e)[:22],str(gt)[:34]))
print('\n%d/%d 일치'%(len(rows)-len(bad),len(rows)))
print('L1 원자료→산출물 %d / L2 산출물→본문 %d / L3 교차파일 %d / X 부가 %d'%(
    sum(1 for r in rows if r[0]=='L1'),sum(1 for r in rows if r[0]=='L2'),
    sum(1 for r in rows if r[0]=='L3'),sum(1 for r in rows if r[0]=='X')))
if bad:
    print('\n실패 항목:')
    for lay,n,e,gt,_ in bad: print('  [%s] %s — expect %s, got %s'%(lay,n,e,gt))
sys.exit(1 if bad else 0)
