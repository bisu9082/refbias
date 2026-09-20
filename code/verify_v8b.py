#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 7 재검증 v8b — 문맥 정규식으로 본문 수치를 v8_final_numbers.json 과 대조"""
import json,re,sys,os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths as P
F=json.load(open(P.out('v8_final_numbers.json')))
H=json.load(open(P.out('human_validation.json')))
tex=open(P.tex(),encoding='utf-8').read()
rows=[]
def chk(name,pat,exp):
    m=re.search(pat,tex); got=m.group(1) if m else None
    rows.append((name,exp,got,got==exp))
hh=H['human_human']; ha=H['human_vs_A']
chk('kappa HH 5class',r'Agreement between them is \$\\kappa = ([0-9.]+)\$','%.3f'%hh['overall_5class']['kappa'])
chk('kappa HH binary',r'and ([0-9.]+) \(0\.853--0\.958\)','%.3f'%hh['overall_binary']['kappa'])
chk('kappa HA binary',r'and ([0-9.]+) \([0-9.]+--[0-9.]+\) on the EVENT boundary\. The block','%.3f'%ha['overall_binary']['kappa'])
chk('kappa HA sec',r'against ([0-9.]+) \([0-9.]+--[0-9.]+\) in the security block','%.3f'%ha['security_binary']['kappa'])
chk('disagree total',r'(Forty) articles are coded differently','Forty')
chk('EVENT boundary n',r'but (\w+) touch the EVENT boundary','twelve')
chk('radiation n',r'and (\w+) of the twelve are the same query','nine')
# table rows
# Table 6 행 — 구간은 쿼리 군집(human_query_tests.json), A 는 구간 없이 점추정만 보고한다
QC=json.load(open(P.v10('human_query_tests.json')))['block_level_query_clustered']['pE']
for disp,b in [('Industrial','industrial'),('Security','security'),('Conventional','conventional')]:
    v=F['blocks'][b]
    m=re.search(disp+r' & (\d+) \((\d+)\) & ([0-9.]+) & ([0-9.]+) & ([0-9.]+) & ([0-9.]+) \[([0-9.]+), ([0-9.]+)\] & ([0-9.]+)--([0-9.]+) & ([0-9.]+) ',tex)
    got=list(m.groups()) if m else [None]*11
    rng=[100*F['consensus_sensitivity'][r][b] for r in F['consensus_sensitivity'] if r!='disagree_all_EVENT']
    ci=QC[b]['ci95_query_cluster']
    exp=['%d'%v['n_drawn'],'%d'%v['n_consensus'],'%.2f'%v['TPR'],'%.2f'%v['FPR'],'%.1f'%(100*v['pE_coderA']),
         '%.1f'%(100*v['pE_human']),'%.1f'%(100*ci[0]),'%.1f'%(100*ci[1]),
         '%.1f'%min(rng),'%.1f'%max(rng),'%.1f'%v['A']]
    for i,lab in enumerate(['n_drawn','n_cons','TPR','FPR','pE_A','pE_H','lo','hi','rng_lo','rng_hi','A']):
        rows.append(('tab %s %s'%(disp,lab),exp[i],got[i],exp[i]==got[i]))
# prose pE
chk('prose pE conv',r'the human event share is ([0-9.]+)\\% \(95\\% CI','%.1f'%(100*F['blocks']['conventional']['pE_human']))
chk('prose pE ind',r'in the conventional block, ([0-9.]+)\\% \(([0-9.]+)--([0-9.]+)\) in the industrial block and','%.1f'%(100*F['blocks']['industrial']['pE_human']))
chk('prose pE sec',r'and ([0-9.]+)\\% \([0-9.]+--[0-9.]+\) in the security block, against','%.1f'%(100*F['blocks']['security']['pE_human']))
# amplification prose
chk('A sec',r'gives \$A = ([0-9.]+)\$ for the security block','%.1f'%F['blocks']['security']['A'])
chk('A ind',r'for the security block, ([0-9.]+) for the industrial block','%.1f'%F['blocks']['industrial']['A'])
chk('A conv',r'for the industrial block and ([0-9.]+) for the conventional block','%.1f'%F['blocks']['conventional']['A'])
# consensus sensitivity prose
cs=F['consensus_sensitivity']
chk('sens all-EVENT',r'all 40 disagreements are coded as events, gives ([0-9.]+)\\%','%.1f'%(100*cs['disagree_all_EVENT']['security']))
chk('sens all-non',r'([0-9.]+)\\% taking H1 alone or resolving every disagreement as a non-event','%.1f'%(100*cs['disagree_all_nonEVENT']['security']))
chk('sens H1',r'([0-9.]+)\\% taking H1 alone','%.1f'%(100*cs['H1_only']['security']))
chk('sens H2',r'and ([0-9.]+)\\% taking H2 alone','%.1f'%(100*cs['H2_only']['security']))
chk('sens consensus',r'four distinct values: ([0-9.]+)\\% on consensus','%.1f'%(100*cs['consensus']['security']))
# corpus RG
# v10: 안보 블록은 점추정을 싣지 않고 상한만 보고하므로 6개 값 + 상한 1개를 확인한다
m=re.search(r'gives ([0-9.]+)\\% \(([0-9.]+)--([0-9.]+)\) in the conventional block and ([0-9.]+)\\% \(([0-9.]+)--([0-9.]+)\) in the industrial block',tex)
got=list(m.groups()) if m else [None]*6
exp=[]
for b in ['conventional','industrial']:
    c=F['blocks'][b]['corpus']; exp+=['%.1f'%(100*c['pE_rg']),'%.1f'%(100*c['ci_rg'][0]),'%.1f'%(100*c['ci_rg'][1])]
for i,l in enumerate(['conv','conv_lo','conv_hi','ind','ind_lo','ind_hi']):
    rows.append(('corpusRG '+l,exp[i],got[i],exp[i]==got[i]))
m2=re.search(r'runs to an upper limit of ([0-9.]+)\\%',tex)
rows.append(('corpusRG sec_upper','%.1f'%(100*F['blocks']['security']['corpus']['ci_rg'][1]),
             m2.group(1) if m2 else None,
             ('%.1f'%(100*F['blocks']['security']['corpus']['ci_rg'][1]))==(m2.group(1) if m2 else None)))
# domestic
d=F['domestic']
chk('domestic human',r'the domestic share of security coverage is ([0-9.]+)\\% \(0\.0--8\.1, resampling whole queries\)','2.5')
chk('domestic coderA',r'on human labels against ([0-9.]+)\\% on coder','%.1f'%(100*d['pE_dom_coderA']))
chk('domestic census',r'All (\d+) security articles that coder A labelled EVENT','%d'%d['coderA_domestic_EVENT_in_census'])
chk('domestic confirmed',r'; (\d+) survive as events to both coders','%d'%d['human_confirmed_in_census'])
chk('rate ci sec TPR',r'carry intervals of ([0-9.]+--[0-9.]+) and','0.28--0.83')
chk('radiation excl range',r'the defensible range at ([0-9.]+--[0-9.]+)\\%','4.8--5.4')
chk('defensible range',r'The defensible range is therefore ([0-9.]+--[0-9.]+)\\%','4.4--6.0')
bad=[r for r in rows if not r[3]]
for n,e,g,ok in rows: print(('OK  ' if ok else 'MISS'),'%-24s expect %-16s tex %s'%(n,e,g))
print('\n%d/%d 일치'%(len(rows)-len(bad),len(rows)))
ph=re.findall(r'\[[A-Z][A-Z0-9-]{3,}[^\]]*\]',tex)+(['GHUSER (GitHub 계정 미확정)'] if 'GHUSER' in tex else [])
print('잔여 플레이스홀더:',ph)
sys.exit(1 if bad else 0)
