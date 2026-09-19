#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 7 재검증 — 본문 수치를 산출물 JSON 과 문맥 포함 대조 (단순 부분문자열 일치 금지)"""
import json,re,sys
B='/home/claude/rb/Step5r/'
hv=json.load(open(B+'human_validation.json')); rg=json.load(open(B+'rg_corrected.json'))
tex=open('/home/claude/refbias_tex/main.tex',encoding='utf-8').read()
rows=[]
def chk(name,pattern,expect_json):
    m=re.search(pattern,tex)
    got=m.group(1) if m else None
    ok=(got is not None) and (got==expect_json)
    rows.append((name,expect_json,got,ok))
hh=hv['human_human']; ha=hv['human_vs_A']
chk('kappa HH 5class',      r'Agreement between them is \$\\kappa = ([0-9.]+)\$','%.3f'%hh['overall_5class']['kappa'])
chk('kappa HH 5class CI',   r'Agreement between them is \$\\kappa = [0-9.]+\$ \(95\\% CI ([0-9.]+--[0-9.]+)\)','%.3f--%.3f'%tuple(hh['overall_5class']['ci95']))
chk('kappa HH binary',      r'and ([0-9.]+) \([0-9.]+--[0-9.]+\) for the EVENT-versus-rest','%.3f'%hh['overall_binary']['kappa'])
chk('raw agreement',        r'with ([0-9.]+)\\% raw agreement','%.1f'%(100*hh['percent_agreement']))
chk('n consensus',          r'The ([0-9]+) articles on which the human coders agree','%d'%hv['n_consensus'])
chk('kappa HA 5class',      r'coder A reaches \$\\kappa = ([0-9.]+)\$ \([0-9.]+--[0-9.]+\) over the five','%.3f'%ha['overall_5class']['kappa'])
chk('kappa HA binary',      r'and ([0-9.]+) \([0-9.]+--[0-9.]+\) on the EVENT boundary','%.3f'%ha['overall_binary']['kappa'])
chk('kappa HA sec binary',  r'against ([0-9.]+) \([0-9.]+--[0-9.]+\) in the security block','%.3f'%ha['security_binary']['kappa'])
# 표 행 대조
for disp,b in [('Industrial','industrial'),('Security','security'),('Conventional','conventional')]:
    v=rg['blocks'][b]; h=hv['pE_human'][b]
    m=re.search(disp+r' & \d+ & ([0-9.]+) & ([0-9.]+) & ([0-9.]+) & ([0-9.]+) \[([0-9.]+), ([0-9.]+)\] & ([0-9.]+) \[([0-9.]+), ([0-9.]+)\] & ([0-9.]+) \[([0-9.]+), ([0-9.]+)\]',tex)
    got=list(m.groups()) if m else [None]*12
    exp=['%.2f'%v['TPR'],'%.2f'%v['FPR'],'%.1f'%(100*h['pE_coderA']),
         '%.1f'%(100*h['pE_human_ipw']),'%.1f'%(100*h['ci95'][0]),'%.1f'%(100*h['ci95'][1]),
         '%.1f'%(100*v['gold']['pE_rg']),'%.1f'%(100*v['gold']['ci_rg'][0]),'%.1f'%(100*v['gold']['ci_rg'][1]),
         '%.1f'%v['amplification']['A'],'%.1f'%v['amplification']['ci95'][0],'%.1f'%v['amplification']['ci95'][1]]
    for i,lab in enumerate(['TPR','FPR','pE_A','pE_H','pE_H_lo','pE_H_hi','RG','RG_lo','RG_hi','A','A_lo','A_hi']):
        rows.append(('tab:human %s %s'%(disp,lab),exp[i],got[i],exp[i]==got[i]))
# 코퍼스 RG
m=re.search(r'gives ([0-9.]+)\\% \(([0-9.]+)--([0-9.]+)\) in the conventional block, ([0-9.]+)\\% \(([0-9.]+)--([0-9.]+)\) in the industrial block and ([0-9.]+)\\% \(([0-9.]+)--([0-9.]+)\) in the security block',tex)
got=list(m.groups()) if m else [None]*9
exp=[]
for b in ['conventional','industrial','security']:
    c=rg['blocks'][b]['corpus']; exp+=['%.1f'%(100*c['pE_rg']),'%.1f'%(100*c['ci_rg'][0]),'%.1f'%(100*c['ci_rg'][1])]
for i,lab in enumerate(['conv','conv_lo','conv_hi','ind','ind_lo','ind_hi','sec','sec_lo','sec_hi']):
    rows.append(('corpus RG '+lab,exp[i],got[i],exp[i]==got[i]))
# 위양성 구조
fp=rg['security_false_positive_structure']
chk('FP: A EVENT calls',r'Of the (\d+) security articles coder A called EVENT','%d'%fp['n_A_EVENT'])
chk('FP: rejected',      r'called EVENT, (\d+) were not events','%d'%fp['n_false_positive'])
chk('FP: IRRELEVANT',    r'and (\d+) of those were coded IRRELEVANT','%d'%fp['by_human_label']['IRRELEVANT'])
chk('FN count',          r'only (\d+) human EVENT articles were missed','%d'%rg['security_false_negative']['n'])
q=fp['by_query']
chk('FP Novichok',r'Novichok query \((\d+)\)','%d'%q['노비촉'])
chk('FP radiation',r'Novichok query \(\d+\), radiation leak \((\d+)\)','%d'%q['방사능 유출'])
chk('FP dprk',r'radiation leak \(\d+\), DPRK nuclear test \((\d+)\)','%d'%q['북한 핵실험'])
chk('FP anthrax',r'DPRK nuclear test \(\d+\) and anthrax \((\d+)\)','%d'%q['탄저균 테러'])
bad=[r for r in rows if not r[3]]
for n,e,g,ok in rows: print(('OK  ' if ok else 'MISS'),'%-26s expect %-14s tex %s'%(n,e,g))
print('\n%d/%d 일치'%(len(rows)-len(bad),len(rows)))
ph=re.findall(r'\[[A-Z][A-Z0-9-]{3,}[^\]]*\]',tex)
print('잔여 플레이스홀더:',ph)
sys.exit(1 if bad else 0)
