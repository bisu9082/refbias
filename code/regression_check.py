#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""배포 전 수치 회귀 점검 — 산출물 JSON 의 주요 값이 고정 기준과 일치하는지 확인한다.
리팩터링이 공개 수치를 조용히 바꾸는 사고(예: 정규식 이스케이프 오류)를 잡기 위한 장치."""
import json,sys,os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import _paths as P
B=os.path.dirname(P.out('v8_final_numbers.json'))+'/'
BASE=[  # (파일, 경로, 기대값, 허용오차)
 ('v8_final_numbers.json',['blocks','security','pE_human'],0.049606,1e-6),
 ('v8_final_numbers.json',['blocks','industrial','pE_human'],0.337725,1e-6),
 ('v8_final_numbers.json',['blocks','conventional','pE_human'],0.379786,1e-6),
 ('v8_final_numbers.json',['blocks','security','A'],22.5304,1e-3),
 ('v8_final_numbers.json',['blocks','security','TPR'],0.484831,1e-6),
 ('v8_final_numbers.json',['blocks','security','FPR'],0.085759,1e-6),
 ('v8_final_numbers.json',['m_block','security'],1.117647,1e-6),
 ('v8_final_numbers.json',['m_block','industrial'],1.748954,1e-6),
 ('v8_final_numbers.json',['rg_identity_max_abs_diff'],0.0,1e-12),
 ('human_validation.json',['human_human','overall_5class','kappa'],0.848150,1e-5),
 ('human_validation.json',['human_vs_A','security_binary','kappa'],0.236443,1e-5),
 ('text_rule_pair_validation.json',None,None,None),  # 별도 처리
 ('revision_A.json',['registry','direct','slope'],-0.068003,1e-5),
 ('conditional_pE.json',['human','security','pE_cond'],0.139123,1e-5),
 ('conditional_pE.json',['human','industrial','pE_cond'],0.520683,1e-5),
 ('conditional_pE.json',['model_query_level','p'],0.009596,1e-5),
]
V10=os.path.dirname(P.v10('human_m_security.json'))+'/'
BASE_V10=[  # v10 추가 산출물
 ('human_m_security.json',['variants','all22','m'],1.833333,1e-5),
 ('human_m_security.json',['variants','census18','m'],1.636364,1e-5),
 ('human_m_security.json',['variants','all22','n_events'],12,0),
 ('human_m_security.json',['variants','all22','A'],36.957754,1e-4),
 ('human_m_security.json',['coderA_m_security'],1.117647,1e-6),
 ('anchor_free_check.json',['hf','rate'],0.107438,1e-6),
 ('anchor_free_check.json',['controls','염소 가스 누출','rate'],0.016598,1e-6),
 ('anchor_free_check.json',['controls','황산 유출 사고','rate'],0.019685,1e-6),
 ('anchor_free_check.json',['controls','질산 누출','rate'],0.037037,1e-6),
]
bad=[]
for f,path,exp,tol in BASE:
    try: d=json.load(open(B+f))
    except Exception as e: bad.append((f,'load fail',str(e))); continue
    if path is None: continue
    v=d
    try:
        for k in path: v=v[k] if not isinstance(v,list) else v[int(k)]
    except Exception as e: bad.append((f,'/'.join(path),'missing: %s'%e)); continue
    ok=abs(float(v)-exp)<=tol
    print(('OK  ' if ok else 'FAIL'),'%-26s %-42s %s (기준 %s)'%(f,'/'.join(path),v,exp))
    if not ok: bad.append((f,'/'.join(path),v))
try:
    t=json.load(open(B+'text_rule_pair_validation.json'))
    row=[r for r in (t if isinstance(t,list) else t.get('rows',[])) if r.get('theta')==0.25 and not r.get('use_loc')]
    if row:
        r=row[0]; ok=abs(r['m_llm']-1.319)<2e-3 and abs(r['pair_f1']-0.361)<2e-3
        print(('OK  ' if ok else 'FAIL'),'text_rule m_llm=%.4f pair_f1=%.4f (기준 1.319 / 0.361)'%(r['m_llm'],r['pair_f1']))
        if not ok: bad.append(('text_rule','m_llm/pair_f1',(r['m_llm'],r['pair_f1'])))
except Exception as e: print('SKIP text_rule:',e)

for f,path,exp,tol in BASE_V10:
    try: d=json.load(open(V10+f))
    except Exception as e: bad.append((f,'load fail',str(e))); continue
    v=d
    try:
        for k in path: v=v[k]
    except Exception as e: bad.append((f,'/'.join(map(str,path)),'missing: %s'%e)); continue
    ok=abs(float(v)-exp)<=max(tol,0)
    print(('OK  ' if ok else 'FAIL'),'%-26s %-42s %s (기준 %s)'%(f,'/'.join(map(str,path)),v,exp))
    if not ok: bad.append((f,'/'.join(map(str,path)),v))

print('\n%s'%('회귀 없음 — 배포 가능' if not bad else 'FAIL %d건: %s'%(len(bad),bad)))
sys.exit(1 if bad else 0)
