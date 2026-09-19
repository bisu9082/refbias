"""4차 패널 R-G 지적: 레지스트리 물질 매칭이 염류·유도체를 포함한다.

원 규칙은 자유기술 사고물질 필드에 대한 부분문자열 일치이므로
질산칼륨/질산암모늄(질산염), 톨루엔 디이소시아네이트(TDI),
차아염소산나트륨 단독(염소가스 아님), 붕불산(플루오린화붕산) 등이 함께 잡힌다.
엄격 재코딩으로 회귀를 다시 돌려 결론이 바뀌는지 확인한다.
또한 원 집계 151 은 물질별 합으로, 혼산 9건이 중복 계상된 값임을 확인한다.
"""
import os, sys, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy import stats
import _paths as P

W_FROM, W_TO = '2021-09-09', '2025-04-28'
c = pd.read_csv(P.data('caris_incidents_raw.csv'), encoding='cp949')
c['dt'] = pd.to_datetime(c['사고일자'], errors='coerce')
c = c[(c.dt >= W_FROM) & (c.dt <= W_TO)].copy()
c['mats'] = (c['제1사고물질'].fillna('') + '|' + c['제2사고물질'].fillna('') + '|' + c['제3사고물질'].fillna(''))

ORIG = {'불산': r'불산|불화수소|플루오르화수소', '암모니아': r'암모니아', '염소': r'^염소|차아염소',
        '황산': r'황산', '톨루엔': r'톨루엔', '질산': r'질산'}
# 엄격 규칙: 해당 물질 자체가 누출된 사고만. 염류·유도체·다른 화합물은 제외한다.
EXCLUDE = {'불산': r'붕불산', '질산': r'질산칼륨|질산암모늄|질산염', '톨루엔': r'디이소시아네이',
           '염소': r'', '황산': r'', '암모니아': r''}
def hits(pat, excl):
    m = c['mats'].str.contains(pat, regex=True, na=False)
    if excl:
        # 제외어만 있고 본 물질 표기가 따로 없는 행을 뺀다
        def keep(s):
            parts = [p for p in re.split(r'[|,]', s) if p.strip()]
            good = [p for p in parts if re.search(pat, p) and not re.search(excl, p)]
            return len(good) > 0
        m = m & c['mats'].map(keep)
    return set(c.index[m])

# 염소 단독 차아염소산나트륨 제외 (염소가스 누출이 아님). 황산과 반응해 염소가 발생한 건은 남긴다.
def chlorine_strict():
    idx = set()
    for i, s in c['mats'].items():
        if re.search(r'^염소|(\|)염소', s):
            idx.add(i); continue
        if '차아염소' in s and '황산' in s:   # 산 + 차아염소산염 → 염소 발생
            idx.add(i)
    return idx

orig = {k: hits(p, '') for k, p in ORIG.items()}
strict = {k: (chlorine_strict() if k == '염소' else hits(ORIG[k], EXCLUDE[k])) for k in ORIG}

news = dict(zip(pd.read_csv(P.data('caris_match.csv'))['query'],
                pd.read_csv(P.data('caris_match.csv'))['보도기사수']))
QMAP = {'불산': '불산 누출', '암모니아': '암모니아 누출 사고', '염소': '염소 가스 누출',
        '황산': '황산 유출 사고', '톨루엔': '톨루엔 누출', '질산': '질산 누출'}

def regress(counts):
    x = np.log(np.array([counts[k] for k in QMAP]))
    y = np.log(np.array([news[QMAP[k]] for k in QMAP]))
    r = stats.linregress(x, y)
    rho = stats.spearmanr(x, y)
    return dict(slope=float(r.slope), p=float(r.pvalue), r=float(r.rvalue),
                stderr=float(r.stderr), spearman_rho=float(rho.statistic), spearman_p=float(rho.pvalue),
                counts={k: int(counts[k]) for k in QMAP})

res = {'window': [W_FROM, W_TO], 'total_incidents_in_window': int(len(c))}
res['original'] = {'per_substance': {k: len(v) for k, v in orig.items()},
                   'sum_over_substances': sum(len(v) for v in orig.values()),
                   'unique_incidents': len(set().union(*orig.values())),
                   'regression': regress({k: len(v) for k, v in orig.items()})}
res['strict'] = {'per_substance': {k: len(v) for k, v in strict.items()},
                 'sum_over_substances': sum(len(v) for v in strict.values()),
                 'unique_incidents': len(set().union(*strict.values())),
                 'regression': regress({k: len(v) for k, v in strict.items()})}
res['dropped_by_strict_recode'] = {k: sorted(c.loc[list(orig[k] - strict[k]), 'mats'].str.strip('|').tolist())
                                   for k in ORIG if orig[k] - strict[k]}
res['double_counted_incidents'] = res['original']['sum_over_substances'] - res['original']['unique_incidents']
json.dump(res, open(P.v10('registry_recode.json'), 'w'), ensure_ascii=False, indent=1)

print('창 내 총 사고 %d' % len(c))
for tag in ['original', 'strict']:
    r = res[tag]
    print('\n[%s] 물질별 %s' % (tag, r['per_substance']))
    print('        합 %d / 고유 %d' % (r['sum_over_substances'], r['unique_incidents']))
    g = r['regression']
    print('        log-log slope %.3f (SE %.3f, p=%.3f, r=%.2f), Spearman rho=%.2f p=%.2f'
          % (g['slope'], g['stderr'], g['p'], g['r'], g['spearman_rho'], g['spearman_p']))
print('\n중복 계상 사고 %d 건' % res['double_counted_incidents'])
for k, v in res['dropped_by_strict_recode'].items():
    print(' 제외 %s (%d):' % (k, len(v)), '; '.join(x[:40] for x in v))
