"""A2: 앵커 비의존 과거사건 인용률.

2차 패널 R-A M4: 대조 위험물(염소·황산·질산)을 HF 의 구미 앵커로 채점하면
"앵커 없는 물질이 남의 앵커를 인용하지 않는다"는 정의상의 결과가 나온다는 지적.

대응: 특정 앵커 어휘에 의존하지 않는 측정을 추가한다.
     기사 본문(제목+발췌)에 수집창 이전 연도(1990-2020)가 사고어와 함께 등장하는 비율을
     30개 질의 전부에 대해 동일 규칙으로 계산한다. 어떤 과거 사건이든 인용하면 잡힌다.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths as P
import re, json, numpy as np, pandas as pd

RNG = np.random.default_rng(20260919)
YEAR = re.compile(r'(?<!\d)(19[9]\d|20[0-1]\d|2020)\s*년?(?!\d)')
INCIDENT = re.compile(r'사고|참사|사태|누출|폭발|테러|공격|유출|피폭|살포|중독')
WIN = 40   # 연도 토큰 기준 좌우 문자수

def cites_past(text):
    if not isinstance(text, str):
        return False
    for m in YEAR.finditer(text):
        seg = text[max(0, m.start()-WIN): m.end()+WIN]
        if INCIDENT.search(seg):
            return True
    return False

df = pd.read_csv(P.data("corpus_bigkinds.csv.gz"),
                 usecols=["qid","block","query","제목","본문"])
df["txt"] = df["제목"].fillna("") + " " + df["본문"].fillna("").str.slice(0, 200)
df["past"] = df.txt.map(cites_past)

def boot(v, B=10000):
    v = np.asarray(v, dtype=bool)
    n = len(v); k = int(v.sum())
    s = RNG.binomial(n, k / n, size=B) / n
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))

rows = []
for (qid, q, b), sub in df.groupby(["qid","query","block"]):
    lo, hi = boot(sub.past.values)
    rows.append(dict(qid=qid, query=q, block=b, n=len(sub),
                     rate=float(sub.past.mean()), lo=lo, hi=hi))
out = pd.DataFrame(rows).sort_values("rate", ascending=False)
out.to_csv(P.v10("generic_past_citation_by_query.csv"), index=False)
print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

ANCHORED = {"불산 누출","사린가스 공격","탄저균 테러","노비촉","방사능 유출"}
CONTROLS = {"황산 유출","질산 누출","염소 가스 누출"}
res = {"rule": "제목+발췌 200자에 1990-2020 연도가 사고어와 40자 이내로 공기하면 과거사건 인용",
       "by_query": out.to_dict("records")}
for name, keys in [("anchored_hazards", ANCHORED), ("industrial_controls", CONTROLS)]:
    s = out[out.query_.isin(keys)] if "query_" in out else out[out["query"].isin(keys)]
    res[name] = {"queries": list(s["query"]), "rate_range": [float(s.rate.min()), float(s.rate.max())],
                 "pooled": float((s.rate * s.n).sum() / s.n.sum())}
    print(name, res[name])
for b, s in out.groupby("block"):
    res.setdefault("by_block", {})[b] = {"pooled": float((s.rate*s.n).sum()/s.n.sum()),
                                         "median_query_rate": float(s.rate.median())}
print(res["by_block"])
json.dump(res, open(P.v10("generic_past_citation.json"),"w"), ensure_ascii=False, indent=1)
