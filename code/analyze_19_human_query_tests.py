"""O1: 블록 대비 유의성 검정을 인간 합의 라벨 위에서 재실행한다.

배경: §4.1 의 쿼리 단위 블록 검정(H(2)=10.06, p=0.0065, eta2=0.30)은 모델 라벨에서
      산출되었고, §4.7 은 그 라벨이 안보 블록에서 인간 기준과 kappa 0.236 으로 갈린다는
      것을 보였다. 세 차례 심사 패널이 공통으로 "중심 주장이 검증되지 않은 라벨 위에 있다"를
      지적했다. 여기서는 동일한 검정을 인간 두 코더가 합의한 360건 위에서 다시 돌린다.

설계: 인간 표본은 블록 x 코더A-EVENT 6셀 층화표집(labels/human_sampling_design.csv).
      쿼리별 사건 비율은 골드셋에서 알려진 쿼리 내 셀 크기로 사후층화한다.
      p_E^q = sum_c N_qc * mean(y|q,c) / sum_c N_qc
검정: Kruskal-Wallis(H, eta2) + 쌍별 Mann-Whitney(Holm 보정, rank-biserial r).
      §4.1 과 동일한 절차이며 라벨만 바뀐다.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths as P
import json, numpy as np, pandas as pd
from scipy import stats

RNG = np.random.default_rng(20260919)
BASE = P.ROOT
BL = {"산업": "industrial", "안보": "security", "비CBRN": "conventional"}

g = pd.read_csv(P.goldset_A())
h1 = pd.read_csv(P.human("H1"))[["sid", "label"]]
h2 = pd.read_csv(P.human("H2"))[["sid", "label"]]
d = h1.merge(h2, on="sid", suffixes=("_1", "_2")).merge(g, on="sid")
cons = d[d.label_1 == d.label_2].copy()
cons["y"] = cons.label_1
cons["cell"] = np.where(cons.label == "EVENT", "EVENT", "non-EVENT")
g["cell"] = np.where(g.label == "EVENT", "EVENT", "non-EVENT")
Nq = g.groupby(["query", "cell"]).size()                      # 쿼리 내 셀 크기 (알려짐)

def ps_query(sub, q, predicate):
    """쿼리 내 사후층화 비율. predicate(y)->bool"""
    num = den = 0.0
    empty = 0
    for cell, gp in sub.groupby("cell"):
        N = Nq.get((q, cell), 0)
        if N == 0 or len(gp) == 0:
            continue
        num += N * predicate(gp).mean(); den += N
    # 표본이 없는 셀은 제외되므로 den 은 쿼리 전체가 아닐 수 있다
    for cell in ["EVENT", "non-EVENT"]:
        if Nq.get((q, cell), 0) > 0 and ((sub.cell == cell).sum() == 0):
            empty += 1
    return (num / den if den else np.nan), empty, den

rows = []
for (blk, q), sub in cons.groupby(["block", "query"]):
    pE, empty, den = ps_query(sub, q, lambda gp: gp.y == "EVENT")
    rel = sub[sub.y != "IRRELEVANT"]
    if len(rel):
        num_c = den_c = 0.0
        for cell, gp in sub.groupby("cell"):
            N = Nq.get((q, cell), 0)
            if N == 0 or len(gp) == 0: continue
            num_c += N * (gp.y == "EVENT").mean()
            den_c += N * (gp.y != "IRRELEVANT").mean()
        pEc = num_c / den_c if den_c else np.nan
    else:
        pEc = np.nan
    rows.append(dict(block=BL[blk], query=q, n=len(sub), n_event=int((sub.y == "EVENT").sum()),
                     pE_human=pE, pE_cond_human=pEc, empty_cells=empty,
                     pE_model=float((g[g["query"] == q].label == "EVENT").mean())))
Q = pd.DataFrame(rows)
Q.to_csv(P.v10("human_query_level.csv"), index=False)
print(Q.sort_values(["block", "pE_human"]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

def eta2_kw(H, k, n):
    return (H - k + 1) / (n - k)

def run(col, tag):
    grp = {b: Q.loc[Q.block == b, col].dropna().values for b in ["industrial", "security", "conventional"]}
    H, p = stats.kruskal(*grp.values())
    k, n = 3, sum(len(v) for v in grp.values())
    out = {"kruskal_H": float(H), "p": float(p), "eta2": float(eta2_kw(H, k, n)),
           "medians": {b: float(np.median(v)) for b, v in grp.items()},
           "means": {b: float(np.mean(v)) for b, v in grp.items()}, "pairs": {}}
    pairs = [("security", "industrial"), ("security", "conventional"), ("industrial", "conventional")]
    raw = []
    for a, b in pairs:
        U, pp = stats.mannwhitneyu(grp[a], grp[b], alternative="two-sided")
        r = 1 - 2 * U / (len(grp[a]) * len(grp[b]))
        raw.append((a, b, U, pp, abs(r)))
    ps = [x[3] for x in raw]
    order = np.argsort(ps); holm = np.empty(3)
    for rank, idx in enumerate(order):
        holm[idx] = min(1.0, max(ps[idx] * (3 - rank), holm[order[rank - 1]] if rank else 0))
    for (a, b, U, pp, r), hp in zip(raw, holm):
        out["pairs"][f"{a} vs {b}"] = {"U": float(U), "p_raw": float(pp), "p_holm": float(hp), "r": float(r)}
    print(f"\n[{tag}] H(2)={H:.2f} p={p:.4f} eta2={out['eta2']:.2f}  medians "
          + ", ".join(f"{b} {100*np.median(v):.0f}%" for b, v in grp.items()))
    for kk, vv in out["pairs"].items():
        print(f"   {kk:32s} U={vv['U']:.1f} p={vv['p_raw']:.4f} Holm={vv['p_holm']:.4f} r={vv['r']:.2f}")
    return out

res = {"design": "쿼리 내 사후층화(골드셋 셀 크기), 인간 두 코더 합의 360건",
       "n_queries": {b: int((Q.block == b).sum()) for b in Q.block.unique()},
       "articles_per_query": {"min": int(Q.n.min()), "median": float(Q.n.median()), "max": int(Q.n.max())},
       "queries_with_empty_cell": int((Q.empty_cells > 0).sum())}
res["human_pE"] = run("pE_human", "인간 라벨 pE")
res["human_pE_conditional"] = run("pE_cond_human", "인간 라벨 조건부 pE")
res["model_pE_rerun"] = run("pE_model", "모델 라벨 pE (동일 30쿼리, 대조용)")

# 블록 중앙값 차이에 대한 쿼리 클러스터 부트스트랩
def boot_gap(col, B=10000):
    out = []
    for _ in range(B):
        med = {}
        for b in ["industrial", "security", "conventional"]:
            v = Q.loc[Q.block == b, col].dropna().values
            med[b] = np.median(RNG.choice(v, len(v), replace=True))
        out.append((min(med["industrial"], med["conventional"]) - med["security"]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))
lo, hi = boot_gap("pE_human")
res["gap_control_min_minus_security"] = {"ci95": [lo, hi],
    "note": "두 대조 블록 중 낮은 쪽의 쿼리 중앙값에서 안보 중앙값을 뺀 차이. 쿼리 재표집 10,000회."}
print(f"\n대조군 최소 - 안보 (중앙값 차) 95% CI: {100*lo:.1f} — {100*hi:.1f} 포인트")
# ---- 블록 수준 인간 pE 를 쿼리 재표집으로 다시 구간화 ----
# §4.7 의 인간 구간은 셀 내 부트스트랩이며 쿼리 군집을 반영하지 않는다.
# §3.3 이 블록 간 비교의 단위를 쿼리로 규정했으므로, 원고와 동일한 블록-셀 사후층화
# 추정량을 유지한 채 쿼리를 재표집한 구간을 병기한다.
NB = g.groupby(["block", "cell"]).size()     # 블록 내 셀 크기 (원고와 동일)
def block_cell_ps(sub, blk, cond=False):
    num = den = 0.0
    for cell, cg in sub.groupby("cell"):
        N = NB.get((blk, cell), 0)
        if N == 0 or len(cg) == 0: continue
        num += N * (cg.y == "EVENT").mean()
        den += N * ((cg.y != "IRRELEVANT").mean() if cond else 1.0)
    return num, den

res["block_level_query_clustered"] = {}
for cond, tag in [(False, "pE"), (True, "pE_cond")]:
    entry = {}
    for blk, b in BL.items():
        sub = cons[cons.block == blk]
        qs = sorted(sub["query"].unique())
        n0, d0 = block_cell_ps(sub, blk, cond)
        pt = n0 / d0 if d0 else np.nan
        draws = []
        for _ in range(4000):
            pick = RNG.choice(qs, len(qs), replace=True)
            rs = pd.concat([sub[sub["query"] == q] for q in pick])
            n_, d_ = block_cell_ps(rs, blk, cond)
            if d_: draws.append(n_ / d_)
        draws = np.array(draws)
        entry[b] = {"point": float(pt),
                    "ci95_query_cluster": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}
        print(f"  [{tag}] {b:13s} {100*pt:5.1f}%  query-cluster CI {100*entry[b]['ci95_query_cluster'][0]:.1f}-{100*entry[b]['ci95_query_cluster'][1]:.1f}")
    ov = [entry["security"]["ci95_query_cluster"][1] < entry[b]["ci95_query_cluster"][0] for b in ["industrial", "conventional"]]
    entry["no_overlap_with_industrial"] = bool(ov[0])
    entry["no_overlap_with_conventional"] = bool(ov[1])
    res["block_level_query_clustered"][tag] = entry
    print(f"  [{tag}] 비중첩 — 산업 {ov[0]}, 재래식 {ov[1]}")
json.dump(res, open(P.v10("human_query_tests.json"), "w"), ensure_ascii=False, indent=1)
