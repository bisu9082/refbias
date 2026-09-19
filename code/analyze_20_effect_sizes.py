"""O8: 블록 검정 효과크기의 편의와 구간.

지적: 블록당 10개 쿼리에서 eta2 = 0.30 은 상향 편의를 가질 수 있고, 30개 쿼리는
      쿼리 모집단의 확률표본이 아니라 세 목록에서 목적표집된 것이다.
대응: (a) eta2 와 epsilon2 를 함께 보고, (b) 순열로 귀무분포를 구해 편의를 실측,
      (c) 쿼리 재표집으로 효과크기 구간을 제시한다.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths as P
import json, numpy as np, pandas as pd
from scipy import stats
RNG = np.random.default_rng(20260919)
BASE = P.ROOT
Q = pd.read_csv(P.v10("human_query_level.csv"))
BLOCKS = ["industrial", "security", "conventional"]

def stats_of(vals):
    H, p = stats.kruskal(*vals)
    n = sum(len(v) for v in vals); k = len(vals)
    return H, p, (H - k + 1) / (n - k), H / (n - 1)      # eta2, epsilon2

res = {}
for col, tag in [("pE_model", "model_pE"), ("pE_human", "human_pE"), ("pE_cond_human", "human_pE_cond")]:
    vals = [Q.loc[Q.block == b, col].dropna().values for b in BLOCKS]
    H, p, eta, eps = stats_of(vals)
    pooled = np.concatenate(vals); sizes = [len(v) for v in vals]
    # 순열 귀무분포
    null_eta, null_eps = [], []
    for _ in range(10000):
        perm = RNG.permutation(pooled); i = 0; gs = []
        for s_ in sizes: gs.append(perm[i:i+s_]); i += s_
        try:
            H0, _, e0, ep0 = stats_of(gs)
            null_eta.append(e0); null_eps.append(ep0)
        except Exception:
            pass
    # 쿼리 재표집 구간 (블록 내 복원추출)
    boot = []
    for _ in range(4000):
        gs = [RNG.choice(v, len(v), replace=True) for v in vals]
        if any(len(np.unique(g)) == 1 for g in gs) and len(set(map(tuple, map(np.sort, gs)))) == 1:
            continue
        try:
            _, _, e, _ = stats_of(gs); boot.append(e)
        except Exception:
            pass
    res[tag] = {"H": float(H), "p": float(p), "eta2": float(eta), "epsilon2": float(eps),
        "null_eta2_mean": float(np.mean(null_eta)), "null_eta2_p95": float(np.percentile(null_eta, 95)),
        "eta2_boot_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
    print(f"[{tag}] H={H:.2f} p={p:.4f} eta2={eta:.2f} eps2={eps:.2f} | 귀무 eta2 평균 {np.mean(null_eta):.3f}, 95분위 {np.percentile(null_eta,95):.2f} | eta2 부트 CI {np.percentile(boot,2.5):.2f}-{np.percentile(boot,97.5):.2f}")
res["_note"] = ("eta2 = (H-k+1)/(n-k) 는 귀무에서 기댓값이 0 근처가 되도록 보정된 형태이고, "
                "epsilon2 = H/(n-1) 는 보정하지 않은 형태다. 순열 귀무분포로 실제 편의를 확인했다. "
                "추론 대상은 이 30개 쿼리이며, 쿼리는 세 목록에서 목적표집된 것이지 쿼리 모집단의 확률표본이 아니다.")
json.dump(res, open(P.v10("effect_sizes.json"), "w"), ensure_ascii=False, indent=1)

# O6: 불일치 처리별 안보 값 정리
hs = json.load(open(P.out("human_ps.json")))["consensus_sensitivity"]
sec = {k: v["security"] for k, v in hs.items()}
uniq = sorted(set(round(v, 6) for v in sec.values()))
print("\n[O6] 안보 블록 처리별 값:", {k: round(100*v, 2) for k, v in sec.items()})
print("     서로 다른 값:", [round(100*u, 1) for u in uniq])
json.dump({"security_by_treatment": sec, "distinct_values_pct": [100*u for u in uniq],
  "defensible_range_pct": [100*min(v for k, v in sec.items() if k != "disagree_all_EVENT"),
                           100*max(v for k, v in sec.items() if k != "disagree_all_EVENT")],
  "note": "disagree_all_EVENT 는 어느 코더도 EVENT 라 하지 않은 기사(40건 중 28건)를 EVENT 로 강제하므로 방어 가능한 상한이 아니다."},
  open(P.v10("disagreement_range.json"), "w"), ensure_ascii=False, indent=1)
