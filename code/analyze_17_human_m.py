"""A1: 인간 합의 EVENT(안보 블록)에 대한 분석자 3차 코딩 기반 m 재산출.

배경: 인간 코더 H1/H2는 ev_date/ev_place/ev_object 필드를 기입하지 않았다(§5.4).
코더 A 라벨 위에서 계산된 m(security)=1.12 는 정밀도 0.23 의 집합에서 나온 값이므로
사건의 속성으로 해석할 수 없다는 지적(2차 패널 R-A M1)에 대응해,
분석자가 제목+발췌만으로 22건에 (event_date, place, object)를 3차 코딩하고
원 코드북과 동일한 entity-and-window 규칙으로 군집화한다.

주의: 이 코딩은 원 코더 2인이 아니라 분석자 1인이 수행했으며 맹검이 아니다.
      군집 배정은 제목만으로 독자가 검증 가능하도록 전량 예치한다.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths as P
import json, numpy as np, pandas as pd

RNG = np.random.default_rng(20260919)

# 분석자 3차 코딩: sid -> (event_key, event_date, place, object)
CODING = {
 "S01428": ("navalny_2024",      "2024-02-16", "Russia",        "Novichok poisoning (death of A. Navalny)"),
 "S01832": ("onagawa_2026",      "2026-05-16", "Miyagi, Japan", "radioactive steam release, Onagawa unit 2"),
 "S00016": ("bupyeong_2022",     "2022-04-25", "Incheon",       "unidentified white powder in a public park"),
 "S02269": ("daegu_powder_2022", "2022-06-09", "Daegu Suseong", "white powder and syringes in a flat ceiling"),
 "S01933": ("daegu_powder_2022", "2022-06-09", "Daegu Suseong", "white powder and syringes in a flat ceiling"),
 "S00212": ("daegu_powder_2022", "2022-06-09", "Daegu Suseong", "white powder and syringes in a flat ceiling"),
 "S00532": ("daegu_powder_2022", "2022-06-09", "Daegu Suseong", "white powder and syringes in a flat ceiling"),
 "S02599": ("daegu_powder_2022", "2022-06-09", "Daegu Suseong", "white powder and syringes in a flat ceiling"),
 "S02339": ("daegu_powder_2022", "2022-06-09", "Daegu Suseong", "white powder and syringes in a flat ceiling"),
 "S02482": ("haeundae_2024",     "2024-08-29", "Busan Haeundae","white-powder letter to a public agency"),
 "S01998": ("parcels_2023",      "2023-07-21", "Haman/nationwide","suspicious international parcels"),
 "S00300": ("jeju_anthrax_2022", "2022-09-28", "Jeju Jocheon",  "suspected anthrax mail"),
 "S01363": ("jeju_anthrax_2022", "2022-09-28", "Jeju Jocheon",  "suspected anthrax mail"),
 "S01004": ("popasna_2022",      "2022-03-13", "Luhansk Popasna","alleged white phosphorus munition"),
 "S01700": ("popasna_2022",      "2022-03-13", "Luhansk Popasna","alleged white phosphorus munition"),
 "S02054": ("popasna_2022",      "2022-03-13", "Luhansk Popasna","alleged white phosphorus munition"),
 "S01623": ("mariupol_2022",     "2022-04-11", "Mariupol",      "alleged chemical weapon use"),
 "S02293": ("mariupol_2022",     "2022-04-11", "Mariupol",      "alleged chemical weapon use"),
 "S01214": ("mariupol_2022",     "2022-04-11", "Mariupol",      "alleged chemical weapon use"),
 "S00512": ("zaporizhzhia_2022", "2022-08-20", "Zaporizhzhia",  "alleged botulinum toxin use"),
 "S00136": ("chloropicrin_apr24","2024-04-07", "Ukraine front", "alleged banned chemical weapon attack"),
 "S02569": ("chloropicrin_may24","2024-05-01", "Ukraine",       "US statement on chloropicrin use"),
}
# 민감도 변형
SPLIT_FOLLOWUP = {"S02339": "daegu_powder_2022_followup"}       # 3주 뒤 국과수 감정 후속을 분리
MERGE_CHLORO   = {"S00136": "chloropicrin_2024", "S02569": "chloropicrin_2024"}

g  = pd.read_csv(P.goldset_A())
h1 = pd.read_csv(P.human("H1"))[["sid","label"]]
h2 = pd.read_csv(P.human("H2"))[["sid","label"]]
d  = h1.merge(h2, on="sid", suffixes=("_1","_2")).merge(g, on="sid")
ev = d[(d.label_1 == d.label_2) & (d.label_1 == "EVENT") & (d.block == "안보")].copy()
assert len(ev) == 22, len(ev)
ev["ekey"] = ev.sid.map(lambda s: CODING[s][0])
ev["census"] = ev.label == "EVENT"          # 코더 A 가 EVENT 로 부른 셀(전수층)
assert ev.census.sum() == 18, ev.census.sum()

def m_of(sub, remap=None):
    k = sub.sid.map(lambda s: (remap or {}).get(s, CODING[s][0]))
    return len(sub) / k.nunique(), len(sub), k.nunique()

def boot_m(sub, remap=None, B=10000):
    k = sub.sid.map(lambda s: (remap or {}).get(s, CODING[s][0])).values
    keys = np.unique(k)
    out = []
    for _ in range(B):
        pick = RNG.choice(keys, size=len(keys), replace=True)
        n_art = sum((k == p).sum() for p in pick)
        out.append(n_art / len(pick))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

res = {"coding_note": "analyst third coding pass, single non-blind coder, titles+excerpt only",
       "n_articles_consensus_event": int(len(ev)), "variants": {}}

for name, sub, remap in [
    ("all22",  ev,              None),
    ("census18", ev[ev.census], None),
    ("all22_split_followup", ev, {**{s: CODING[s][0] for s in CODING}, **SPLIT_FOLLOWUP}),
    ("all22_merge_chloropicrin", ev, {**{s: CODING[s][0] for s in CODING}, **MERGE_CHLORO}),
]:
    m, na, nev = m_of(sub, remap)
    lo, hi = boot_m(sub, remap)
    res["variants"][name] = {"m": m, "n_articles": na, "n_events": nev, "ci95": [lo, hi]}
    print(f"{name:28s} m={m:.3f} [{lo:.3f}, {hi:.3f}]  ({na} articles / {nev} events)")

# A = m / pE 재계산 (pE 는 사후층화 인간 기준값, v8_final_numbers.json)
v8 = json.load(open(P.out("v8_final_numbers.json")))
pE = v8["security_human"]["pE_human"] if "security_human" in v8 else None
if pE is None:
    pE = json.load(open(P.out("human_ps.json")))["pE_human_ps"]["security"]["pE"]
pE_ci = json.load(open(P.out("human_ps.json")))["pE_human_ps"]["security"]["ci95"]
res["pE_human_security"] = {"pE": pE, "ci95": pE_ci}
for name, v in res["variants"].items():
    v["A"] = v["m"] / pE
    v["A_range_from_pE_ci"] = [v["m"] / pE_ci[1], v["m"] / pE_ci[0]]
    print(f"{name:28s} A={v['A']:.1f}  (pE CI 만 반영: {v['A_range_from_pE_ci'][0]:.1f}–{v['A_range_from_pE_ci'][1]:.1f})")


# ---- 원 코드북 군집 함수를 실제로 적용한다 (규칙 적용 사실을 코드로 보증) ----
import importlib.util as _il
_sp=_il.spec_from_file_location("_cf", os.path.join(os.path.dirname(os.path.abspath(__file__)),"_cluster_funcs.py"))
_cf=_il.module_from_spec(_sp); _sp.loader.exec_module(_cf)
cl = ev.assign(ev_date=[CODING[s][1].replace("-","") for s in ev.sid],
               ev_place=[CODING[s][2] for s in ev.sid],
               ev_object=[CODING[s][3] for s in ev.sid])
rule_ids = _cf.cluster_gold(cl, window=3)
n_rule = pd.Series(rule_ids).nunique()
res["rule_applied"] = {"function":"_cluster_funcs.cluster_gold(window=3)",
    "n_events_by_rule": int(n_rule), "n_events_by_key": int(ev.ekey.nunique()),
    "match": bool(n_rule == ev.ekey.nunique()),
    "note":"군집은 ev_date(사건일) 기준이며 기사 게재일 기준이 아니다. 원 코드북과 동일."}
print("rule-applied events:", n_rule, "== key events:", ev.ekey.nunique())

# ---- 설계가중 m (전수층 pi=1, 비전수층 pi=150/805) ----
PI_NONEVENT = 150/805
w = np.where(ev.census.values, 1.0, 1.0/PI_NONEVENT)
k = ev.ekey.values
wt_articles = w.sum()
wt_events = sum(max(w[k==key]) for key in np.unique(k))   # 사건은 그 사건을 담은 기사 중 최대 가중으로 1회 계상
res["variants"]["all22_design_weighted"] = {"m": float(wt_articles/wt_events),
    "n_articles": float(wt_articles), "n_events": float(wt_events),
    "note":"기사에 설계가중을 주고 사건은 최대가중 1회로 계상. pE 와 추정량 기반을 맞추기 위한 대안."}
print("design-weighted m = %.3f"%(wt_articles/wt_events))

# ---- leave-one-cluster-out ----
loo={}
for key in np.unique(k):
    sub = ev[ev.ekey != key]
    loo[key] = len(sub)/sub.ekey.nunique()
res["leave_one_cluster_out"]={"values":{a:float(b) for a,b in loo.items()},
    "range":[float(min(loo.values())),float(max(loo.values()))]}
print("LOO m range %.2f-%.2f (min drops %s)"%(min(loo.values()),max(loo.values()),min(loo,key=loo.get)))

res["coderA_m_security"] = 1.1176470588235294
_ix = ev.set_index("sid")
_has_title = "제목" in ev.columns   # 공개 저장소의 라벨 파일에는 기사 제목이 없다(아카이브 저작권)
pd.DataFrame([{"sid": s, "event_key": CODING[s][0], "event_date": CODING[s][1],
               "place": CODING[s][2], "object": CODING[s][3],
               "coderA_label": _ix.label.get(s),
               "article_date": _ix["일자"].get(s),
               "title": (_ix["제목"].get(s) if _has_title else "")} for s in ev.sid]
             ).to_csv(P.v10("human_event_coding_security.csv"), index=False)
json.dump(res, open(P.v10("human_m_security.json"), "w"), ensure_ascii=False, indent=1)
print("\n코더 A 기준 m =", res["coderA_m_security"])
