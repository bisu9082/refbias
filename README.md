# refbias

Replication material for **"How much of hazard news is news? Decomposing open-source coverage of chemical accidents and CBRN threats for hazard surveillance"** (submitted to *Safety Science*).

The study decomposes 109,610 Korean news articles retrieved by 30 hazard queries into five components — new event, past-event reference, institutional activity, other, off-topic — and asks how much of hazard news volume is evidence that something happened. On a human reference standard, 5.0% of CBRN security articles report an event, against 38% for conventional terrorism and 34% for industrial chemical accidents.

## What is here

| Path | Contents |
|---|---|
| `labels/` | Component labels: two model coders, two human coders, and the sampling design for the human subset |
| `data/` | Article identifiers, query definitions, and the national chemical-accident registry extract |
| `code/` | Analysis scripts, in the order given below |
| `outputs/` | Every JSON and CSV the manuscript's numbers are read from |
| `figures/` | Figures 1–5 and the graphical abstract |
| `paper/` | LaTeX source, bibliography, and compiled PDFs |
| `codebook_v2.md` | The coding instruction given to every coder, human and model |

## What is not here, and why

**Article text.** The corpus was retrieved from BigKinds (bigkinds.or.kr), whose terms do not permit redistribution of article titles or body text. `data/corpus_article_ids.csv.gz` and `labels/goldset_labels_coderA.csv` therefore carry the BigKinds news identifier (`뉴스 식별자`), date, outlet and query for every article, but no text. A reader with BigKinds access can rejoin the text on that identifier and reproduce every result; a reader without it can still verify every computation that operates on labels, which is most of the paper.

**The model coders' prompt wrapper and interface-returned model identifiers.** These were not retained. The codebook that constituted the coding instruction is deposited in full (`codebook_v2.md`), and both model coders and both human coders worked from that same text, but the wrapper that presented each article to the model is gone. This limits exact reproduction of the coding runs, and the manuscript says so.

## Reproducing the results

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# label-only analyses (run without BigKinds access)
python code/analyze_12_human.py        # human validation: kappas, IPW event shares, misclassification rates
python code/analyze_14_human_ps.py     # post-stratified estimates, consensus-rule sensitivity, domestic share
python code/finalize_v8_numbers.py     # the numbers reported in the manuscript
python code/analyze_15_conditional_pE.py  # event share conditional on topical relevance
python code/analyze_16_v9_additions.py    # matched agreement table, 2x2, conditional anchor rates, power, attenuation
python code/analyze_19_human_query_tests.py # block tests on human labels, query-clustered intervals, domestic share
python code/analyze_22_registry_recall.py  # recall audit of the registry substance matching (S14)
python code/analyze_23_zero_cell_ci.py     # Jeffreys treatment of the zero-count cell (S13)
python code/integrity_audit.py            # three-way audit: raw data -> outputs -> manuscript
python code/verify_v8b.py                 # checks every human-validation number in main.tex against outputs/
python code/verify_from_raw.py            # recomputes headline numbers from labels/ and data/ alone, no JSONs
python code/regression_check.py           # guards the deposited JSONs against silent drift
python code/make_manifest.py --check      # byte size and SHA-256 of every deposited file against MANIFEST.md
```

`integrity_audit.py` runs three checks in one pass: it recomputes 24 quantities from the label files and compares them with the deposited JSONs, matches 60 values in `paper/main.tex` against those JSONs by context-anchored regex, and checks 16 quantities that appear in more than one file for mutual consistency. `verify_v8b.py` adds 65 further manuscript checks. `verify_from_raw.py` is the independent one: it reads only `labels/` and `data/`, recomputes 18 headline quantities from scratch — the gold-set shares, the post-stratified human shares, the inter-coder kappa and the whole registry regression — and compares them with what `paper/main.tex` says, so an error in the deposited JSONs cannot hide it. `regression_check.py` pins the headline quantities to fixed baselines; it exists because a refactor once changed a published number without anyone noticing.

The remaining scripts (`analyze_01`–`analyze_11`, `infer_corpus.py`, `make_figset*.py`) need the article text and will fail without `data/corpus_bigkinds.csv.gz`, which is not distributable.

## Headline quantities

| Block | n drawn / consensus | Coder A | Human (post-stratified) | 95% CI | Range over disagreement rules | On-topic only | Amplification |
|---|---|---|---|---|---|---|---|
| Industrial | 78 / 74 | 43.2% | 33.8% | 22.9–39.9 | 33.5–38.5 | 52.1% | 5.2 |
| Security (CBRN) | 245 / 219 | 10.6% | **5.0%** | 0.6–10.9 | 4.4–12.1 | **13.9%** | **22.5** |
| Conventional | 77 / 67 | 40.9% | 38.0% | 30.2–40.9 | 35.4–48.2 | 59.6% | 3.1 |

The three query sets retrieve with different precision (25.6% of security articles are off-topic against 9.2% industrial), so the last-but-one column gives the event share among on-topic articles only. The contrast holds: the security interval (1.9–31.2) reaches neither control's. All intervals in this table resample whole queries within the block, which is the unit used for the between-block comparisons; amplification carries no interval, because inverting a share interval whose lower end is 0.6% does not produce a usable one.

Inter-human agreement is $\kappa = 0.848$ over the five components and 0.905 on the event boundary. Agreement between the human standard and the model coder is 0.938 for conventional terrorism, 0.689 for industrial accidents and 0.236 for CBRN security threats: the model counted topically adjacent occurrences that the codebook excludes, and the paper reports which of its own results that failure does and does not license.

Amplification is a hybrid quantity — a human event share over a model-derived duplication factor — because the human coders recorded the component but not the event date, place and object. It is a lower bound.

## Licensing

Code is released under the MIT Licence (`LICENSE`). Labels, derived outputs and figures are released under CC BY 4.0 (`LICENSE-DATA`). Article identifiers are facts about the BigKinds archive and carry no licence from us; the article text they point to remains subject to BigKinds terms. The registry extract derives from the Chemical Safety Agency's public incident records.

## Citation

Kim, M., Cha, J., Shin, M., Choi, G., Kang, K. How much of hazard news is news? Decomposing open-source coverage of chemical accidents and CBRN threats for hazard surveillance. Submitted.

Corresponding author: Ku Kang (bisu9082@gmail.com).

## 검증 스크립트 실행

저장소 루트에서 그대로 실행된다. 경로는 `code/_paths.py` 가 저자 작업 트리와 이 저장소 양쪽을 해석한다.

```
python3 code/integrity_audit.py     # 3단 감사: 원자료→산출물, 산출물→원고, 교차파일
python3 code/regression_check.py    # 고정 기준값 회귀 점검
python3 code/verify_v8b.py          # 원고 표·본문 수치 대조
```

`integrity_audit.py` 는 현재 자리표시자 항목 하나를 의도적으로 실패시킨다(`bisu9082`, GitHub 계정 미확정).
그 외 항목이 실패하면 예치물과 원고가 어긋난 것이다.

분석 스크립트도 루트에서 실행된다. 다만 기사 텍스트(`data/corpus_bigkinds.csv.gz`)는 아카이브
이용약관상 재배포할 수 없으므로 텍스트에 의존하는 산출(앵커 비율, 분류기, `analyze_18`)은
빅카인즈 접근권이 있는 독자만 재현할 수 있다. 라벨에서 계산되는 값은 전부 재현 가능하다.
