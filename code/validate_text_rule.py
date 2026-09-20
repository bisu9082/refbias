import re,sys,json,itertools,os
import pandas as pd, numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths as _P
BASE=_P.ROOT
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# analyze_10_cluster_m.py 의 군집 함수를 정식 모듈 임포트로 재사용한다 (exec 제거, V-2 통과)
import importlib.util as _ilu
_spec=_ilu.spec_from_file_location('_cl',BASE+'/code/_cluster_funcs.py')
_cl=_ilu.module_from_spec(_spec); _spec.loader.exec_module(_cl)
cluster_gold=_cl.cluster_gold; cluster_text=_cl.cluster_text; norm=_cl.norm
ns={'cluster_gold':cluster_gold,'cluster_text':cluster_text,'norm':norm}

g=pd.read_csv(BASE+'/data/goldset_labeled_A.csv',dtype=str)
g['text']=g['제목'].fillna('')+' ▁ '+g['본문'].fillna('').str.replace(r'\s+',' ',regex=True).str[:200]
corpus=pd.read_csv(BASE+'/data/corpus_bigkinds.csv.gz',dtype=str,usecols=['뉴스 식별자','위치']).drop_duplicates('뉴스 식별자')
gold_ev=g[g.label=='EVENT'].copy(); gold_ev['d']=pd.to_datetime(gold_ev['일자'],format='%Y%m%d',errors='coerce')
gold_ev=gold_ev.merge(corpus.rename(columns={'위치':'loc'}),on='뉴스 식별자',how='left'); gold_ev['loc']=gold_ev['loc'].fillna('')
ref=(ns['cluster_gold'](gold_ev,3,True,True).reindex(gold_ev.index).astype(str)+'_'+gold_ev['qid']).values
def pairs(lbl):
    d={}; s=set()
    for i,l in enumerate(lbl): d.setdefault(l,[]).append(i)
    for ix in d.values():
        for a,b in itertools.combinations(ix,2): s.add((a,b))
    return s
P_ref=pairs(ref); out=[]
for theta in [0.20,0.25,0.35]:
    for use_loc in [True,False]:
        cid=pd.concat([ns['cluster_text'](s,3,theta,use_loc) for _,s in gold_ev.groupby('qid')]).reindex(gold_ev.index)
        lbl=(cid.astype(str)+'_'+gold_ev['qid']).values; P=pairs(lbl); tp=len(P&P_ref)
        prec=tp/len(P) if P else 0; rec=tp/len(P_ref)
        out.append(dict(theta=theta,use_loc=use_loc,pairs_text=len(P),pairs_llm=len(P_ref),pair_precision=prec,pair_recall=rec,pair_f1=(2*prec*rec/(prec+rec) if prec+rec else 0),
                        m_text=len(gold_ev)/len(set(lbl)),m_llm=len(gold_ev)/len(set(ref))))
        print({k:(round(v,3) if isinstance(v,float) else v) for k,v in out[-1].items()})
json.dump(out,open(BASE+'/Step5r/text_rule_pair_validation.json','w'),indent=1)
