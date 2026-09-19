"""저자 작업 트리와 공개 저장소 양쪽에서 동일하게 동작하는 경로 해석기.

저자 트리:  <root>/{data,code,Step4/outputs,Step5r,Step5v10}, 원고는 /home/claude/refbias_tex/main.tex
저장소:     <root>/{data,code,labels,outputs,outputs/Step5v10,paper/main.tex}
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _first(*cands):
    for c in cands:
        if os.path.exists(c):
            return c
    return cands[0]

def out(name=""):
    """Step5r 산출물 (저장소에서는 outputs/)"""
    return _first(os.path.join(ROOT, "Step5r", name), os.path.join(ROOT, "outputs", name))

def out4(name=""):
    """Step4/outputs 산출물 (저장소에서는 outputs/)"""
    return _first(os.path.join(ROOT, "Step4", "outputs", name), os.path.join(ROOT, "outputs", name))

def _in_existing_dir(*cands):
    """존재하는 파일이 있으면 그 경로, 없으면 존재하는 디렉토리 쪽 경로를 준다.
    새 산출물이 저장소 레이아웃에서 엉뚱한 루트에 떨어지는 것을 막는다."""
    for c in cands:
        if os.path.exists(c):
            return c
    for c in cands:
        if os.path.isdir(os.path.dirname(c)):
            return c
    return cands[0]

def v10(name=""):
    return _in_existing_dir(os.path.join(ROOT, "Step5v10", name),
                            os.path.join(ROOT, "outputs", "Step5v10", name))

def data(name=""):
    return _first(os.path.join(ROOT, "data", name), os.path.join(ROOT, "labels", name))

def goldset_A():
    return _first(os.path.join(ROOT, "data", "goldset_labeled_A.csv"),
                  os.path.join(ROOT, "labels", "goldset_labels_coderA.csv"))

def human(tag):
    """tag: 'H1' | 'H2'"""
    return _first(os.path.join(ROOT, "labels", "human_labels_%s.csv" % tag),
                  "/home/claude/refbias_repo/labels/human_labels_%s.csv" % tag,
                  os.path.join(ROOT, "Step4", "human_labeling", "human_labels_%s.xlsx" % tag))

def tex():
    return _first(os.path.join(ROOT, "paper", "main.tex"), "/home/claude/refbias_tex/main.tex")

def texlog():
    return _first(os.path.join(ROOT, "paper", "main.log"), "/home/claude/refbias_tex/main.log")
