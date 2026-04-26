from __future__ import annotations

import numpy as np
from .modern_reranker import ModernReranker

class TailReranker:
    def __init__(self, base_ranker, tail_ranker, top_k:int=10):
        self.base_ranker=base_ranker
        self.tail_ranker=tail_ranker
        self.top_k=top_k
    def score_titles(self, article:str, titles:list[str]) -> np.ndarray:
        n=len(titles)
        if n==0: return np.array([],dtype=float)
        base_scores=np.asarray(self.base_ranker.score_titles(article,titles),dtype=float)
        order=list(np.argsort(-base_scores))
        if n==1: return np.array([1.0])
        k=min(max(2,self.top_k),n)
        fixed=[order[0]]
        tail=order[1:k]
        rest=order[k:]
        tail_titles=[titles[i] for i in tail]
        tail_scores=np.asarray(self.tail_ranker.score_titles(article,tail_titles),dtype=float)
        tail_order=[tail[i] for i in np.argsort(-tail_scores)]
        final=fixed+tail_order+rest
        out=np.zeros(n,dtype=float)
        for r,idx in enumerate(final): out[idx]=float(n-r)
        return out

def build_tail_aux(name:str):
    name=name.lower().strip()
    if name=='bge':
        return ModernReranker()
    return None
