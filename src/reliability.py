"""Reliability-weighted fusion with explicit abstention."""
import numpy as np

def fuse_reliability(results,base_weights=None,reliability=None,threshold=.65):
    if not np.isfinite(threshold) or not 0<=threshold<=1:
        raise ValueError('Threshold must lie in [0,1].')
    base_weights=base_weights or {};reliability=reliability or {}
    evidence={}
    for modality,result in results.items():
        score=result.get('score_norm')
        if score is None or np.isnan(float(score)):
            continue
        score=float(score);weight=float(base_weights.get(modality,1));quality=float(reliability.get(modality,1))
        if not np.isfinite([score,weight,quality]).all() or not 0<=score<=1 or weight<0 or not 0<=quality<=1:
            raise ValueError('Scores and reliability must be in [0,1]; weights finite and nonnegative.')
        evidence[modality]={'score':score,'effective_weight':weight*quality}
    total=sum(item['effective_weight'] for item in evidence.values())
    if total<=0:
        return {'final_score':None,'final_label':None,'reason':'Abstain: no reliable evidence','by_modality':evidence}
    for item in evidence.values():item['effective_weight']/=total
    score=sum(item['score']*item['effective_weight'] for item in evidence.values())
    return {'final_score':score,'final_label':int(score>=threshold),
            'reason':'Reliability-weighted score; no strong-modality override','by_modality':evidence}
