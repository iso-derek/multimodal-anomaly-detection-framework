"""Aligned event-level fusion study, using disjoint calibration/validation/test groups."""
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score
from src.reliability import fuse_reliability
MODALITIES=['tabular','timeseries','image','video']

def synthetic_cases(n=1800,seed=42):
    rng=np.random.default_rng(seed)
    labels=rng.binomial(1,.25,n)
    df=pd.DataFrame({'case_id':[f'case-{i}' for i in range(n)],'group_id':[f'group-{i}' for i in range(n)],
                     'split':np.repeat(['calibration','validation','test'],[n//3,n//3,n-2*(n//3)]),'label':labels})
    latent=np.clip(.25+.5*labels+rng.normal(0,.13,n),0,1)
    for m,noise in zip(MODALITIES,[.18,.2,.25,.3]):
        quality=rng.uniform(.55,1,n)
        df['quality_'+m]=quality
        df['score_'+m]=np.clip(quality*(latent+rng.normal(0,noise,n))+(1-quality)*rng.random(n),0,1)
    return df

def validate(df):
    required=['case_id','group_id','split','label']+['score_'+m for m in MODALITIES]+['quality_'+m for m in MODALITIES]
    if not set(required)<=set(df):raise ValueError('Missing aligned-score columns.')
    if df[['case_id','group_id','split','label']].isna().any().any() or df.case_id.duplicated().any():
        raise ValueError('Unique case IDs and complete grouping/labels required.')
    if not df.label.isin([0,1]).all() or set(df.split)!= {'calibration','validation','test'}:
        raise ValueError('Binary labels and calibration/validation/test splits required.')
    if df.groupby('group_id').split.nunique().max()>1:raise ValueError('A group cannot cross partitions.')
    for m in MODALITIES:
        scores=df['score_'+m];quality=df['quality_'+m]
        if not (scores.isna() | scores.between(0,1)).all() or not quality.between(0,1).all():
            raise ValueError('Scores may be missing, otherwise scores and qualities must be in [0,1].')
    for _,part in df.groupby('split'):
        if len(part)<20 or part.label.nunique()!=2:raise ValueError('Each split needs 20 cases and both classes.')

def score_frame(df,priors,adaptive):
    predictions=[]
    for _,row in df.iterrows():
        result=fuse_reliability({m:{'score_norm':row['score_'+m]} for m in MODALITIES},priors,
                                {m:row['quality_'+m] if adaptive else 1 for m in MODALITIES})
        predictions.append(result['final_score'])
    return np.array([np.nan if p is None else p for p in predictions])

def run_study(df,target_fpr=.05,seed=42):
    validate(df)
    if not 0<target_fpr<1:raise ValueError('Target FPR must be in (0,1).')
    cal=df[df.split.eq('calibration')];val=df[df.split.eq('validation')];test=df[df.split.eq('test')]
    priors={}
    for m in MODALITIES:
        observed=cal['score_'+m].notna()
        priors[m]=float(1/(.05+np.mean((cal.loc[observed,'score_'+m]-cal.loc[observed,'label'])**2))) if observed.sum()>=10 else 0.
    methods={'equal':({m:1 for m in MODALITIES},False),'fixed_reliability':(priors,False),'quality_adaptive':(priors,True)}
    thresholds={}
    for method,(weights,adaptive) in methods.items():
        scores=score_frame(val,weights,adaptive)
        normal=scores[(val.label.to_numpy()==0)&np.isfinite(scores)]
        if len(normal)<10:raise ValueError('Too few covered validation negatives to freeze a threshold.')
        thresholds[method]=float(np.nextafter(np.quantile(normal,1-target_fpr,method='higher'),np.inf))
    scenarios={'clean':test.copy()}
    for m in MODALITIES:
        corrupted=test.copy();corrupted['score_'+m]=np.nan;corrupted['quality_'+m]=0
        scenarios['missing_'+m]=corrupted
    for noise in [.25,.5,.75,1.]:
        corrupted=test.copy();rng=np.random.default_rng(seed)
        corrupted['score_video']=(1-noise)*corrupted.score_video+noise*rng.random(len(test))
        corrupted['quality_video']*=1-noise
        scenarios[f'video_noise_{noise:g}']=corrupted
    unknown=scenarios['video_noise_1'].copy();unknown['quality_video']=test.quality_video
    scenarios['video_noise_quality_unknown']=unknown
    records=[];predictions=[]
    for scenario,part in scenarios.items():
        for method,(weights,adaptive) in methods.items():
            scores=score_frame(part,weights,adaptive);covered=np.isfinite(scores);y=part.label.to_numpy()[covered]
            pred=(scores[covered]>=thresholds[method]).astype(int)
            records.append({'scenario':scenario,'method':method,'coverage':float(covered.mean()),
                            'auc':float(roc_auc_score(y,scores[covered])) if len(np.unique(y))==2 else None,
                            'average_precision':float(average_precision_score(y,scores[covered])) if y.sum() else None,
                            'f1':float(f1_score(y,pred,zero_division=0)) if len(y) else None,
                            'false_positive_rate':float(pred[y==0].mean()) if (y==0).any() else None})
            predictions.extend({'case_id':case,'scenario':scenario,'method':method,'score':None if not np.isfinite(s) else float(s),'label':int(label)} for case,s,label in zip(part.case_id,scores,part.label))
    meta={'seed':seed,'target_validation_fpr':target_fpr,'priors':priors,'thresholds':thresholds,
          'input_sha256':hashlib.sha256(df.to_csv(index=False).encode()).hexdigest(),
          'note':'Scores are detector outputs, not calibrated probabilities. Synthetic quality knows injected corruption.'}
    return pd.DataFrame(records),pd.DataFrame(predictions),meta
