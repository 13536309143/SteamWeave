"""Causal curve-history features and training-only model selection for SteamWeave v2."""
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor,ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures,StandardScaler

VERSION='curve-history-v2'
SHAPE_DIM=10
HISTORY_DIM=SHAPE_DIM*2+4

@dataclass(frozen=True)
class Candidate:
    name:str
    family:str
    view:str='geometry'
    loss:str='squared_error'
    leaves:int=31
    minimum:int=12
    iterations:int=450
    l2:float=1.

# Frozen before viewing new outer-fold predictions. No external API calls.
CANDIDATES=(
    Candidate('geometry_hgb_l2','hgb'),
    Candidate('geometry_hgb_l1','hgb',loss='absolute_error'),
    Candidate('history_hgb_l2','hgb','history'),
    Candidate('history_hgb_l1','hgb','history',loss='absolute_error'),
    Candidate('semantic_history_hgb_l1','hgb','semantic',loss='absolute_error'),
    Candidate('history_hgb_smooth','hgb','history',leaves=15,minimum=25,l2=4.),
    Candidate('history_extra_trees','extra','history'),
    Candidate('geometry_polynomial_ridge','ridge'),
)

def curve_shape(curves):
    """Fit a quadratic to observed image geometry; no target or global normalization."""
    curves=np.asarray(curves,float)
    out=np.full((len(curves),SHAPE_DIM),np.nan);grid=np.linspace(0,1,32)
    for i,row in enumerate(curves):
        z=row[:32];valid=np.isfinite(z)
        if valid.sum()<5:continue
        x=grid[valid];y=z[valid];a,b,c=np.polyfit(x,y,2)
        vertex=-b/(2*a) if abs(a)>1e-8 else np.nan
        depth=c-b*b/(4*a) if np.isfinite(vertex) else np.nan
        residual=np.sqrt(np.mean((np.polyval([a,b,c],x)-y)**2))
        out[i]=[a,b,c,vertex,depth,residual,np.mean(y),np.std(y),x[0],x[-1]]
    return out

def causal_curve_history(records,features):
    """Strictly earlier curves, reset on observed repair; invariant to targets/future rows."""
    shape=curve_shape(features['image'][:,2:]);out=np.full((len(records),HISTORY_DIM),np.nan)
    by_device=defaultdict(list)
    for i,r in enumerate(records):by_device[r.device].append(i)
    for indices in by_device.values():
        prior=[];repair=None
        for i in sorted(indices,key=lambda j:records[j].time):
            r=records[i]
            if repair is not None and r.last_repair>repair+1e-5:prior=[]
            repair=r.last_repair
            out[i,-4:]=[0.,0.,0.,0.]
            if prior:
                j=prior[-1];age=r.time-records[j].time
                out[i,:SHAPE_DIM]=shape[j];out[i,-4:]=[age,1.,0.,float(len(prior))]
                if len(prior)>1:
                    k=prior[-2];gap=records[j].time-records[k].time
                    out[i,SHAPE_DIM:2*SHAPE_DIM]=(shape[j]-shape[k])/gap
                    out[i,-2]=gap
            if features['quality'][i,0]>0 and np.isfinite(shape[i]).all():prior.append(i)
    return out

def matrix(features,history,view):
    numeric=features['numeric'];shape=curve_shape(features['image'][:,2:])
    # Numeric maintenance counters and current curve geometry remain available to every view.
    pieces=[numeric,shape,features['quality'][:,[0,1,4,5,6,7,9,10,11]]]
    if view!='geometry':pieces.extend([history,features['maintenance'][:,10:]])
    if view=='semantic':pieces.extend([features['text'][:,2:8],features['quality'][:,2:4]])
    # Explicit horizon interactions help represent future degradation rather than current severity.
    pieces.append(shape[:,[2,4,6]]*numeric[:,0,None])
    return np.column_stack(pieces)

class ForecastHead:
    """Learn missing-column handling solely from fitting rows."""
    def __init__(self,candidate,seed=17):self.candidate=candidate;self.seed=seed
    def fit(self,x,y,sample_weight=None):
        x=np.asarray(x,float);self.columns_=np.flatnonzero(np.isfinite(x).any(axis=0));self.width_=x.shape[1]
        xx=x[:,self.columns_] if len(self.columns_) else np.zeros((len(x),1));c=self.candidate
        if c.family=='hgb':
            self.model_=HistGradientBoostingRegressor(loss=c.loss,max_iter=c.iterations,learning_rate=.05,
                max_leaf_nodes=c.leaves,min_samples_leaf=c.minimum,l2_regularization=c.l2,early_stopping=False,random_state=self.seed)
            self.model_.fit(xx,y,sample_weight=sample_weight)
        elif c.family=='extra':
            self.model_=ExtraTreesRegressor(n_estimators=300,min_samples_leaf=2,max_features=1.,n_jobs=2,random_state=self.seed)
            self.model_.fit(xx,y,sample_weight=sample_weight)
        elif c.family=='ridge':
            self.model_=make_pipeline(SimpleImputer(add_indicator=True),StandardScaler(),PolynomialFeatures(2,include_bias=False),Ridge(alpha=100.))
            self.model_.fit(xx,y,ridge__sample_weight=sample_weight)
        else:raise ValueError('Unknown forecast candidate family')
        return self
    def predict(self,x):
        x=np.asarray(x,float)
        if x.shape[1]!=self.width_:raise ValueError('Forecast feature width mismatch')
        xx=x[:,self.columns_] if len(self.columns_) else np.zeros((len(x),1))
        return self.model_.predict(xx)

def weights_for(records,indices,groups):
    counts=defaultdict(int)
    keys=[(groups[i],records[i].next_inspection) for i in indices]
    for k in keys:counts[k]+=1
    w=np.array([1/counts[k] for k in keys]);return w/w.mean()

def development_splits(records,train,groups,temporal=False):
    if not temporal:
        for a,b in GroupKFold(n_splits=3).split(train,groups=groups[train]):yield train[a],train[b]
    else:
        # Forward event validation wholly inside outer training: future labels never enter a fit.
        for fraction in (.5,.65,.8):
            fit=[];val=[]
            for d in sorted({records[i].device for i in train}):
                ix=[i for i in train if records[i].device==d]
                events=sorted({records[i].next_inspection for i in ix});cut=events[max(0,int(len(events)*fraction)-1)]
                end=events[min(len(events)-1,max(1,int(len(events)*(fraction+.15))-1))]
                fit.extend(i for i in ix if records[i].next_inspection<=cut)
                val.extend(i for i in ix if records[i].time>=cut and cut<records[i].next_inspection<=end)
            if not fit or not val:raise ValueError('Insufficient training events for forward validation')
            yield np.array(fit),np.array(val)

def select_head(records,features,history,y,train,groups,seed=17,temporal=False,progress=None):
    splits=list(development_splits(records,train,groups,temporal));audit=[];scores=[]
    views={v:matrix(features,history,v) for v in {c.view for c in CANDIDATES}}
    for fit,val in splits:
        assert not set(fit)&set(val)
        if not temporal:assert not set(groups[fit])&set(groups[val])
        audit.append({'fit_indices':fit.tolist(),'validation_indices':val.tolist()})
    for c in CANDIDATES:
        fold_scores=[]
        for fit,val in splits:
            m=ForecastHead(c,seed).fit(views[c.view][fit],y[fit],weights_for(records,fit,groups))
            fold_scores.append(float(np.mean(np.abs(m.predict(views[c.view][val])-y[val]))))
        score={'candidate':c.name,'inner_mae_adu':float(np.mean(fold_scores)),'fold_mae_adu':fold_scores}
        scores.append(score)
        if progress:progress(c.name,score['inner_mae_adu'])
    winner=min(range(len(scores)),key=lambda j:scores[j]['inner_mae_adu']);candidate=CANDIDATES[winner]
    m=ForecastHead(candidate,seed).fit(views[candidate.view][train],y[train],weights_for(records,train,groups))
    return m,{'version':VERSION,'selected':candidate.name,'candidates':scores,'inner_folds':audit,'selection_metric':'mean inner-validation row MAE'}
