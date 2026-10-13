"""Resume/update pressure tests independently from expensive neural training."""
import json
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from sg_agents.config import ARTIFACTS
from sg_agents.data import load_dataset,trajectory_groups
from .run_suite import make_splits,fit_steam,record_stress,delayed_report_features,transformed,steam_predict,score

def main():
    rows=load_dataset();features=joblib.load(ARTIFACTS/'features.joblib')['features'];y=np.array([r.target for r in rows])
    mapping=trajectory_groups(rows);groups=np.array([mapping[r.device] for r in rows])
    delayed={lag:delayed_report_features(rows,features,lag) for lag in (10,25,50)}
    with threadpool_limits(limits=2):
        for fold,train,cal,test,info in make_splits(rows,'grouped_cv',17):
            dest=ARTIFACTS/'research'/'runs'/'grouped_cv'/'seed_17'/f'fold_{fold}';dest.mkdir(parents=True,exist_ok=True)
            for variant in ('full','without_modality_dropout'):
                print('Extended pressure tests',fold,variant,flush=True)
                fitted=fit_steam(rows,features,y,train,cal,groups,17,variant)
                stress=record_stress(rows,features,fitted,y,test,groups,17+fold)
                for lag,ff in delayed.items():
                    lf,_=transformed(ff,variant);pred,_,_,_=steam_predict(fitted,lf,test)
                    stress.append({'type':'synthetic_report_delay','lag_atu':lag,**score(y[test],pred)})
                name='stress.json' if variant=='full' else 'stress_without_modality_dropout.json'
                (dest/name).write_text(json.dumps(stress,indent=2))
    print('Extended pressure tests complete',flush=True)

if __name__=='__main__':main()
