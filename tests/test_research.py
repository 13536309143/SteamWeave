import importlib.util
import numpy as np
import pytest
from sg_agents.data import load_dataset,trajectory_groups

torch=pytest.importorskip('torch')
from research.run_suite import make_splits,score,transformed
from research.paper_baseline import MultiBranchDNN

def test_grouped_cv_uses_each_group_once_and_never_crosses_partitions():
    rows=load_dataset();mapping=trajectory_groups(rows)
    seen=[]
    for fold,train,cal,test,info in make_splits(rows,'grouped_cv',17):
        groups=[{mapping[rows[i].device] for i in ix} for ix in (train,cal,test)]
        assert not(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
        seen.extend(groups[2])
    assert sorted(seen)==list(range(1,31))

def test_paper_style_protocol_is_explicitly_diagnostic():
    rows=load_dataset();infos=[info for *_,info in make_splits(rows,'paper_device_cv',17)]
    assert all(info['diagnostic_only'] for info in infos)
    assert any(info['duplicate_groups_crossing_train_test'] for info in infos)
    import json
    assert all(json.dumps(info) for info in infos)

def test_paper_dnn_branch_dimensions_and_gradient():
    model=MultiBranchDNN(np.random.default_rng(1).normal(size=(20,100)).astype(np.float32))
    image=torch.zeros(2,1,325,194);text=torch.ones(2,27,dtype=torch.long);numeric=torch.zeros(2,4)
    p=model(image,text,numeric);assert p.shape==(2,)
    p.sum().backward();assert model.output.weight.grad is not None

def test_paper_metrics_include_mse_and_percent_relative_error():
    s=score(np.array([10.,20.]),np.array([12.,18.]))
    assert s['mae_adu']==2 and s['mse_adu2']==4 and s['mre_percent']==pytest.approx(15)
