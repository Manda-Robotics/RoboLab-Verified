"""Actual P129 terminal poses: wrong-order left stacks reject; correct stacks retain."""
import json
from types import SimpleNamespace
import torch
from test_geometry_base import ROOT, hull, load_function

helper=load_function('robolab/core/task/predicate_logic.py','geometry_centroid_height',{
    'lowest_world_z':hull.lowest_world_z,
    '_untransformed_hull':lambda w,n:SimpleNamespace(centroid=w.centroids[n])})
guard=load_function('robolab/core/task/conditionals.py','_object_center_above_container',{
    'atomic':lambda f:f,'geometry_centroid_height':helper})

def test_recorded_terminal_orders_and_origin_rebasing():
    rows=json.loads((ROOT/'offline_tests/fixtures/bowl_order_p129.json').read_text())
    assert len(rows)==6
    for row in rows:
        bodies=row['bodies'];w=SimpleNamespace(centroids={n:torch.tensor(v['centroid']) for n,v in bodies.items()})
        poses={n:torch.tensor([v['pose']]) for n,v in bodies.items()}
        w.get_pose=lambda n,env_id=None: ((poses[n][:,:3],poses[n][:,3:]) if env_id is None else (poses[n][env_id,:3],poses[n][env_id,3:]))
        assert guard(w,row['upper'],row['lower']).item()==row['expected'],row
        assert guard(w,row['upper'],row['lower'],env_id=0)==row['expected'],row
        # Translation of local origin along local z; compensate world position
        # with the rotated local displacement, preserving actual world geometry.
        n=row['upper'];q=poses[n][0,3:];a,b,c,d=q/q.norm();shift=torch.tensor([2*(b*d+a*c),2*(c*d-a*b),1-2*(b*b+c*c)])*.2
        poses[n][0,:3]+=shift;w.centroids[n][2]-=.2
        assert guard(w,row['upper'],row['lower']).item()==row['expected'],row
