"""P129: test the actual geometry helper and guard without importing Isaac Sim."""
import ast, importlib.util, math
from pathlib import Path
from types import SimpleNamespace
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('base_hull', ROOT/'robolab/core/task/hull_check.py')
hull=importlib.util.module_from_spec(spec);spec.loader.exec_module(hull)

def load_function(file, name, namespace):
    fn=next(n for n in ast.parse((ROOT/file).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(file),'exec'),namespace)
    return namespace[name]

ns={'lowest_world_z':hull.lowest_world_z,'_untransformed_hull':lambda w,n:SimpleNamespace(vertices=w.vertices[n])}
geometry_bottom=load_function('robolab/core/task/predicate_logic.py','geometry_bottom_height',ns)
guard=load_function('robolab/core/task/conditionals.py','_object_above_container_base',{'atomic':lambda f:f,'geometry_bottom_height':geometry_bottom})

def box(lo,hi):
    return torch.tensor([[x,y,z] for x in [-.05,.05] for y in [-.05,.05] for z in [lo,hi]])

def world(va,pa,vb,pb,qa=None,qb=None):
    ident=torch.tensor([[1.,0,0,0]])
    w=SimpleNamespace(vertices={'object':va,'container':vb}, poses={
        'object':(torch.tensor([pa]),ident if qa is None else qa),
        'container':(torch.tensor([pb]),ident if qb is None else qb)})
    w.get_pose=lambda n,env_id=None:tuple(x if env_id is None else x[env_id] for x in w.poses[n])
    return w

def test_centred_crate_accepts_low_object_and_rejects_below_base():
    w=world(box(-.018,.018),[0.,0.,.074],box(-.073,.073),[0.,0.,.123])
    assert guard(w,'object','container').tolist()==[True]
    assert guard(w,'object','container',env_id=0) is True
    w.poses['object'][0][0,2]=.045
    assert not guard(w,'object','container',env_id=0)

def test_butter_origin_below_mesh_is_valid():
    w=world(box(.026,.052),[0.,0.,.03],box(0.,.15),[0.,0.,.05])
    assert guard(w,'object','container').item()

def test_correct_and_inverted_identical_bowls():
    b=box(-.0275,.0275)
    w=world(b,[0.,0.,.1175],b,[0.,0.,.0775])
    assert guard(w,'object','container').item()
    assert not guard(w,'container','object').item()

def test_asset_origin_rebasing_preserves_decision_with_rotation():
    q=torch.tensor([[math.sqrt(.5),math.sqrt(.5),0.,0.]])
    w=world(box(-.02,.02),[0.,0.,.11],box(-.05,.05),[0.,0.,.10],qa=q,qb=q)
    before=guard(w,'object','container').clone()
    for name,delta in [('object',torch.tensor([.3,.2,-.4])),('container',torch.tensor([-.2,.5,.1]))]:
        # 90 degrees around X maps (x,y,z) -> (x,-z,y).
        w.vertices[name]=w.vertices[name]-delta
        w.poses[name][0][0]+=torch.tensor([delta[0],-delta[2],delta[1]])
    assert torch.equal(before,guard(w,'object','container'))

def test_rotated_vertices_not_rotated_bounding_box():
    v=torch.tensor([[0.,0.,1.],[1.,0.,0.],[1.,0.,1.]])
    q=torch.tensor([math.cos(math.pi/8),0.,-math.sin(math.pi/8),0.])
    value=hull.lowest_world_z(v,torch.zeros(3),q)
    assert torch.isclose(value,torch.tensor(math.sqrt(.5)),atol=1e-6)

def test_tilted_container_and_batch_independence():
    v=box(0.,.2)
    qs=torch.tensor([[1.,0,0,0],[math.sqrt(.5),0.,math.sqrt(.5),0.]])
    h=hull.lowest_world_z(v,torch.tensor([[0.,0.,.05],[0.,0.,.2]]),qs)
    assert torch.allclose(h,torch.tensor([.05,.15]),atol=1e-6)

def test_guard_retains_existing_one_centimetre_tolerance():
    w=world(box(0.,.1),[0.,0.,.041],box(0.,.2),[0.,0.,.05])
    assert guard(w,'object','container').item()
    w.poses['object'][0][0,2]=.039
    assert not guard(w,'object','container').item()
