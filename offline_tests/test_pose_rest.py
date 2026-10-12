"""P128: stability despite velocity jitter, without accepting excursions/reset history."""
import sys, types, math
import pytest
import torch
import robolab.constants as C
from robolab.core.task.rest import PoseRestTracker
from robolab.core.task.confirm import SuccessConfirmer

@pytest.fixture
def rig(monkeypatch):
    env = types.SimpleNamespace(num_envs=2, device='cpu', step_dt=.1, episode_length_buf=torch.zeros(2,dtype=torch.long), common_step_counter=0)
    world = types.SimpleNamespace(pos=torch.zeros(2,3), quat=torch.tensor([[1.,0,0,0]]*2))
    world.get_pose=lambda *a,**k:(world.pos,world.quat)
    world.get_velocity=lambda *a,**k:torch.ones(2,6)*.04
    m=types.ModuleType('robolab.core.world.world_state');m.get_world=lambda e:world
    monkeypatch.setitem(sys.modules,'robolab.core.world.world_state',m)
    monkeypatch.setattr(C,'REST_METRIC','pose')
    def step(x=0.,angle=0.,reset_second=False):
        env.common_step_counter+=1;env.episode_length_buf+=1
        if reset_second:env.episode_length_buf[1]=1
        world.pos[0,0]=x
        world.quat[0]=torch.tensor([math.cos(angle/2),0,0,math.sin(angle/2)])
    return env,world,step

def test_jitter_and_no_extra_wait_on_goal(rig):
    env,w,step=rig; raw=torch.zeros(2,dtype=torch.bool)
    c=SuccessConfirmer(lambda e:raw,.2,.02,['banana'])
    for x in [0.,.0002,-.0002]:step(x);assert not c(env).any()
    raw[:]=True;step(.0001)
    assert c(env).all() # high reported velocity does not block a stable pose

def test_return_to_start_bounce_rejected(rig):
    env,w,step=rig;t=PoseRestTracker(env)
    for x in [0.,.01,0.]:step(x);out=t.stable('banana',.2)
    assert out.tolist()==[False,True]

def test_slide_and_rotation_rejected(rig):
    env,w,step=rig;t=PoseRestTracker(env)
    for x in [0.,.002,.004]:step(x);out=t.stable('banana',.2)
    assert not out[0]
    for a in [0.,.05,.1]:step(.004,a);out=t.stable('banana',.2)
    assert not out[0]

def test_duplicate_calls_reset_and_quaternion_sign(rig):
    env,w,step=rig;t=PoseRestTracker(env)
    step()
    for _ in range(10):assert not t.stable('banana',.2).any()
    step();t.stable('banana',.2)
    step(reset_second=True);w.quat[0]*=-1
    assert t.stable('banana',.2).tolist()==[True,False]
    step();assert t.stable('banana',.2).tolist()==[True,False]
    step();assert t.stable('banana',.2).all()

def test_missing_steps_require_new_window(rig):
    env,w,step=rig;t=PoseRestTracker(env)
    for _ in range(3):step();t.stable('banana',.2)
    env.common_step_counter+=3;step()
    assert not t.stable('banana',.2).any()
