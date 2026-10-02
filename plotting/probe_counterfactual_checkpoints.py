"""Read-only checkpoint probes for the two-agent CF comparison.

PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python plotting/probe_counterfactual_checkpoints.py
All seeds/policies use the same 64 reset keys. Dense and sparse tasks have
identical dynamics and observations, so both are probed in one dense simulator.
The 1024-step prefix is the original task; 2048 steps is a time-budget intervention.
The scripted manager is a privileged, coordinated intervention, not a baseline.
"""
import csv
import os
from pathlib import Path
import time
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import jax
import jax.numpy as jnp
import numpy as np
from flax.serialization import msgpack_restore
from algorithms.mappo_jax.network import MAPPOActor
from algorithms.simplified_feudal_mappo_jax import waypoints as wp
from environments.mjx_suite.multi_box_push_mjx import MultiBoxPushMJX

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'plotting/feudal_goal_analysis/counterfactual_2026-09-30'
PREFIX='simplified_feudal_tanh_'
MODELS=['mlp']+[PREFIX+x for x in ['relative_input','relative_input_cf','relative_input_cf_hold','local_input','local_input_cf']]
N=64


def build_probe(env,mode):
    actor=MAPPOActor(action_dim=2,hidden_dim=168,discrete=False)
    reset=jax.vmap(env.reset); step=jax.vmap(env.step)
    pos_fn=jax.vmap(env.goal_state); entity=jax.vmap(env.entity_state)
    agent_pos=jax.vmap(env._agent_pos); box_pose=jax.vmap(env._box_pose)

    @jax.jit
    def probe(worker,manager,keys):
        obs,state=reset(keys)
        def advance(carry,t):
            obs,state,waypoint,target,done=carry
            active=~done
            pos=pos_fn(state)
            def decide(args):
                waypoint,target=args
                if mode=='script':
                    ap=agent_pos(state.data); bp,_=box_pose(state.data)
                    distance=jnp.linalg.norm(bp-ap.mean(axis=1)[:,None,:],axis=-1)
                    nearest=jnp.argmin(jnp.where(state.delivered,jnp.inf,distance),axis=-1)
                    old_done=jnp.take_along_axis(state.delivered,jnp.maximum(target,0)[:,None],axis=1)[:,0]
                    target=jnp.where((target<0)|old_done,nearest,target)
                    box=jnp.take_along_axis(bp,target[:,None,None],axis=1)[:,0]
                    half=env._box_half[target]
                    slots=jnp.array([-.45,.45])
                    stages=box[:,None,:]+jnp.stack([jnp.broadcast_to(slots,(N,2)),jnp.broadcast_to(-(half+.6)[:,None],(N,2))],axis=-1)
                    ds=stages-ap
                    close=jnp.linalg.norm(ds,axis=-1)<.85
                    desired=jnp.where(close[...,None],ap+jnp.array([0.,4.5]),stages)
                    # When approaching a new box from above, route around its side
                    # before descending; aiming through a heavy box is not feasible.
                    bp_delta=ap-box[:,None,:]
                    above=ap[...,1] > stages[...,1]+.25
                    sign=jnp.where(bp_delta[...,0]>=0,1.,-1.)
                    bypass_x=box[:,None,0]+sign*(half[:,None]+1.2)
                    beside=jnp.abs(bp_delta[...,0]) >= half[:,None]+.8
                    around=jnp.stack([bypass_x,jnp.where(beside,stages[...,1],ap[...,1])],axis=-1)
                    desired=jnp.where(above[...,None],around,desired)
                    delta=(desired-ap)/jnp.array([env.world_width,env.world_height])
                    waypoint=jnp.clip(pos+jnp.clip(delta,-.15,.15),-.5,.5)
                elif mode!='mlp':
                    x=wp.manager_actor_input_relative(*entity(state)) if mode=='relative' else wp.manager_actor_input_local(obs)
                    mu,_=actor.apply(manager,x)
                    waypoint=wp.waypoint_from_action(pos,mu,.15,'tanh')
                return waypoint,target
            waypoint,target=jax.lax.cond(t%32==0,decide,lambda a:a,(waypoint,target))
            x=obs if mode=='mlp' else wp.worker_actor_input(obs,wp.goal_error(waypoint,pos,.15),wp.remaining_fraction(t%32,32))
            action,_=actor.apply(worker,x)
            no,ns,_,term,_,info=step(state,action)
            new=(ns.delivered & ~state.delivered)&active[:,None]
            touch=info['agents_2_objects'].sum(axis=-1)
            heavy_live=(~state.delivered)[:,2:]
            alone=((touch[:,2:]==1)&heavy_live).any(axis=1)&active
            together=((touch[:,2:]>=2)&heavy_live).any(axis=1)&active
            ap=agent_pos(state.data); bp,_=box_pose(state.data)
            d=jnp.linalg.norm(bp[:,None,:,:]-ap[:,:,None,:],axis=-1)
            nearest=jnp.argmin(jnp.where(state.delivered[:,None,:],jnp.inf,d),axis=-1)
            same=(nearest[:,0]==nearest[:,1])&active
            # Finished episodes may continue physically, but every measurement is masked.
            done=done|term
            return (no,ns,waypoint,target,done),(new,jnp.where(active,info['task_reward'],0.),alone,together,same,active)
        init=(obs,state,pos_fn(state),jnp.full((N,),-1,dtype=jnp.int32),jnp.zeros(N,dtype=bool))
        _,(new,reward,alone,together,same,active)=jax.lax.scan(advance,init,jnp.arange(2048))
        times=jnp.min(jnp.where(new,jnp.arange(1,2049)[:,None,None],99999),axis=0)
        return dict(times=times,return1024=reward[:1024].sum(axis=0),alone_steps=alone[:1024].sum(axis=0),together_steps=together[:1024].sum(axis=0),same_steps=same[:1024].sum(axis=0),active_steps=active[:1024].sum(axis=0))
    return probe


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    env=MultiBoxPushMJX(n_agents=2,n_objects=4,coupling_def=[1,1,2,2],variant='trunc',use_global_state=True,max_steps=2048)
    funcs={mode:build_probe(env,mode) for mode in ['mlp','relative','local','script']}
    keys=jax.random.split(jax.random.PRNGKey(20260930),N)
    rows=[];episodes=[]
    for name,data in [('checkpoint_probes.csv',rows),('checkpoint_probe_episodes.csv',episodes)]:
        if (OUT/name).exists():
            data.extend(csv.DictReader((OUT/name).open()))
            for r in data:
                if r['intervention']=='script':r['intervention']='script_direct'
    completed={(r['env'],r['model'],str(r['seed']),r['intervention']) for r in rows}
    for suffix in ['', '_sparse']:
        ename='mjx_2a_4o_1122_1024_gs'+suffix
        for model in MODELS:
            for seed in range(5):
                path=ROOT/'experiments/results'/ename/model/str(seed)/'models/models_finished.msgpack'
                p=jax.tree.map(jnp.asarray,msgpack_restore(path.read_bytes()))
                worker=p['actor'] if model=='mlp' else p['worker_actor']
                manager=worker if model=='mlp' else p['manager_actor']
                mode='mlp' if model=='mlp' else ('local' if 'local_input' in model else 'relative')
                for intervention in (['learned'] if model=='mlp' else ['learned','script_routed']):
                    if (ename,model,str(seed),intervention) in completed:continue
                    start=time.time();res=jax.device_get(funcs[mode if intervention=='learned' else 'script'](worker,manager,keys))
                    t=res['times'];at1024=t<=1024;at2048=t<=2048
                    r=dict(env=ename,model=model,seed=seed,intervention=intervention,episodes=N,boxes1024=at1024.sum(axis=1).mean(),light1024=at1024[:,:2].sum(axis=1).mean(),heavy1024=at1024[:,2:].sum(axis=1).mean(),success1024=at1024.all(axis=1).mean(),boxes2048=at2048.sum(axis=1).mean(),success2048=at2048.all(axis=1).mean(),return1024=res['return1024'].mean() if not suffix else 100*at1024.sum(axis=1).mean(),alone_heavy_step_frac=res['alone_steps'].sum()/max(res['active_steps'].sum(),1),together_heavy_step_frac=res['together_steps'].sum()/max(res['active_steps'].sum(),1),same_nearest_step_frac=res['same_steps'].sum()/max(res['active_steps'].sum(),1))
                    for k in range(4):
                        kth=np.sort(t,axis=1)[:,k];valid=kth<=1024
                        r[f'delivery{k+1}_median_step']=np.median(kth[valid]) if valid.any() else np.nan
                    rows.append(r)
                    for ep in range(N):episodes.append(dict(env=ename,model=model,seed=seed,intervention=intervention,episode=ep,**{f'box{k}_time':int(t[ep,k]) for k in range(4)}))
                    for name,data in [('checkpoint_probes.csv',rows),('checkpoint_probe_episodes.csv',episodes)]:
                        with (OUT/name).open('w',newline='') as f:
                            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
                    print(f'{ename} {model} seed {seed} {intervention}: boxes={r["boxes1024"]:.2f} light={r["light1024"]:.2f} heavy={r["heavy1024"]:.2f} extended={r["boxes2048"]:.2f}; {time.time()-start:.1f}s',flush=True)
    print('done',flush=True)

if __name__=='__main__':main()
