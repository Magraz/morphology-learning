"""Reproduce the 2026-09-30 CF comparison from trusted local pickle logs.

Run: .venv/bin/python plotting/analyze_counterfactual_runs.py
Only eval_time > 0 rows count as evaluations. Seeds, not evaluation episodes,
are the independent units. No training/checkpoint files are modified.
"""
import csv
import pickle
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'plotting/feudal_goal_analysis/counterfactual_2026-09-30'
ENVS = [f'mjx_{task}_1024_gs{suffix}' for task in ('1a_3o_111', '2a_4o_1122') for suffix in ('', '_sparse')]
PREFIX = 'simplified_feudal_tanh_'
MODELS = {'mlp':'Flat MAPPO', PREFIX+'relative_input':'Relative', PREFIX+'relative_input_cf':'Relative CF', PREFIX+'relative_input_cf_hold':'Relative CF hold', PREFIX+'local_input':'Local', PREFIX+'local_input_cf':'Local CF'}
COLORS = ['#111827','#2563eb','#06b6d4','#9333ea','#ea580c','#16a34a']
DIAG = ['manager_cf_beta','manager_cf_adv_var_ratio','manager_cf_model_ev','manager_cf_goal_sensitivity','manager_cf_correction_std','manager_explained_variance','worker_explained_variance','waypoint_reached_frac','waypoint_final_error','waypoint_offset','manager_action_saturation','intrinsic_reward','manager_window_return']


def write_csv(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/name).open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs={}; inventory=[]
    for env in ENVS:
        for modeldir in sorted((ROOT/'experiments/results'/env).iterdir()):
            if modeldir.name != 'mlp' and not modeldir.name.startswith('simplified_feudal'): continue
            for trial in sorted(modeldir.iterdir()):
                p=trial/'logs/training_stats_finished.pkl'
                if not p.exists(): continue
                s={k:np.asarray(v) for k,v in pickle.load(p.open('rb')).items()}
                t=s['total_steps']; ev=s['eval_time']>0
                assert len(t)==len(ev) and np.all(np.diff(t)>0)
                assert np.isfinite(s['reward'][ev]).all()
                runs[env,modeldir.name,trial.name]=s
                inventory.append(dict(env=env,model=modeldir.name,seed=trial.name,path=str(p.relative_to(ROOT)),last_step=int(t[-1]),evaluations=int(ev.sum())))
    write_csv('inventory.csv',inventory)
    selected=[s for (e,m,k),s in runs.items() if m in MODELS]
    assert len(selected)==120, f'Expected 4 tasks x 6 models x 5 seeds, found {len(selected)}'
    end=min(s['total_steps'][-1] for s in selected)
    rows=[]
    for (env,model,seed),s in runs.items():
        t=s['total_steps']; ev=s['eval_time']>0
        # Interpolation at a common step grid eliminates evaluation cadence bias.
        et=t[ev]; er=s['reward'][ev]; grid=np.linspace(0,end,1001)
        auc=float(np.trapezoid(np.interp(grid,et,er),grid)/end)
        for lo,hi in [(0,20),(20,40),(40,60),(60,80),(80,90),(90,100),(80,100)]:
            mask=(t>=lo*1e6)&(t<hi*1e6); em=mask&ev
            r=dict(env=env,model=model,seed=seed,lower_m=lo,upper_m=hi,evaluations=int(em.sum()),reward=float(s['reward'][em].mean()),auc_0_common_end=auc,common_end=int(end))
            for k in DIAG:
                if k in s:r[k]=float(s[k][mask].mean())
            if 'manager_cf_adv_var_ratio' in s:
                # A constant-zero advantage gives the logged ratio zero, not a meaningful variance reduction.
                informative=mask & (np.abs(s['manager_cf_model_loss'])>1e-8)
                r['cf_var_ratio_nonzero_loss']=float(s['manager_cf_adv_var_ratio'][informative].mean()) if informative.any() else ''
            rows.append(r)
    write_csv('per_seed_windows.csv',rows)
    summary=[]
    for env in ENVS:
        print('\n'+env)
        for model in sorted({m for e,m,k in runs if e==env}):
            rr=[r for r in rows if r['env']==env and r['model']==model and r['lower_m']==90]
            v=np.array([r['reward'] for r in rr]); a=np.array([r['auc_0_common_end'] for r in rr])
            row=dict(env=env,model=model,n_seeds=len(v),late_mean=v.mean(),late_sd=v.std(ddof=1),late_median=np.median(v),auc_mean=a.mean(),auc_sd=a.std(ddof=1),seed_scores=';'.join(f'{x:.3f}' for x in v))
            for k in DIAG:
                if k in rr[0]:row[k]=float(np.mean([r[k] for r in rr]))
            summary.append(row)
            if model in MODELS:print(f'{MODELS[model]:18} {v.mean():6.1f} ± {v.std(ddof=1):5.1f}; AUC {a.mean():.1f}; {np.round(v,1)}')
    write_csv('summary.csv',summary)
    pairs=[]
    for env in ENVS:
        for cf,parent in [(PREFIX+'relative_input_cf',PREFIX+'relative_input'),(PREFIX+'relative_input_cf_hold',PREFIX+'relative_input'),(PREFIX+'relative_input_cf_hold',PREFIX+'relative_input_cf'),(PREFIX+'local_input_cf',PREFIX+'local_input')]:
            rr={r['seed']:r for r in rows if r['env']==env and r['model']==cf and r['lower_m']==90}
            pp={r['seed']:r for r in rows if r['env']==env and r['model']==parent and r['lower_m']==90}
            for metric in ['reward','auc_0_common_end']:
                d=np.array([rr[k][metric]-pp[k][metric] for k in sorted(rr)])
                # Descriptive paired-t interval; n=5 and failure mixtures limit inference.
                se=d.std(ddof=1)/np.sqrt(len(d)); hw=2.776445105*se
                pairs.append(dict(env=env,branch=cf,parent=parent,metric=metric,delta=d.mean(),sd_delta=d.std(ddof=1),ci95_low=d.mean()-hw,ci95_high=d.mean()+hw,n_positive=int((d>0).sum()),seed_deltas=';'.join(f'{x:.3f}' for x in d)))
    write_csv('paired_deltas.csv',pairs)
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for ax,env in zip(axes.flat,ENVS):
        for (model,label),color in zip(MODELS.items(),COLORS):
            curves=[]
            for seed in range(5):
                s=runs[env,model,str(seed)]; t=s['total_steps']/1e6; ev=s['eval_time']>0
                curves.append([s['reward'][ev&(t>=lo)&(t<lo+5)].mean() for lo in range(0,100,5)])
            a=np.array(curves); x=np.arange(2.5,100,5)
            ax.plot(x,a.mean(axis=0),label=label,color=color,lw=2)
            ax.fill_between(x,a.mean(axis=0)-a.std(axis=0,ddof=1),a.mean(axis=0)+a.std(axis=0,ddof=1),color=color,alpha=.08)
        ax.set_title(('1 agent / 3 boxes' if '1a_' in env else '2 agents / 4 boxes')+(' — sparse' if env.endswith('sparse') else ' — dense'))
        ax.set_xlabel('Environment steps (millions)');ax.set_ylabel('Evaluation team return');ax.grid(alpha=.2);ax.set_ylim(bottom=0)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=6)
    fig.suptitle('Counterfactual branches: mean ± seed SD; five independent training seeds\n5M-step bins, actual evaluations only')
    fig.savefig(OUT/'learning_curves.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for ax,env in zip(axes.flat,ENVS):
        for i,((model,label),color) in enumerate(zip(MODELS.items(),COLORS)):
            s=next(r for r in summary if r['env']==env and r['model']==model)
            v=np.array([float(x) for x in s['seed_scores'].split(';')])
            ax.bar(i,v.mean(),color=color,alpha=.75)
            ax.errorbar(i,v.mean(),yerr=v.std(ddof=1),fmt='none',color='#111827',capsize=3)
            ax.scatter(i+np.linspace(-.15,.15,len(v)),v,s=20,color='#111827',zorder=3)
        ax.set_xticks(range(6),[s.replace(' ','\n') for s in MODELS.values()],fontsize=9)
        ax.set_title(('1 agent' if '1a_' in env else '2 agents')+(' — sparse' if env.endswith('sparse') else ' — dense'));ax.set_ylabel('Evaluation team return');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Late performance (90–100M steps): dots are training seeds, bars are mean ± SD')
    fig.savefig(OUT/'late_returns.png',dpi=180);plt.close(fig)
    print(f'\nSaved {OUT}; common AUC end = {end}')

if __name__=='__main__':main()
