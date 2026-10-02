"""Aggregate saved checkpoint probes and append them to the comparison report."""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
out=root/'plotting/feudal_goal_analysis/counterfactual_2026-09-30'
rows=list(csv.DictReader((out/'checkpoint_probes.csv').open()))
groups=defaultdict(list)
for r in rows:groups[r['env'],r['model'],r['intervention']].append(r)
models=['mlp']+['simplified_feudal_tanh_'+s for s in ['relative_input','relative_input_cf','relative_input_cf_hold','local_input','local_input_cf']]
labels=['Flat MAPPO','Relative','Relative CF','Relative CF hold','Local','Local CF']
metrics=['boxes1024','light1024','heavy1024','success1024','boxes2048','success2048','alone_heavy_step_frac','together_heavy_step_frac','same_nearest_step_frac']
ss=[]
for (env,model,intervention),rr in groups.items():
 r=dict(env=env,model=model,intervention=intervention,n_seeds=len(rr))
 for k in metrics:
  v=np.array([float(x[k]) for x in rr]);r[k]=v.mean();r[k+'_sd']=v.std(ddof=1) if len(v)>1 else 0
 ss.append(r)
with (out/'checkpoint_probe_summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(ss[0]));w.writeheader();w.writerows(ss)
lookup={(r['env'],r['model'],r['intervention']):r for r in ss}
envs=['mjx_2a_4o_1122_1024_gs'+s for s in ['', '_sparse']]
assert all(lookup[e,m,'learned']['n_seeds']==5 for e in envs for m in models)
fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
for ax,e in zip(axes,envs):
 rs=[lookup[e,m,'learned'] for m in models]
 light=np.array([r['light1024'] for r in rs]);heavy=np.array([r['heavy1024'] for r in rs]);x=np.arange(len(models))
 ax.bar(x,light,color='#60a5fa',label='Requires one agent')
 ax.bar(x,heavy,bottom=light,color='#7c3aed',label='Requires both agents')
 ax.errorbar(x,light+heavy,yerr=[r['boxes1024_sd'] for r in rs],fmt='none',color='#111827',capsize=3)
 ax.set_xticks(x,[s.replace(' ','\n') for s in labels]);ax.set_ylim(0,4.25);ax.set_ylabel('Delivered boxes per episode (maximum 4)');ax.grid(axis='y',alpha=.2)
 ax.set_title('Two agents — '+('sparse' if e.endswith('sparse') else 'dense'))
 for i,r in enumerate(rs):ax.text(i,r['boxes1024']+.12,f'{r["boxes1024"]:.2f}',ha='center',fontsize=9)
fig.suptitle('Which boxes are missing? Final checkpoints, 1024-step episodes\nFive training seeds × 64 shared reset states; errors: seed SD of total boxes')
h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=2)
fig.savefig(out/'deliveries_by_coupling.png',dpi=180);plt.close(fig)
lines=['**Fresh checkpoint interventions: direct evidence about the two-agent failure**','', 'These are new deterministic rollouts of final checkpoints, five training seeds × 64 identical reset states per policy. They are separate from the 90–100M log averages above. Each environment has two boxes requiring one agent and two requiring both agents. Dense/sparse configurations differ only in reward; both checkpoint sets were evaluated in the same simulator dynamics. Sparse return is reconstructed as 100 × deliveries. Each rollout runs to 2048 steps: the 1024-step prefix is the original task, and the second half is an episode-budget intervention without retraining. The policies do not observe the changed time limit.', '', '| Policy | Dense: light / heavy boxes | Sparse: light / heavy boxes | Sparse total, 2048 steps |', '|---|---:|---:|---:|']
for m,label in zip(models,labels):
 d=lookup[envs[0],m,'learned'];s=lookup[envs[1],m,'learned']
 lines.append(f'| {label} | {d["light1024"]:.2f} / {d["heavy1024"]:.2f} | {s["light1024"]:.2f} / {s["heavy1024"]:.2f} | {s["boxes2048"]:.2f} |')
lines += ['', 'The two-agent sparse relative and relative-CF policies deliver **zero cooperation-required boxes in all 320 episodes per branch**. This is specific evidence of a missing cooperative strategy, not merely a drop in aggregate return. Doubling the time limit changes their total delivery count only from 1.35 to 1.39 and 1.42 to 1.44. Dense relative CF benefits more from time (2.69 → 3.18), but still trails MAPPO at the original limit (3.83). Thus time is a contributor in dense runs and does not explain the sparse relative failure.', '', 'A second intervention replaces only the manager with a deterministic privileged controller: it latches a shared box, assigns distinct positions below it, routes around the side if approaching from above, and issues ordinary radius-limited waypoints every 32 steps. Every branch keeps its own trained worker. This is an existence/diagnostic test, not an information-matched learned baseline or an optimal controller.', '', '| Sparse checkpoint workers | Learned manager: total / heavy | Scripted manager: total / heavy | Scripted manager total, 2048 steps |', '|---|---:|---:|---:|']
for m,label in zip(models[1:],labels[1:]):
 s=lookup[envs[1],m,'learned'];q=lookup[envs[1],m,'script_routed'];assert q['n_seeds']==5
 lines.append(f'| {label} | {s["boxes1024"]:.2f} / {s["heavy1024"]:.2f} | {q["boxes1024"]:.2f} / {q["heavy1024"]:.2f} | {q["boxes2048"]:.2f} |')
lines += ['', 'For the sparse relative branches, this rescue shows that their trained workers and waypoint interface can deliver cooperation-required boxes when given suitable coordinated goals. The learned manager is a major bottleneck there. This is not a general proof that the workers are optimal: the script is not a consistent improvement on dense checkpoints, and it jointly changes target selection, role assignment and approach geometry. A preliminary direct-to-staging script was poor because it could aim through boxes; its completed results are retained as `script_direct` and are not used to claim worker incapacity. The revised controller and all measurements are saved for inspection.', '', 'Contact diagnostics support the coordination interpretation. Dense relative CF spends about 29% of active steps with one agent alone touching an undelivered heavy box and 15% with both touching one; MAPPO is about 18% / 28%. Sparse relative CF almost never has both touching an undelivered heavy box (about 0.3% of active steps). These are descriptive fractions; they are affected by the states each policy visits and should not be read as independent causal effects.', '', 'Files: `checkpoint_probes.csv` (per seed), `checkpoint_probe_episodes.csv` (first delivery time per box; 99999 means not delivered by 2048), `checkpoint_probe_summary.csv`, and `deliveries_by_coupling.png`.', '']
report=out/'report.md';text=report.read_text().replace('[ _sparse]','[_sparse]')
marker='**Fresh checkpoint interventions:'
if marker in text:text=text[:text.index(marker)]
report.write_text(text+'\n'+'\n'.join(lines))
for r in ss:
 if r['intervention']!='script_direct':print(r['env'],r['model'],r['intervention'],{k:round(r[k],3) for k in metrics[:6]})
