"""Aggregate completed local LunarLander experiments; no published scores."""
from __future__ import annotations
import argparse
import base64
import csv
import html
import json
from pathlib import Path
import numpy as np

LABELS={'gym':'Gymnasium LunarLander-v3 (Box2D)', 'rust':'RustLander-v0 (custom surrogate)'}
METHODS={'ppo':'PPO','a2c':'A2C','ddqn':'Double DQN'}
COLORS={'ppo':'#2563eb','a2c':'#d97706','ddqn':'#059669'}


def seed_ci(values):
    """Percentile bootstrap over independent training-seed means, not episodes."""
    values=np.asarray(values,dtype=float)
    if len(values)<2:return [None,None]
    rng=np.random.default_rng(20261113)
    draws=values[rng.integers(len(values),size=(20000,len(values)))].mean(axis=1)
    return [float(x) for x in np.quantile(draws,[.025,.975])]


def load_runs(directory,experiment):
    expected={(b,m,int(s)) for b in experiment['backends'] for m in METHODS for s in experiment['seeds']}
    runs={}
    for path in sorted(directory.glob('*/summary.json')):
        row=json.loads(path.read_text())
        cfg=row['config']; key=(cfg['backend'],cfg['method'],int(cfg['seed']))
        if key not in expected:continue
        if key in runs:raise ValueError(f'Duplicate run for {key}')
        if row['trained_steps']!=experiment['steps_per_run']:
            raise ValueError(f'Wrong transition budget: {path}')
        episodes=row['test']['episodes']
        if len(episodes)!=cfg['eval_episodes']:
            raise ValueError(f'Incomplete test episodes: {path}')
        if not np.isfinite([x['return_'] for x in episodes]).all():
            raise ValueError(f'Nonfinite test return: {path}')
        row['_path']=path.parent
        row['_key']=key
        runs[key]=row
    missing=expected-set(runs)
    if missing:raise ValueError('Experiments still incomplete: '+', '.join(map(str,sorted(missing))))
    if experiment.get('failures',0):raise ValueError('Launcher reported failed jobs')
    if 'total_wall_seconds' not in experiment:raise ValueError('Launcher has not finished; total wall time is missing')
    return list(runs.values())


def aggregate(runs):
    per_seed=[]
    for r in runs:
        b,m,seed=r['_key']; episodes=r['test']['episodes']
        per_seed.append(dict(backend=b,environment=LABELS[b],method=METHODS[m],training_seed=seed,
                             trained_steps=r['trained_steps'],test_episodes=len(episodes),
                             mean_return=float(np.mean([e['return_'] for e in episodes])),
                             success_rate=float(np.mean([e['success'] for e in episodes])),
                             mean_episode_length=float(np.mean([e['length'] for e in episodes])),
                             training_seconds=r['training_seconds'],training_cpu_seconds=r['training_cpu_seconds'],
                             transitions_per_second=r['transitions_per_second'],wall_seconds=r['wall_seconds']))
    groups=[]
    for b in dict.fromkeys(r['backend'] for r in per_seed):
        for m in METHODS:
            subset=[r for r in per_seed if r['backend']==b and r['method']==METHODS[m]]
            if not subset:continue
            returns=[r['mean_return'] for r in subset]
            low,high=seed_ci(returns)
            groups.append(dict(backend=b,environment=LABELS[b],method=METHODS[m],training_seeds=len(subset),
                               steps_per_seed=subset[0]['trained_steps'],test_episodes_per_seed=subset[0]['test_episodes'],
                               mean_return=float(np.mean(returns)),return_seed_std=float(np.std(returns,ddof=1)) if len(returns)>1 else None,
                               seed_bootstrap_ci95_low=low,seed_bootstrap_ci95_high=high,
                               mean_success_rate=float(np.mean([r['success_rate'] for r in subset])),
                               mean_training_seconds=float(np.mean([r['training_seconds'] for r in subset])),
                               mean_training_cpu_seconds=float(np.mean([r['training_cpu_seconds'] for r in subset])),
                               mean_transitions_per_second=float(np.mean([r['transitions_per_second'] for r in subset])),
                               mean_run_wall_seconds=float(np.mean([r['wall_seconds'] for r in subset]))))
    return groups,per_seed


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def plot_curves(directory,runs,experiment):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    backends=experiment['backends']
    fig,axes=plt.subplots(len(backends),2,figsize=(12,4.1*len(backends)),squeeze=False,layout='constrained')
    for row,b in enumerate(backends):
        for m in METHODS:
            subset=[r for r in runs if r['_key'][:2]==(b,m)]
            series=[]
            for r in subset:
                records=[json.loads(line) for line in (r['_path']/'progress.jsonl').read_text().splitlines() if line.strip()]
                series.append(records)
            grid=np.linspace(0,experiment['steps_per_run'],11)
            for col,metric in enumerate(['mean_return','success_rate']):
                ys=np.array([np.interp(grid,[x['steps'] for x in s],[x['validation'][metric] for x in s]) for s in series])
                mean=ys.mean(axis=0)
                axes[row,col].plot(grid,mean,label=METHODS[m],color=COLORS[m],linewidth=2)
                axes[row,col].fill_between(grid,ys.min(axis=0),ys.max(axis=0),color=COLORS[m],alpha=.13)
                axes[row,col].set_xlabel('Training environment transitions')
                axes[row,col].set_ylabel('Validation return (symlog; linear within +/-100)' if col==0 else 'Validation successful-landing fraction')
                if col==0: axes[row,col].set_yscale('symlog',linthresh=100)
                axes[row,col].set_title(LABELS[b])
                axes[row,col].grid(alpha=.2)
        axes[row,0].legend()
        axes[row,1].set_ylim(-.03,1.03)
    fig.suptitle('Measured local training: mean and range across training seeds\nValidation checkpoints interpolated to a shared grid; shading is not a confidence interval',fontsize=12)
    fig.savefig(directory/'learning_curves.png',dpi=150)
    plt.close(fig)


def table(rows,columns):
    head='<tr>'+''.join('<th>'+html.escape(label)+'</th>' for _,label in columns)+'</tr>'
    body=''
    for row in rows:
        cells=[]
        for key,_ in columns:
            value=row[key]
            if value is None: value='n/a'
            elif isinstance(value,float): value=f'{value:.3f}' if 'success' in key else f'{value:,.1f}'
            cells.append('<td>'+html.escape(str(value))+'</td>')
        body+='<tr>'+''.join(cells)+'</tr>'
    return '<div class="scroll"><table><thead>'+head+'</thead><tbody>'+body+'</tbody></table></div>'


def write_html(directory,groups,per_seed,experiment,extras):
    png=base64.b64encode((directory/'learning_curves.png').read_bytes()).decode('ascii')
    sections=[]
    for b in experiment['backends']:
        subset=[r for r in groups if r['backend']==b]
        sections.append('<h2>'+html.escape(LABELS[b])+'</h2>')
        if b=='rust':sections.append('<p class="note">Custom Rust dynamics, not a Box2D port. These returns and landing criteria cannot be compared numerically with Gymnasium returns.</p>')
        sections.append(table(subset,[('method','Method'),('mean_return','Mean test return'),('return_seed_std','SD across seed means'),('seed_bootstrap_ci95_low','95% CI low'),('seed_bootstrap_ci95_high','95% CI high'),('mean_success_rate','Landing fraction'),('mean_training_seconds','Train seconds/run'),('mean_transitions_per_second','Transitions/s')]))
    details=table(per_seed,[('environment','Environment'),('method','Method'),('training_seed','Train seed'),('mean_return','Test return'),('success_rate','Landing fraction'),('training_seconds','Train seconds'),('wall_seconds','Run wall seconds')])
    metadata=html.escape(json.dumps(experiment,indent=2))
    document='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Local LunarLander benchmark</title>
<style>body{font:16px/1.55 system-ui,sans-serif;color:#172033;background:#f4f6fa;margin:0}main{max-width:1200px;margin:auto;padding:36px 24px}h1,h2{line-height:1.2}h1{font-size:34px}h2{margin-top:38px}.lead{font-size:19px}.note{border-left:4px solid #d97706;padding:12px 18px;background:#fff5de}.scroll{overflow:auto}table{border-collapse:collapse;background:white;width:100%;font-variant-numeric:tabular-nums;font-size:14px}th,td{padding:11px;text-align:right;border-bottom:1px solid #dce1eb}th:first-child,td:first-child{text-align:left}th{background:#e9eef7}img{width:100%;background:white;border-radius:8px}code,pre{background:#e9eef7}pre{padding:18px;overflow:auto}li{margin:8px 0}.muted{color:#526078}a{color:#185cc5}</style><main>
<h1>Measured local LunarLander benchmark</h1><p class="lead">Actual CPU training and held-out evaluation from this machine. No published performance scores are included.</p>'''
    document+=f'<p>{len(per_seed)} completed runs; {experiment["steps_per_run"]:,} transitions per run; {len(experiment["seeds"])} independent training seeds per method/environment. Up to {experiment["jobs"]} concurrent processes, one BLAS/OpenMP thread each. Whole experiment wall time: <strong>{experiment["total_wall_seconds"]:.1f} seconds</strong>.</p>'
    document+=''.join(sections)
    if extras.get('stochastic'):
        document+='<h2>Supplementary stochastic policy evaluation</h2><p>The same final PPO/A2C checkpoints are also evaluated by sampling their categorical action distributions. This is a fixed-checkpoint diagnostic; it does not replace the deterministic primary table or reselect checkpoints. Environment test seeds are the same; action RNG is separately seeded for each episode.</p>'
        document+=table(extras['stochastic'],[('environment','Environment'),('method','Method'),('training_seeds','Train seeds'),('mean_return','Mean sampled-policy return'),('mean_success_rate','Landing fraction'),('seed_bootstrap_ci95_low','95% CI low'),('seed_bootstrap_ci95_high','95% CI high')])
    if extras.get('parallel_timing'):
        timing=extras['parallel_timing']
        document+='<h2>Matched serial versus parallel timing</h2><p>One measured pair of matched workloads on this machine; this is not a general scaling estimate.</p>'
        document+=table([timing],[('serial_seconds','Serial wall seconds'),('parallel_seconds','Parallel wall seconds'),('measured_speedup','Measured speedup'),('parameters_identical','Parameters identical')])
    if extras.get('baselines'):
        rows=[]
        for backend,policies in extras['baselines'].items():
            for policy,metrics in policies.items():
                rows.append(dict(environment=LABELS.get(backend,backend),policy=policy,**metrics))
        document+='<h2>Measured non-learning baselines</h2><p>Random and hand-written heuristic policies, evaluated locally on the held-out episode seeds. These have no training budget.</p>'
        document+=table(rows,[('environment','Environment'),('policy','Policy'),('mean_return','Mean return'),('success_rate','Landing fraction'),('episodes','Episodes')])
    if extras.get('throughput'):
        throughput=extras['throughput']
        rows=[dict(mode=mode,**metrics) for mode,metrics in throughput['summary'].items()]
        document+='<h2>Isolated simulator throughput</h2><p>'+html.escape(throughput['note'])+'</p>'
        document+=table(rows,[('mode','Workload'),('median_steps_per_second','Median steps/s'),('min_steps_per_second','Min steps/s'),('max_steps_per_second','Max steps/s')])
        document+='<p>'+html.escape(throughput['timing_scope'])+'</p><p>'+html.escape(throughput['load_note'])+'</p>'
    document+='<h2>Learning curves</h2><img alt="Validation return and successful landing rate versus training transitions, separated by environment" src="data:image/png;base64,'+png+'">'
    document+='''<h2>How to read this comparison</h2><ul>
<li>Test metrics evaluate each final fixed-budget checkpoint deterministically on held-out environment seeds starting at 2,000,000,000. Validation seeds start at 1,000,000,000 and are not test seeds.</li>
<li>Mean return gives equal weight to each training seed. The 95% interval bootstraps independent training-seed means with 20,000 draws. With only three training seeds, uncertainty is poorly resolved; 300 test episodes are not 300 independent training replicates. Intervals do not capture all environment-seed uncertainty.</li>
<li>Landing fraction uses explicit success in the Rust simulator, or Gymnasium true termination with final reward at least 99. It is distinct from the fraction of episodes scoring at least 200, and time-limit truncation is not a landing.</li>
<li>Training seconds include rollout simulation and optimizer updates, and exclude validation, final evaluation and file-writing outside the timed training block. Transitions/s equals training transitions divided by those seconds. Run wall seconds include evaluation and checkpoint/report work after environment initialization.</li>
<li>All learners use NumPy on CPU. Rust accelerates the custom simulator only. The Rust model differs from Box2D in physical dynamics, contact handling and task difficulty; this is not an equivalent-engine speed comparison.</li>
<li>Methods receive equal transition budgets, not equal optimizer steps or compute. PPO/A2C use one 64-unit tanh hidden layer and rewards scaled by 0.01 for optimization; Double DQN uses two 64-unit ReLU layers and unscaled rewards. Reported rewards are always raw. The comparison concerns these implementations and settings, not algorithms in isolation.</li>
<li>Hyperparameters were fixed for these runs. A finite budget does not establish convergence or a universal winner. Six concurrent processes contend for CPU resources; per-run times are measured under this workload. Any optional matched serial/parallel measurement is a separate fixed-workload pair; per-run contended timings alone are not a serial speedup estimate.</li></ul>'''
    document+='<h2>Every training seed</h2>'+details+'<h2>Machine and execution metadata</h2><pre>'+metadata+'</pre>'
    document+='<p class="muted">Machine-readable companions: benchmark_summary.csv, per_seed.csv, summary.json. Raw run directories contain configurations, test episode CSVs, progress JSONL, training JSONL and final checkpoints.</p></main></html>'
    (directory/'report.html').write_text(document,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',nargs='?',type=Path,default=Path('lunar_bench/results/benchmark_500k'))
    args=parser.parse_args()
    experiment=json.loads((args.directory/'experiment.json').read_text())
    try:runs=load_runs(args.directory,experiment)
    except ValueError as error:parser.error(str(error))
    groups,per_seed=aggregate(runs)
    write_csv(args.directory/'benchmark_summary.csv',groups)
    write_csv(args.directory/'per_seed.csv',per_seed)
    extras={}
    for key,path in [('baselines',args.directory.parent/'baselines'/'summary.json'),('throughput',args.directory.parent/'throughput_isolated.json'),('parallel_timing',args.directory.parent/'parallel_timing'/'summary.json')]:
        if path.exists():extras[key]=json.loads(path.read_text())
    stochastic=[]
    for run in runs:
        path=run['_path']/'stochastic_summary.json'
        if path.exists():
            result=json.loads(path.read_text())
            stochastic.append(result)
    if stochastic:
        extras['stochastic']=[]
        for backend in experiment['backends']:
            for method in ['ppo','a2c']:
                subset=[r for r in stochastic if r['config']['backend']==backend and r['config']['method']==method]
                if not subset:continue
                values=[r['test']['mean_return'] for r in subset]
                low,high=seed_ci(values)
                extras['stochastic'].append(dict(environment=LABELS[backend],method=METHODS[method],training_seeds=len(subset),
                                                mean_return=float(np.mean(values)),mean_success_rate=float(np.mean([r['test']['success_rate'] for r in subset])),
                                                seed_bootstrap_ci95_low=low,seed_bootstrap_ci95_high=high))
    summary=dict(experiment=experiment,groups=groups,per_seed=per_seed,extras=extras,
                 interval='95% percentile bootstrap over independent training-seed means, 20000 draws; not pooled episode uncertainty',
                 note='RustLander-v0 is a custom surrogate, not equivalent to Gymnasium Box2D; raw scores are not cross-environment comparable.')
    (args.directory/'summary.json').write_text(json.dumps(summary,indent=2))
    plot_curves(args.directory,runs,experiment)
    write_html(args.directory,groups,per_seed,experiment,extras)
    print(json.dumps(groups,indent=2))
    print('Report: '+str((args.directory/'report.html').resolve()))

if __name__=='__main__':main()
