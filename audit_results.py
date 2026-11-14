"""Audit completed benchmark outputs and repair optimizer-count labeling only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def audit(directory, expected_runs=18, correct=True):
    directory=Path(directory)
    experiment=json.loads((directory/'experiment.json').read_text())
    methods=experiment.get('methods',['ppo','a2c','ddqn'])   # older experiments have no method list
    expected=[f'{b}_{m}_seed{s}' for b in experiment['backends']
              for m in methods for s in experiment['seeds']]
    errors=[];pending=[];checked=[];corrections=[]
    if len(expected)!=expected_runs:
        errors.append(f'Expected {expected_runs} runs, protocol specifies {len(expected)}')
    source_hashes={}
    def source_hash(name):
        if name not in ('agents.py','agents_extra.py'):raise ValueError(f'Unknown algorithm source {name}')
        if name not in source_hashes:
            source_hashes[name]=hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        return source_hashes[name]
    for name in expected:
        run=directory/name
        required=['config.json','summary.json','final.npz','training.jsonl','test_episodes.csv']
        absent=[f for f in required if not (run/f).exists()]
        if absent:
            pending.append(dict(run=name,missing=absent));continue
        try:
            cfg=json.loads((run/'config.json').read_text())
            summary=json.loads((run/'summary.json').read_text())
            budget=experiment['steps_per_run']
            assert cfg['steps']==summary['trained_steps']==budget, 'Transition budget mismatch'
            assert summary['config']==cfg, 'Summary and standalone configs differ'
            assert cfg['algorithm_source_sha256']==source_hash(cfg.get('algorithm_source','agents.py')), 'Algorithm source changed since run'
            for key in ('training_seconds','training_cpu_seconds','transitions_per_second','wall_seconds'):
                assert np.isfinite(summary[key]) and summary[key]>0, f'Invalid {key}'
            last=None
            with (run/'training.jsonl').open() as f:
                for line in f:
                    if line.strip():last=json.loads(line)
            assert last and last['steps']==budget, 'Training log does not reach budget'
            with np.load(run/'final.npz',allow_pickle=False) as data:
                metadata=json.loads(str(data['metadata']))
                assert metadata==cfg, 'Checkpoint metadata differs from config'
                optimizer_updates=int(data['updates'])
                assert optimizer_updates>0, 'No optimizer steps recorded'
                params=[k for k in data.files if k.startswith('p_')]
                assert params, 'No trained model parameters'
                for key in data.files:
                    array=data[key]
                    if np.issubdtype(array.dtype,np.number):
                        assert np.isfinite(array).all(), f'Nonfinite checkpoint array {key}'
            rows=summary['test']['episodes']
            assert len(rows)==cfg['eval_episodes'], 'Test count differs from protocol'
            assert [r['seed'] for r in rows]==list(range(2000000000,2000000000+len(rows))), 'Test seeds mismatch'
            assert all(np.isfinite(r['return_']) and 1<=r['length']<=1000 for r in rows), 'Invalid test episode'
            assert np.isclose(summary['test']['mean_return'],np.mean([r['return_'] for r in rows])), 'Test mean mismatch'
            with (run/'test_episodes.csv').open(newline='') as f:csv_rows=list(csv.DictReader(f))
            assert len(csv_rows)==len(rows), 'CSV episode count mismatch'
            assert all(int(c['seed'])==r['seed'] and np.isclose(float(c['return_']),r['return_'])
                       for c,r in zip(csv_rows,rows)), 'CSV and summary differ'
            old_count=summary['optimizer_updates']
            if correct:
                summary.setdefault('update_calls',old_count)
                summary['optimizer_updates']=optimizer_updates
                summary['optimizer_count_audit']={
                    'source':'final.npz updates',
                    'note':'update_calls counts runner update calls; optimizer_updates counts actual Adam steps.'}
                temporary=run/'summary.audit.tmp'
                temporary.write_text(json.dumps(summary,indent=2))
                temporary.replace(run/'summary.json')
            if old_count!=optimizer_updates:
                corrections.append(dict(run=name,old_count=old_count,actual_adam_steps=optimizer_updates,applied=correct))
            checked.append(dict(run=name,trained_steps=budget,optimizer_updates=optimizer_updates,
                                update_calls=summary.get('update_calls',old_count),finite_checkpoint=True))
        except (AssertionError,KeyError,ValueError,OSError) as exc:
            errors.append(f'{name}: {exc}')
    report=dict(passed=not errors and not pending and len(checked)==expected_runs,
                expected_runs=expected_runs,completed_checked=len(checked),pending=pending,
                errors=errors,corrections=corrections,runs=checked)
    (directory/'audit.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--expected-runs',type=int,default=18)
    p.add_argument('--no-correct',action='store_true')
    args=p.parse_args()
    result=audit(args.directory,args.expected_runs,not args.no_correct)
    print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))
    raise SystemExit(0 if result['passed'] else 1)

if __name__=='__main__':main()
