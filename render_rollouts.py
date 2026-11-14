"""Render predetermined actual Gymnasium rollouts from final seed-0 checkpoints.

Run after all three training/evaluation jobs finish. This does not select the best
checkpoint or episode: every method uses final.npz and episode seed 2000000000.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('SDL_AUDIODRIVER','dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .agents import PPO,A2C,DoubleDQN
from .agents_extra import DiscreteSAC,GRPO,GFlowNetTB

METHODS={'ppo':PPO,'a2c':A2C,'ddqn':DoubleDQN,'sac':DiscreteSAC,'grpo':GRPO,'gfn':GFlowNetTB}
DEFAULT_METHODS=('ppo','a2c','ddqn')
EPISODE_SEED=2000000000
TRAINING_SEED=0
MAX_STEPS=1000
SAMPLE_EVERY=3


def annotated_frame(array,method,step,total):
    frame=Image.fromarray(np.asarray(array,dtype=np.uint8)).convert('RGB')
    draw=ImageDraw.Draw(frame)
    draw.rectangle((0,0,frame.width,36),fill=(12,16,25))
    draw.text((8,5),f'{method.upper()} | final checkpoint | training seed 0 | test seed {EPISODE_SEED}',fill='white')
    draw.text((8,20),f'LunarLander-v3 | step {step} | return {total:.1f}',fill='white')
    return frame


def render_one(method,run_dir,output_dir,use_mp4):
    import gymnasium as gym
    checkpoint=run_dir/'final.npz'
    model=METHODS[method].load(checkpoint)
    config=json.loads((run_dir/'config.json').read_text())
    summary=json.loads((run_dir/'summary.json').read_text())
    if config.get('method')!=method or config.get('backend')!='gym' or config.get('seed')!=TRAINING_SEED:
        raise ValueError(f'Unexpected checkpoint configuration: {run_dir}')
    if summary.get('trained_steps')!=config.get('steps'):
        raise ValueError(f'Incomplete training budget: {run_dir}')
    env=gym.make('LunarLander-v3',render_mode='rgb_array',max_episode_steps=MAX_STEPS)
    frames=[];frame_steps=[];transitions=[];total=0.;term=False;trunc=False;last_reward=0.
    try:
        obs,_=env.reset(seed=EPISODE_SEED)
        frames.append(annotated_frame(env.render(),method,0,total));frame_steps.append(0)
        initial_observation=np.asarray(obs).tolist()
        for step in range(1,MAX_STEPS+1):
            action=(int(model.act(obs[None,:],epsilon=0)[0]) if method=='ddqn' else
                    int(model.act(obs[None,:],deterministic=True)[0][0]))
            obs,reward,term,trunc,info=env.step(action)
            last_reward=float(reward);total+=last_reward
            transitions.append(dict(step=step,action=action,reward=last_reward,cumulative_return=total,
                                    observation=np.asarray(obs).tolist(),terminated=bool(term),truncated=bool(trunc)))
            if step%SAMPLE_EVERY==0 or term or trunc:
                frames.append(annotated_frame(env.render(),method,step,total));frame_steps.append(step)
            if term or trunc:break
        fps=float(env.metadata.get('render_fps',50))
    finally:env.close()
    stem=f'gym_{method}_seed0_episode{EPISODE_SEED}'
    if use_mp4:
        import imageio.v2 as imageio
        artifact=output_dir/(stem+'.mp4')
        with imageio.get_writer(str(artifact),fps=fps/SAMPLE_EVERY,codec='libx264',quality=7,macro_block_size=1) as writer:
            for frame in frames:writer.append_data(np.asarray(frame))
    else:
        artifact=output_dir/(stem+'.gif')
        frames[0].save(artifact,save_all=True,append_images=frames[1:],duration=round(1000*SAMPLE_EVERY/fps),loop=0,optimize=False,disposal=2)
    expected=next((row for row in summary['test']['episodes'] if row['seed']==EPISODE_SEED),None)
    matches_test=bool(expected is not None and np.isclose(total,expected['return_'],rtol=1e-6,atol=1e-5) and len(transitions)==expected['length'])
    metadata=dict(environment='LunarLander-v3',method=method,training_seed=TRAINING_SEED,
                  episode_seed=EPISODE_SEED,selection='Predetermined first held-out seed; final fixed-budget checkpoint; no episode selection.',
                  checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  trained_steps=summary['trained_steps'],return_=total,length=len(transitions),
                  terminated=bool(term),truncated=bool(trunc),success=bool(term and last_reward>=99),
                  success_definition='True termination with final reward >=99, matching benchmark evaluator.',
                  frame_count=len(frames),frame_steps=frame_steps,frame_stride=SAMPLE_EVERY,source_fps=fps,
                  playback_fps=fps/SAMPLE_EVERY,artifact=str(artifact),
                  matches_recorded_test_episode=matches_test,recorded_test_episode=expected,
                  initial_observation=initial_observation,transitions=transitions)
    (output_dir/(stem+'.json')).write_text(json.dumps(metadata,indent=2))
    if not matches_test:
        raise RuntimeError(f'Rendered episode differs from recorded deterministic test episode for {method}; metadata preserved for investigation')
    print(json.dumps({k:metadata[k] for k in ['method','return_','length','success','artifact','matches_recorded_test_episode']}),flush=True)
    return {k:metadata[k] for k in ['method','return_','length','success','artifact','matches_recorded_test_episode']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,default=Path('lunar_bench/results/benchmark_500k'))
    p.add_argument('--output',type=Path)
    p.add_argument('--methods',nargs='+',choices=sorted(METHODS),default=list(DEFAULT_METHODS))
    a=p.parse_args()
    output=a.output or a.directory/'previews'
    if output.exists():p.error('Use a fresh output directory to preserve previous previews')
    for method in a.methods:
        run=a.directory/f'gym_{method}_seed0'
        for filename in ['final.npz','summary.json','config.json']:
            if not (run/filename).is_file():p.error(f'Wait for completed benchmark: missing {run/filename}')
    use_mp4=importlib.util.find_spec('imageio_ffmpeg') is not None and importlib.util.find_spec('imageio') is not None
    output.mkdir(parents=True)
    results=[render_one(method,a.directory/f'gym_{method}_seed0',output,use_mp4) for method in a.methods]
    (output/'index.json').write_text(json.dumps(dict(episode_seed=EPISODE_SEED,training_seed=TRAINING_SEED,format='mp4' if use_mp4 else 'gif',previews=results),indent=2))

if __name__=='__main__':main()
