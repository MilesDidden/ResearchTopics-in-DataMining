import argparse
import copy
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from src.components.buffer import ReplayBuffer
from src.components.horizon import HorizonPredictor, HorizonSelector
from src.components.normalizer import normalizer
from src.components.policy_checkpoint import normalize_vector, save_policy_checkpoint
from src.envs import register_envs
from src.train_offline import HORIZON_TUNED, HER_RATIO_TUNED, Policy, build_env, discounted_return


def convert_legacy_checkpoint(
    source, destination, dataset, env, task, variant, seed,
    calibration_batches=200, batch_size=512,
):
    source, destination, dataset = Path(source), Path(destination), Path(dataset)
    if destination.exists() or destination.resolve() == source.resolve():
        raise FileExistsError('conversion needs a new output path; the original policy must remain untouched')
    weights = torch.load(source, map_location='cpu', weights_only=True)
    if not weights or not all(torch.is_tensor(value) for value in weights.values()):
        raise ValueError('legacy conversion expects a policy-only state dictionary')
    with dataset.open('rb') as handle:
        data = pickle.load(handle)
    states, actions, achieved_goals = data['o'], data['u'], data['ag']
    count, length, state_dim = states.shape
    goal_dim = achieved_goals.shape[-1]
    if states.shape[2:] != env.observation_space['observation'].shape:
        raise ValueError('calibration dataset observations do not match the selected task')
    if achieved_goals.shape[2:] != env.observation_space['desired_goal'].shape:
        raise ValueError('calibration dataset goals do not match the selected task')
    config = {
        'state_shape': (state_dim,),
        'goal_dim': goal_dim,
        'timestep_embedding_dim': 32,
        'hidden_dim': 256,
        'output_dim': 2 * env.action_space.shape[0],
        'max_path_length': length,
        'max_action': float(env.action_space.high[0]),
    }
    policy = Policy(**config)
    policy.load_state_dict(weights)
    buffer = ReplayBuffer(
        (state_dim,), states.dtype, actions.shape[-1], actions.dtype,
        goal_dim, achieved_goals.dtype, length, HER_RATIO_TUNED[task][variant], count,
    )
    for index in range(count):
        buffer.add_trajectory(states[index], actions[index], data['g'][index, 0], achieved_goals[index])
    observation_normalizer, goal_normalizer = normalizer(state_dim), normalizer(goal_dim)
    random_state = np.random.get_state()
    try:
        np.random.seed(seed)
        for _ in range(calibration_batches):
            observations, _, goals, _, _, _ = buffer.sample_batch(batch_size)
            observation_normalizer.update(observations)
            goal_normalizer.update(goals)
        observation_normalizer.recompute_stats()
        goal_normalizer.recompute_stats()
    finally:
        np.random.set_state(random_state)
    metadata = {
        'task': task,
        'variant': variant,
        'seed': int(seed),
        'training_steps': None,
        'fixed_horizon': HORIZON_TUNED[task][variant],
        'normalization_source': 'dataset_calibration',
        'source_policy': str(source.resolve()),
        'dataset_path': str(dataset.resolve()),
        'calibration_batches': calibration_batches,
        'calibration_her_ratio': float(buffer.her_ratio),
        'calibration_batch_size': batch_size,
        'calibration_seed': seed,
    }
    checkpoint = save_policy_checkpoint(
        destination, policy, observation_normalizer, goal_normalizer, config, metadata,
    )
    print(f'Converted policy: {destination}')
    print('Normalization source: dataset calibration')
    return checkpoint


def load_policy(checkpoint, env, task, variant, device):
    if checkpoint.get('checkpoint_type') != 'merlin_policy' or checkpoint.get('format_version') != 1:
        raise ValueError('evaluation requires a complete Merlin policy checkpoint')
    metadata = checkpoint['metadata']
    if metadata['task'] != task or metadata['variant'] != variant:
        raise ValueError('policy checkpoint task or dataset variant does not match evaluation')
    config = checkpoint['policy_config']
    if len(config['state_shape']) != 1:
        raise ValueError('this evaluator supports vector observations only')
    if tuple(config['state_shape']) != env.observation_space['observation'].shape:
        raise ValueError('policy observation dimensions do not match the simulator')
    if config['goal_dim'] != env.observation_space['desired_goal'].shape[0]:
        raise ValueError('policy goal dimensions do not match the simulator')
    if config['output_dim'] != 2 * env.action_space.shape[0]:
        raise ValueError('policy action dimensions do not match the simulator')
    policy = Policy(**config).to(device)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval().requires_grad_(False)
    return policy


def initial_success(env, observation, info):
    if 'is_success' in info:
        return bool(np.asarray(info['is_success']).item())
    base = env.unwrapped
    if hasattr(base, '_is_success'):
        return bool(np.asarray(base._is_success(observation['achieved_goal'], observation['desired_goal'])).item())
    return bool(np.asarray(base._get_info()['angle_success']).item())


def reset_episode(env, seed, saved_state):
    base = env.unwrapped
    if saved_state is not None:
        base.set_env_state(copy.deepcopy(saved_state))
    np.random.seed(seed)
    if callable(getattr(base, 'seed', None)):
        base.seed(seed)
    else:
        base.np_random = np.random.default_rng(seed)
    env.action_space.seed(seed)
    return env.reset()


def summarize_episodes(records):
    first_success_steps = [r['first_success_step'] for r in records if r['first_success_step'] is not None]
    total_steps = sum(r['steps'] for r in records)
    horizons = [horizon for record in records for horizon in record['horizons']]
    inference_seconds = sum(record['inference_seconds'] for record in records)
    mean_first_success = None
    if first_success_steps:
        mean_first_success = float(np.mean(first_success_steps))
    summary = {
        'success_rate': float(np.mean([r['success'] for r in records])),
        'ever_success_rate': float(np.mean([r['ever_success'] for r in records])),
        'discounted_return': float(np.mean([r['discounted_return'] for r in records])),
        'undiscounted_return': float(np.mean([r['undiscounted_return'] for r in records])),
        'mean_first_success_step': mean_first_success,
        'first_success_sample_count': len(first_success_steps),
        'mean_horizon': float(np.mean(horizons)),
        'mean_inference_ms': 1000 * inference_seconds / total_steps,
    }
    return {'summary': summary, 'episodes': records}


def evaluate_episode(env, policy, normalization, selector, observation, info, max_steps, device):
    record = {
        'initial_observation': np.asarray(observation['observation']).tolist(),
        'initial_goal': np.asarray(observation['desired_goal']).tolist(),
        'initial_achieved_goal': np.asarray(observation['achieved_goal']).tolist(),
    }
    first_success = 0 if initial_success(env, observation, info) else None
    rewards, horizons, actions = [], [], []
    inference_seconds = 0.0
    selector.reset()
    for step in range(max_steps):
        started = time.perf_counter()
        horizon = selector.select(observation['observation'], observation['desired_goal'], step)
        state = normalize_vector(observation['observation'], normalization['observation'])
        goal = normalize_vector(observation['desired_goal'], normalization['goal'])
        with torch.no_grad():
            action, _ = policy(
                torch.as_tensor(state, dtype=torch.float32, device=device),
                torch.as_tensor(goal, dtype=torch.float32, device=device),
                torch.tensor([horizon], dtype=torch.float32, device=device),
            )
        action = action.cpu().numpy()
        inference_seconds += time.perf_counter() - started
        if not np.isfinite(action).all():
            raise ValueError('policy produced a nonfinite action')
        observation, reward, terminated, truncated, info = env.step(action)
        reward = float(np.asarray(reward).item())
        if not np.isfinite(reward):
            raise ValueError('environment returned a nonfinite reward')
        success = bool(np.asarray(info['is_success']).item())
        if success and first_success is None:
            first_success = step + 1
        rewards.append(reward)
        horizons.append(int(horizon))
        actions.append(action.tolist())
        if terminated or truncated:
            break
    discounted, undiscounted = discounted_return(rewards, 0.98, reward_offset=True)
    record.update(
        success=success,
        ever_success=first_success is not None,
        first_success_step=first_success,
        steps=len(rewards),
        horizons=horizons,
        actions=actions,
        discounted_return=float(discounted),
        undiscounted_return=float(undiscounted),
        inference_seconds=inference_seconds,
    )
    return record


def evaluate_policy(env, policy, normalization, selector, episodes, evaluation_seed, max_steps):
    base = env.unwrapped
    saved_state = copy.deepcopy(base.get_env_state()) if hasattr(base, 'get_env_state') else None
    random_state = np.random.get_state()
    device = next(policy.parameters()).device
    records = []
    try:
        for episode in range(episodes):
            seed = evaluation_seed + episode
            observation, info = reset_episode(env, seed, saved_state)
            record = {'episode': episode, 'seed': seed}
            record.update(evaluate_episode(
                env, policy, normalization, selector, observation, info, max_steps, device,
            ))
            records.append(record)
    finally:
        np.random.set_state(random_state)
        if saved_state is not None:
            base.set_env_state(saved_state)
    return summarize_episodes(records)


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['FetchReach', 'SawyerDoor'], required=True)
    parser.add_argument('--variant', choices=['expert', 'random'], default='expert')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--policy-checkpoint', type=Path)
    parser.add_argument('--horizon-checkpoint', type=Path)
    parser.add_argument('--modes', nargs='+', choices=['fixed', 'learned-episode', 'learned-step'], default=['fixed'])
    parser.add_argument('--fixed-horizon', type=int)
    parser.add_argument('--update-interval', type=int, default=1)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--eval-seed', type=int, default=10000)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--overwrite-results', action='store_true')
    parser.add_argument('--calibrate-legacy', action='store_true')
    parser.add_argument('--converted-checkpoint', type=Path)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--calibration-batches', type=int, default=200)
    parser.add_argument('--calibration-batch-size', type=int, default=512)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args(argv)
    for name in ('episodes', 'update_interval', 'calibration_batches', 'calibration_batch_size', 'threads'):
        if getattr(args, name) < 1:
            parser.error(f'--{name.replace("_", "-")} must be positive')
    if args.max_steps is not None and args.max_steps < 1:
        parser.error('--max-steps must be positive')
    if args.fixed_horizon is not None and args.fixed_horizon < 1:
        parser.error('--fixed-horizon must be positive')
    if args.seed < 0 or args.seed >= 2**32 or args.eval_seed < 0 or args.eval_seed + args.episodes - 1 >= 2**32:
        parser.error('seeds and episode count must fit the NumPy seed range')
    if len(set(args.modes)) != len(args.modes):
        parser.error('--modes must not contain duplicates')
    return args


def run(args):
    if args.output.exists() and not args.overwrite_results:
        raise FileExistsError('results already exist; choose a new --output or use --overwrite-results')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; use --device cpu')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    register_envs()
    checkpoint_name = f'{args.task}_{args.variant}_seed{args.seed}.pt'
    frozen_path = Path('checkpoints/policy') / checkpoint_name
    legacy_path = (
        Path('logging/state_obs/checkpoints') / args.task / args.variant / str(args.seed)
        / 'forward_diffusion_model/best_policy.pt'
    )
    training_path = legacy_path.with_name('best_policy_with_normalization.pt')
    policy_path = args.policy_checkpoint
    if policy_path is None:
        policy_path = legacy_path
        for candidate in (training_path, frozen_path):
            if candidate.exists():
                policy_path = candidate
                break
    checkpoint = torch.load(policy_path, map_location='cpu', weights_only=True)
    env = build_env(args.task)
    try:
        if checkpoint.get('checkpoint_type') != 'merlin_policy':
            if not args.calibrate_legacy:
                raise ValueError(
                    'this old checkpoint has no normalization; run once with --calibrate-legacy '
                    'or use a newly trained complete checkpoint'
                )
            policy_path = args.converted_checkpoint or frozen_path
            checkpoint = convert_legacy_checkpoint(
                args.policy_checkpoint or legacy_path, policy_path,
                args.dataset or Path('offline_data') / args.variant / args.task / 'buffer.pkl',
                env, args.task, args.variant, args.seed,
                args.calibration_batches, args.calibration_batch_size,
            )
        policy = load_policy(checkpoint, env, args.task, args.variant, args.device)
        max_steps = args.max_steps or policy.max_path_length
        if max_steps > policy.max_path_length:
            raise ValueError('--max-steps exceeds the checkpoint episode length')
        fixed_horizon = args.fixed_horizon or checkpoint['metadata']['fixed_horizon']
        if fixed_horizon > policy.max_path_length:
            raise ValueError('fixed horizon exceeds the policy training range')
        predictor = None
        horizon_path = None
        if any(mode != 'fixed' for mode in args.modes):
            horizon_path = args.horizon_checkpoint or Path('checkpoints/horizon') / checkpoint_name
            policy_config = checkpoint['policy_config']
            predictor = HorizonPredictor(
                horizon_path, args.task, args.variant,
                state_dim=policy_config['state_shape'][0],
                goal_dim=policy_config['goal_dim'],
                max_horizon=policy.max_path_length,
                device=args.device,
            )
        report = dict(
            task=args.task, variant=args.variant, policy_seed=checkpoint['metadata']['seed'],
            evaluation_seed=args.eval_seed, episode_count=args.episodes, max_steps=max_steps,
            fixed_horizon=fixed_horizon, update_interval=args.update_interval,
            policy_checkpoint=str(policy_path.resolve()),
            policy_sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest(),
            normalization_source=checkpoint['metadata']['normalization_source'],
            policy_her_ratio=checkpoint['metadata'].get('her_ratio'),
            horizon_checkpoint=str(horizon_path.resolve()) if horizon_path else None,
            horizon_sha256=hashlib.sha256(horizon_path.read_bytes()).hexdigest() if horizon_path else None,
            horizon_target_metadata=predictor.target_metadata if predictor else None,
            modes={},
        )
        for mode in args.modes:
            selector = HorizonSelector(mode, fixed_horizon, predictor, args.update_interval)
            result = evaluate_policy(
                env, policy, checkpoint['normalization'], selector,
                episodes=args.episodes, evaluation_seed=args.eval_seed, max_steps=max_steps,
            )
            report['modes'][mode] = result
            summary = result['summary']
            print(
                f"{mode} | success {summary['success_rate']:.3f} | "
                f"return {summary['discounted_return']:.3f} | "
                f"mean h {summary['mean_horizon']:.2f}"
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('w' if args.overwrite_results else 'x', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
        print(f'Results: {args.output.resolve()}')
        return report
    finally:
        env.close()


def main():
    run(get_args())


if __name__ == '__main__':
    main()
