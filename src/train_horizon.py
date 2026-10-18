import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.components.buffer import ReplayBuffer
from src.components.horizon import HorizonModel
from src.components.horizon_targets import build_first_hit_table, mean_first_hit_target, sample_first_hit_batch


def load_dataset(dataset_path):
    with Path(dataset_path).open('rb') as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict) or not {'o', 'ag'}.issubset(data):
        raise ValueError("dataset must contain observations ('o') and achieved goals ('ag')")
    states = np.asarray(data['o'])
    achieved_goals = np.asarray(data['ag'])
    if states.ndim != 3 or achieved_goals.ndim != 3:
        raise ValueError('only vector observations and goals shaped (trajectories, time, features) are supported')
    if states.shape[:2] != achieved_goals.shape[:2]:
        raise ValueError('observations and achieved goals must share trajectory and time dimensions')
    trajectory_count, path_length, state_dim = states.shape
    goal_dim = achieved_goals.shape[-1]
    if trajectory_count < 2 or path_length < 2 or state_dim < 1 or goal_dim < 1:
        raise ValueError('dataset needs at least two trajectories, two time slots, and nonempty features')
    lengths = np.asarray(data.get('lengths', np.full(trajectory_count, path_length)))
    if lengths.shape != (trajectory_count,) or not np.issubdtype(lengths.dtype, np.integer):
        raise ValueError('optional lengths must contain one integer per trajectory')
    if np.any(lengths < 0) or np.any(lengths > path_length):
        raise ValueError('trajectory lengths must fit within the dataset time dimension')

    buffer = ReplayBuffer(
        (state_dim,), np.float32, 1, np.float32, goal_dim, np.float32,
        path_length, her_ratio=0.0, buffer_size=trajectory_count,
    )
    actions = np.zeros_like(buffer._actions[0])
    desired_goal = np.zeros_like(buffer._desired_states[0])
    for index, length in enumerate(lengths):
        buffer.add_trajectory(states[index], actions, desired_goal, achieved_goals[index], int(length))
        trajectory_states = buffer._states[index, :length]
        trajectory_goals = buffer._achieved_states[index, :length]
        if not np.isfinite(trajectory_states).all() or not np.isfinite(trajectory_goals).all():
            raise ValueError('valid trajectory states and achieved goals must be finite')
    return buffer


def fit_normalization(buffer, trajectory_indices):
    sums = {
        'state': np.zeros(buffer._states.shape[-1]),
        'goal': np.zeros(buffer._achieved_states.shape[-1]),
    }
    squared_sums = {name: np.zeros_like(values) for name, values in sums.items()}
    count = 0
    for index in trajectory_indices:
        length = buffer._length_of_traj[index]
        values = {
            'state': buffer._states[index, :length].astype(np.float64),
            'goal': buffer._achieved_states[index, :length].astype(np.float64),
        }
        for name, samples in values.items():
            sums[name] += samples.sum(axis=0)
            squared_sums[name] += np.square(samples).sum(axis=0)
        count += length
    if count == 0:
        raise ValueError('normalization requires valid training states')
    normalization = {'clip_range': 5.0}
    for name in sums:
        mean = sums[name] / count
        variance = np.maximum(squared_sums[name] / count - np.square(mean), 0)
        normalization[name + '_mean'] = mean.astype(np.float32)
        normalization[name + '_std'] = np.maximum(np.sqrt(variance), 1e-6).astype(np.float32)
    return normalization


def normalize_batch(states, goals, normalization):
    clip = normalization['clip_range']
    states = (states - normalization['state_mean']) / normalization['state_std']
    goals = (goals - normalization['goal_mean']) / normalization['goal_std']
    return np.clip(states, -clip, clip), np.clip(goals, -clip, clip)


def prepare_batch(batch, normalization, device):
    states, goals, targets = batch
    states, goals = normalize_batch(states, goals, normalization)
    states = torch.as_tensor(states, dtype=torch.float32, device=device)
    goals = torch.as_tensor(goals, dtype=torch.float32, device=device)
    targets = torch.as_tensor(targets, dtype=torch.float32, device=device)
    return states, goals, targets


@torch.no_grad()
def evaluate(model, batches, constant_horizon):
    model.eval()
    loss_sum = error_sum = baseline_sum = 0.0
    count = 0
    zero_count = 0
    zero_error_sum = nonzero_error_sum = one_baseline_sum = zero_baseline_sum = 0.0
    for states, goals, targets in batches:
        predictions = model(states, goals)
        loss_sum += F.smooth_l1_loss(predictions, targets, reduction='sum').item()
        errors = (predictions - targets).abs()
        error_sum += errors.sum().item()
        baseline_sum += (targets - constant_horizon).abs().sum().item()
        zero_baseline_sum += targets.abs().sum().item()
        one_baseline_sum += (targets - 1).abs().sum().item()
        zero_targets = targets == 0
        zero_count += zero_targets.sum().item()
        zero_error_sum += errors[zero_targets].sum().item()
        nonzero_error_sum += errors[~zero_targets].sum().item()
        count += targets.numel()
    nonzero_count = count - zero_count
    return {
        'validation_loss': loss_sum / count,
        'validation_mae': error_sum / count,
        'constant_baseline_mae': baseline_sum / count,
        'zero_baseline_mae': zero_baseline_sum / count,
        'one_baseline_mae': one_baseline_sum / count,
        'zero_target_fraction': zero_count / count,
        'zero_target_mae': zero_error_sum / zero_count if zero_count else None,
        'nonzero_target_mae': nonzero_error_sum / nonzero_count if nonzero_count else None,
    }


def reference_data_setup(buffer, dataset, checkpoint_path):
    reference = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    source_dataset = Path(reference['dataset_path'])
    if source_dataset.resolve() != Path(dataset).resolve():
        raise ValueError('reference checkpoint must use the same dataset path')
    config = reference['model_config']
    if (config['state_dim'] != buffer._states.shape[-1]
            or config['goal_dim'] != buffer._achieved_states.shape[-1]):
        raise ValueError('reference checkpoint feature dimensions do not match the dataset')
    train_indices = reference['train_trajectory_indices'].cpu().numpy()
    validation_indices = reference['validation_trajectory_indices'].cpu().numpy()
    eligible_indices = buffer._horizon_trajectory_indices(None)
    for indices in (train_indices, validation_indices):
        checked = buffer._horizon_trajectory_indices(indices)
        if len(checked) != len(indices) or len(np.unique(indices)) != len(indices):
            raise ValueError('reference splits must contain unique eligible trajectories')
    split_indices = np.concatenate((train_indices, validation_indices))
    if not np.array_equal(np.sort(split_indices), eligible_indices):
        raise ValueError('reference splits must partition the eligible dataset trajectories')
    normalization = {
        name: value.cpu().numpy().copy() if torch.is_tensor(value) else value
        for name, value in reference['normalization'].items()
    }
    expected = fit_normalization(buffer, train_indices)
    for name in ('state_mean', 'state_std', 'goal_mean', 'goal_std', 'clip_range'):
        if not np.array_equal(normalization[name], expected[name]):
            raise ValueError('reference normalization does not match the current training data')
    return train_indices, validation_indices, normalization


def save_checkpoint(output, model, normalization, train_indices, validation_indices, metadata, metrics):
    checkpoint = metadata.copy()
    checkpoint.update(
        model_state_dict={name: value.detach().cpu() for name, value in model.state_dict().items()},
        normalization={
            name: torch.from_numpy(value.copy()) if isinstance(value, np.ndarray) else value
            for name, value in normalization.items()
        },
        train_trajectory_indices=torch.from_numpy(train_indices.copy()),
        validation_trajectory_indices=torch.from_numpy(validation_indices.copy()),
        metrics=metrics.copy(),
    )
    torch.save(checkpoint, output)


def train_horizon(args):
    output = args.output.resolve()
    metrics_path = output.with_suffix('.metrics.json')
    if ((args.target == 'first-hit' or args.reference_checkpoint is not None)
            and (output.exists() or metrics_path.exists())):
        raise FileExistsError('experiment outputs already exist; choose a new --output filename')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device_name = args.device
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device_name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is not available; use --device cpu')
    device = torch.device(device_name)

    buffer = load_dataset(args.dataset)
    if args.reference_checkpoint is None:
        train_indices, validation_indices = buffer.split_horizon_trajectories(
            args.validation_fraction, args.seed,
        )
        normalization = fit_normalization(buffer, train_indices)
    else:
        train_indices, validation_indices, normalization = reference_data_setup(
            buffer, args.dataset, args.reference_checkpoint,
        )
    first_hit_targets = None
    if args.target == 'first-hit':
        first_hit_targets = build_first_hit_table(buffer, args.goal_tolerance, args.inclusive_tolerance)

    def sample_batch(indices):
        if first_hit_targets is None:
            return buffer.sample_horizon_batch(args.batch_size, indices)
        return sample_first_hit_batch(buffer, args.batch_size, indices, first_hit_targets)

    validation_batches = [
        prepare_batch(sample_batch(validation_indices), normalization, device)
        for _ in range(args.validation_batches)
    ]
    if first_hit_targets is None:
        constant_horizon = float(buffer._length_of_traj[train_indices].mean() / 2)
    else:
        constant_horizon = mean_first_hit_target(buffer, train_indices, first_hit_targets)

    model_config = {
        'state_dim': buffer._states.shape[-1],
        'goal_dim': buffer._achieved_states.shape[-1],
        'hidden_size': args.hidden_size,
    }
    model = HorizonModel(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    output.parent.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    for name in ('dataset', 'output', 'reference_checkpoint'):
        path = getattr(args, name)
        config[name] = str(path.resolve()) if path is not None else None
    if args.target == 'step-gap':
        target_definition = 'future achieved goal at j; target is j - i'
    else:
        target_definition = (
            'future achieved goal at j; target is first tolerance hit '
            'from i within [i, j], including zero'
        )
    dataset_sha256 = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    reference_sha256 = None
    if args.reference_checkpoint is not None:
        reference_sha256 = hashlib.sha256(args.reference_checkpoint.read_bytes()).hexdigest()
    max_horizon = int(buffer._length_of_traj[train_indices].max() - 1)
    checkpoint_metadata = {
        'format_version': 1,
        'model_config': model_config,
        'horizon_range': {'min': 1, 'max': max_horizon},
        'target_definition': target_definition,
        'target_type': args.target,
        'target_range': {'min': 0 if first_hit_targets is not None else 1, 'max': max_horizon},
        'goal_tolerance': args.goal_tolerance,
        'tolerance_inclusive': args.inclusive_tolerance,
        'dataset_path': config['dataset'],
        'dataset_sha256': dataset_sha256,
        'reference_checkpoint': config['reference_checkpoint'],
        'reference_sha256': reference_sha256,
        'training_config': config,
        'constant_horizon': constant_horizon,
    }
    best_mae = float('inf')
    best_update = 0
    running_loss = 0.0
    running_updates = 0
    history = []
    print(f'Train: {len(train_indices)} | validation: {len(validation_indices)} | '
          f'target: {args.target} | device: {device}')

    for update in range(args.updates + 1):
        if update > 0:
            model.train()
            states, goals, targets = prepare_batch(sample_batch(train_indices), normalization, device)
            optimizer.zero_grad()
            loss = F.smooth_l1_loss(model(states, goals), targets)
            if not torch.isfinite(loss):
                raise ValueError('training produced a nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()
            running_updates += 1

        if update % args.eval_interval != 0 and update != args.updates:
            continue
        metrics = evaluate(model, validation_batches, constant_horizon)
        metrics.update(update=update, train_loss=running_loss / running_updates if running_updates else None)
        history.append(metrics)
        if metrics['validation_mae'] < best_mae:
            best_mae = metrics['validation_mae']
            best_update = update
            save_checkpoint(
                output, model, normalization, train_indices, validation_indices,
                metadata=checkpoint_metadata, metrics=metrics,
            )
        training_text = '-' if metrics['train_loss'] is None else f"{metrics['train_loss']:.4f}"
        print(f"Update {update:>6} | loss {training_text} | val MAE {metrics['validation_mae']:.3f}")
        running_loss = 0.0
        running_updates = 0

    training_history = {
        'training_config': config,
        'history': history,
        'best_update': best_update,
        'best_validation_mae': best_mae,
    }
    with metrics_path.open('w', encoding='utf-8') as handle:
        json.dump(training_history, handle, indent=2)
    print(f'Best checkpoint: {output} (update {best_update}, MAE {best_mae:.3f} steps)')
    print(f'Training history: {metrics_path}')
    return {'best_validation_mae': best_mae, 'best_update': best_update,
            'initial_validation_mae': history[0]['validation_mae'],
            'constant_baseline_mae': history[0]['constant_baseline_mae']}


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target', choices=['step-gap', 'first-hit'], default='step-gap')
    parser.add_argument('--reference-checkpoint', type=Path)
    parser.add_argument('--goal-tolerance', type=float)
    parser.add_argument('--inclusive-tolerance', action='store_true')
    parser.add_argument('--updates', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--validation-fraction', type=float, default=0.2)
    parser.add_argument('--validation-batches', type=int, default=20)
    parser.add_argument('--eval-interval', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args(argv)
    for name in ('updates', 'batch_size', 'hidden_size', 'validation_batches', 'eval_interval', 'threads'):
        if getattr(args, name) < 1:
            parser.error(f'--{name.replace("_", "-")} must be positive')
    if not np.isfinite(args.lr) or args.lr <= 0:
        parser.error('--lr must be finite and positive')
    if not 0 < args.validation_fraction < 1:
        parser.error('--validation-fraction must be between zero and one')
    if args.target == 'first-hit':
        if args.reference_checkpoint is None:
            parser.error('--target first-hit requires --reference-checkpoint for a controlled comparison')
        if args.goal_tolerance is None or not np.isfinite(args.goal_tolerance) or args.goal_tolerance <= 0:
            parser.error('--target first-hit requires a finite positive --goal-tolerance')
    elif args.goal_tolerance is not None or args.inclusive_tolerance:
        parser.error('goal tolerance options require --target first-hit')
    return args


def main():
    train_horizon(get_args())


if __name__ == '__main__':
    main()
