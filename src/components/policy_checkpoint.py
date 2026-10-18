from pathlib import Path

import numpy as np
import torch


def snapshot_normalizer(normalizer):
    if normalizer is None:
        return None
    return {
        'mean': torch.as_tensor(normalizer.mean.copy()),
        'std': torch.as_tensor(normalizer.std.copy()),
        'clip_range': float(normalizer.default_clip_range),
        'epsilon': 1e-6,
    }


def save_policy_checkpoint(
    path, policy, observation_normalizer, goal_normalizer, policy_config, metadata,
):
    # Don't let later updates change these weights.
    weights = {}
    for name, value in policy.state_dict().items():
        weights[name] = value.detach().cpu().clone()
    normalization = {
        'observation': snapshot_normalizer(observation_normalizer),
        'goal': snapshot_normalizer(goal_normalizer),
    }
    checkpoint = {
        'checkpoint_type': 'merlin_policy',
        'format_version': 1,
        'policy_config': policy_config,
        'policy_state_dict': weights,
        'normalization': normalization,
        'metadata': metadata,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return checkpoint


def save_training_sidecar(trainer, training_steps):
    policy = trainer.policy
    checkpoint_dir = Path(trainer.checkpoint_dir)
    dataset_path = Path(trainer.dataset_path)
    policy_config = {
        'state_shape': tuple(trainer.state_shape),
        'goal_dim': int(trainer.goal_dim),
        'timestep_embedding_dim': int(policy.timestep_embedding_dim),
        'hidden_dim': int(policy.dense_layers[0].out_features),
        'output_dim': int(policy.output_dim),
        'max_path_length': int(policy.max_path_length),
        'max_action': float(policy.max_action),
    }
    metadata = {
        'task': dataset_path.parent.name,
        'variant': dataset_path.parent.parent.name,
        'seed': int(checkpoint_dir.parent.name),
        'training_steps': int(training_steps),
        'fixed_horizon': int(trainer.test_horizon),
        'her_ratio': float(trainer.her_ratio),
        'diffusion_rollout': trainer.diffusion_rollout,
        'diffusion_nonparam': trainer.diffusion_nonparam,
        'normalization_source': 'training',
    }
    observation_normalizer = None if trainer.image_obs else trainer.o_norm
    path = checkpoint_dir / 'best_policy_with_normalization.pt'
    return save_policy_checkpoint(
        path, policy, observation_normalizer, trainer.g_norm, policy_config, metadata,
    )


def normalize_vector(values, statistics):
    if statistics is None:
        raise ValueError('vector evaluation requires saved normalization statistics')
    mean = statistics['mean'].cpu().numpy()
    std = statistics['std'].cpu().numpy()
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError('normalization statistics must be finite with positive standard deviations')
    values = np.asarray(values)
    if values.shape[-1:] != mean.shape:
        raise ValueError('input feature dimensions do not match the saved normalizer')
    normalized = (values - mean) / (std + statistics['epsilon'])
    clip = statistics['clip_range']
    return np.clip(normalized, -clip, clip)
