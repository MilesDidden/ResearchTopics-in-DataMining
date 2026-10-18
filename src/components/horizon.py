from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class HorizonModel(nn.Module):
    def __init__(self, state_dim, goal_dim, hidden_size=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + goal_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1),
            nn.Softplus(),
        )

    def forward(self, states, goals):
        inputs = torch.cat([states, goals], dim=-1)
        return self.network(inputs) + 1e-5


class HorizonPredictor:
    def __init__(
        self, checkpoint_path, task, variant, state_dim, goal_dim,
        max_horizon, device='cpu',
    ):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        required = {'dataset_path', 'model_config', 'model_state_dict', 'normalization', 'horizon_range'}
        if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
            raise ValueError('expected a horizon-estimator checkpoint')
        self.target_metadata = {
            'target_type': checkpoint.get('target_type', 'step-gap'),
            'target_definition': checkpoint.get('target_definition'),
            'goal_tolerance': checkpoint.get('goal_tolerance'),
            'tolerance_inclusive': checkpoint.get('tolerance_inclusive', False),
            'reference_checkpoint': checkpoint.get('reference_checkpoint'),
        }
        dataset_path = Path(checkpoint['dataset_path'])
        if dataset_path.parent.name != task or dataset_path.parent.parent.name != variant:
            raise ValueError('horizon checkpoint task or dataset variant does not match evaluation')
        model_config = checkpoint['model_config']
        if model_config['state_dim'] != state_dim or model_config['goal_dim'] != goal_dim:
            raise ValueError('horizon checkpoint feature dimensions do not match the policy')
        self.device = torch.device(device)
        self.model = HorizonModel(**model_config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval().requires_grad_(False)
        self.statistics = {}
        for name, value in checkpoint['normalization'].items():
            if torch.is_tensor(value):
                value = value.cpu().numpy()
            self.statistics[name] = value
        for name, expected_size in (('state', state_dim), ('goal', goal_dim)):
            mean = self.statistics[name + '_mean']
            std = self.statistics[name + '_std']
            if mean.shape != (expected_size,) or std.shape != (expected_size,):
                raise ValueError('horizon normalization feature dimensions do not match')
            if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
                raise ValueError('horizon normalization must be finite with positive standard deviations')
        # Zero is a valid label, but Merlin needs h >= 1.
        self.minimum = max(1, int(checkpoint['horizon_range']['min']))
        self.maximum = min(int(max_horizon), int(checkpoint['horizon_range']['max']))
        if self.maximum < self.minimum:
            raise ValueError('horizon checkpoint has no usable horizon range')

    @torch.no_grad()
    def predict(self, state, goal):
        clip = self.statistics['clip_range']
        state = (state - self.statistics['state_mean']) / self.statistics['state_std']
        goal = (goal - self.statistics['goal_mean']) / self.statistics['goal_std']
        state = np.clip(state, -clip, clip)
        goal = np.clip(goal, -clip, clip)
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        goal = torch.as_tensor(goal, dtype=torch.float32, device=self.device)
        predicted_horizon = self.model(state, goal).item()
        if not np.isfinite(predicted_horizon):
            raise ValueError('horizon estimator produced a nonfinite prediction')
        return int(np.clip(round(predicted_horizon), self.minimum, self.maximum))


class HorizonSelector:
    def __init__(self, mode, fixed_horizon, predictor=None, update_interval=1):
        if fixed_horizon < 1:
            raise ValueError('fixed horizon must be positive')
        if mode != 'fixed' and predictor is None:
            raise ValueError('learned horizon modes require a predictor')
        self.mode = mode
        self.fixed_horizon = fixed_horizon
        self.predictor = predictor
        self.update_interval = update_interval
        self.reset()

    def reset(self):
        self.cached_horizon = None

    def select(self, state, goal, step):
        if self.mode == 'fixed':
            return self.fixed_horizon
        needs_prediction = self.cached_horizon is None or (
            self.mode == 'learned-step' and step % self.update_interval == 0
        )
        if needs_prediction:
            self.cached_horizon = self.predictor.predict(state, goal)
        return self.cached_horizon
