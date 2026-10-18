import numpy as np


def build_first_hit_table(buffer, tolerance, inclusive=False):
    """Store first-hit steps for each trajectory, start, and future goal."""
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('first-hit tolerance must be finite and positive')
    trajectory_count = buffer.current_buffer_size
    path_length = buffer.max_trajectory_length
    targets = np.full((trajectory_count, path_length, path_length), -1, dtype=np.int32)
    for trajectory in range(trajectory_count):
        length = int(buffer._length_of_traj[trajectory])
        if length < 2:
            continue
        achieved_goals = buffer._achieved_states[trajectory, :length].astype(np.float64)
        offsets = achieved_goals[:, None, :] - achieved_goals[None, :, :]
        distances = np.linalg.norm(offsets, axis=-1)
        if inclusive:
            within_tolerance = distances <= tolerance
        else:
            within_tolerance = distances < tolerance
        state_indices = np.arange(length)
        starts = state_indices[:, None]
        ends = state_indices[None, :]
        next_hit_indices = np.where(within_tolerance, starts, length)
        # Scan backward to find the nearest hit for each start and goal.
        reverse_hits = next_hit_indices[::-1]
        next_hit_indices = np.minimum.accumulate(reverse_hits, axis=0)[::-1]
        valid_pairs = ends > starts
        steps_to_hit = next_hit_indices - starts
        trajectory_targets = targets[trajectory, :length, :length]
        trajectory_targets[valid_pairs] = steps_to_hit[valid_pairs]
    return targets


def sample_first_hit_batch(buffer, batch_size, trajectory_indices, targets):
    if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer')
    indices = buffer._horizon_trajectory_indices(trajectory_indices)
    trajectories = np.random.choice(indices, batch_size)
    lengths = buffer._length_of_traj[trajectories]
    gaps = np.random.randint(1, lengths)
    starts = np.random.randint(0, lengths - gaps)
    ends = starts + gaps
    labels = targets[trajectories, starts, ends]
    if np.any(labels < 0) or np.any(labels > gaps):
        raise ValueError('first-hit table does not contain valid labels for sampled pairs')
    states = buffer._states[trajectories, starts]
    goals = buffer._achieved_states[trajectories, ends]
    labels = labels.astype(np.float32)[:, None]
    return states, goals, labels


def mean_first_hit_target(buffer, trajectory_indices, targets):
    indices = buffer._horizon_trajectory_indices(trajectory_indices)
    weights_by_length = {}
    total = 0.0
    for trajectory in indices:
        length = int(buffer._length_of_traj[trajectory])
        if length not in weights_by_length:
            times = np.arange(length)
            gaps = times[None, :] - times[:, None]
            valid_pairs = gaps > 0
            # Each gap is equally likely; its valid starts share that weight.
            starts_per_gap = length - gaps[valid_pairs]
            weights = 1.0 / ((length - 1) * starts_per_gap)
            weights_by_length[length] = valid_pairs, weights
        valid_pairs, weights = weights_by_length[length]
        trajectory_targets = targets[trajectory, :length, :length]
        labels = trajectory_targets[valid_pairs]
        if np.any(labels < 0):
            raise ValueError('training trajectories must have valid first-hit targets')
        total += float(np.dot(labels, weights))
    return total / len(indices)
