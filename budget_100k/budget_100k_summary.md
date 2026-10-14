# Adaptive Horizon Results: 100K Updates

## Setup

- Plain Merlin, state observations, expert datasets.
- Tasks: FetchReach and SawyerDoor.
- Training seeds: 0, 1, 2; 100,000 optimizer updates per run.
- Checkpoint selection: highest discounted return during periodic training evaluation, not necessarily the final update.
- Hindsight ratio: 0.2 for both tasks.
- Evaluation: 500 episodes per policy and method; episode seeds 30000-30499.
- Methods: fixed horizon, learned once per episode, learned at every step.
- Fixed horizons: FetchReach 1; SawyerDoor 5.
- One frozen first-hit MLP per task, MLP training seed 0, shared across all policy seeds.
- First-hit label tolerances: FetchReach distance < 0.05; SawyerDoor distance < 0.02.
- Policy normalization comes from each saved training checkpoint; MLP normalization comes from its own checkpoint.
- Initial observations, achieved goals, desired goals, and episode seeds were checked and match across methods within each policy run.

## Results

Values are mean +/- sample standard deviation across three policy training seeds (ddof=1). Success is measured at the final episode step; its standard deviation is in percentage points. Higher return and success are better.

| Task | Method | Discounted Return | Final Success (%) | Mean Horizon |
|---|---|---:|---:|---:|
| FetchReach | Fixed | 30.075 +/- 0.032 | 100.00 +/- 0.00 | 1.000 |
| FetchReach | Learned-episode | 26.179 +/- 0.164 | 99.93 +/- 0.12 | 7.896 |
| FetchReach | Learned-step | 27.921 +/- 0.134 | 100.00 +/- 0.00 | 1.469 |
| SawyerDoor | Fixed | 23.896 +/- 0.224 | 88.47 +/- 7.35 | 5.000 |
| SawyerDoor | Learned-episode | 19.576 +/- 0.398 | 73.93 +/- 2.84 | 17.978 |
| SawyerDoor | Learned-step | 20.736 +/- 0.455 | 93.87 +/- 5.56 | 4.123 |

## Individual Seeds

| Task | Seed | Fixed Return | Episode Return | Step Return | Fixed Success (%) | Episode Success (%) | Step Success (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| FetchReach | 0 | 30.078 | 26.361 | 28.005 | 100.0 | 100.0 | 100.0 |
| FetchReach | 1 | 30.106 | 26.134 | 27.991 | 100.0 | 99.8 | 100.0 |
| FetchReach | 2 | 30.042 | 26.043 | 27.767 | 100.0 | 100.0 | 100.0 |
| SawyerDoor | 0 | 23.667 | 19.159 | 20.310 | 93.2 | 72.0 | 87.6 |
| SawyerDoor | 1 | 24.114 | 19.951 | 21.215 | 80.0 | 72.6 | 95.8 |
| SawyerDoor | 2 | 23.906 | 19.618 | 20.682 | 92.2 | 77.2 | 98.2 |

## Interpretation

- Fixed horizons have higher discounted return than either learned method for all three seeds in both tasks.
- On FetchReach, learned-step maintains 100% final success but reaches the goal later: mean first-success step is about 5.02 versus 2.72 for fixed.
- On SawyerDoor, learned-step raises average final success by 5.40 percentage points, with an improvement in two of three seeds. Its average return is lower by 3.160, and mean first-success step is later: about 15.90 versus 12.45 for fixed.
- Every SawyerDoor episode reaches strict success at least once under every method. Differences in final success therefore concern whether the episode ends in the success region, not whether it ever reaches it.
- SawyerDoor's reward tolerance (distance <= 0.06) differs from its stricter success tolerance (distance < 0.02). Discounted return and final success measure different aspects of behavior.
- Learned-episode performs worse than learned-step on both task-level mean metrics.
- First-hit prediction is not guaranteed to identify the best conditioning horizon for the frozen action policy. This is a possible explanation, not an established cause.