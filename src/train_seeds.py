import argparse
import ast
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HORIZONS = {'FetchReach': 1, 'SawyerDoor': 5}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def build_jobs(args, project_root):
    jobs = []
    for task in args.tasks:
        for seed in args.seeds:
            checkpoint_dir = (
                args.run_root / 'logging/state_obs/checkpoints' / task / 'expert' / str(seed)
                / 'forward_diffusion_model'
            )
            record_dir = args.run_root / 'runs' / task / str(seed)
            command = [
                sys.executable, '-u', str(project_root / 'src/train_offline.py'),
                '--task', task, '--variant', 'expert', '--seed', str(seed),
                '--max-timesteps', str(args.max_timesteps), '--max-path-length', '50',
            ]
            jobs.append({
                'task': task,
                'seed': seed,
                'checkpoint_dir': checkpoint_dir,
                'record_dir': record_dir,
                'command': command,
            })
    return jobs


def protocol(args, project_root):
    trainer = project_root / 'src/train_offline.py'
    tuned_settings = {}
    for node in ast.parse(trainer.read_text(encoding='utf-8')).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in ('HER_RATIO_TUNED', 'HORIZON_TUNED'):
                tuned_settings[target.id] = ast.literal_eval(node.value)
    for task, horizon in HORIZONS.items():
        if (tuned_settings['HER_RATIO_TUNED'][task]['expert'] != 0.2
                or tuned_settings['HORIZON_TUNED'][task]['expert'] != horizon):
            raise ValueError('selected expert task settings must match paper Table 3')
    sources = list((project_root / 'src').rglob('*.py'))
    environment = project_root / 'environment.yml'
    if environment.exists():
        sources.append(environment)
    packages = {}
    for name in ('torch', 'numpy', 'gymnasium', 'gymnasium-robotics', 'mujoco', 'mujoco-py', 'mpi4py'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    source_hashes = {}
    for path in sorted(sources):
        source_hashes[str(path.relative_to(project_root))] = file_hash(path)
    dataset_hashes = {}
    for task in HORIZONS:
        dataset = project_root / 'offline_data/expert' / task / 'buffer.pkl'
        dataset_hashes[task] = file_hash(dataset)
    return dict(
        method='plain Merlin', observation_type='state', variant='expert',
        max_timesteps=args.max_timesteps, max_path_length=50, batch_size=512,
        optimizer='Adam', learning_rate=0.0005, her_ratio=0.2, fixed_horizons=HORIZONS,
        training_evaluation_episodes=100, training_evaluation_interval=2000,
        checkpoint_selection='original best discounted return at training evaluation intervals',
        paper_seed_ids='not specified; seed IDs are chosen by the reproduction experiment',
        project_root=str(project_root), python=sys.version, executable=sys.executable,
        platform=platform.platform(), packages=packages,
        source_sha256=source_hashes, dataset_sha256=dataset_hashes,
        threads=1,
    )


def completed_checkpoint(job, budget):
    import torch

    checkpoint_dir = job['checkpoint_dir']
    original_path = checkpoint_dir / 'best_policy.pt'
    sidecar_path = checkpoint_dir / 'best_policy_with_normalization.pt'
    original_weights = torch.load(original_path, map_location='cpu', weights_only=True)
    checkpoint = torch.load(sidecar_path, map_location='cpu', weights_only=True)
    if checkpoint.get('checkpoint_type') != 'merlin_policy':
        raise ValueError('expected a complete best-policy checkpoint')
    metadata = checkpoint['metadata']
    expected = {
        'task': job['task'],
        'variant': 'expert',
        'seed': job['seed'],
        'her_ratio': 0.2,
        'fixed_horizon': HORIZONS[job['task']],
        'diffusion_rollout': False,
        'diffusion_nonparam': False,
        'normalization_source': 'training',
    }
    if any(metadata.get(name) != value for name, value in expected.items()):
        raise ValueError('best-policy checkpoint metadata does not match the run')
    best_update = metadata['training_steps']
    if not 0 < best_update <= budget or best_update % 2000:
        raise ValueError('best-policy update is outside the requested training budget')
    config = checkpoint['policy_config']
    if (config['hidden_dim'] != 256 or config['timestep_embedding_dim'] != 32
            or config['max_path_length'] != 50):
        raise ValueError('policy architecture does not match the reproduction protocol')
    checkpoint_weights = checkpoint['policy_state_dict']
    if set(original_weights) != set(checkpoint_weights) or any(
            not torch.equal(value, original_weights[name]) for name, value in checkpoint_weights.items()):
        raise ValueError('complete checkpoint weights differ from original best_policy.pt')
    return {
        'best_training_steps': best_update,
        'best_policy_sha256': file_hash(original_path),
        'complete_policy_sha256': file_hash(sidecar_path),
    }


def preflight_jobs(jobs, args):
    pending = []
    for job in jobs:
        record_path = job['record_dir'] / 'status.json'
        if not record_path.exists():
            if job['record_dir'].exists() or job['checkpoint_dir'].exists():
                raise FileExistsError('existing run artifacts would trigger loading or overwrite; choose a new --run-root')
            pending.append(job)
            continue
        record = json.loads(record_path.read_text(encoding='utf-8'))
        if not args.skip_completed or record.get('status') != 'complete':
            raise FileExistsError(
                'run already has a status record; use --skip-completed for complete '
                'runs or a new --run-root for interrupted runs'
            )
        if record.get('max_timesteps') != args.max_timesteps:
            raise ValueError('completed run used a different training budget')
        verified = completed_checkpoint(job, args.max_timesteps)
        if any(record.get(name) != value for name, value in verified.items()):
            raise ValueError('completed checkpoint has changed since the recorded run')
        print(f"Skip completed: {job['task']} seed {job['seed']}")
    return pending


def stream_training(command, cwd, env, log_path):
    with log_path.open('x', encoding='utf-8') as log:
        process = subprocess.Popen(
            command, cwd=cwd, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
        )
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
            return process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            process.stdout.close()


def run_job(job, args, env):
    job['record_dir'].mkdir(parents=True)
    record_path = job['record_dir'] / 'status.json'
    log_path = job['record_dir'] / 'training.log'
    record = {
        'status': 'running',
        'task': job['task'],
        'variant': 'expert',
        'seed': job['seed'],
        'max_timesteps': args.max_timesteps,
        'started': timestamp(),
        'command': job['command'],
    }
    write_json(record_path, record)
    start = time.monotonic()
    print(f"\nTraining {job['task']} seed {job['seed']}", flush=True)
    try:
        exit_code = stream_training(job['command'], args.run_root, env, log_path)
        if exit_code:
            raise RuntimeError(f'trainer exited with status {exit_code}; inspect {log_path}')
        training_log = log_path.read_text(encoding='utf-8')
        if (f'Timesteps: {args.max_timesteps},' not in training_log
                or f'Iterations: {args.max_timesteps},' not in training_log):
            raise ValueError('trainer did not log optimizer updates and evaluation at the full budget')
        record.update(completed_checkpoint(job, args.max_timesteps))
        record.update(status='complete', finished=timestamp(), elapsed_seconds=time.monotonic() - start)
        write_json(record_path, record)
        print(
            f"Completed {job['task']} seed {job['seed']} | "
            f"{record['elapsed_seconds'] / 60:.1f} min | best update {record['best_training_steps']}",
            flush=True,
        )
    except BaseException as error:
        record.update(
            status='failed', finished=timestamp(),
            elapsed_seconds=time.monotonic() - start, error=str(error),
        )
        write_json(record_path, record)
        raise


def run(args, project_root=PROJECT_ROOT):
    project_root = Path(project_root).resolve()
    args.run_root = args.run_root.resolve()
    data_root = project_root / 'offline_data'
    protected = [data_root, project_root / 'src', project_root / 'logging', project_root / 'checkpoints']
    if (args.run_root == project_root or args.run_root in data_root.parents
            or any(args.run_root.is_relative_to(path) for path in protected)):
        raise ValueError('--run-root must be a separate output directory')
    manifest_path = args.run_root / 'protocol.json'
    specification = protocol(args, project_root)
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding='utf-8')) != specification:
            raise ValueError('recorded protocol, source, datasets, or environment differs; use a new --run-root')
    elif args.run_root.exists() and any(args.run_root.iterdir()):
        raise FileExistsError('--run-root must be empty or managed by this runner')
    jobs = build_jobs(args, project_root)
    print(f"Requested: {len(jobs)} policy runs; {len(jobs) * args.max_timesteps:,} optimizer updates")
    print(f'Isolated output root: {args.run_root}')
    if args.dry_run:
        pending = preflight_jobs(jobs, args)
        for job in pending:
            print(f"{job['task']} seed {job['seed']}: {subprocess.list2cmdline(job['command'])}")
        print('Dry run.')
        return
    args.run_root.mkdir(parents=True, exist_ok=True)
    lock_path = args.run_root / 'batch.lock'
    with lock_path.open('x', encoding='utf-8') as lock:
        lock.write(f'pid={os.getpid()} started={timestamp()}\n')
    try:
        pending = preflight_jobs(jobs, args)
        link = args.run_root / 'offline_data'
        if link.exists() or link.is_symlink():
            if not link.is_symlink() or link.resolve() != data_root.resolve():
                raise ValueError('isolated offline_data must point to the original dataset directory')
        else:
            link.symlink_to(data_root, target_is_directory=True)
        if not manifest_path.exists():
            write_json(manifest_path, specification)
        env = os.environ.copy()
        python_paths = [str(project_root), str(project_root / 'src'), env.get('PYTHONPATH', '')]
        env['PYTHONPATH'] = os.pathsep.join(python_paths)
        env['OMP_NUM_THREADS'] = env['MKL_NUM_THREADS'] = '1'
        env['PYTHONUNBUFFERED'] = '1'
        for job in pending:
            run_job(job, args, env)
        print('Requested batch complete.')
    finally:
        lock_path.unlink()


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, default=Path('reproduction/full_budget'))
    parser.add_argument('--tasks', nargs='+', choices=list(HORIZONS), default=list(HORIZONS))
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(10)))
    parser.add_argument('--max-timesteps', type=int, default=500000)
    parser.add_argument('--skip-completed', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.max_timesteps < 2000 or args.max_timesteps % 2000:
        parser.error('--max-timesteps must be a positive multiple of the 2000-update evaluation interval')
    if any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error('seeds must be in the NumPy seed range')
    if len(set(args.tasks)) != len(args.tasks) or len(set(args.seeds)) != len(args.seeds):
        parser.error('tasks and seeds must not contain duplicates')
    return args


def main():
    run(get_args())


if __name__ == '__main__':
    main()
