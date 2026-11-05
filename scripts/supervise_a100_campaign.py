"""Durable local supervision for the approved November 2026 A100 campaign."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT.parents[2] / 'programs/soflow-a100-campaign-20261105'
RUN = ROOT / 'runs/soflow_a100_campaign_20261105'
RUNNER = ROOT / 'scripts/prospective_command.sh'
UUID = 'GPU-5b21247d-a82b-11f1-bf7c-810566ab0ea6'
CUTOFF = dt.datetime(2026, 11, 7, 6, 59, tzinfo=dt.UTC).timestamp()
KST = dt.timezone(dt.timedelta(hours=9))
ARMS = (
    ('S3 60k', 'configs/soflow_a100_s3_60k.json', 99_682),
    ('FM25 40k', 'configs/soflow_a100_c3_fm25_40k.json', 62_952),
)


def timestamp(epoch, zone=dt.UTC):
    return dt.datetime.fromtimestamp(epoch, zone).isoformat(timespec='seconds')


def atomic(path, contents):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(contents)
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def command(*args):
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=True,
                          timeout=5).stdout.strip()


def session_processes(sid):
    result = subprocess.run(['ps', '-s', str(sid), '-o', 'pid=,args='],
                            capture_output=True, text=True, timeout=5)
    return [line.strip().split(None, 1) for line in result.stdout.splitlines() if line.strip()]


def terminate_session(sid):
    for sig, seconds in ((signal.SIGTERM, 12), (signal.SIGKILL, 2)):
        until = time.monotonic() + seconds
        while True:
            pids = session_processes(sid)
            if not pids:
                return
            for pid, _ in pids:
                try:
                    os.kill(int(pid), sig)
                except ProcessLookupError:
                    pass
            if time.monotonic() >= until:
                break
            time.sleep(0.5)


def main():
    os.chdir(ROOT)
    commit = command('git', 'rev-parse', 'HEAD')
    if command('git', 'status', '--porcelain'):
        raise RuntimeError('Campaign requires a clean worktree')
    if hashlib.sha256(RUNNER.read_bytes()).hexdigest() != (
            'e0056022130a5ada7d28c5fcf1c6e32bcddf554cc7e63b37a0201992fb94cc50'):
        raise RuntimeError('Verified runbook bytes changed')
    RUN.mkdir()
    started = time.time()
    identity = {
        'supervisor_pid': os.getpid(), 'started_utc': timestamp(started),
        'started_kst': timestamp(started, KST), 'commit': commit,
        'cutoff_utc': timestamp(CUTOFF), 'cutoff_kst': timestamp(CUTOFF, KST),
    }
    attempts = []
    stopping = False
    release = None

    def request_stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def publish(state, step, progress, active=None, eta=None):
        now = time.time()
        pids = [os.getpid()] if state == 'running' else []
        if active:
            pids.extend(active['pids'])
        elapsed = sum(row.get('elapsed_seconds', now - row['started_unix']) for row in attempts)
        status = {
            'id': 'soflow-a100-campaign-20261105', 'state': state, 'updated': timestamp(now),
            'step': step, 'progress': progress, 'eta': eta,
            'cost': {'gpu_hours': elapsed / 3600, 'wall_seconds': now - started, 'spending': 0},
            'resources': {
                'ids': [f'cs30708b-15/GPU0/{UUID}'],
                'reservation': 'active campaign' if state == 'running' else
                               'released' if release else 'release check incomplete',
                'processes': pids, 'release_evidence': release,
            },
            'alerts': [progress] if state in ('failed', 'stopped') else [],
            'needs': None,
        }
        atomic(PROGRAM / 'status.json', json.dumps(status, indent=2) + '\n')
        evidence = {'identity': identity, 'status': status, 'attempts': attempts, 'active': active}
        atomic(RUN / 'campaign.json', json.dumps(evidence, indent=2) + '\n')
        lines = [
            '# A100 SoFlow deadline campaign', '', '## Summary', '',
            f'- State {state}. {progress}',
            f'- Updated {timestamp(now)} / {timestamp(now, KST)}.',
            f'- Cutoff {identity["cutoff_utc"]} / {identity["cutoff_kst"]}.',
            f'- Supervisor PID {os.getpid()}. Active PIDs {pids}.',
            f'- Worktree revision `{commit}`. GPU wall allocation {elapsed / 3600:.4f} h so far.',
            '', '## Attempts', '',
        ]
        for row in attempts:
            lines.append(f'- {row["arm"]} started {row["started_utc"]} / {row["started_kst"]}. '
                         f'Runner PID {row["pid"]}, cap {row["cap_seconds"]} seconds. '
                         f'Exit {row.get("returncode", "running")}. '
                         f'[{row["run_dir"]}]({ROOT / row["run_dir"]}) · [console]({row["log"]}).')
        if eta:
            lines.append(f'- Current arm conservative end bound {eta["utc"]} / {eta["kst"]}.')
        lines.extend(('', f'[Live campaign evidence]({RUN / "campaign.json"}).'))
        if release:
            lines.append(f'[GPU release]({release}).')
        atomic(PROGRAM / 'report.md', '\n'.join(lines) + '\n')

    def run_arm(label, config_path, cap):
        config = read_json(ROOT / config_path)
        if config['budget_seconds'] != cap or config['final_eval_reserve_seconds'] != 3600:
            raise RuntimeError(f'{label} allocation changed')
        remaining = CUTOFF - time.time()
        if remaining < cap + 60:
            return f'{label} skipped. {remaining / 3600:.2f} h remain before the reserve.'
        if stopping:
            raise InterruptedError('Stop requested before launch')
        log = RUN / f'{Path(config_path).stem}.log'
        arm_start = time.time()
        arm_timer = time.monotonic()
        arm_end = arm_start + cap
        with log.open('x') as output:
            process = subprocess.Popen(['bash', str(RUNNER), config_path], stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            attempt = {'arm': label, 'started_unix': arm_start, 'started_utc': timestamp(arm_start),
                       'started_kst': timestamp(arm_start, KST), 'pid': process.pid,
                       'cap_seconds': cap, 'run_dir': config['run_dir'], 'config': config_path,
                       'log': str(log)}
            attempts.append(attempt)
            eta = {'utc': timestamp(arm_end), 'kst': timestamp(arm_end, KST)}
            run_dir = ROOT / config['run_dir']
            try:
                while process.poll() is None:
                    if stopping:
                        raise InterruptedError(f'{label} stopped by supervisor signal')
                    seconds_left = min(cap - (time.monotonic() - arm_timer), CUTOFF - time.time())
                    if seconds_left <= 15:
                        raise TimeoutError(f'{label} reached its shutdown allowance')
                    session = session_processes(process.pid)
                    active = {'pids': [int(pid) for pid, _ in session], 'processes': session}
                    active['train_status'] = read_json(run_dir / 'status.json')
                    active['latest_checkpoint_exists'] = (run_dir / 'checkpoints/latest.pt').is_file()
                    step = active['train_status'].get('step', 0)
                    progress = f'{label} at step {step}/{config["total_steps"]}.'
                    if step == config['total_steps']:
                        progress += ' Paired evaluation and export are underway.'
                    publish('running', label, progress, active, eta)
                    time.sleep(min(15 if step else 5, max(0.1, seconds_left - 15)))
            finally:
                if session_processes(process.pid):
                    terminate_session(process.pid)
                result = process.wait(timeout=5)
                elapsed = time.monotonic() - arm_timer
                attempt.update(returncode=result, elapsed_seconds=elapsed, ended_utc=timestamp(time.time()),
                               ended_kst=timestamp(time.time(), KST))
        if result != 0 or elapsed > cap:
            raise RuntimeError(f'{label} failed or exceeded budget. Runner exit {result}. See {log}.')
        terminal = read_json(run_dir / 'status.json')
        metrics = list((run_dir / 'paired_dev/evaluations').glob('*/metrics.json'))
        observed = {(row['identity']['nfe'], row['identity']['seed']) for row in map(read_json, metrics)}
        expected = {(nfe, seed) for nfe in (1, 4) for seed in range(4101, 4105)}
        if (terminal.get('reason') != 'schedule_complete' or terminal.get('step') != config['total_steps']
                or observed != expected or not (run_dir / 'export/SHA256SUMS.json').is_file()):
            raise RuntimeError(f'{label} lacks required terminal, paired evaluation or export evidence')
        attempt['paired_metrics'] = [str(path) for path in sorted(metrics)]
        attempt['export'] = str(run_dir / 'export')
        return f'{label} completed in {elapsed / 3600:.2f} h with paired evaluation and verified export.'

    state = 'failed'
    progress = 'Campaign did not start.'
    try:
        publish('running', 'preflight', 'Checking launch conditions.')
        if command('nvidia-smi', '-i', '0', '--query-gpu=uuid', '--format=csv,noheader') != UUID:
            raise RuntimeError('Physical GPU UUID changed')
        if command('nvidia-smi', '-i', '0', '--query-compute-apps=pid', '--format=csv,noheader,nounits'):
            raise RuntimeError('GPU0 already has an active compute process')
        summaries = []
        for label, config, cap in ARMS:
            summaries.append(run_arm(label, config, cap))
            if len(attempts) != len(summaries):
                break
        state, progress = 'done', ' '.join(summaries)
    except InterruptedError as error:
        state, progress = 'stopped', str(error)
    except Exception as error:
        state, progress = 'failed', str(error)
    finally:
        try:
            gpu = command('nvidia-smi', '-i', '0', '--query-gpu=uuid,memory.used,utilization.gpu',
                          '--format=csv')
            active = command('nvidia-smi', '-i', '0', '--query-compute-apps=pid',
                             '--format=csv,noheader,nounits')
            if active:
                raise RuntimeError(f'GPU compute PIDs remain: {active}')
            command('flock', '-n', str(ROOT / 'runs/gpu_0.lock'), 'true')
            command('flock', '-n', '/home/jw/.local/share/fastgen-soflow-a100/gpu-5b21247d.lock', 'true')
            release_path = RUN / 'release.txt'
            atomic(release_path, f'{timestamp(time.time())}\n{gpu}\nNo compute processes. Both leases available.\n')
            release = str(release_path)
        except Exception as error:
            progress += f' Release check: {error}.'
        publish(state, 'terminal', progress)
        print(progress, flush=True)


if __name__ == '__main__':
    main()
