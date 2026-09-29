"""Prepare and durably supervise a declared scratch-training experiment."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.launch import run_gpu_job, used_seconds
from soflow_fastgen.common import ROOT, atomic_json, digest, source_provenance
from soflow_fastgen.evaluation import prepare_reference
from soflow_fastgen.paired_evaluation import environment_identity, fingerprint, image_manifest
from soflow_fastgen.training_study import check_inputs, check_prepared, evaluate_study, load_study


def process_identity():
    return {'pid': os.getpid(), 'started_unix': time.time(),
            'proc_start_ticks': Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def accounts(protocol):
    run = Path(protocol['run_dir'])
    return {'qualification': run/'accounting/qualification',
            'training': run, 'evaluation': run/'accounting/evaluation'}


def consumed(protocol):
    values = {name: used_seconds(path) for name, path in accounts(protocol).items()}
    return {**values, 'total': sum(values.values())}


def prepare(protocol, training):
    parameters = check_inputs(protocol, training)
    run = Path(protocol['run_dir'])
    if (run/'record.json').exists() or (run/'checkpoints').exists():
        raise ValueError('Existing experiment: inspect or recover it; never overwrite preparation')
    run.mkdir(parents=True, exist_ok=True)
    reference = prepare_reference(training)
    if fingerprint(image_manifest(reference)) != protocol['reference_sha256']:
        raise ValueError('Fresh study reference preprocessing differs from EXP-1')
    provenance = source_provenance()
    record = {
        'experiment': protocol['experiment'], 'state': 'prepared',
        'authorization_reference': protocol['authorization_reference'],
        'initialization': 'scratch', 'starting_checkpoint': None, 'initial_step': 0,
        'protocol_sha256': fingerprint(protocol),
        'training_config_sha256': digest(protocol['training_config']),
        'source_sha256': fingerprint(provenance['source_sha256']),
        'prepared_git_commit': provenance['git_commit'], 'environment': environment_identity(),
        'prepared_unix': time.time(), 'gpu': 1, 'parameters': parameters,
        'attempts': [], 'completed_stages': [], 'unfinished_job': None,
        'gpu_job_seconds': consumed(protocol), 'token_expenditure': 'unavailable',
    }
    atomic_json(run/'protocol.json', protocol)
    atomic_json(run/'provenance.json', provenance)
    atomic_json(run/'record.json', record)
    print(json.dumps({'state': record['state'], 'parameters': parameters,
                      'protocol_sha256': record['protocol_sha256'], 'run_dir': str(run)}))


def summarize(protocol):
    run = Path(protocol['run_dir'])
    selection = json.loads((run/'selection.json').read_text())
    confirmation = json.loads((run/'confirmation.json').read_text())
    status = json.loads((run/'status.json').read_text())
    metrics = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    stats = {
        'experiment': protocol['experiment'], 'training_steps': status['step'],
        'logged_update_records': len(metrics),
        'optimizer_seconds': sum(r['seconds'] for r in metrics),
        'logged_image_presentations': len(metrics)*128,
        'peak_training_allocated_bytes': max(r['allocated_bytes'] for r in metrics),
        'peak_training_reserved_bytes': max(r['reserved_bytes'] for r in metrics),
        'peak_host_rss_bytes': max(r['host_peak_rss_bytes'] for r in metrics),
        'peak_sampled_process_tree_pss_bytes': max(r.get('process_tree_pss_bytes', 0) for r in metrics),
        'gpu_job_seconds': consumed(protocol),
        'reference_training_gpu_job_seconds': protocol['reference_training_gpu_job_seconds'],
        'prior_research_gpu_job_seconds': protocol['prior_research_gpu_job_seconds'],
        'selection': selection['decisions'], 'confirmation': confirmation['decisions'],
        'protocol_sha256': fingerprint(protocol), 'official_score': False,
    }
    atomic_json(run/'result_summary.json', stats)


def supervise(protocol_path, protocol, training, resume):
    run = Path(protocol['run_dir'])
    with (run/'workflow.lock').open('a') as workflow, (ROOT/'runs/gpu_1.lock').open('a') as gpu:
        fcntl.flock(workflow, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = json.loads((run/'record.json').read_text())
        check_prepared(protocol, record)
        check_inputs(protocol, training)
        if record['unfinished_job'] is not None:
            raise ValueError('Reconcile unfinished job identity and accounting before recovery')
        if record['state'] == 'completed':
            raise ValueError('Experiment already completed')
        if record['attempts'] and not resume:
            raise ValueError('Existing attempts require explicit same-run recovery')
        if record['gpu_job_seconds'] != consumed(protocol):
            raise ValueError('Durable record and budget files disagree; reconcile before recovery')
        active = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-compute-apps=pid',
                                          '--format=csv,noheader,nounits'], text=True).strip()
        if active:
            raise RuntimeError(f'GPU 1 already has compute processes: {active}')
        record['worker'] = process_identity()
        atomic_json(run/'record.json', record)
        atomic_json(run/'process.json', record['worker'])
        (run/'exit_status').write_text('running\n')
        stopped = False
        current_account = None

        def stop_worker(signum, frame):
            nonlocal stopped
            stopped = True
            if current_account is not None:
                active_path = current_account/'active_job.json'
                if active_path.exists():
                    job = json.loads(active_path.read_text())
                    pid = job.get('pid')
                    if pid and Path(f'/proc/{pid}/stat').exists():
                        ticks = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
                        if ticks == job['proc_start_ticks']:
                            os.killpg(pid, signal.SIGTERM)

        signal.signal(signal.SIGTERM, stop_worker)
        signal.signal(signal.SIGINT, stop_worker)
        for stage in ('qualification', 'control', 'training', 'selection', 'confirmation'):
            if stage in record['completed_stages']:
                continue
            check_prepared(protocol, record)
            prior = [a for a in record['attempts'] if a['stage'] == stage]
            retry = bool(prior)
            if retry and sum(a['retry'] for a in record['attempts']) >= protocol['max_retries']:
                raise ValueError('The bounded experiment retry allowance is exhausted')
            kind = stage if stage in ('qualification', 'training') else 'evaluation'
            current_account = accounts(protocol)[kind]
            current_account.mkdir(parents=True, exist_ok=True)
            limit = protocol[f'{kind}_budget_seconds']
            before = used_seconds(current_account)
            allowance = limit-before-protocol['shutdown_reserve_seconds']
            if allowance <= 0:
                raise ValueError(f'{kind} allowance exhausted')
            if stage == 'qualification':
                command = [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v']
            elif stage == 'training':
                command = [sys.executable, 'train.py', '--config', protocol['training_config'], '--skip-eval']
                checkpoint = run/'checkpoints/latest.pt'
                if checkpoint.exists():
                    if not prior:
                        raise ValueError('A new training experiment cannot warm-start')
                    command += ['--resume', str(checkpoint)]
            else:
                command = [sys.executable, '-m', 'scripts.training_study', '--config', protocol_path,
                           '--evaluate', stage]
            job = {'stage': stage, 'account': kind, 'command': command, 'retry': retry,
                   'started_unix': time.time(), 'worker': record['worker'],
                   'used_seconds_before': before, 'timeout_seconds': allowance,
                   'log': str(run/f'{stage}.log')}
            record.update(state='running', unfinished_job=job, current_stage=stage)
            atomic_json(run/'record.json', record)
            returned, result = False, None
            try:
                with Path(job['log']).open('a') as log:
                    result = run_gpu_job(command, {'budget_seconds': limit-protocol['shutdown_reserve_seconds']},
                                         current_account, stdout=log, stderr=subprocess.STDOUT)
                returned = True
            finally:
                used = used_seconds(current_account)
                atomic_json(current_account/'budget.json', {'used_seconds': used, 'limit_seconds': limit})
                record['attempts'].append({**job, 'returncode': result,
                                           'used_seconds_after': used, 'ended_unix': time.time()})
                record.update(gpu_job_seconds=consumed(protocol),
                              unfinished_job=None if returned else job,
                              state='prepared' if returned and result == 0 else 'failed')
                atomic_json(run/'record.json', record)
            if stopped:
                record['state'] = 'stopped'
                atomic_json(run/'record.json', record)
                (run/'exit_status').write_text('stopped\n')
                return
            if result != 0:
                (run/'exit_status').write_text('failed\n')
                raise RuntimeError(f'{stage} failed; inspect {job["log"]}')
            if stage == 'training':
                status = json.loads((run/'status.json').read_text())
                if status['step'] != protocol['expected_steps'] or status['reason'] != 'schedule_complete':
                    record['state'] = 'training_incomplete'
                    atomic_json(run/'record.json', record)
                    (run/'exit_status').write_text('training_incomplete\n')
                    return
            record['completed_stages'].append(stage)
            atomic_json(run/'record.json', record)
        summarize(protocol)
        record.update(state='completed', current_stage=None, completed_unix=time.time(),
                      gpu_job_seconds=consumed(protocol))
        atomic_json(run/'record.json', record)
        (run/'exit_status').write_text('completed\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--evaluate', choices=('control', 'selection', 'confirmation'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    protocol, training = load_study(args.config)
    run = Path(protocol['run_dir'])
    if args.prepare:
        prepare(protocol, training)
    elif args.evaluate:
        record = json.loads((run/'record.json').read_text())
        if record['state'] != 'running' or record['current_stage'] != args.evaluate:
            raise ValueError('Evaluation must belong to the recorded supervised stage')
        evaluate_study(protocol, args.evaluate, record)
    elif args.worker:
        supervise(args.config, protocol, training, args.resume)
    else:
        if not (run/'record.json').exists():
            raise ValueError('Prepare and freeze the experiment before launch')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1',
                   OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MPLBACKEND='Agg')
        command = [sys.executable, '-m', 'scripts.training_study', '--config', args.config, '--worker']
        if args.resume:
            command += ['--resume']
        with (run/'console.log').open('a') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True, cwd=ROOT, env=env)
        time.sleep(1)
        if process.poll() is not None:
            raise SystemExit(f'Workflow startup failed; inspect {run/"console.log"}')
        print(json.dumps({'workflow_pid': process.pid, 'run_dir': str(run), 'log': str(run/'console.log')}))


if __name__ == '__main__':
    main()
