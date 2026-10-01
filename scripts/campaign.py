"""Bounded supervisor; durable allowance includes every failed/replayed attempt."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from flow_sid.common import ROOT, atomic_json, append_json, object_hash
from flow_sid.gpu import process_identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True)
    args = parser.parse_args()
    sequence = ['T', 'S1', 'A1', 'S2', 'A2', 'S3', 'A3', 'S4', 'A4']
    assert all(r in sequence for r in args.runs)
    config = json.loads((ROOT / 'configs/campaign.json').read_text())
    ledger_path = ROOT / 'runs/campaign/allowance.json'
    ledger = (json.loads(ledger_path.read_text()) if ledger_path.exists() else
              {'config_sha256': object_hash(config), 'runs': {}, 'started_at': time.time()})
    assert ledger['config_sha256'] == object_hash(config)
    active_child = None
    def stop(signum, frame):
        if active_child is not None and active_child.poll() is None:
            active_child.send_signal(signal.SIGTERM)
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for run in args.runs:
        folder = ROOT / 'runs/campaign' / run
        folder.mkdir(parents=True, exist_ok=True)
        entry = ledger['runs'].setdefault(run, {'consumed_seconds': 0., 'attempts': []})
        status = folder / 'status.json'
        if status.exists() and json.loads(status.read_text())['phase'] == 'complete':
            continue
        preceding = sequence[:sequence.index(run)]
        for prev in preceding:
            p = ROOT / 'runs/campaign' / prev / 'status.json'
            assert p.exists() and json.loads(p.read_text())['phase'] == 'complete', prev
        limit = config['watchdogs']['teacher_seconds'] if run == 'T' else config['watchdogs']['student_seconds'][run[1:]]
        while True:
            # An orphan attempt cannot be replaced while its process identity remains live.
            for a in entry['attempts']:
                if 'end' not in a:
                    try:
                        if process_identity(a['pid']) == a['identity']:
                            raise RuntimeError('Recorded worker still exists; refusing duplicate')
                    except FileNotFoundError:
                        pass
                    a['end'] = time.time()
                    a['outcome'] = 'external interruption, conservative allowance charge'
                    elapsed = a['end'] - a['start']
                    a['seconds'] = elapsed
                    entry['consumed_seconds'] += elapsed
            if len(entry['attempts']) >= 1 + config['watchdogs']['max_exact_recoveries_per_run']:
                raise RuntimeError(f'{run}: recovery allowance exhausted')
            remaining = limit - entry['consumed_seconds']
            if remaining <= 0:
                raise RuntimeError(f'{run}: finite watchdog exhausted')
            start = time.time()
            logfile = folder / f'worker_{len(entry["attempts"]):02d}.log'
            with logfile.open('w') as log:
                active_child = subprocess.Popen(['bash', 'scripts/run_gpu.sh', 'scripts/train_stage.py', '--run', run],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                attempt = {'pid': active_child.pid, 'identity': process_identity(active_child.pid),
                           'start': start, 'log': str(logfile.relative_to(ROOT)), 'allowance_at_start': remaining}
                entry['attempts'].append(attempt)
                atomic_json(ledger_path, ledger)
                last_progress = time.time()
                last_marker = None
                try:
                    while active_child.poll() is None:
                        if status.exists():
                            current = json.loads(status.read_text())
                            marker = (current['phase'], current['step'])
                            if marker != last_marker:
                                last_marker = marker
                                last_progress = time.time()
                        if time.time() - start >= remaining:
                            raise TimeoutError('Full run watchdog expired')
                        if time.time() - last_progress > config['watchdogs']['no_progress_seconds']:
                            raise TimeoutError('No-progress watchdog expired')
                        time.sleep(10)
                    code = active_child.returncode
                except BaseException as exc:
                    if active_child.poll() is None:
                        # Known owned worker only; never issue a device-wide action.
                        assert process_identity(active_child.pid) == attempt['identity']
                        active_child.terminate()
                        try: active_child.wait(timeout=30)
                        except subprocess.TimeoutExpired: active_child.kill(); active_child.wait()
                    attempt['outcome'] = str(exc)
                    code = active_child.returncode
                    raise
                finally:
                    attempt['end'] = time.time()
                    attempt['seconds'] = attempt['end'] - start
                    attempt['returncode'] = active_child.returncode
                    entry['consumed_seconds'] += attempt['seconds']
                    atomic_json(ledger_path, ledger)
                    active_child = None
            if code == 0:
                break
            # Only external interruptions permit exact checkpoint recovery. Numerical,
            # data, code and ownership failures are surfaced, not silently retrained.
            if code not in [-signal.SIGTERM, -signal.SIGINT, 130, 143] or not (folder / 'latest.pt').exists():
                raise RuntimeError(f'{run} failed (exit {code}); see {logfile}')
        print(json.dumps({'finished': run, 'charged_seconds': entry['consumed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
