"""Start the full run in a session independent of the invoking terminal."""
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
run_dir = root / 'runs' / 'cifar10_gpu0'
run_dir.mkdir(parents=True, exist_ok=True)
with (run_dir / 'console.log').open('ab') as log:
    process = subprocess.Popen(
        ['bash', str(root / 'scripts' / 'train_cifar_gpu0.sh')],
        cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
print(f'Launcher PID: {process.pid}; progress: {run_dir / "console.log"}')
