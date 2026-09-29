import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from scripts.launch import run_gpu_job, used_seconds
from soflow_fastgen.paired_evaluation import PairedAdapter
from soflow_fastgen.training_study import decide, validate_pairing, validate_protocol


class StudyContracts(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(Path('configs/research_s1_20260930.json').read_text())
        self.training = json.loads(Path(self.protocol['training_config']).read_text())

    def test_scratch_initialization_and_complete_horizon_are_required(self):
        validate_protocol(self.protocol, self.training)
        for key, value in (('initialization', 'warm_start'), ('starting_checkpoint', 'old.pt'),
                           ('expected_steps', 20000)):
            candidate = dict(self.protocol, **{key: value})
            with self.assertRaises(ValueError):
                validate_protocol(candidate, self.training)
        changed = dict(self.training, checkpoint_retain_steps=[10000, 30000])
        with self.assertRaises(ValueError):
            validate_protocol(self.protocol, changed)

    def test_confirmation_seeds_cannot_overlap_selection(self):
        changed = copy.deepcopy(self.protocol)
        changed['confirmation_seeds'][0] = changed['selection_seeds'][0]
        with self.assertRaises(ValueError):
            validate_protocol(changed, self.training)

    def rows(self, seeds):
        result = []
        for label in ('reference', 'step_030000', 'step_040000'):
            for nfe in (1, 4):
                for seed in seeds:
                    result.append({'checkpoint_label': label, 'nfe': nfe, 'seed': seed,
                                   'fid': 10. if label == 'reference' else 1. if label == 'step_030000'
                                          else 9. if nfe == 1 else 11.,
                                   'noise_sha256': [f'{seed}:0'] + [f'{seed}:{i}' for i in range(1, nfe)]})
        return result

    def test_final_checkpoint_and_nfe_tracks_are_decided_independently(self):
        decisions = decide(self.rows(self.protocol['selection_seeds']), self.protocol, 'selection')
        self.assertTrue(decisions['1']['eligible'])
        self.assertFalse(decisions['4']['eligible'])
        self.assertEqual(decisions['4']['candidate_mean'], 11.)

    def test_confirmation_requires_every_reserved_seed(self):
        rows = self.rows(self.protocol['confirmation_seeds'])
        decisions = decide(rows, self.protocol, 'confirmation', [1])
        self.assertEqual(decisions['1']['outcome'], 'confirmed_under_protocol')
        rows[-2]['fid'] = float('nan')
        with self.assertRaises(ValueError):
            decide(rows, self.protocol, 'confirmation', [1])
        rows = self.rows(self.protocol['confirmation_seeds'])
        rows = [r for r in rows if not (r['checkpoint_label'] == 'step_040000'
                                       and r['nfe'] == 1 and r['seed'] == 4104)]
        with self.assertRaises(KeyError):
            decide(rows, self.protocol, 'confirmation', [1])

    def test_pairing_rejects_changed_draws_and_duplicate_rows(self):
        rows = self.rows(self.protocol['selection_seeds'])
        rows[-1]['noise_sha256'][-1] = 'different draw'
        with self.assertRaises(ValueError):
            validate_pairing(rows)
        rows = self.rows(self.protocol['selection_seeds'])
        with self.assertRaises(ValueError):
            validate_pairing(rows + [rows[0]])


class ToyNetwork(torch.nn.Module):
    def forward(self, x, t, s, labels, augmentation):
        return x * .5


class ToyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = ToyNetwork()

    def solution_operator(self, x, t, s, labels, augmentation):
        return self.network(x, t, s, labels, augmentation)


class PairedNoiseCPU(unittest.TestCase):
    def test_actual_draws_repeat_across_checkpoints_and_initial_draws_across_nfes(self):
        a = PairedAdapter(ToyBackbone(), 1, [], 3101)
        b = PairedAdapter(ToyBackbone(), 4, [.75, .5, .25], 3101)
        c = PairedAdapter(ToyBackbone(), 4, [.75, .5, .25], 3101)
        for size in (2, 1):
            labels = torch.zeros(size, dtype=torch.long)
            a.sample((size, 3, 64, 64), device='cpu', category=labels)
            x = b.sample((size, 3, 64, 64), device='cpu', category=labels)
            y = c.sample((size, 3, 64, 64), device='cpu', category=labels)
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        self.assertEqual(a.noise_sha256()[0], b.noise_sha256()[0])
        self.assertEqual(b.noise_sha256(), c.noise_sha256())
        self.assertEqual((a.network_calls, b.network_calls), (2, 8))


class SupervisionCPU(unittest.TestCase):
    def test_child_identity_output_and_elapsed_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            with (run/'child.log').open('w') as log:
                result = run_gpu_job([sys.executable, '-c', 'print("retained output")'],
                                     {'budget_seconds': 5}, run, stdout=log, stderr=subprocess.STDOUT)
            self.assertEqual(result, 0)
            self.assertIn('retained output', (run/'child.log').read_text())
            self.assertGreater(used_seconds(run), 0)
            job = json.loads((run/'active_job.json').read_text())
            self.assertIsNone(job['pid'])
            self.assertIn('proc_start_ticks', job)
            self.assertIn('boot_id', job)

    def test_timeout_is_charged_without_renewing_allowance(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            result = run_gpu_job([sys.executable, '-c', 'import time; time.sleep(10)'],
                                 {'budget_seconds': .1}, run)
            self.assertIsNone(result)
            before = used_seconds(run)
            self.assertGreaterEqual(before, .1)
            self.assertIsNone(run_gpu_job([sys.executable, '-c', 'raise RuntimeError()'],
                                         {'budget_seconds': .1}, run))
            self.assertEqual(used_seconds(run), before)
