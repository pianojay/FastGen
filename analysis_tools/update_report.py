"""Evidence-backed campaign tables and standalone quality/cost SVG plots."""
import html
import json
import time
from flow_sid.common import ROOT, atomic_json
from scripts.report import curve

RUNS = ['T', 'S1', 'A1', 'S2', 'A2', 'S3', 'A3', 'S4', 'A4']


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


def plot(records, key, path, title, xlabel):
    points = [r for name, r in records.items() if name != 'T' and r.get('confirmation_fid') is not None]
    if not points:
        return
    w, h, left, right, top, bottom = 800, 440, 72, 22, 58, 66
    xmin, xmax = (0, 4.3) if key == 'nfe' else (0, max(r[key] for r in points) * 1.12)
    ymin, ymax = 0, max(r['confirmation_fid'] for r in points) * 1.14
    xy = lambda x, y: (left + (w-left-right)*(x-xmin)/(xmax-xmin), h-bottom-(h-top-bottom)*(y-ymin)/(ymax-ymin))
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{left}" y="25" font-family="sans-serif" font-size="18">{html.escape(title)}</text>',
             '<g font-family="sans-serif" font-size="12" fill="#222">',
             f'<path d="M{left} {top} V{h-bottom} H{w-right}" stroke="#333" fill="none"/>']
    for i in range(6):
        value = ymax*i/5
        _, y = xy(0, value)
        lines += [f'<path d="M{left} {y:.2f} H{w-right}" stroke="#e8e8e8"/>',
                  f'<text x="{left-9}" y="{y+4:.2f}" text-anchor="end">{value:.1f}</text>']
    xticks = [1, 2, 3, 4] if key == 'nfe' else [xmax*i/5 for i in range(6)]
    for value in xticks:
        x, y = xy(value, 0)
        lines.append(f'<text x="{x:.2f}" y="{y+20:.2f}" text-anchor="middle">{value:.1f}</text>')
    for method, color in [('S', '#2166ac'), ('A', '#d95f02')]:
        selected = sorted([p for p in points if p['run'].startswith(method)], key=lambda p: p['nfe'])
        coords = [xy(p[key], p['confirmation_fid']) for p in selected]
        if key == 'nfe' and len(coords) > 1:
            lines.append('<polyline points="' + ' '.join(f'{x:.2f},{y:.2f}' for x,y in coords) + f'" stroke="{color}" fill="none"/>')
        for p, (x, y) in zip(selected, coords):
            lines += [f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{color}"/>',
                      f'<text x="{x+8:.2f}" y="{y-8:.2f}" fill="{color}">{p["run"]}: {p["confirmation_fid"]:.2f}</text>']
    lines += [f'<text x="{w/2}" y="{h-22}" text-anchor="middle">{html.escape(xlabel)}</text>',
              '<text x="18" y="215" transform="rotate(-90 18 215)" text-anchor="middle">Confirmation development FID ↓</text>',
              '<text x="520" y="44" fill="#2166ac">SiD</text><text x="575" y="44" fill="#d95f02">SiDA</text>',
              '</g></svg>']
    path.write_text('\n'.join(lines))


def main():
    now = time.time()
    ledger = read(ROOT / 'runs/campaign/allowance.json', {'runs': {}})
    records = {}
    report = ['# Flow SiD/SiDA campaign results', '',
              'Local development evaluation: 3,020 fixed train-derived reference images and 20 generated images per class. '
              'Selection seed 1729; confirmation seed 271828 without reselection. One training realization per arm.', '',
              '| Run | State / completed updates | Selected update | NFE | Selection FID | Final FID | Confirmation FID | Pixel flags / 151 | Stage hours |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for run in RUNS:
        folder = ROOT / 'runs/campaign' / run
        status = read(folder / 'status.json', {})
        selected = read(folder / 'selected.json', {})
        export = read(folder / 'export.json', {})
        conf = export.get('confirmation', {})
        entry = ledger['runs'].get(run, {})
        seconds = entry.get('consumed_seconds', 0) + sum(now-a['start'] for a in entry.get('attempts', []) if 'end' not in a)
        metrics = rows(folder / 'metrics.jsonl')
        evals = rows(folder / 'evaluations.jsonl')
        final_step = 30000 if run == 'T' else 4000
        final_fid = next((e['fid'] for e in reversed(evals) if e['step'] == final_step), None)
        memories = rows(folder / 'memory.jsonl')
        if 'memory' in status:
            memories.append(status['memory'])
        r = {'run': run, 'nfe': 32 if run == 'T' else int(run[1:]), 'status': status,
             'selected': selected, 'confirmation_fid': conf.get('fid'), 'stage_hours': seconds/3600,
             'final_fid': final_fid, 'evaluations': evals,
             'optimization_seconds': sum(m['seconds'] for m in metrics),
             'generation_and_fid_seconds': sum(e['seconds'] for e in evals) + conf.get('seconds', 0),
             'attempts': len(entry.get('attempts', [])),
             'peak_allocated': max((m['peak_allocated_bytes'] for m in memories), default=0),
             'peak_reserved': max((m['peak_reserved_bytes'] for m in memories), default=0),
             'device_MiB': max((int(m['device']['used_MiB']) for m in memories), default=0),
             'host_RSS': max((m['host']['Rss_bytes'] for m in memories), default=0),
             'host_PSS': max((m['host']['Pss_bytes'] for m in memories), default=0),
             'host_HWM': max((m['host']['VmHWM_bytes'] for m in memories), default=0)}
        records[run] = r
        f = lambda v: '—' if v is None else f'{v:.4f}'
        flags = selected.get('diversity', {}).get('potential_collapse_classes')
        report.append(f'| {run} | {status.get("phase", "not started")} / {status.get("step", 0)} | '
                      f'{selected.get("step", "—")} | {r["nfe"]} | {f(selected.get("fid"))} | {f(final_fid)} | '
                      f'{f(conf.get("fid"))} | {len(flags) if flags is not None else "—"} | {seconds/3600:.3f} |')
        unique_updates = list({m['step']: m for m in metrics}.values())
        if run != 'T':
            r['g_presentations_by_ladder_position'] = [64 * sum(m['step_g'] == i for m in unique_updates) for i in range(r['nfe'])]
            r['fake_presentations_by_ladder_position'] = [64 * sum(m['step_fake'] == i for m in unique_updates) for i in range(r['nfe'])]
        for key in ['fm_loss', 'sid_loss', 'fake_fm_loss', 'd_bce_loss', 'g_gan_loss', 'g_grad_norm', 'fake_D_grad_norm', 'seconds']:
            curve(metrics, key, folder / (key + '.svg'))
    report += ['', 'Active-stage hours include elapsed work; completed-stage hours use the external allowance ledger. '
               'Stage cost includes optimization, evaluation, diagnostics, checkpoint I/O, startup and any failed/replayed attempts.', '',
               '| NFE | Selection ΔFID (A−S) | Final ΔFID (A−S) | Confirmation ΔFID (A−S) | Δ median relative pixel diversity (A−S) |',
               '|---:|---:|---:|---:|---:|']
    for n in range(1, 5):
        s, a = records['S'+str(n)], records['A'+str(n)]
        diff = lambda av, sv: '—' if av is None or sv is None else f'{av-sv:+.4f}'
        report.append(f'| {n} | {diff(a["selected"].get("fid"),s["selected"].get("fid"))} | '
                      f'{diff(a["final_fid"],s["final_fid"])} | '
                      f'{diff(a["confirmation_fid"],s["confirmation_fid"])} | '
                      f'{diff(a["selected"].get("diversity",{}).get("median_relative_diversity"),s["selected"].get("diversity",{}).get("median_relative_diversity"))} |')
    report += ['', '| NFE | Matched updates | G and fake presentations each | SiD FID | SiDA FID | ΔFID (A−S) |',
               '|---:|---:|---:|---:|---:|---:|']
    for n in range(1, 5):
        for step in [1000, 2000, 4000]:
            values = [next((e['fid'] for e in reversed(records[m+str(n)]['evaluations']) if e['step'] == step), None) for m in ['S', 'A']]
            if any(v is not None for v in values):
                report.append(f'| {n} | {step} | {step*64:,} | {f(values[0])} | {f(values[1])} | {diff(values[1],values[0])} |')
    report += ['', '| Run | Teacher presentations | G / fake / real D presentations | G presentations by ladder position |',
               '|---|---:|---:|---|']
    for run, r in records.items():
        exp = r['status'].get('exposures', {})
        if not exp:
            report.append(f'| {run} | — | — | — |')
            continue
        counts = ' / '.join(f'{exp[k]:,}' for k in ['g_images', 'fake_images', 'd_real_images'])
        positions = ', '.join(f'{v:,}' for v in r.get('g_presentations_by_ladder_position', [])) or '—'
        report.append(f'| {run} | {exp["teacher_images"]:,} | {counts} | {positions} |')
    report += ['', 'Same-NFE initial-baseline FID: **unavailable**. The teacher’s 32-call evaluation is context, '
               'and the unrelated untrained plumbing FID is not a scientific control. '
               'Equal student cycles match total G/fake exposure; per-position exposure and compute differ across NFE. '
               'Interpretation follows [the feasibility focus](FEASIBILITY_FOCUS.md), written after the first S1 result; '
               'it is not a retrospective preregistered success threshold.']
    report += ['', '| Run | Optimization hours | Generation + FID hours | Allocated / reserved GB | Sampled device GB | Worker host RSS / PSS / HWM GB | Attempts |',
               '|---|---:|---:|---:|---:|---:|---:|']
    for run, r in records.items():
        if not r['status']:
            report.append(f'| {run} | — | — | — | — | — | 0 |')
            continue
        report.append(f'| {run} | {r["optimization_seconds"]/3600:.3f} | {r["generation_and_fid_seconds"]/3600:.3f} | '
                      f'{r["peak_allocated"]/1e9:.3f} / {r["peak_reserved"]/1e9:.3f} | {r["device_MiB"]*2**20/1e9:.3f} | '
                      f'{r["host_RSS"]/1e9:.3f} / {r["host_PSS"]/1e9:.3f} / {r["host_HWM"]/1e9:.3f} | {r["attempts"]} |')
    setup = read(ROOT / 'runs/setup_cost.json', {})
    setup_seconds = setup.get('pre_teacher_wall_seconds', 0)
    additional = [e for e in rows(ROOT / 'runs/cost.jsonl') if e.get('event') == 'end'
                  and not e['stage'].startswith('run_') and e['time'] > setup.get('teacher_worker_start', float('inf'))]
    extra_seconds = sum(e['seconds'] for e in additional)
    stage_hours = sum(r['stage_hours'] for r in records.values())
    report += ['', f'Recorded pipeline cost: **{stage_hours+(setup_seconds+extra_seconds)/3600:.3f} hours** = '
               f'{stage_hours:.3f} run-stage hours (teacher once) + {setup_seconds/3600:.3f} pre-teacher setup wall hours '
               f'+ {extra_seconds/3600:.3f} subsequent qualification/export/analysis hours. Setup includes overlapping installation, '
               'preprocessing and disposable probes; those components are not added twice. Pre-environment read-only review was not instrumented. '
               'This accounting sums run-stage and ancillary task durations; CPU analysis can overlap training, '
               'so it is neither end-to-end elapsed wall time nor measured GPU-active time. '
               'The finite component allowances sum to 152 hours.', '',
               'Every teacher, student and full fake-score/discriminator framework has **61,823,875 learned parameters**; '
               'the shared encoder’s pooled discriminator output adds zero parameters. EMA storage and optimizer states are included '
               'in measured simultaneous memory. The strict allocated-memory limit is 20,000,000,000 bytes. '
               'Device usage is sampled; allocated/reserved and worker host HWM are high-water measurements. '
               'Host statistics cover the training/evaluation worker process; an aggregate host high-water mark '
               'including data-loader and CPU archiver processes was not instrumented.', '',
               'Pixel flags use the predeclared within-class duplication/diversity rule. They do not certify semantic recognition. '
               'Nearest optimization examples are same-class neighbors in 16×16 pixel space, not a perceptual test.', '']
    reviews = read(ROOT / 'resources/gallery_reviews.json', {})
    for key, review in reviews.items():
        report.append(f'- **{key}:** {review["observation"]} [Gallery](../{review["gallery"]})')
    plots = ROOT / 'artifacts/report'
    plots.mkdir(parents=True, exist_ok=True)
    for run, r in records.items():
        r['pipeline_hours'] = r['stage_hours'] + (records['T']['stage_hours'] if run != 'T' else 0)
    plot(records, 'nfe', plots / 'quality_vs_nfe.svg', 'Selected students: quality versus inference calls', 'Exact learned calls per image')
    plot(records, 'pipeline_hours', plots / 'quality_vs_pipeline_cost.svg', 'Selected students: quality versus pipeline cost', 'Teacher + this student stage hours (stand-alone cost)')
    if (plots / 'quality_vs_nfe.svg').exists():
        report += ['', '[Quality versus NFE](../artifacts/report/quality_vs_nfe.svg) · '
                   '[Quality versus stand-alone pipeline cost](../artifacts/report/quality_vs_pipeline_cost.svg). '
                   'Plots use confirmation FID; these are sampling-seed checks, not independent training repetitions.']
    report += ['', 'Raw loss/gradient/timing records and SVG trends, all-class diversity, saturation, same-class neighbor audits '
               'and fixed-noise galleries are retained under `runs/campaign/<run>/`. '
               'Completed student exports are written under `artifacts/exports/`; minimal inference sources are packaged at the milestones.', '',
               'Interpretation is limited to this single teacher and training realization. The treatment adds the full adversarial package '
               '(real/fake D learning and G adversarial loss) while preserving the SiD coefficient. '
               'This experiment cannot establish that teacher supervision prevents GAN collapse without an adversarial-only control. '
               'External reported scores use incompletely aligned settings and are contextual, not controlled comparisons.', '',
               'Protocol: [frozen recipe](FROZEN_PROTOCOL.md), [implementation/deviations](IMPLEMENTATION.md), '
               '[source identity](../resources/campaign_source.json).', '']
    (ROOT / 'docs/RESULTS.md').write_text('\n'.join(report))
    atomic_json(ROOT / 'resources/campaign_results.json', {'updated_at': now, 'runs': records,
                'run_stage_hours_teacher_once': stage_hours, 'setup_seconds': setup_seconds,
                'subsequent_checks_seconds': extra_seconds,
                'recorded_pipeline_hours': stage_hours+(setup_seconds+extra_seconds)/3600})
    print('\n'.join(report[:16]))


if __name__ == '__main__':
    main()
