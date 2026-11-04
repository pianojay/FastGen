"""TA-001 frozen direct color transfer experiment; NumPy/Pillow only, CPU only.

CIELAB: D65 2-degree reference white (0.95047, 1, 1.08883); standard
sRGB piecewise EOTF/OETF and RGB-to-XYZ matrix. The inverse is computed
from that same matrix to avoid inverse-coefficient rounding mismatch.
This adapts the pinned community moment idea, not its uint8 Lab or ratio.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import os
import platform
import resource
import sys
import time
import traceback
from datetime import datetime, timezone
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BUNDLE = ROOT / 'transfer_aug/shared/cohort_v1'
DATA = ROOT / 'data/pokemon-generation-one-22k'
ALPHAS = [0., .25, .5, 1.]
LABELS = ['alpha_0', 'alpha_0.25', 'alpha_0.5', 'alpha_1']
PINS = {
    'contract_sha256': '0c57c5f6e23c269957c4bd18287fca6a8b959c3af5594efae19f9b85162ab7b3',
    'common_protocol_sha256': '1be27b900ee25ad95aed707427fb047e2456c122c6a0d2c4cf1feaa02fe51425',
    'cohort_sha256': 'da15bf04c3281a170e18ffc6116a298366d0287356c1ba19c6765027be7b5a6c',
    'bundle_manifest_sha256': '46d27966191a14746aa8ec212509aefb0c46b905de0e674947e9b2e094203bda',
}
MATRIX = np.array([[.4124564, .3575761, .1804375], [.2126729, .7151522, .0721750],
                   [.0193339, .1191920, .9503041]], dtype=np.float64)
MATRIX_INV = np.linalg.inv(MATRIX)
WHITE = np.array([.95047, 1., 1.08883])
DELTA = 6 / 29
START = time.monotonic()
ASSIGNMENT_START = '2026-10-07T11:51:00+00:00'


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def status(stage, **kw):
    save_json(HERE / 'status.json', {'experiment_id': 'TA-001', 'stage': stage,
              'updated_utc': stamp(), 'pid': os.getpid(), 'process_elapsed_seconds': time.monotonic()-START,
              'assignment_start_utc': ASSIGNMENT_START, **kw})


def rgb_to_lab(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)
    linear = np.empty_like(rgb)
    low = rgb <= .04045
    linear[low] = rgb[low] / 12.92
    linear[~low] = ((rgb[~low]+.055)/1.055)**2.4
    xyz = (linear @ MATRIX.T) / WHITE
    f = np.empty_like(xyz)
    low = xyz <= DELTA**3
    f[low] = xyz[low] / (3*DELTA**2) + 4/29
    f[~low] = np.cbrt(xyz[~low])
    return np.stack([116*f[..., 1]-16, 500*(f[..., 0]-f[..., 1]),
                     200*(f[..., 1]-f[..., 2])], axis=-1)


def lab_to_rgb(lab):
    fy = (lab[..., 0]+16)/116
    f = np.stack([fy+lab[..., 1]/500, fy, fy-lab[..., 2]/200], axis=-1)
    xyz = np.where(f > DELTA, f**3, 3*DELTA**2*(f-4/29)) * WHITE
    linear = xyz @ MATRIX_INV.T
    rgb = np.empty_like(linear)
    low = linear <= .0031308  # Includes negative values; no invalid fractional powers.
    rgb[low] = 12.92*linear[low]
    rgb[~low] = 1.055*linear[~low]**(1/2.4)-.055
    return rgb


def moments(lab):
    # Shift before reduction to retain precision for almost-constant channels.
    # Population moments are invariant to this translation (no regularization).
    anchor = lab[0, 0]
    centered = lab-anchor
    return anchor+centered.mean(axis=(0, 1)), centered.std(axis=(0, 1), ddof=0)


def transfer_lab(c, d, alpha):
    mc, sc = moments(c)
    md, sd = moments(d)
    scale = np.ones(3)
    good = sc >= 1e-6
    scale[good] = sd[good] / sc[good]
    target = (c-mc)*scale+md
    return (1-alpha)*c+alpha*target


def transfer(c, d, alpha):
    if alpha == 0:
        return c.copy()  # Exact bypass including dtype; donor not touched.
    return lab_to_rgb(transfer_lab(rgb_to_lab(c), rgb_to_lab(d), alpha))


def exported(raw):
    return np.rint(np.clip(raw, 0, 1)*255).astype(np.uint8)


def png_bytes(arr):
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format='PNG')
    return b.getvalue()


def integrity(label):
    t = time.monotonic()
    status('integrity_'+label)
    manifest = json.loads((BUNDLE/'manifest.json').read_text())
    cohort = json.loads((BUNDLE/'cohort.json').read_text())
    checked = {}
    pinned_paths = {'contract_sha256': HERE/'CONTRACT_v1.md',
                    'common_protocol_sha256': ROOT/'transfer_aug/shared/PROTOCOL_V1.md',
                    'cohort_sha256': BUNDLE/'cohort.json',
                    'bundle_manifest_sha256': BUNDLE/'manifest.json'}
    for name, path in pinned_paths.items():
        actual = sha(path)
        assert actual == PINS[name], (name, actual)
        checked[str(path.relative_to(ROOT))] = actual
    for name, expected in manifest['files'].items():
        actual = sha(BUNDLE/name)
        assert actual == expected, name
        checked[str((BUNDLE/name).relative_to(ROOT))] = actual
    for name, expected in manifest['protected_inputs'].items():
        actual = sha(ROOT/name)
        assert actual == expected, name
        checked[name] = actual
    snapshot = json.loads((BUNDLE/'source_stat_snapshot.json').read_text())
    mismatch = []
    for name, expected in snapshot.items():
        st = (DATA/name).stat()
        if [st.st_size, st.st_mtime_ns] != expected:
            mismatch.append(name)
    assert not mismatch, mismatch[:10]
    selected = {}
    for row in cohort['rows']:
        for arm in ['content', 'intra', 'inter']:
            name = row[arm+'_path']
            expected = row[arm+'_file_sha256']
            if name not in selected:
                selected[name] = sha(DATA/name)
            assert selected[name] == expected, name
    for arm in ['content', 'intra', 'inter']:
        array = np.load(BUNDLE/(arm+'_uint8.npy'), mmap_mode='r')
        assert array.shape == (604, 64, 64, 3) and array.dtype == np.uint8
        for i, row in enumerate(cohort['rows']):
            assert hashlib.sha256(array[i].tobytes()).hexdigest() == row[arm+'_pixel_sha256']
    audit = ROOT/'transfer_aug/audit_sources/color/color_transfer/__init__.py'
    assert sha(audit) == '51ed90fd02e2bfd96790506621dea1ac883733e878b3f298a9a536815122a99a'
    result = {'pass': True, 'timestamp_utc': stamp(), 'seconds': time.monotonic()-t,
              'frozen_and_protected_hashes': checked, 'source_stat_count': len(snapshot),
              'source_stat_mismatches': mismatch, 'selected_A1_unique_file_count': len(selected),
              'selected_A1_source_hashes': selected, 'bundle_pixel_checks': 1812,
              'audit_source_sha256': sha(audit)}
    save_json(HERE/'verification'/('integrity_'+label+'.json'), result)
    return result


def numerical_checks(content, donor):
    status('numerical_checks')
    t = time.monotonic()
    palette = np.array([[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                        [.5, .5, .5], [.04045, .0031308, .9]], dtype=np.float64).reshape(1, 7, 3)
    rterr = float(np.max(np.abs(lab_to_rgb(rgb_to_lab(palette))-palette)))
    assert rterr <= 1e-6
    known = np.array([[0, 0, 0], [100, 0, 0], [53.2408, 80.0925, 67.2032],
                      [87.7347, -86.1827, 83.1793], [32.2970, 79.1875, -107.8602]])
    known_error = float(np.max(np.abs(rgb_to_lab(palette)[0, :5]-known)))
    assert known_error < .001
    max_rt = max_self = max_lsb = max_pre_mean = max_pre_std = 0.
    degenerate_channels = 0
    for i, cu8 in enumerate(content):
        c, d = cu8.astype(np.float64)/255, donor[i].astype(np.float64)/255
        cl, dl = rgb_to_lab(c), rgb_to_lab(d)
        max_rt = max(max_rt, float(np.max(np.abs(lab_to_rgb(cl)-c))))
        mc, sc = moments(cl)
        md, sd = moments(dl)
        degenerate_channels += int(np.sum(sc < 1e-6))
        assert np.array_equal(transfer(cu8, donor[i], 0), cu8)
        assert np.array_equal(transfer(c, d, 0), c)
        for a in ALPHAS[1:]:
            self_out = transfer(c, c, a)
            max_self = max(max_self, float(np.max(np.abs(self_out-c))))
            max_lsb = max(max_lsb, float(np.max(np.abs(exported(self_out).astype(int)-cu8))))
            lab = transfer_lab(cl, dl, a)
            em = (1-a)*mc+a*md
            es = (1-a)*sc+a*np.where(sc >= 1e-6, sd, sc)
            m, s = moments(lab)
            max_pre_mean = max(max_pre_mean, float(np.max(np.abs(m-em))))
            max_pre_std = max(max_pre_std, float(np.max(np.abs(s-es))))
        if i % 100 == 0:
            status('numerical_checks', processed_sources=i+1)
    assert max_rt <= 1e-6 and max_self <= 1e-6 and max_lsb <= 1
    assert max_pre_mean <= 1e-8 and max_pre_std <= 1e-8
    fixtures = []
    grid = np.linspace(-1, 1, 64*64).reshape(64, 64, 1)
    lab_donor = np.broadcast_to(np.array([60., -20., 30.]), (64, 64, 3)).copy()
    lab_donor += grid*np.array([15., 10., 12.])
    for name, amplitude in [('constant', 0.), ('near_constant_below_threshold', 1e-8),
                             ('near_constant_above_threshold', 1e-5)]:
        c = np.broadcast_to(np.array([50., 5., -4.]), (64, 64, 3)).copy()
        c += grid*amplitude
        mc, sc = moments(c)
        md, sd = moments(lab_donor)
        for a in ALPHAS:
            out = transfer_lab(c, lab_donor, a)
            m, s = moments(out)
            expected_std = (1-a)*sc+a*np.where(sc >= 1e-6, sd, sc)
            me = float(np.max(np.abs(m-((1-a)*mc+a*md))))
            se = float(np.max(np.abs(s-expected_std)))
            raw = lab_to_rgb(out)
            assert me <= 1e-8 and se <= 1e-8 and np.isfinite(raw).all()
            fixtures.append({'fixture': name, 'alpha': a, 'source_std': sc.tolist(),
                             'fallback_channels': (sc < 1e-6).tolist(), 'mean_error': me,
                             'std_error': se, 'finite': True})
    # Constant and near-constant RGB fixtures exercise both conversions and transfer.
    rgbfixtures = []
    for name, amplitude in [('constant_rgb', 0.), ('near_constant_rgb', 1e-10)]:
        c = np.full((64,64,3), .4)+grid*amplitude
        d = np.broadcast_to(np.array([.8,.3,.6]), c.shape).copy()
        for a in ALPHAS:
            o = transfer(c,d,a)
            assert np.isfinite(o).all()
            rgbfixtures.append({'fixture': name, 'alpha': a, 'finite': True,
                                'source_Lab_std': moments(rgb_to_lab(c))[1].tolist()})
    report = {'pass': True, 'seconds': time.monotonic()-t, 'roundtrip_palette_max_abs_rgb': rterr,
              'known_Lab_anchor_max_error': known_error, 'roundtrip_cohort_max_abs_rgb': max_rt,
              'self_donor_count': 1812, 'self_donor_max_abs_rgb': max_self,
              'self_donor_max_exported_LSB': max_lsb, 'alpha0_uint8_and_float_exact_count': 604,
              'pre_inverse_max_mean_error': max_pre_mean, 'pre_inverse_max_std_error': max_pre_std,
              'cohort_degenerate_Lab_channels': degenerate_channels,
              'Lab_fixtures': fixtures, 'RGB_fixtures': rgbfixtures,
              'thresholds': {'roundtrip_rgb': 1e-6, 'self_donor_rgb': 1e-6,
                             'self_donor_LSB': 1, 'Lab_moments': 1e-8}}
    save_json(HERE/'verification/numerical_checks.json', report)
    return report


def edge(x):
    return .5*(np.mean(np.abs(x[1:]-x[:-1]))+np.mean(np.abs(x[:,1:]-x[:,:-1])))


def boundary(x):
    return .5*(np.mean(np.abs(x[0]-x[-1]))+np.mean(np.abs(x[:,0]-x[:,-1])))


FREQ = np.fft.fftfreq(64)*64
HF = np.maximum(np.abs(FREQ[:,None]), np.abs(FREQ[None,:])) >= 16


def hf(x):
    e = np.abs(np.fft.fft2(x, axes=(0,1)))**2
    e[0,0] = 0
    total = e.sum()
    return float(e[HF].sum()/total) if total > 1e-20 else 0.


def measurement(c, raw, prelab, cl, dl):
    displayed = np.clip(raw,0,1)
    mc, sc = moments(cl)
    md, sd = moments(dl)
    mp, sp = moments(prelab)
    mpost, spost = moments(rgb_to_lab(displayed))
    return {'rms_change': float(np.sqrt(np.mean((displayed-c)**2))),
            'clipped_fraction': float(np.mean((raw<0)|(raw>1))),
            'max_excursion': float(max(0., -raw.min(), raw.max()-1)),
            'rgb_mean_shift_l2': float(np.linalg.norm(displayed.mean(axis=(0,1))-c.mean(axis=(0,1)))),
            'edge_energy_delta': float(edge(displayed)-edge(c)),
            'boundary_jump_delta': float(boundary(displayed)-boundary(c)),
            'hf_fraction_delta': float(hf(displayed)-hf(c)),
            'pre_Lab_mean_to_donor_l2': float(np.linalg.norm(mp-md)),
            'pre_Lab_std_to_donor_l2': float(np.linalg.norm(sp-sd)),
            'postclip_Lab_mean_to_donor_l2': float(np.linalg.norm(mpost-md)),
            'postclip_Lab_std_to_donor_l2': float(np.linalg.norm(spost-sd)),
            'gamut_Lab_mean_error_l2': float(np.linalg.norm(mpost-mp)),
            'gamut_Lab_std_error_l2': float(np.linalg.norm(spost-sp))}


def output_row(i, row, a, c, d):
    cl, dl = rgb_to_lab(c), rgb_to_lab(d)
    pre = cl if a == 0 else transfer_lab(cl,dl,a)
    raw = c.copy() if a == 0 else lab_to_rgb(pre)
    assert np.isfinite(raw).all()
    m = measurement(c,raw,pre,cl,dl)
    mc, sc = moments(cl)
    md, sd = moments(dl)
    mp, sp = moments(pre)
    a_expected_mean = (1-a)*mc+a*md
    a_expected_std = (1-a)*sc+a*np.where(sc>=1e-6,sd,sc)
    m.update({'pre_Lab_expected_mean_max_abs_error': float(np.max(np.abs(mp-a_expected_mean))),
              'pre_Lab_expected_std_max_abs_error': float(np.max(np.abs(sp-a_expected_std)))})
    return raw, {'sample_id': row['sample_id'], 'category': row['category'], 'donor_arm': 'intra',
                 'alpha': a, 'condition': LABELS[ALPHAS.index(a)], 'cohort_index': i, **m}


def csv_bytes(rows):
    s = io.StringIO(newline='')
    w = csv.DictWriter(s, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    return s.getvalue().encode()


def aggregate(rows):
    keys = [k for k,v in rows[0].items() if isinstance(v,float) and k!='alpha']
    result = []
    for a in ALPHAS:
        rs = [r for r in rows if r['alpha'] == a]
        result.append({'alpha': a, 'donor_arm': 'intra', 'count': len(rs),
                       'metrics': {k: {'median': float(np.median([r[k] for r in rs])),
                                       'p95': float(np.quantile([r[k] for r in rs],.95)),
                                       'max': float(np.max([r[k] for r in rs]))} for k in keys}})
    return result


def sheet(row, content, donor, raw_array, index, group, selected=None):
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    width, height = 1344, 370
    canvas = Image.new('RGB',(width,height),'#efefef')
    draw = ImageDraw.Draw(canvas)
    title = f"TA-001 | {group} | {row['sample_id']} {row['category']} | intra-class donor"
    if selected is not None:
        title += f' | selected alpha={selected:g}'
    draw.text((12,8),title,font=font,fill='black')
    images = [content[index],donor[index]]+[exported(raw_array[j,index]) for j in [1,2,3]]
    labels = ['Content / alpha=0','Intra donor','alpha=0.25','alpha=0.5','alpha=1.0']
    for col,(arr,label) in enumerate(zip(images,labels)):
        x = 12+col*267
        canvas.paste(Image.fromarray(arr).resize((256,256),Image.Resampling.NEAREST),(x,68))
        draw.text((x,42),label,font=font,fill='black')
    draw.text((12,338),'Actual 64x64 RGB; nearest-neighbor 4x display; strength interpolates in CIELAB; uint8 exports shown.',font=small,fill='black')
    return png_bytes(np.asarray(canvas))


def generate_or_recompute(mode):
    for d in ['verification','arrays','png','sheets/representative','sheets/tail','logs']:
        (HERE/d).mkdir(parents=True,exist_ok=True)
    times = {}
    integrity('before' if mode=='generate' else 'recompute_before')
    cohort = json.loads((BUNDLE/'cohort.json').read_text())
    content = np.load(BUNDLE/'content_uint8.npy',mmap_mode='r')
    donor = np.load(BUNDLE/'intra_uint8.npy',mmap_mode='r')
    if mode=='generate':
        numerical_checks(content,donor)
        save_json(HERE/'environment.json',{'python': sys.version, 'numpy': np.__version__, 'pillow': PIL.__version__,
                  'platform': platform.platform(), 'executable': sys.executable,
                  'threads': {k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},
                  'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
                  'device': 'cpu','learned_parameters': 0,'models_loaded': False,'training_applicable': False})
        save_json(HERE/'config.json',{'experiment_id':'TA-001','contract_revision':1,**PINS,
                  'alphas':ALPHAS,'donor_arm':'intra','std_ddof':0,'std_fallback_threshold':1e-6,
                  'dtype':'float64','matrix_RGB_to_XYZ':MATRIX.tolist(),'white_D65':WHITE.tolist(),
                  'inverse_matrix':'np.linalg.inv of forward matrix','display':'np.rint(np.clip(raw,0,1)*255).astype(uint8)',
                  'assignment_start_utc':ASSIGNMENT_START,
                  'source_adaptation':'Donor/content std ratio; true float CIELAB; mean-only fallback; no Lab clipping; no OpenCV.'})
    path = HERE/'arrays/raw_rgb.npy'
    if mode=='generate':
        assert not path.exists(), 'Do not overwrite previous successful outputs.'
        raw_array = np.lib.format.open_memmap(path,mode='w+',dtype='float64',shape=(4,604,64,64,3))
    else:
        raw_array = np.load(path,mmap_mode='r')
    rows = []
    t = time.monotonic()
    compared_arrays = compared_pngs = 0
    for j,a in enumerate(ALPHAS):
        (HERE/'png'/LABELS[j]).mkdir(exist_ok=True)
        for i,row in enumerate(cohort['rows']):
            c,d = content[i].astype(np.float64)/255,donor[i].astype(np.float64)/255
            raw, measurement_row = output_row(i,row,a,c,d)
            rows.append(measurement_row)
            pp = HERE/'png'/LABELS[j]/(row['sample_id']+'.png')
            pb = png_bytes(exported(raw))
            if mode=='generate':
                raw_array[j,i] = raw
                pp.write_bytes(pb)
            else:
                assert np.array_equal(raw_array[j,i],raw), (j,i,'raw')
                assert pp.read_bytes() == pb, (j,i,'png')
                compared_arrays += 1
                compared_pngs += 1
            if i%50 == 0:
                status(mode, alpha=a, processed_outputs=j*604+i+1,total_outputs=2416)
    times['generation_metrics_png_seconds' if mode=='generate' else 'array_png_metric_recompute_seconds'] = time.monotonic()-t
    metrics = csv_bytes(rows)
    conditions = aggregate(rows)
    tails = []
    seen = set()
    ranked = sorted((r for r in rows if r['alpha']>0),
                    key=lambda r:(-r['clipped_fraction'],-r['rms_change'],r['sample_id'],r['alpha']))
    for r in ranked:
        if r['sample_id'] not in seen:
            tails.append(r)
            seen.add(r['sample_id'])
        if len(tails)==8:
            break
    if mode=='generate':
        raw_array.flush()
        (HERE/'per_output_metrics.csv').write_bytes(metrics)
        save_json(HERE/'condition_metrics.json',conditions)
        save_json(HERE/'tail_selection.json',{'rule':'descending clipped_fraction, descending rms_change, ascending sample_id/alpha; distinct content IDs','rows':tails})
        save_json(HERE/'arrays/layout.json',{'path':'arrays/raw_rgb.npy','shape':list(raw_array.shape),'dtype':str(raw_array.dtype),
                  'axes':['condition','cohort_row','y','x','RGB_channel'], 'condition_order':ALPHAS,
                  'cohort_order':'../../shared/cohort_v1/cohort.json rows', 'raw_unclipped':True})
    else:
        assert (HERE/'per_output_metrics.csv').read_bytes() == metrics
        assert json.loads((HERE/'condition_metrics.json').read_text()) == conditions
        assert json.loads((HERE/'tail_selection.json').read_text())['rows'] == tails
    sheet_paths = []
    for group, cases in [('representative',[(sid,None) for sid in cohort['representative_review_ids']]),
                         ('tail',[(r['sample_id'],r['alpha']) for r in tails])]:
        for sid, selected in cases:
            i = next(i for i,r in enumerate(cohort['rows']) if r['sample_id']==sid)
            p = HERE/'sheets'/group/(sid+'.png')
            data = sheet(cohort['rows'][i],content,donor,raw_array,i,group,selected)
            if mode=='generate':
                p.write_bytes(data)
            else:
                assert p.read_bytes() == data, str(p)
            sheet_paths.append(str(p.relative_to(HERE)))
    gallery = '<!doctype html><html><head><meta charset="utf-8"><title>TA-001</title></head><body><h1>TA-001: intra-class CIELAB color transfer</h1><p>Frozen A1 cohort; 64-pixel originals; unblinded AI visual review. No training or accuracy claim.</p>'
    for p in sheet_paths:
        gallery += f'<p><a href="{p}">{p}</a></p><img src="{p}" style="max-width:100%" alt="{p}">'
    gallery += '</body></html>\n'
    if mode=='generate':
        (HERE/'gallery.html').write_text(gallery)
    else:
        assert (HERE/'gallery.html').read_text() == gallery
    integrity('after_generation' if mode=='generate' else 'after')
    resource_report = {'device':'cpu','parameters':0,'gpu_seconds':0,'gpu_memory_bytes':0,
                        'peak_host_memory_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                        'process_elapsed_seconds':time.monotonic()-START,'stage_timings':times,
                        'cpu_user_seconds':resource.getrusage(resource.RUSAGE_SELF).ru_utime,
                        'cpu_system_seconds':resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                        'completed_utc':stamp(),'pid':os.getpid()}
    save_json(HERE/'verification'/('resources_'+mode+'.json'),resource_report)
    if mode=='recompute':
        save_json(HERE/'verification/determinism.json',{'pass':True,'raw_output_arrays_exact':compared_arrays,
                  'display_png_bytes_exact':compared_pngs,'metrics_csv_bytes_exact':True,'condition_metrics_exact':True,
                  'tail_selection_exact':True,'contact_sheet_png_bytes_exact':len(sheet_paths),
                  'gallery_html_exact':True,'successful_outputs_rewritten':False,'timestamp_utc':stamp()})
    status('awaiting_visual_review' if mode=='generate' else 'recomputed',completed_outputs=2416)
    print(json.dumps({'mode':mode,'pass':True,'outputs':2416,'tails':[r['sample_id'] for r in tails],
                       'resources':resource_report},indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode',choices=['generate','recompute'])
    args = ap.parse_args()
    try:
        generate_or_recompute(args.mode)
    except Exception:
        fail = {'timestamp_utc':stamp(),'mode':args.mode,'traceback':traceback.format_exc()}
        (HERE/'verification').mkdir(exist_ok=True)
        save_json(HERE/'verification'/('failed_attempt_'+str(time.time_ns())+'.json'),fail)
        status('failed',error=fail['traceback'])
        raise
