"""Independent proof of report-only clipped-float definition correction."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;B=R/'report_corrections/clipped_float_v1'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
proof={}
for name,exclude in [('per_output.json',{'clipped_rgb_rms'}),('source_aggregates.json',{'clipped_rgb_rms','diversity_clipped_rgb_rms'})]:
 rows=json.loads((R/'eda'/name).read_text());h=hashlib.sha256(json.dumps([{k:v for k,v in x.items() if k not in exclude} for x in rows],sort_keys=True).encode()).hexdigest();assert h==(B/(name+'.unchanged_fields.sha256')).read_text().strip();proof[name+'_unaffected_fields_exact']=True
outputs=json.loads((R/'eda/per_output.json').read_text());sources=json.loads((R/'eda/source_aggregates.json').read_text())
proof['max_raw_minus_clipped_source_rms']=max(abs(x['rgb_rms']-x['clipped_rgb_rms']) for x in outputs)
proof['max_raw_minus_clipped_diversity']=max(abs(x['diversity_rgb_rms']-x['diversity_clipped_rgb_rms']) for x in sources)
proof['max_clipping_mean_absolute']=max(x['clipping_mean_absolute'] for x in outputs)
proof['max_outside_range_fraction']=max(x['outside_range_fraction'] for x in outputs)
assert all(proof[k]==0 for k in ['max_raw_minus_clipped_source_rms','max_raw_minus_clipped_diversity','max_clipping_mean_absolute','max_outside_range_fraction'])
old=json.loads((B/'tail_selection.json').read_text());new=json.loads((R/'eda/tail_selection.json').read_text());assert [x['source_eval_index'] for x in old]==[x['source_eval_index'] for x in new]
proof['tail_source_selection_unchanged']=True;proof['tail_indices']=[x['source_eval_index'] for x in new]
prev=json.loads((B/'output_hashes.json').read_text());verified=[]
for path,expected in prev.items():
 if path.startswith(('outputs/','checkpoints/')) or (path.startswith('eda/') and path.endswith('.png')):
  assert digest(R/path)==expected,path;verified.append(path)
proof['original_output_checkpoint_and_sheet_files_verified_unchanged']=len(verified)
review=json.loads((R/'visual_review.json').read_text());assert all(digest(Path(x['path']))==x['sha256'] for x in review['sheets']);proof['all_22_previously_reviewed_sheet_hashes_unchanged']=True
proof['definition']='Clipped RGB means float64 np.clip((raw+1)/2,0,1), no uint8 rounding. Descriptor/PCA inputs remain quantized PNG.'
proof['status']='passed';(B/'verification.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2))
