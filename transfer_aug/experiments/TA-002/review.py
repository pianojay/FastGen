"""Expand manually authored sheet-specific visual judgments, never metric labels."""
import csv,json,time
from pathlib import Path
H=Path(__file__).resolve().parent
cohort=json.loads((H/'../../shared/cohort_v1/cohort.json').read_text())
manual=json.loads((H/'visual_judgments_manual.json').read_text())
fields=['sample_id','condition','arm','review_group','recognizable_identity','anatomy_markings','background_contamination','artifacts','observation','sheet','inspection_tool','judgment_basis']
rows=[]
codes={'R':'retained','D':'disrupted','U':'uncertain','N':'none','V':'visible'}
for entry in manual['representative']:
 assert len(entry['flags'])==12 and len(entry['observations'])==4
 for ri,r in enumerate([0,1,2,4]):
  for ai,a in enumerate([.25,.5,1.]):
   code=entry['flags'][ri*3+ai];assert len(code)==4
   observation=entry['observations'][ri][ai] if isinstance(entry['observations'][ri],list) else entry['observations'][ri]
   rows.append(dict(zip(fields,[entry['sample_id'],f'r{r}_a{a:g}',entry['arm'],'representative',codes[code[0]],codes[code[1]],codes[code[2]],codes[code[3]],observation,f"sheets/{entry['sample_id']}_{entry['arm']}.png",'tools.view_image via functions.exec','Unblinded AI visual inspection relative to content; not category accuracy'])))
for entry in manual['tails']:
 code=entry['flags'];rows.append(dict(zip(fields,[entry['sample_id'],entry['condition'],entry['arm'],'tail',codes[code[0]],codes[code[1]],codes[code[2]],codes[code[3]],entry['observation'],f"sheets/tails_{entry['arm']}.png",'tools.view_image via functions.exec','Unblinded AI visual inspection relative to content; selected by clipping rank'])))
assert len(rows)==392
assert len({(x['sample_id'],x['arm'],x['condition']) for x in rows if x['review_group']=='representative'})==384
expected={(sid,arm,f'r{r}_a{a:g}') for sid in cohort['representative_review_ids'] for arm in ['intra','inter'] for r in [0,1,2,4] for a in [.25,.5,1.]}
assert expected=={(x['sample_id'],x['arm'],x['condition']) for x in rows if x['review_group']=='representative'}
expected_tail={(x['sample_id'],x['arm'],x['condition']) for x in json.loads((H/'tail_selection.json').read_text())}
assert expected_tail=={(x['sample_id'],x['arm'],x['condition']) for x in rows if x['review_group']=='tail'}
for x in rows:
 assert (H/x['sheet']).is_file()
 assert x['recognizable_identity'] in ['retained','disrupted','uncertain']
 assert x['anatomy_markings'] in ['retained','disrupted','uncertain']
 assert x['background_contamination'] in ['none','visible','uncertain']
 assert x['artifacts'] in ['none','visible','uncertain']
with (H/'visual_reviews.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
(H/'visual_reviews.json').write_text(json.dumps(rows,indent=2)+'\n')
counts={}
for arm in ['intra','inter']:
 counts[arm]={}
 for group in ['representative','tail']:
  subset=[x for x in rows if x['arm']==arm and x['review_group']==group]
  counts[arm][group]={'count':len(subset),**{k:{v:sum(x[k]==v for x in subset) for v in (['retained','disrupted','uncertain'] if k in ['recognizable_identity','anatomy_markings'] else ['none','visible','uncertain'])} for k in ['recognizable_identity','anatomy_markings','background_contamination','artifacts']}}
(H/'verification/visual_review_checks.json').write_text(json.dumps({'pass':True,'representative_count':384,'tail_count':8,'sheets_actually_inspected':manual['sheets_actually_inspected'],'counts':counts,'manual_source':'visual_judgments_manual.json','uncertainty_policy':'Judgments are relative to visible input at 64px; no classifier, class correctness, training utility or FID inferred.'},indent=2)+'\n')
print(json.dumps(counts,indent=2))
