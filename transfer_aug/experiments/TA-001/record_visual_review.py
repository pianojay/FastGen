"""Transcribe actual tool-assisted, unblinded AI visual observations; no metric-derived labels."""
import csv
import io
import json
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256

HERE = Path(__file__).resolve().parent
cohort = json.loads((HERE/'../../shared/cohort_v1/cohort.json').read_text())
lookup = {r['sample_id']:r for r in cohort['rows']}

# Each row is identity, anatomy/markings, background contamination, artifacts,
# observation; observations were made after opening every listed actual sheet.
R = 'retained'; U = 'uncertain'; D = 'disrupted'; N = 'none'; V = 'visible'
representatives = {
 '010_0': [
  (R,R,V,N,'Wing outlines and red eyes remain clear; a faint cyan cast reaches the formerly white background.'),
  (R,R,V,N,'Wings and eye shapes stay recognizable; cyan background and cooler body colors are stronger.'),
  (R,R,V,N,'Wing pattern and body silhouette remain; red eye colors are muted toward tan and the background becomes pale cyan.')],
 '016_0': [
  (R,R,V,N,'Ears, face and belly outline remain; background checkerboard and body are slightly darker.'),
  (R,R,V,N,'Facial marks and outlines remain in place; the whole image gains a darker warm-grey cast.'),
  (R,R,V,N,'The same silhouette and facial marks remain visible despite substantial darkening of subject and checkerboard.')],
 '017_0': [
  (R,R,V,N,'Face, curled forehead marking and rainbow item remain; pale pink spreads into the background.'),
  (R,R,V,N,'The outlined facial marks and accessory remain; stronger pink affects subject and background.'),
  (R,R,V,N,'Recognizable face, ear outline and accessory remain; broad pink tint and darker outlines dominate.')],
 '019_0': [
  (R,R,V,N,'Skull shape and eye opening remain; the photographed setting is mildly brighter.'),
  (R,R,V,N,'Skull/eye contours remain; greens and brown surroundings become lighter.'),
  (R,U,V,V,'The skull stays recognizable but highlights flatten to white and skull-surface shading is difficult to recover.')],
 '027_0': [
  (R,R,V,N,'Curved body and head remain traceable; both the card scene and subject become paler.'),
  (R,R,V,U,'The body curve and head remain recognizable, but pastel wash reduces head/background contrast.'),
  (U,U,V,V,'A broad pale wash nearly merges the head and body shading into the background; fine features are uncertain.')],
 '029_0': [
  (R,R,V,N,'Three heads and their pale noses remain; card background blues shift toward purple.'),
  (R,R,V,N,'The three faces retain their layout; both card background and creatures acquire a warmer cast.'),
  (R,R,V,U,'All three heads and nose marks remain, though a brown haze lowers overall contrast.')],
 '034_0': [
  (R,U,V,N,'The egg cluster is still identifiable relative to input; original low-resolution card blur limits facial-mark assessment.'),
  (R,U,V,U,'Egg outlines remain but added lavender tint reduces already weak facial contrast.'),
  (U,U,V,V,'Lavender wash covers eggs and card background; pale facial details are difficult to distinguish at this resolution.')],
 '050_0': [
  (R,R,V,N,'Serpentine body, fins and open mouth remain; white background becomes light grey-blue.'),
  (R,R,V,U,'Body and belly segments remain clear; head and scales become markedly more saturated and darker.'),
  (R,U,V,V,'Silhouette is recognizable, but dark blue head regions and saturated belly lose tonal separation; background turns grey-blue.')],
 '056_0': [
  (R,R,V,N,'Eyes and forehead spots remain; green background and red flower become a little brighter.'),
  (R,R,V,N,'Eye shapes and dark forehead marks remain visible as the whole scene brightens.'),
  (R,R,V,U,'Facial marks remain recognizable; very bright yellow-green background and eye highlights make clipping damage uncertain.')],
 '105_0': [
  (R,R,U,N,'Belly spiral, eyes and mouth remain clear; any background change is subtle at this display size.'),
  (R,R,V,N,'Belly spiral and face remain; the surface and surrounding photograph become brighter and cooler.'),
  (R,R,V,U,'Spiral and facial marks remain legible; near-white ground loses some tonal variation.')],
 '110_0': [
  (R,R,V,N,'Raised fist, nose and eyes remain; originally black background becomes dark grey.'),
  (R,U,V,V,'Pose and nose remain, but the pale face approaches white and eye contrast is markedly weakened.'),
  (U,D,V,V,'White flattening removes the distinctive eye marks; the body outline and nose survive against a grey background.')],
 '122_0': [
  (R,R,V,N,'Large eye and fin spots remain; a faint warm cast reaches the pale fin/background regions.'),
  (R,R,V,N,'Eye, dark head patch and fin spots stay clear with stronger orange coloration.'),
  (R,R,V,N,'Characteristic spots and eye remain; white fins/background turn peach while body orange becomes stronger.')],
 '124_0': [
  (R,R,V,N,'Shell plates, eyes and tongue remain; the teal setting darkens.'),
  (R,R,V,U,'Shell and eyes remain legible, though mouth and shell shadows are increasingly flat and dark.'),
  (R,U,V,V,'Eyes and tongue still identify the shape, but very dark shell and mouth areas lose shading; background is strongly dimmed.')],
 '134_0': [
  (R,R,V,N,'Head and two tentacles remain; red head patches intensify and black background becomes bluish-grey.'),
  (R,U,V,V,'Head/legs remain, but bright cyan and magenta flatten much of the head shading.'),
  (R,D,V,V,'The head/tentacle silhouette survives, while large cyan and magenta regions erase much of the original head-surface detail.')],
 '139_0': [
  (U,U,V,N,'The complex card layout remains; original blur prevents confident independent identity or fine-marking judgment.'),
  (U,U,V,U,'Card layout persists but higher chroma emphasizes coarse texture; fine anatomy remains unresolved.'),
  (U,U,V,U,'Strong cyan/magenta coloration affects the whole card; existing low-resolution clutter prevents confident new-artifact attribution.')],
 '146_0': [
  (R,R,U,N,'Eyes, mouth and leaf outlines remain; any background tint change is subtle.'),
  (R,R,V,N,'Eyes and leaf contours remain; yellow body and surrounding green field become more saturated.'),
  (R,U,V,V,'The face and leaf silhouette remain recognizable, but near-black mouth and very bright body lose tonal shading.')],
}
tails = {
 '082_3': (R,R,N,N,'Mewtwo silhouette and purple tail/belly remain almost unchanged; the white background appears unchanged despite the high strict clipping count.'),
 '104_3': (R,R,N,N,'Horns and toothed mouth remain clear; subject is mildly warmer and the white background looks unchanged.'),
 '018_2': (R,R,V,V,'Shell spikes and inner face remain; yellow background turns white and faint pale text near the top is no longer discernible.'),
 '128_1': (R,R,N,U,'Beak, dark neck patch and feather markings remain, but a pale wash weakens contrast across the bird.'),
 '013_1': (R,U,N,V,'Wings and orange body remain recognizable; washed highlights obscure the already small facial details.'),
 '020_0': (U,D,V,V,'The white creature merges into a near-white field; tail/flipper shading and much of the facial/body detail disappear.'),
 '093_0': (R,R,N,N,'Leaf veins, eyes and mouth remain; leaves turn muted olive and the body purple while the white field looks unchanged.'),
 '086_2': (R,U,V,V,'Horn, tail and arms remain, but bright chest/head regions lose shading and the checkerboard background largely disappears.'),
}
rows = []
sheet_records = []
for group, items in [('representative',representatives),('tail',tails)]:
    for sid, judgments in items.items():
        sp = Path('sheets')/group/(sid+'.png')
        sheet_records.append({'path':str(sp),'sample_id':sid,'review_group':group,
                              'tool':'tools.view_image','actual_tool_view_completed':True,
                              'sha256':sha256((HERE/sp).read_bytes()).hexdigest()})
        alphas = [.25,.5,1.] if group=='representative' else [1.]
        if group=='tail': judgments=[judgments]
        for a, (identity,anatomy,bg,artifact,obs) in zip(alphas,judgments):
            rows.append({'sample_id':sid,'category':lookup[sid]['category'],'condition':f'alpha_{a:g}',
                         'alpha':a,'donor_arm':'intra','review_group':group,
                         'recognizable_identity':identity,'anatomy_markings':anatomy,
                         'background_contamination':bg,'artifacts':artifact,'observation':obs,
                         'sheet':str(sp)})
assert len(rows)==56 and len(sheet_records)==24
assert set(representatives)==set(cohort['representative_review_ids'])
assert set(tails)=={r['sample_id'] for r in json.loads((HERE/'tail_selection.json').read_text())['rows']}
s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(HERE/'visual_review.csv').write_text(s.getvalue())
counts=[]
for group in ['representative','tail']:
    for a in ([.25,.5,1.] if group=='representative' else [1.]):
        rs=[r for r in rows if r['review_group']==group and r['alpha']==a]
        counts.append({'review_group':group,'alpha':a,'count':len(rs),
                       **{k:dict(Counter(r[k] for r in rs)) for k in
                          ['recognizable_identity','anatomy_markings','background_contamination','artifacts']}})
report={'reviewer':'TA-001 worker; unblinded AI visual judgment',
        'recorded_utc':datetime.now(timezone.utc).isoformat(),
        'required_sheets':24,'viewed_sheets':24,'required_review_rows':56,'completed_review_rows':56,
        'inspection_order':'Four representative batches of four sheets, then two tail batches of four sheets, each opened individually with tools.view_image.',
        'label_basis':'Direct visual comparison to content at actual 64px nearest-neighbor enlargement. Background contamination means apparent whole-field tint/brightness transfer, not imported donor objects. Artifact labels concern visible flattening, lost detail, or other appearance defects. Uncertainty includes pre-existing source blur and ambiguity between style shift and damage.',
        'limitations':['Unblinded AI judgments, not category accuracy or blinded human evaluation.',
                       'Only 16 fixed representative samples and eight selected diagnostic outputs were reviewed.',
                       'Existing blur and card/screenshot backgrounds limit fine anatomy judgments.',
                       'Tail selection was frozen numerical ranking; labels were not assigned by metrics.'],
        'sheets':sheet_records,'counts':counts}
(HERE/'visual_inspection.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(counts,indent=2))
