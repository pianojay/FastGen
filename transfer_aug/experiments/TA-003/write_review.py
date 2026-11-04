"""Persist direct unblinded AI inspection notes from all nine displayed sheets."""
import json
from pathlib import Path
from runner import ROOT,write,utc
reviews=json.loads((ROOT/'outputs/review_manifest.json').read_text())['records']
sheets=json.loads((ROOT/'eda/sheet_manifest.json').read_text())
notes=[
'Butterfree wing layout and antennae remain; darker wing divisions and body shading change fine markings.',
'Butterfree frontal pose remains, with much darker body and thicker wing divisions; fine facial detail is uncertain.',
'Clefable crop, ears and curled outline remain; red-brown contours and shading strengthen.',
'Clefable plush shape remains, with stronger orange-red shading and darkened small facial details.',
'Clefairy scene layout remains; pink body shifts toward cream and foreground/background contrast increases.',
'Clefairy photographed figure and street scene remain recognizable in layout; shadows and background become darker.',
'Cubone card-like scene retains skull and eye placement; orange/brown saturation and dark outlines increase.',
'Cubone street pose remains; body and street shadows darken, with small skull/face details uncertain.',
'Dratini outline is visible but broad dark colored bars and speckled edges contaminate the previously plain background, especially late.',
'Dratini curved body remains; orange background becomes strongly red and body contour more pronounced.',
'Dugtrio three-part silhouette remains but brown forms become blue-black with bright highlights; colors/markings are substantially altered.',
'Dugtrio grouped forms and scene remain; darker colored edges and mottled background texture are added.',
'Exeggcute grouped faces remain arranged similarly; facial outlines, expression marks and pink shading strengthen.',
'Exeggcute cluster remains broadly recognizable but fine eyes/face outlines and several contour junctions change amid colored grain.',
'Gyarados upright silhouette remains; strong dark and blue outlines replace softer source edges.',
'Gyarados side pose remains; neck/belly and contours become markedly darker and more contrasty.',
'Ivysaur bud, leaf and body layout remains, with heavy dark leaf outlines and grained background/body shading.',
'Ivysaur head and bud remain; eyes and face are still visible, but blue shading and dark leaves strengthen.',
'Poliwag illustrative scene remains arranged similarly, with darker accents and slightly changed facial shading.',
'Faceted rear-view Poliwag form and tail remain, but illumination/color and background change strongly; class identity is uncertain from this view.',
'Primeape frontal pose remains; darker lower outline and stronger facial lines are visible.',
'Primeape model pose remains; background and floor become darker and body highlights/shadows strengthen.',
'Seaking is still locatable but a pale patterned background becomes harsh white/dark texture with border artifacts; small markings are uncertain.',
'Seaking toy-like fish shape and scene remain; orange shading and background contrast strengthen.',
'Shellder circular shell and central opening remain, with purple shifting toward warmer brown and increased highlights.',
'Shellder close-up contours remain largely aligned; the mouth interior becomes darker.',
'Tentacool frontal head and tentacle arrangement remain; shadows, contrast and background bands strengthen.',
'Tentacool plain white background becomes black with a bright surrounding halo; body/tentacle edges and fine anatomy are altered or uncertain.',
'Venusaur body/flower layout remains; blue body shading and shadow contrasts strengthen.',
'Venusaur flower and open-mouth pose remain; greens, reds and mouth/face shading change.',
'Weepinbell body, eyes and mouth remain recognizable, but background gains dense grain and the mouth rim/interior darkens.',
'Weepinbell form remains similar to the source; dense colored background texture and stronger mouth/eye outlines are introduced.',
'Butterfree wings remain locatable but black background changes to bright white/blue haze, with washed-out markings and colored halos.',
'Butterfree frontal silhouette persists but black background becomes almost white and wing/body outlines are strongly faded.',
'Clefable broad model pose remains; background lightens and the body becomes very pale, reducing facial contrast.',
'Clefable is reduced to a very pale silhouette; face and body outlines are difficult to inspect after black-to-white background change.',
'Clefairy outline and mouth remain but black background becomes pale and several body markings/edges wash out.',
'Clefairy pose remains; background replacement and bright body wash reduce fine facial and limb detail.',
'Cubone scene layout remains; image is softened and less contrasty, with small skull details uncertain.',
'Cubone pixel-art outline and pose remain; black contours lighten and body shading becomes paler.',
'Dratini composition remains; blue scene/body colors shift toward pale mint with reduced contrast.',
'Dratini curved pose remains on white; fine dark contours lighten and blues soften.',
'Dugtrio three heads and colorful scene remain; texture and shading soften with mild color shifts.',
'Dugtrio three-mole arrangement remains; brown bodies and facial contrast become paler against a hazy background.',
'Exeggcute scene retains its broad arrangement but individual small faces blur; fine identity/anatomy judgments are uncertain.',
'Exeggcute black background becomes pale, with upper cluster contours and facial lines changed; preservation of all small head details is uncertain.',
'Gyarados head/open mouth remains; teeth and small outlines soften considerably in the busy scene.',
'Gyarados curved red silhouette remains on a mostly dark background, with softened edges and subtle colored haze.',
'Ivysaur face/bud and scene remain; scene contrast softens while some green/cyan colors change.',
'Ivysaur model shape remains; dark body/leaf shading becomes a pale washed texture.',
'Poliwag frontal shape and spiral remain visible, though the spiral and body shading lose contrast.',
'Poliwag raised-tail scene remains; purple background and spiral outlines are paler and softer.',
'Primeape pose remains; body, face and outlines become pale, and background shading changes.',
'Primeape black background becomes cream; body/facial details wash out and colored edge halos appear.',
'Seaking pose and illustration layout remain; contrast and fine details soften.',
'Seaking drawn fish outline remains broadly aligned, but orange/black markings and fin outlines become paler.',
'Shellder model and tongue remain recognizable; colors and shadow transitions soften.',
'Shellder face/tongue composition remains; purples and line contrast soften without obvious gross displacement.',
'Tentacool slender form remains, but black background acquires a dark olive cast and the head/limb edges fade.',
'Tentacool pair stays in place; formerly black background gains blue/green tones and head shading changes.',
'Venusaur circular object stays in place but becomes hazier; the tiny internal character is too small for a confident identity/anatomy judgment.',
'Venusaur card layout remains broadly aligned, but text and fine central character details blur; precise identity/marking preservation is uncertain.',
'Weepinbell pose and mouth remain; illustration is softened and yellows/greens shift toward muted colors.',
'Weepinbell dark-background form remains; edges soften and dark colors/background hues shift slightly.'
]
assert len(notes)==len(reviews)==64
summaries={
('D0_D1',1):'Mostly retains coarse pose/layout while strengthening contours, saturation and shading; small facial details may change.',
('D0_D1',2):'Added edge/texture contrast includes severe Dratini background bars and large Dugtrio color changes; fine facial details can be altered.',
('D0_D1',3):'Coarse scene arrangements mostly remain, but stronger shading/grain and Seaking background artifacts prevent treating the shift as pure style transfer.',
('D0_D1',4):'Some forms remain clear, but Tentacool white-to-black background replacement and Weepinbell graining are clear failures of the background/detail invariants.',
('D1_D0',1):'Several black backgrounds become pale/white and figures are washed out; reduced contrast erases fine outlines and facial/body details.',
('D1_D0',2):'Many broad forms remain with softer shading; some small faces/cluster contours blur and Exeggcute background replacement is substantial.',
('D1_D0',3):'Many poses and distinctive coarse markings remain, but fine lines fade and one Primeape background becomes bright with severe washout.',
('D1_D0',4):'Many coarse forms remain; background casts, fading, and blurred tiny card/figurine details make some identity judgments uncertain.'}
records=[]
for s in sheets:
    entry=dict(s,directly_inspected=True,inspected_utc=utc())
    if s['kind']=='representative':
        page=int(Path(s['sheet']).stem.rsplit('_',1)[1]);entry['finding']=summaries[s['direction'],page]
        entry['rows']=[dict(review_index=i,path=reviews[i]['path'],category=reviews[i]['category'],note=notes[i],coarse_identity=('uncertain at this resolution/view' if i in (19,60,61) else 'broad source form remains recognizable; class accuracy not evaluated'),anatomy_markings='see row note; fine details at 64px can be uncertain',background_and_artifacts='see row note') for i in s['review_indices']]
        entry['temporal_observation']='Initial outputs are noise-like; later outputs acquire source form. Some artifacts persist or worsen at later milestones, so no monotonic quality or convergence claim.'
    else:
        tails=json.loads((ROOT/'eda/tail_selection.json').read_text());tail_notes=[
            'Persian: tiny source becomes surrounded by a large yellow-green halo on black; anatomy/markings and identity are uncertain in the altered region.',
            'Wigglytuff: black background becomes white and pale body blends into it; eyes/ears remain locatable but other detail is lost.',
            'Drowzee: background becomes black with a broad yellow halo; body remains locatable but face/outline detail is altered.',
            'Exeggcute: black becomes pale and outlines/face details change; overlaps one representative example.',
            'Machoke: white becomes black with cyan halo and faint/doubled-looking edges; limb and torso details are uncertain.',
            'Meowth: white becomes black with yellow outline/halo and strong internal lines; broad pose remains but fine markings change.',
            'Mewtwo: white becomes black, with heavy colored contours and changed internal shading; broad silhouette is visible, exact anatomy/detail uncertain.',
            'Marowak: black becomes pale and the skull/bone contours partly wash out; anatomy/detail preservation is uncertain.'
        ]
        entry['rows']=[dict(path=r['path'],category=r['category'],direction=r['direction'],rgb_rms=r['rgb_rms'],note=note) for r,note in zip(tails,tail_notes)]
        entry['finding']='All eight selected large-change tails show substantial background replacement/inversion and loss or corruption of details. Cycle and identity controls often resemble sources despite poor translated outputs; these controls do not establish preservation in the translation.'
    records.append(entry)
write(ROOT/'visual_review.json',dict(method='Small unblinded AI visual review of displayed contact sheets, without independent raters or a semantic classifier.',
    scope=dict(representative_sources=64,representative_categories=16,representative_sheets=8,tail_entries=8,tail_sheets=1,tail_representative_overlap=1,unique_review_sources=71,milestone_columns=[0,1000,5000,10000,20000]),
    rubric=['recognizable identity relative to source','anatomy and markings','background contamination','visible artifacts'],
    uncertainty='No class-accuracy estimates. Fine 64px details and obscured/atypical source views can be uncertain. Selected tails are not representative incidence estimates.',
    conclusion='Both directions shift the fixed appearance axis but do not reliably isolate appearance from background and fine-content changes. Completion at 20000 updates is not a convergence claim.',sheets=records))
print('Recorded direct review of',len(records),'sheets and64 representative +8 tail entries.')
