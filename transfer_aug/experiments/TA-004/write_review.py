"""Persist observations from direct inspection of all nine generated sheets."""
import json
from pathlib import Path
R=Path(__file__).resolve().parent
notes={
'representative_D0_1.png':[
'Wing/body layout remains source-like; darker doubled wing lines and speckle introduce uncertain fine markings.',
'Wing silhouette and central body remain; saturation and dark outline increase, with softened fine details.',
'Large source outlines remain but red/orange edging and foreground-background color bleeding alter markings.',
'Pose remains visible; stronger orange highlights and dark edging change the face and body shading.',
'Source figure layout remains; background becomes much darker/stronger green with thick black outlines.',
'Photographic toy scene remains broadly recognizable; contrast darkening and texture changes affect facial detail.',
'Card/object layout remains; background and skull edges strongly darken, small text/detail is uncertain.',
'Toy and street layout remain; strong shadow darkening and rough edge texture alter appearance.'],
'representative_D0_2.png':[
'Pale background changes to black. Serpentine outline remains, but bright halo, stronger color and altered face detail reduce fidelity.',
'Serpentine pose remains; pale card background becomes strongly saturated orange/red with thickened edges.',
'Three-head layout remains approximately visible, but gray bodies become glossy black/blue and facial markings are uncertain.',
'Group shape remains; yellow/black contrast intensifies and fine face/body detail is uncertain.',
'Egg-group placement remains; facial outlines thicken and surface/background texture appears.',
'Checker/pale background changes to black; egg group remains with strong halos and changed facial detail.',
'White background changes to black; serpentine silhouette remains but thin appendages and markings are uncertain.',
'Pose remains but blue background becomes pale and body darkens; complex small anatomy is uncertain.'],
'representative_D0_3.png':[
'Body/flower layout remains; deep leaf shadows, heavier outlines and saturated bud alter markings.',
'Body/leaf silhouette remains; leaf turns near-black and skin darkens; facial/skin details change.',
'Round creature/card layout remains, with heavy outlines and textured/color-shifted surroundings; fine anatomy uncertain.',
'Round body/tail layout remains; gray background becomes dark brown and smooth planes gain bright texture.',
'Raised-arm layout remains; thick dark edges and facial shading alter fine details.',
'Pose remains; background shifts darker and body highlights blur, making fine limbs/face uncertain.',
'Fish/card layout remains but pale background gains strong dark specks, edging and increased texture.',
'Fish pose remains; darker contours and rougher surface texture affect detailed markings.'],
'representative_D0_4.png':[
'Round shell layout remains; dark background is preserved/darkened, body color and rim highlights change.',
'Shell contour remains; strong blue/dark contrast and extra texture alter shell markings.',
'Head/tentacle layout remains; tentacles become much darker and head surface gains textured lines.',
'White background changes to black; bright head and thin tentacle outlines remain but solid limb detail is lost/uncertain.',
'Large body/flower structure remains; shading darkens and surface texture alters small markings.',
'Frontal body/flower layout remains; leaf and mouth shading darken substantially and fine lines are uncertain.',
'Open-mouth form remains; green scene gets darker/textured and face/leaf boundaries roughen.',
'Open-mouth form remains; thick mouth edging and textured yellow/green surfaces alter the clean source shading.'],
'representative_D1_1.png':[
'Black background and wing/body layout largely remain; mild color shifts/edge softening alter small details.',
'Wing silhouette and dark frame remain; stronger body coloring and small line changes visible.',
'Pink figure retains pose, but highlights wash out much facial/body detail; exact facial fidelity uncertain.',
'Pose remains with pale highlights and heavier outlines; fine face details soften.',
'Pose and black background remain; brighter body shading and some fine-line distortion visible.',
'Pose and black background remain; heavier shading/outline alters small face and limb details.',
'Card layout remains; slight palette/contrast change and soft small markings.',
'Sprite-like pose remains; orange/bone colors shift with thicker dark edging.'],
'representative_D1_2.png':[
'Serpentine form and blue scene remain; small contour/color changes with fine details uncertain.',
'Curling pose remains; body/background lighten and outlines deepen.',
'Group arrangement remains; pink/yellow tones and edges shift slightly.',
'Three-head outline remains; brown shading and face edges change modestly.',
'Egg/card layout remains; highlights and background tones shift, small markings soften.',
'Egg outline and black background remain; facial lines and shell shading alter.',
'Open-mouth/card scene remains; color and contrast changes, fine anatomical details uncertain.',
'Thin red serpentine silhouette and black background remain; small appendage fidelity uncertain at64px.'],
'representative_D1_3.png':[
'Plant/body composition remains; modest color/outline changes, some small markings altered.',
'Frontal plant/body form remains with slightly changed green shading and outlines.',
'Round body and spiral remain; blue becomes more saturated/purple and spiral line thickness changes.',
'Round pose and spiral remain; small color/contour changes on body and background.',
'Pose/background layout remains; body highlights/face become smoother and somewhat washed out.',
'Raised-arm pose remains; brighter body/face and dark edge changes make fine expression uncertain.',
'Fish pose and spotted fin remain; heavier edge/color treatment modifies spots.',
'Fish outline remains; stronger orange/black contours and subtle fin-line changes.'],
'representative_D1_4.png':[
'Toy shape and scene remain; slight blue/purple and tongue color changes, softer face edges.',
'Shell/tongue drawing remains; outlines thicken and colors shift subtly.',
'Thin tentacles and black scene remain; bright head/line softness makes small anatomy uncertain.',
'Head and tentacle positions remain; neighboring cropped object persists, fine lines soften.',
'Circular object scene remains; tiny central figure is too small for confident anatomy/identity judgment.',
'Card layout persists, but tiny text/figure details are blurred and cannot support confident identity judgment.',
'Plant/face layout remains; modest brightening and dark outlines change fine markings.',
'Leaf/body form and dark background remain; shading/edge changes visible.'],
 'tails.png':[
'White background becomes black; tiny source silhouette persists but facial/ear detail blurs and fidelity is uncertain.',
'White background becomes black; muscular silhouette persists but body/face details become softened and mottled.',
'White background becomes black; body outline remains but face and hand detail wash out.',
'White background becomes black; solid arms become hollow curved bright outlines and face detail disappears: substantial anatomy/marking damage.',
'White background becomes black; upright outline remains with halos and major loss/alteration of facial detail.',
'White background becomes black; tentacles become thin bright traces with lost solid shading and altered head marks.',
'White background becomes black; wings and mouth remain broadly placed but wing edges/texture change and a stray upper mark appears.',
'White background becomes black; reclining outline remains but many stripes disappear and face/limb details become uncertain.']}
inv=json.loads((R/'review_inventory.json').read_text())
obs=[]
for item in inv:
    obs.append({**item,'observation':notes[item['sheet']][item['row']-1],
                'identity_judgment':'Broad source correspondence assessed descriptively; no semantic class verification. Fine-detail uncertainties are noted.',
                'reviewed_columns':'source, initial,1000,5000,10000,final' if item['sheet']!='tails.png' else 'source,final,nearest same-class D1 optimization (pixel neighbor, not semantic match)'})
v={'review_method':'Small unblinded AI visual review by the implementation worker, direct inspection of all nine contact sheets; no human panel or class-accuracy measurement.',
'reviewed_representative_examples':64,'reviewed_representative_translations':32,'reviewed_representative_identity_controls':32,'reviewed_tail_examples':8,
'reviewed_sheets':list(notes),'observations':obs,
'overall':{'representative_translations':'Broad source layouts usually remain visible, with stronger contrast, dark thick contours, altered colors, halos/speckle and uncertain small markings. Four of32 representative translations visibly change pale/white/checker backgrounds to black (Dratini36,Exeggcute47,Gyarados78,Tentacool20).',
'identity_controls':'Generally smaller visual changes than translations, but not exact identities: color/highlight changes, softened or thickened edges and small marking distortion remain. Tiny Venusaur/card details are uncertain.',
'high_change_tails':'All eight deliberately selected largest-RMS translations replace white backgrounds with black. Broad outlines often remain; Geodude loses solid arm/face structure and Arcanine loses stripes. Facial/limb fidelity in several others is uncertain.',
'milestones':'Initial outputs are colored noise. Early updates recover broad source structure; later milestones change palettes/contrast and sometimes switch backgrounds. No monotonic quality or convergence claim follows.',
'interpretation':'The appearance projection shifts strongly toward D1, but background replacement and marking damage violate the desired invariants in visible cases. This is not evidence of safe augmentation or lower generative FID.',
'limitations':'Representatives are32 translation sources/32 descriptive identity controls in16 fixed classes; tails are outcome-selected and may overlap representatives. Small unblinded AI judgments are not population prevalence or class accuracy.'}}
(R/'visual_review.json').write_text(json.dumps(v,indent=2)+'\n')
print({'observations':len(obs),'sheets':len(notes)})
