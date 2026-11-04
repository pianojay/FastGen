"""Persist observations from direct worker inspection of every required sheet."""
import json,datetime
from pathlib import Path
R=Path(__file__).resolve().parent;ev=json.loads((R.parents[1]/'shared/round3/evaluation_manifest.json').read_text())['records']
notes={
20:'Wing outlines remain suggestive, but body/face and wing markings are reconstructed incorrectly; latent palettes become yellow/green and cluttered.',
21:'Butterfly-like silhouette persists, with new circular wing marks and altered body proportions; self-control already changes the toy-like source.',
32:'Reference colors/backgrounds follow donors, while torso/face lines and limbs shift; latent outputs include saturated blue/purple and partially erased forms.',
33:'Rounded body stays recognizable as the source motif, but face, ears and limbs mutate; latent colors/backgrounds vary strongly.',
34:'Curl/limbs and body outline are warped; self-control loses the original sprite structure. Latent variants include near-silhouette loss.',
35:'Reference variants retain pinkish palette but add/reshape limbs and face; latent variants become blue or dark silhouettes.',
38:'Skull-like head motif persists, but skull openings, body texture and setting change; latent colors are not conservative.',
39:'Source pose is roughly retained, but face/mask and protrusions alter. Reference backgrounds are copied in broad appearance; latent output can erase body.',
54:'Small curled shape becomes a bulky folded form; donor colors transfer with substantial face/body deformation.',
55:'Long neck pose remains only approximately; face/outline and backdrop change, with some latent outputs strongly obscured.',
58:'Three-head organization is weakened or merged; reference colors and backgrounds transfer, but faces and rock base are rebuilt.',
59:'Clustered head silhouette persists loosely, with distorted head boundaries and faces; latent variants add aggressive colors and texture.',
68:'Egg group count/boundaries and faces change; latent variants include a dark almost-empty body and unrelated colors.',
69:'Grouped circular layout remains, but facial arrangement and individual eggs merge; donor influence does not preserve markings.',
100:'Serpentine contour remains approximate, but jaw, belly segmentation and fins change; latent outputs become unrelated colors.',
101:'Reference black/blue settings transfer clearly; fish/dragon outline is heavily reshaped, with face and belly damage.',
112:'Plant/bulb and body merge into a distorted radial form even in self-control; latent variants amplify contour distortion.',
113:'Head silhouette partially survives but flower/leaves and face details collapse; references change palette/background and latent colors vary widely.',
210:'Spiral/eye and tail structure are displaced or missing; outputs resemble reconstructed fish shapes more than faithful source renderings.',
211:'Simple low-poly ball becomes a new detailed creature-like form; identity preservation is especially uncertain.',
220:'Raised-arm motif remains, but face and limbs are reconfigured; latent variants are saturated and visibly nonconservative.',
221:'Standing pose roughly survives; reference arms/legs and facial features differ. Latent outputs distort proportions and palette.',
244:'Tiny fish is expanded into a differently shaped object with altered fins/markings; latent variants include silhouettes and clutter.',
245:'Rounded body persists but fins, eye and mouth move; reference palettes/backgrounds transfer while latent variants become unrelated colors.',
248:'Shell opening/face/tongue region is restructured; reference broad purple palette stays, latent colors change drastically.',
249:'Shell-like outline remains but interior/edges are redrawn and tongue information lost; latent outputs can become mostly blank contrasting shapes.',
268:'Head and tentacles change shape, jewels become eye-like patterns; source geometry is altered even with self-style.',
269:'Hanging layout remains but head/eye proportions and tentacles differ; latent colors/backgrounds alter identity cues.',
278:'Flower/leaf structure and face are largely lost; body becomes a generic folded shape and latent outputs include severe disappearance.',
279:'Central body and plant silhouette deform; references follow donor colors but not anatomical details, latent palettes diverge strongly.',
292:'Large open mouth is relocated/reshaped and leaves become limb-like; donor scenes/palettes transfer but markings/anatomy do not stay fixed.',
293:'Mouth/eyes and side leaves are substantially reinterpreted; latent variants display color variation with distorted face.',
322:'Butterfly changes toward an upright winged shape with shifted body/face; pale target backgrounds and donor palettes transfer.',
323:'Wing silhouette remains but patterns/body are replaced by smooth colored regions; references and latent outputs alter facial cues.',
334:'Pink silhouette roughly remains under references, but face/torso lines disappear; latent colors often become brown, white or yellow.',
335:'Source outline is partially retained, yet belly/face details wash out or deform; several references nearly erase internal structure.',
336:'Facial features and ears are shifted/erased; latent variants strongly recolor foreground/background and distort the body.',
337:'Similar source is reconstructed with altered ears/body and face; pale reference output does not preserve the original markings.',
340:'Skull/face and body transform into an uncertain simplified shape; target white backgrounds remove source setting, latent colors alter class cues.',
341:'Skull and eye shape change; toy-like broad outline survives some references while latent blues/yellows alter palette.',
356:'Coiled creature becomes a rounded distorted form; face/neck detail is lost despite donor palette following.',
357:'Coil is loosely preserved but head/neck and inner outline shift; some reference outputs become a washed-out silhouette.',
360:'Three-head motif survives more clearly than many cases, but faces/mouths and base geometry shift; latent colors vary strongly.',
361:'Grouped heads remain approximately; mouth/eye positions and body subdivisions change, including in self-control.',
370:'Multiple eggs fuse into a new elongated blob-like structure; recognizable identity is highly uncertain across both arms.',
371:'Central face motif remains weakly, but egg boundaries and facial placement change; donor backgrounds/colors transfer.',
402:'Large open mouth and segmented body are lost; translated shape is a different curled object with donor-like backgrounds.',
403:'Thin serpentine sprite becomes a rounded fish/tadpole-like form. Black-to-pale background replacement also drives large RMS.',
414:'Face and bulb/leaf details change markedly; reference and latent colors/backgrounds vary while partial spatial layout persists.',
415:'Frontal body silhouette survives, but face and plant/bulb details distort in self, reference and latent outputs.',
512:'Round-body motif survives, but spiral becomes a malformed circular feature and eyes move; latent palettes differ strongly.',
513:'Fish/tail layout remains, but spiral, facial details and fins change; target backgrounds and colors replace source appearance.',
522:'Raised arms/body motif survives loosely, but face becomes a blank/altered region and hands/body edges warp.',
523:'Rounded outline remains, with altered mouth/eyes and limb details; reference scenes/palettes transfer and latent colors differ.',
546:'Fish eye/fins and body shape deform; black/white/color regions shift beyond a simple rendering change.',
547:'Fish body becomes elongated and fin/markings are lost or detached; target backgrounds and palettes substantially change.',
550:'Shell and protruding face remain approximate, but eyes/tongue and outline change; latent outputs strongly alter colors.',
551:'Shell/tongue motif remains weakly while eyes and shell edges warp; dark references obscure structure and latent colors change.',
570:'Long tentacle layout remains but head becomes face-like and legs fuse/change; identity cues are uncertain despite layout retention.',
571:'Dangling silhouette survives, but jewel/face regions and tentacle structure alter; self-control is also visibly changed.',
580:'Circular enclosing object remains, but the small inner Pokemon becomes unrecognizable and recolored; preserving the boundary is insufficient.',
581:'Card layout and creature anatomy collapse into layered blocks; reference colors transfer while latent forms/backgrounds differ radically.',
594:'Head/leaf motif is partially recognizable, but mouth changes to a closed circular snout and eyes alter; latent palettes depart from source.',
595:'Source face/mouth and leaf geometry transform into an elongated new shape; both arms change background and anatomy.'}
rows=[]
for i,q in enumerate(ev):
 if not q['representative']:continue
 assert i in notes
 rows.append({'source_id':i,'category':q['category'],'source_path':q['source_path'],'reviewed_arms':{'reference_variants':[0,1,2,3],'latent_variants':[0,1,2,3],'self_control':True},'recognizable_identity':'uncertain','anatomy_markings':'visible_changes','background_contamination':'visible_donor_or_latent_background_changes','plausible_augmentation':'not_established','notes':notes[i]})
extra={456:'Small upright creature outline remains only roughly; facial/limb proportions change. Black-to-white replacement contributes strongly to the RMS tail.',82:'Raised-arm rock motif remains, but face/hands and contours change. Background reversal and saturation dominate several variants.',590:'Broad upright pose survives some references, with head, belly and limbs changed. Black-to-white replacement helps explain high RMS.',363:'Lowest-relative diversity still includes clear blue/orange/yellow recoloring and face changes; it is not an exact-collapse example.',420:'Dress/hair silhouette roughly persists but eyes, face and hair detail change; even relatively low-diversity latent variants remain differently colored.'}
tails=json.loads((R/'tails.json').read_text())
for t in tails:t['observation']=notes.get(t['source_id'],extra.get(t['source_id']));t['recognizable_identity']='uncertain';t['plausible_augmentation']='not_established';assert t['observation']
trace=json.loads((R/'outputs/trace_indices.json').read_text())
v={'reviewer':'worker AI, small unblinded direct visual inspection','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'all_required_sheets_inspected':True,'sheet_count':22,'representative_source_count':64,'representative_generated_outputs_reviewed':576,'tail_source_count':8,'tail_generated_outputs_reviewed':72,'tail_representative_overlap':3,'milestone_source_count':32,'milestone_translation_count':320,'sheets':[p.name for p in sorted((R/'sheets').glob('*.png'))],'representative_reviews':rows,'tail_reviews':tails,'milestone_reviews':[{'source_id':i,'milestones':['initial','1000','5000','10000','final'],'arms':['reference0','latent0'],'observation':'Initial colorful noise and1000-step blur give way to structured but distorted shapes by5000;10000/final sharpen contours without establishing preserved anatomy. Final-specific observation: '+notes[i]} for i in trace],'overall_observations':['Reference outputs visibly follow donor palette and background, but transfer is entangled with faces, markings and shape changes.','Latent outputs exhibit strong visible variation including species-associated recoloring, background replacement, clutter and distorted forms.','Self-style controls also change details, showing a reconstruction/preservation limitation beyond cross-domain styling.','Largest RMS tails include strong background inversion; low-diversity tails still vary visibly and are not exact-collapse evidence.'],'limits':['These labels are descriptive observations, not class accuracy or a prevalence estimate.','Source identity is uncertain without independent verification; no semantic-diversity, FID or augmentation-benefit claim.','Milestones were inspected descriptively; final checkpoint was fixed by iteration count, not selected for appearance.']}
(R/'visual_review.json').write_text(json.dumps(v,indent=2)+'\n')
