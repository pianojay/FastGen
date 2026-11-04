from pathlib import Path
from PIL import Image, ImageDraw
import torch
from torchvision.transforms.functional import to_pil_image
from .sampling import sample


def sheet(images, labels, path, columns=4):
    images = images.detach().float().cpu()
    height = 84
    canvas = Image.new('RGB', (64*columns, height*((len(images)+columns-1)//columns)), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (im, label) in enumerate(zip(images, labels)):
        x, y = (i%columns)*64, (i//columns)*height
        canvas.paste(to_pil_image(((im+1)/2).clamp(0,1)), (x,y))
        draw.text((x+2,y+65), str(label)[:11], fill='black')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def demo(backbone, output, mapping):
    device = next(backbone.parameters()).device
    inverse = {v:k for k,v in mapping.items()}
    with torch.random.fork_rng(devices=[device.index or 0]):
        for nfe in (1,2,3,4):
            images, labels = [], []
            for category in (0,20,40,60,80,100,120,140):
                generator = torch.Generator(device=device).manual_seed(901)
                images.append(sample(backbone, (4,3,64,64), device=device,
                    category=torch.full((4,),category,device=device,dtype=torch.long), nfe=nfe, generator=generator))
                labels += [inverse[category]]*4
            sheet(torch.cat(images), labels, Path(output)/f'nfe{nfe}.png')
