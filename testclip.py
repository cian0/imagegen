from PIL import Image
from clip_interrogator import Interrogator, Config
image = Image.open('/home/ian/projs/stable-diffusion-webui/outputs/txt2img-images/00038-993863794-portrait of paularedygg person as fire elemental, ghostly form, transparent, d & d, golden!!! palette, highly detailed, portrait.png').convert('RGB')
ci = Interrogator(Config(clip_model_name="ViT-L/14"))
print(ci.interrogate(image))

