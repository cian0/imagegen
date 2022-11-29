import os
import requests
import random 
import uuid

import boto3 
import subprocess
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler, EulerAncestralDiscreteScheduler, StableDiffusionImg2ImgPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

from prompts import man_prompts, woman_prompts, handsome_prompts

# from diffusers.safety_checker import StableDiffusionSafetyChecker

from transformers import CLIPFeatureExtractor
# from IPython.display import display

BUCKET = os.getenv('MODEL_BUCKET')
S3_PATH = os.getenv('MODEL_PATH')
MODEL_ID = os.getenv('MODEL_ID')

BATCH_ITER = int(os.getenv('BATCH_ITER'))
BATCH_SAMPLES = int(os.getenv('BATCH_SAMPLES'))
SAMPLING_STEPS = int(os.getenv('SAMPLING_STEPS'))

TG_API_KEY = os.environ.get('TG_API_KEY')
TG_CHANNEL_ID = os.environ.get('TG_CHANNEL_ID')
PROMPTLIST = os.environ.get('PROMPTLIST')
model_class = os.getenv('MODEL_CLASS')

model_path = '/workspace/model' #WEIGHTS_DIR             # If you want to use previously trained model saved in gdrive, replace this with the full path of model in gdrive

# scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
scheduler = EulerAncestralDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
# scheduler = EulerAncestralDiscreteScheduler()
pipe = StableDiffusionPipeline.from_pretrained(model_path, scheduler=scheduler
#                                                , safety_checker=StableDiffusionSafetyChecker
#                                                , feature_extractor=CLIPFeatureExtractor
                                               , torch_dtype=torch.float16).to("cuda")

img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_path,
    revision="fp16", 
    scheduler=scheduler,
#     safety_checker=StableDiffusionSafetyChecker,
#     feature_extractor=CLIPFeatureExtractor,
    torch_dtype=torch.float16,
).to("cuda")

g_cuda = None

key = f"{MODEL_ID} person"
prompt = f"{MODEL_ID} person" #@param {type:"string"}
negative_prompt = "blurry, pixelated, faceless, sexy, cleavage, sensual, titties" #@param {type:"string"}
num_samples = BATCH_SAMPLES #@param {type:"number"}
guidance_scale = 7.5 #@param {type:"number"}
# num_inference_steps = 70 #@param {type:"number"}
num_inference_steps = SAMPLING_STEPS #@param {type:"number"}
num_inference_steps_imgtoimg = SAMPLING_STEPS
height = 512 #@param {type:"number"}
width = 512 #@param {type:"number"}


if model_class == 'man':
    promptsarr = man_prompts
else:
    promptsarr = woman_prompts

if PROMPTLIST != None and PROMPTLIST != '':
    promptsarr = PROMPTLIST.split('||')

uploadedCtr = 0
ctr = 0
num_batches = BATCH_ITER
s3 = boto3.resource('s3')

def send_post_req(input, host):
    try:
        headers = {"Content-Type": "application/json; charset=utf-8"}
        r = requests.post(host, headers=headers, json=input) 
        return r.json()
    except ConnectionError as e:    # This is the correct syntax
        print(e)
        return {'error': e}

def telegram_bot_notif(msg) :
    print(msg)
    input = {
        "text" : msg,
        "chat_id" : TG_CHANNEL_ID,
        "disable_notification" : True
    }
    print(send_post_req(input, f'https://api.telegram.org/{TG_API_KEY}/sendMessage'))


telegram_bot_notif(f'starting image gen for {MODEL_ID}')

while ctr < num_batches:
    realprompt = random.choice(promptsarr).replace('_KEYS_', key)
    with autocast("cuda"), torch.inference_mode():
        images = pipe(
            realprompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_samples,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=g_cuda
        ).images

    with autocast("cuda"), torch.inference_mode():
        for img in images:
            # display(img)
            img = img.resize((1024,1024))
            img2img_output = img2img_pipe(
                prompt=realprompt, 
                init_image=img, 
                num_inference_steps=num_inference_steps_imgtoimg,
                strength=0.25,
                guidance_scale=8).images
            unique_filename = str(uuid.uuid4())
            img2img_output[0].save('/workspace/imagegen/outputs/' +unique_filename + ".png")

            # aws s3 cp /workspace/sdw/outputs s3://$MODEL_BUCKET/$/outputs --recursive 
            # s3.Bucket(BUCKET).upload_file('/workspace/imagegen/outputs/' +unique_filename + ".png", f"{S3_PATH}/{MODEL_ID}/outputs/{unique_filename}.png")

            subprocess.run([
                "/workspace/imagegen/face_detect_purge_diffusers.sh"
            ])
            subprocess.run([
                "aws", 
                "s3", 
                "sync", 
                # f"/workspace/matched",
                "/workspace/matcher/similar/outputs",
                f"s3://aimodels-cyian/models/{MODEL_ID}/outputs",
            ])
            # display(img2img_output[0])
    ctr+=1

telegram_bot_notif(f'Done with image gen for {MODEL_ID}')