import os
import requests
import random 
import json

from time import sleep
from prompts import man_prompts, woman_prompts, handsome_prompts

from requests.exceptions import ConnectionError
from json.decoder import JSONDecodeError
# this works for 
# 4b22ec413843cfc787bf5fced1193f77bb5cf0b6 automatic1111 last commit

# Get environment variables -- uncomment these for actual use
# MODEL_ID = os.getenv('MODEL_ID')
# MODEL_KEY = os.environ.get('MODEL_KEY')
# MODEL_CLASS = os.environ.get('MODEL_CLASS')
# MODEL_STEPS = int(os.environ.get('MODEL_STEPS'))
# BATCH_SAMPLES = int(os.environ.get('BATCH_SAMPLES'))
# CFG_SCALE = int(os.environ.get('CFG_SCALE'))
# SAMPLING_STEPS = int(os.environ.get('SAMPLING_STEPS'))
# SAMPLING_METHOD = os.environ.get('SAMPLING_METHOD')
# RANDOM_PROMPTS = int(os.environ.get('RANDOM_PROMPTS'))
# WIDTH = int(os.environ.get('WIDTH'))
# HEIGHT = int(os.environ.get('HEIGHT'))
# RESTORE_FACES = int(os.environ.get('RESTORE_FACES'))

# BATCH_ITER = int(os.environ.get('BATCH_ITER'))
# HIGHRES_FIX = int(os.environ.get('HIGHRES_FIX'))
# TG_API_KEY = os.environ.get('TG_API_KEY')
# TG_CHANNEL_ID = os.environ.get('TG_CHANNEL_ID')
# prm = os.environ.get('PR')


# MODEL_ID = 'mmharnelle'#os.getenv('MODEL_ID')
# MODEL_KEY = 'parkminyoung'#os.environ.get('MODEL_KEY')
# MODEL_CLASS = 'woman'#os.environ.get('MODEL_CLASS')
MODEL_ID = 'shad_rdg'#os.getenv('MODEL_ID')
MODEL_KEY = 'portrait of shad_rdg'#os.environ.get('MODEL_KEY')
MODEL_CLASS = 'man'#os.environ.get('MODEL_CLASS')
MODEL_STEPS = 20 #int(os.environ.get('MODEL_STEPS'))
BATCH_SAMPLES = 1 #int(os.environ.get('BATCH_SAMPLES'))
CFG_SCALE = 9 #int(os.environ.get('CFG_SCALE'))
SAMPLING_STEPS = 20 #int(os.environ.get('SAMPLING_STEPS'))
SAMPLING_METHOD = 'Euler a' #os.environ.get('SAMPLING_METHOD')
RANDOM_PROMPTS = 1 #int(os.environ.get('RANDOM_PROMPTS'))
# WIDTH = 1024 #int(os.environ.get('WIDTH'))
# HEIGHT = 1024 #int(os.environ.get('HEIGHT'))
WIDTH = 512 #int(os.environ.get('WIDTH'))
HEIGHT = 512 #int(os.environ.get('HEIGHT'))
RESTORE_FACES = 0 #int(os.environ.get('RESTORE_FACES'))

BATCH_ITER = 4 #int(os.environ.get('BATCH_ITER'))
HIGHRES_FIX = 0 #int(os.environ.get('HIGHRES_FIX'))
HIGHRES_FIX_DENOISING = 0.3 #int(os.environ.get('HIGHRES_FIX'))
TG_API_KEY = ''#os.environ.get('TG_API_KEY')
TG_CHANNEL_ID = ''# os.environ.get('TG_CHANNEL_ID')
# prm = '1^2^3^1^2^3^1^2^3^1^2^3^1^2^3^1^2^3^'#os.environ.get('PR')
prm = '1^2^3^1^2^3^1^2^3^1^2^3^1^2^3^1^2^3^1^2^3^1^2^3^1^2^3^1^2^3^1^2^3^1^2^3^1^2^3^1^2^3^1^2^3^1^2^3^1^2^3^1^2^3^1^2^3^1^2^3^1^2^3^1^2^3^1^2^3^1^2^3^1^2^3^1^2^3^1^2^3^1^2^3^1^2^3^1^2^3^1^2^3^1^2^3^1^2^3^1^2^3^1^2^3^1^2^3^'#os.environ.get('PR')

# if RESTORE_FACES == 0: 
#     RESTORE_FACES = False
# else:
#     RESTORE_FACES = True

# MODEL_ID = 'klao'# os.getenv('MODEL_ID')
# MODEL_KEY = 'ahhjaehyun'#os.environ.get('MODEL_KEY')
# MODEL_CLASS = 'man'#os.environ.get('MODEL_CLASS')
# MODEL_STEPS = 3 #int(os.environ.get('MODEL_STEPS'))
# BATCH_SAMPLES = 1#int(os.environ.get('BATCH_SAMPLES'))
# CFG_SCALE = 6#int(os.environ.get('CFG_SCALE'))
# SAMPLING_STEPS = 30#int(os.environ.get('SAMPLING_STEPS'))
# SAMPLING_METHOD = "Euler a"# os.environ.get('SAMPLING_METHOD')
# RANDOM_PROMPTS = 1#int(os.environ.get('RANDOM_PROMPTS'))
# WIDTH = 512#int(os.environ.get('WIDTH'))
# HEIGHT = 512#int(os.environ.get('HEIGHT'))
# RESTORE_FACES = 0#int(os.environ.get('RESTORE_FACES'))

# BATCH_ITER = 4 #int(os.environ.get('BATCH_ITER'))
# HIGHRES_FIX = 0 #int(os.environ.get('HIGHRES_FIX'))
# TG_API_KEY = 'bot1702872815:AAF5NfKt3099gpx1h2vHTLce0bN_JI-zfkk'#os.environ.get('TG_API_KEY')
# TG_CHANNEL_ID = '-1001597965870'# os.environ.get('TG_CHANNEL_ID')
# prm = '1^2^3^1^2^3^1^2^3^1^2^3^1^2^3^1^2^3^'#os.environ.get('PR')

# prm = "_KEYS_ some string ^ _KEYS_ another string"
prm = prm.replace("_KEYS_", f"{MODEL_KEY} {MODEL_CLASS}")
splittedPrompts = prm.split("^")


# // let API_URI = "http://192.168.1.7:7860/api/predict/"
# ;//http://0.0.0.0:8888/api/predict/
# API_URI = "http://0.0.0.0:8888/api/predict/"
API_URI = "http://0.0.0.0:7860/api/predict/"
# API_URI = "http://localhost:7860/api/predict/"
# API_URI = "http://192.168.1.7:7860/api/predict/"

def send_post_req(input, host):
    try:
        headers = {"Content-Type": "application/json; charset=utf-8"}
        r = requests.post(host, headers=headers, json=input) 
        return r.json()
    except ConnectionError as e:    # This is the correct syntax
        print(e)
        return {'error': e}

def automatic1111_run_txt2img(body):
    input = generate_body(body)
    return send_post_req(input, API_URI)

def search_lexica(search_key):
    try:
        headers = {"Content-Type": "application/json; charset=utf-8"}
        r = requests.get(f'https://lexica.art/api/v1/search?q={search_key.replace(" ", "+")}', headers=headers) 
        return r.json()
    except ConnectionError as e:    # This is the correct syntax
        print(e)
        return {'error': e}
    except JSONDecodeError as e:
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

def generate_body(params):
    body = {
        "fn_index": 51,
        "data":
        [
            params['prompt'],
            params['negative_prompt'],
            'None',
            'None',
            int(params['sampling_steps']), #//sampling steps 
            params['sampling_method'], #//sampling method 
            params['restore_faces'] == 1,  #//restore faces  
            False,
            int(params['batch_count']), #//batch count 
            int(params['batch_sizes']), #//batch sizes 
            int(params['cfg_scale']), #//cfg scale 
            int(params['seed']), #//seed 
            -1,
            0,
            0,
            0,
            False,
            int(params['height']), #height 
            int(params['width']), #width 
            params['highres_fix'] == 1, #highres fix 
            params['highres_fix_denoising'] , #//denoising_strength (for highres fix only)
            0,
            0,
            "None",
            False,
            False,
            False,
            # None, what is this?
            "",
            "Seed",
            "",
            "Nothing",
            "",
            True,
            False,
            False,
            None,
            "",
            ""
        ] 
    }
    return body 

while automatic1111_run_txt2img({
                'prompt': 'test',
                'negative_prompt': '',
                'sampling_steps': 1,
                'sampling_method': SAMPLING_METHOD,
                'restore_faces': False,
                'batch_count': 1,
                'batch_sizes': 1,
                'cfg_scale': CFG_SCALE,
                'highres_fix': HIGHRES_FIX,
                'highres_fix_denoising': HIGHRES_FIX_DENOISING,

                'seed': -1,
                'height': 10,
                'width': 10,
            }).get('error') != None:
    sleep(1)
    print('Error connecting, retrying...')

# telegram_bot_notif(f'Starting image generation process for {MODEL_ID}')

prompts_ext_data = requests.get('https://cian0.github.io/test.txt')

for prompt in splittedPrompts:
    # for each batch and iteration, run the image generation process
    if RANDOM_PROMPTS == 1:
        if MODEL_CLASS == 'man':
            promptsarr = man_prompts
            # promptsarr = woman_prompts
        else:
            promptsarr = woman_prompts
            # promptsarr = handsome_prompts
        # promptsarr = man_prompts

        realprompt = random.choice(promptsarr).replace('_KEYS_', MODEL_KEY + ' ' + MODEL_CLASS)
    else:
        # print(r.status_code)
        # print(r.text)
        data = json.loads(prompts_ext_data.text)
        # print
        # generate prompts based on 
        promptsarr = data[prompt]
        realprompt = random.choice(promptsarr).replace('_KEYS_', MODEL_KEY + ' ' + MODEL_CLASS)

    ctr = 0
    while ctr < BATCH_SAMPLES:
        print(realprompt)
        ctr += 1
        automatic1111_run_txt2img({
                    'prompt': realprompt,
                    # 'negative_prompt': '((((((((real life photo)))))))) ((((ugly)))), (((duplicate))), ((morbid)), ((mutilated)), out of frame, extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck)))',
                    'negative_prompt': 'sex, kissing, skin, naked, blowjob, titties, sexy, explicit, sensual, breast, group, harem, love making, cock, penis',
                    'sampling_steps': SAMPLING_STEPS,
                    'sampling_method': SAMPLING_METHOD,
                    # 'sampling_method': 'DPM adaptive',
                    'restore_faces': RESTORE_FACES,
                    'batch_count': 1,
                    'batch_sizes': BATCH_ITER,
                    'cfg_scale': CFG_SCALE,
                    'highres_fix': HIGHRES_FIX,
                    'highres_fix_denoising': HIGHRES_FIX_DENOISING,
                    'seed': -1,
                    'height': HEIGHT,
                    'width': WIDTH,
                })

# telegram_bot_notif(f'Done with image generation for {MODEL_ID}')

# telegram_bot_notif('sasa')
