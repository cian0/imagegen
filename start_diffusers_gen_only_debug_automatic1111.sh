#!/bin/bash

git clone https://github.com/ShivamShrirao/diffusers.git sdw
cd /workspace/sdw/examples/dreambooth
wget -q https://github.com/ShivamShrirao/diffusers/raw/main/examples/dreambooth/train_dreambooth.py \
    && wget -q https://github.com/ShivamShrirao/diffusers/raw/main/scripts/convert_diffusers_to_original_stable_diffusion.py \
    && pip install -qq git+https://github.com/ShivamShrirao/diffusers@219e279b0376d60382fce6a993641f806710ac44  \
    && pip install -U --pre triton==2.0.0.dev20221105 \
    && pip install accelerate==0.12.0 transformers==4.24.0 ftfy==6.1.1 bitsandbytes gradio natsort==8.2.0

export MODEL_ID=$_MODEL_ID
export MODEL_KEY=$_MODEL_KEY
export MODEL_CLASS=$_MODEL_CLASS
export MODEL_STEPS=$_MODEL_STEPS
export MODEL_PATH=$_MODEL_PATH
export MODEL_BUCKET=$_MODEL_BUCKET
export REPO_ID=$_REPO_ID
export BATCH_SAMPLES=$_BATCH_SAMPLES
export CFG_SCALE=$_CFG_SCALE
export SAMPLING_STEPS=$_SAMPLING_STEPS
export SAMPLING_METHOD=$_SAMPLING_METHOD
export RANDOM_PROMPTS=$_RANDOM_PROMPTS
export WIDTH=$_WIDTH
export HEIGHT=$_HEIGHT
export RESTORE_FACES=$_RESTORE_FACES
export BATCH_ITER=$_BATCH_ITER
export HIGHRES_FIX=$_HIGHRES_FIX
export PROMPTLIST=$PROMPTLIST
# export _HIGHRES_FIX_DENOISING=$_HIGHRES_FIX_DENOISING
export TG_API_KEY=$_TG_API_KEY
export TG_CHANNEL_ID=$_TG_CHANNEL_ID

cd /workspace/sdw
git reset --hard
# lets use the old commit since new ones are broken. 
git checkout 219e279b0376d60382fce6a993641f806710ac44

cd /workspace/
git clone https://github.com/$REPO_ID/imagegen.git imagegen
chmod +x /workspace/imagegen/face_detect_purge_diffusers.sh
curl -X POST \
     -H "Content-Type: application/json" \
     -d "{\"chat_id\": \"$TG_CHANNEL_ID\", \"text\": \"Starting processing for $MODEL_ID $MODEL_KEY $MODEL_CLASS\", \"disable_notification\": true}" \
     https://api.telegram.org/$TG_API_KEY/sendMessage

cd /workspace

mkdir -p ~/.huggingface
echo -n "$HUGGINGFACE_TOKEN" > ~/.huggingface/token

pip install awscli --upgrade --user
mv /root/.local/bin/aws* /bin

export AWS_ACCESS_KEY_ID=$S3_AK_ID
export AWS_SECRET_ACCESS_KEY=$S3_AKS
export AWS_DEFAULT_REGION="ap-southeast-1"
export AWS_DEFAULT_OUTPUT="json"

aws configure set aws_access_key_id AWS_ACCESS_KEY_ID
aws configure set aws_secret_access_key AWS_SECRET_ACCESS_KEY
aws configure set region "ap-southeast-1"
aws configure set output "json"

GPU_NAME=`nvidia-smi --query-gpu=gpu_name --format=csv,noheader`

aws s3 cp s3://$MODEL_BUCKET/xformerwheels/ ./ --recursive

# aws s3 cp s3://$MODEL_BUCKET/class_images/person /content/data/person --recursive

if [[ $GPU_NAME == *"3090"* ]]; then
    export FORCE_CUDA="1" && \
        export TORCH_CUDA_ARCH_LIST=8.6 && \
        CUDA_VISIBLE_DEVICES=0 pip install /workspace/xformers-0.0.14.dev0-cp39-cp39-linux_x86_64.whl
fi

if [[ $GPU_NAME == *"A100"* ]]; then    
    export FORCE_CUDA="1" && \
        CUDA_VISIBLE_DEVICES=0 pip install /workspace/a100/xformers-0.0.15.dev0+fd21b40.d20221115-cp39-cp39-linux_x86_64.a100.cuda11.8.whl
fi

mkdir -p /content/data/$MODEL_ID
cd /content/data/$MODEL_ID/
aws s3 cp s3://$MODEL_BUCKET/$MODEL_PATH/$MODEL_ID/$TRAINING_PATH ./training_samples --recursive

mv ./training_samples /workspace/cropped

# Start image gen!!

cd /workspace/
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
git clone https://github.com/CompVis/stable-diffusion.git

aws s3 cp s3://$MODEL_BUCKET/$MODEL_PATH/$MODEL_ID/$MODEL_ID.ckpt /workspace/$MODEL_ID.ckpt 

cd stable-diffusion-webui
mv /workspace/$MODEL_ID.ckpt /workspace/stable-diffusion-webui/$MODEL_ID.ckpt
./webui.sh

# ./vast destroy instance ${VAST_CONTAINERLABEL:2}

sleep infinity
wait