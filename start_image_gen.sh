#!/bin/bash


export MODEL_ID=$_MODEL_ID
export MODEL_KEY=$_MODEL_KEY
export MODEL_CLASS=$_MODEL_CLASS
export MODEL_STEPS=$_MODEL_STEPS
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
# export _HIGHRES_FIX_DENOISING=$_HIGHRES_FIX_DENOISING
export TG_API_KEY=$_TG_API_KEY
export TG_CHANNEL_ID=$_TG_CHANNEL_ID

curl -X POST \
     -H "Content-Type: application/json" \
     -d "{\"chat_id\": \"$TG_CHANNEL_ID\", \"text\": \"Starting processing for $MODEL_ID $MODEL_KEY $MODEL_CLASS\", \"disable_notification\": true}" \
     https://api.telegram.org/$TG_API_KEY/sendMessage

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


# cd /workspace/sdw
# git reset --hard
# # lets use the old commit since new ones are broken. 
# git checkout 219e279b0376d60382fce6a993641f806710ac44

git clone https://github.com/ShivamShrirao/diffusers.git sdw
cd /workspace/sdw/examples/dreambooth
wget -q https://github.com/ShivamShrirao/diffusers/raw/main/examples/dreambooth/train_dreambooth.py \
    && wget -q https://github.com/ShivamShrirao/diffusers/raw/main/scripts/convert_diffusers_to_original_stable_diffusion.py \
    && pip install -qq git+https://github.com/ShivamShrirao/diffusers@219e279b0376d60382fce6a993641f806710ac44  \
    && pip install -q -U --pre triton \
    && pip install -q accelerate==0.12.0 transformers ftfy bitsandbytes gradio natsort

cd /workspace/
git clone https://github.com/$REPO_ID/stable-diffusion-webui.git stable-diffusion-webui
git clone https://github.com/$REPO_ID/imagegen.git

chmod +x /workspace/imagegen/face_detect_purge.sh

cd /workspace/stable-diffusion-webui
git checkout last_working

python3 -m venv venv
source venv/bin/activate

cd /workspace
# install dlib first
apt-get -y install build-essential cmake pkg-config libx11-dev libatlas-base-dev libgtk-3-dev libboost-python-dev
git clone https://github.com/davisking/dlib
cd dlib
python3 setup.py install

#install face recognition
pip3 install face_recognition

cd /workspace/

GPU_NAME=`nvidia-smi --query-gpu=gpu_name --format=csv,noheader`

aws s3 cp s3://$MODEL_BUCKET/xformerwheels/ ./ --recursive

if [[ $GPU_NAME == *"3090"* ]]; then
    export FORCE_CUDA="1" && \
        export TORCH_CUDA_ARCH_LIST=8.6 && \
        CUDA_VISIBLE_DEVICES=0 pip install /workspace/xformers-0.0.14.dev0-cp39-cp39-linux_x86_64.whl
fi

if [[ $GPU_NAME == *"A100"* ]]; then    
    export FORCE_CUDA="1" && \
        CUDA_VISIBLE_DEVICES=0 pip install /workspace/a100/xformers-0.0.15.dev0+fd21b40.d20221115-cp39-cp39-linux_x86_64.a100.cuda11.8.whl
fi

aws s3 cp s3://$MODEL_BUCKET/$MODEL_PATH/$MODEL_ID/$MODEL_ID.ckpt /workspace/stable-diffusion-webui/model.ckpt 
aws s3 cp s3://$MODEL_BUCKET/$MODEL_PATH/$MODEL_ID/$TRAINING_PATH /workspace/training_samples --recursive

python3 /workspace/imagegen/face_cropper.py /workspace/training_samples /workspace/cropped

# install stable diffusion webui
cd /workspace/stable-diffusion-webui

./webui.sh &

# wait for the api to become available
while ! curl --output /dev/null --silent --head --fail http://0.0.0.0:7860; do sleep 1 && echo -n .; done; 

python /workspace/imagegen/generateimages_local.py

aws s3 cp /workspace/sdw/outputs s3://$MODEL_BUCKET/$MODEL_PATH/$MODEL_ID/outputs --recursive 

aws s3 cp "/workspace/logs" s3://$MODEL_BUCKET/$MODEL_PATH/$MODEL_ID/logs.txt

apt-get install -y wget; wget https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast; chmod +x vast;
./vast destroy instance ${VAST_CONTAINERLABEL:2}

sleep infinity
wait