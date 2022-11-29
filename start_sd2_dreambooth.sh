#!/bin/bash

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


aws s3 cp s3://$MODEL_BUCKET/xformerwheels/ ./ --recursive

aws s3 cp s3://$MODEL_BUCKET/class_images/person /content/data/person --recursive

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
aws s3 cp s3://$MODEL_BUCKET/$MODEL_PATH/$MODEL_ID/$TRAINING_PATH /content/data/$MODEL_ID/training_samples --recursive


# cd /workspace/sdw
# git reset --hard
# # lets use the old commit since new ones are broken. 
# git checkout 219e279b0376d60382fce6a993641f806710ac44

git clone https://github.com/huggingface/diffusers sdw
cd /workspace/sdw/examples/dreambooth
# wget -q https://github.com/ShivamShrirao/diffusers/raw/main/examples/dreambooth/train_dreambooth.py \
#     && wget -q https://github.com/ShivamShrirao/diffusers/raw/main/scripts/convert_diffusers_to_original_stable_diffusion.py \
#     && pip install -qq git+https://github.com/ShivamShrirao/diffusers@219e279b0376d60382fce6a993641f806710ac44  \
#     && pip install -q -U --pre triton \
#     && pip install -q accelerate==0.12.0 transformers ftfy bitsandbytes gradio natsort

pip install -qq git+https://github.com/huggingface/diffusers

cd /workspace/sdw/examples/dreambooth
pip install -r requirements.txt 

cd /workspace 

mkdir -p ~/.cache/huggingface/accelerate/

cat <<EOT > ~/.cache/huggingface/accelerate/default_config.yaml
compute_environment: LOCAL_MACHINE
deepspeed_config: {}
distributed_type: 'NO'
downcast_bf16: 'no'
fsdp_config: {}
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 1
use_cpu: false
EOT

# for 512 x 512

export MODEL_NAME="stabilityai/stable-diffusion-2"
export INSTANCE_DIR="/content/data/$MODEL_ID/training_samples"
export CLASS_DIR="/content/data/person"
export OUTPUT_DIR="/workspace/sdw/examples/dreambooth/stable_diffusion_weights/output" 

accelerate launch --mixed_precision="fp16" train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of eenygg person" \
  --class_prompt="a photo of person" \
  --resolution=512 \ # can be 768 
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \ 
  --gradient_checkpointing \ # needed as mem rans out
  --learning_rate=2e-6 \ # should be 1e-6 for constant lr_schedule
  --lr_scheduler="polynomial" \ # constant or polynomial
  --lr_warmup_steps=0 \
  --num_class_images=80 \
  --max_train_steps=2200