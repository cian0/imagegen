#!/bin/bash

git clone https://github.com/ShivamShrirao/diffusers.git sdw
cd /workspace/sdw/examples/dreambooth
wget -q https://github.com/ShivamShrirao/diffusers/raw/main/examples/dreambooth/train_dreambooth.py \
    && wget -q https://github.com/ShivamShrirao/diffusers/raw/main/scripts/convert_diffusers_to_original_stable_diffusion.py \
    && pip install -qq git+https://github.com/ShivamShrirao/diffusers@219e279b0376d60382fce6a993641f806710ac44  \
    && pip install -q -U --pre triton \
    && pip install -q accelerate==0.12.0 transformers ftfy bitsandbytes gradio natsort

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

cd /workspace/sdw
git reset --hard
# lets use the old commit since new ones are broken. 
git checkout 219e279b0376d60382fce6a993641f806710ac44

cd /workspace/
git clone https://github.com/$REPO_ID/imagegen.git imagegen

curl -X POST \
     -H "Content-Type: application/json" \
     -d "{\"chat_id\": \"$TG_CHANNEL_ID\", \"text\": \"Starting processing for $MODEL_ID $MODEL_KEY $MODEL_CLASS\", \"disable_notification\": true}" \
     https://api.telegram.org/$TG_API_KEY/sendMessage

cd /workspace
# install dlib first
# apt-get -y install build-essential cmake pkg-config libx11-dev libatlas-base-dev libgtk-3-dev libboost-python-dev
# git clone https://github.com/davisking/dlib
# cd dlib
# python3 setup.py install

# #install face recognition
# pip3 install face_recognition

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
aws s3 cp s3://$MODEL_BUCKET/$MODEL_PATH/$MODEL_ID/$TRAINING_PATH ./training_samples --recursive

# mv ./training_samples ./uncropped

# python3 /workspace/imagegen/face_cropper.py /content/data/$MODEL_ID/uncropped /content/data/$MODEL_ID/training_samples

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

cd /workspace/

pip install -qq git+https://github.com/ShivamShrirao/diffusers@219e279b0376d60382fce6a993641f806710ac44
pip install -q -U --pre triton
pip install -q accelerate==0.12.0 transformers ftfy bitsandbytes gradio natsort

cd /workspace/sdw/
git checkout 219e279b0376d60382fce6a993641f806710ac44

export MODEL_NAME="runwayml/stable-diffusion-v1-5" 
export OUTPUT_DIR="/workspace/sdw/examples/dreambooth/stable_diffusion_weights/output" 

cat <<EOT > /workspace/sdw/examples/dreambooth/concepts_list.json
    [{
        "instance_prompt":      "photo of $MODEL_ID person",
        "class_prompt":         "photo of a person",
        "instance_data_dir":    "/content/data/$MODEL_ID/training_samples/",
        "class_data_dir":       "/content/data/person"
    }]
EOT

cd /workspace/sdw/examples/dreambooth/

curl -X POST \
     -H "Content-Type: application/json" \
     -d "{\"chat_id\": \"$TG_CHANNEL_ID\", \"text\": \"Starting training for $MODEL_ID $MODEL_KEY $MODEL_CLASS\", \"disable_notification\": true}" \
     https://api.telegram.org/$TG_API_KEY/sendMessage


export GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | tee /dev/tty)

if [[ $GPU_NAME == *"3090"* ]]; then
    accelerate launch train_dreambooth.py \
        --pretrained_model_name_or_path="$MODEL_NAME" \
        --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
        --output_dir=$OUTPUT_DIR \
        --revision="fp16" \
        --with_prior_preservation --prior_loss_weight=1.0 \
        --seed=1337 \
        --resolution=512 \
        --train_batch_size=1 \
        --train_text_encoder \
        --center_crop \
        --mixed_precision="fp16" \
        --gradient_accumulation_steps=2 \
        --learning_rate=1e-6 \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --num_class_images=80 \
        --sample_batch_size=4 \
        --max_train_steps=2200 \
        --save_interval=2200 \
        --save_sample_prompt="photo of $MODEL_ID person" \
        --concepts_list="concepts_list.json"
fi

if [[ $GPU_NAME == *"A100"* ]]; then    
    accelerate launch train_dreambooth.py \
        --pretrained_model_name_or_path=$MODEL_NAME \
        --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
        --output_dir=$OUTPUT_DIR \
        --revision="fp16" \
        --with_prior_preservation --prior_loss_weight=1.0 \
        --seed=1337 \
        --resolution=512 \
        --train_batch_size=24 \ #change to 8 if 40GB alloc
        --train_text_encoder \
        --center_crop \
        --mixed_precision="fp16" \
        --gradient_accumulation_steps=2 \
        --learning_rate=1e-6 \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --num_class_images=80 \
        --sample_batch_size=4 \
        --max_train_steps=1600 \
        --save_interval=1200 \
        --save_sample_prompt="photo of $MODEL_ID person" \
        --concepts_list="concepts_list.json"
fi

curl -X POST \
     -H "Content-Type: application/json" \
     -d "{\"chat_id\": \"$TG_CHANNEL_ID\", \"text\": \"Finished training for $MODEL_ID $MODEL_KEY $MODEL_CLASS\", \"disable_notification\": true}" \
     https://api.telegram.org/$TG_API_KEY/sendMessage

python convert_diffusers_to_original_stable_diffusion.py --model_path "$OUTPUT_DIR/2200"  --checkpoint_path "$OUTPUT_DIR/2200/model.ckpt" --half

aws s3 cp "$OUTPUT_DIR/2200/model.ckpt" s3://$MODEL_BUCKET/$MODEL_PATH/$MODEL_ID/$MODEL_ID.ckpt

curl -X POST \
     -H "Content-Type: application/json" \
     -d "{\"chat_id\": \"$TG_CHANNEL_ID\", \"text\": \"Done uploading model for $MODEL_ID $MODEL_KEY $MODEL_CLASS\", \"disable_notification\": true}" \
     https://api.telegram.org/$TG_API_KEY/sendMessage

# Start image gen!!

cd /workspace/
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
git clone https://github.com/CompVis/stable-diffusion.git

wget -q https://github.com/ShivamShrirao/diffusers/raw/main/scripts/convert_original_stable_diffusion_to_diffusers.py

pip install omegaconf
pip install boto3 

# python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path "./$MODEL_ID.ckpt" --original_config_file "/workspace/stable-diffusion/configs/stable-diffusion/v1-inference.yaml" --dump_path "./model"

mv "$OUTPUT_DIR/2200" "/workspace/model"

curl -X POST \
     -H "Content-Type: application/json" \
     -d "{\"chat_id\": \"$TG_CHANNEL_ID\", \"text\": \"Starting image gen for $MODEL_ID $MODEL_KEY $MODEL_CLASS\", \"disable_notification\": true}" \
     https://api.telegram.org/$TG_API_KEY/sendMessage

# do the image gen here

cd /workspace/imagegen
mkdir outputs
accelerate launch diffusers_image_gen.py

curl -X POST \
     -H "Content-Type: application/json" \
     -d "{\"chat_id\": \"$TG_CHANNEL_ID\", \"text\": \"Finished image gen for $MODEL_ID $MODEL_KEY $MODEL_CLASS\", \"disable_notification\": true}" \
     https://api.telegram.org/$TG_API_KEY/sendMessage

aws s3 cp "/workspace/logs" s3://$MODEL_BUCKET/$MODEL_PATH/$MODEL_ID/logs.txt

# end instance
cd ~
cat ~/.ssh/authorized_keys | md5sum | awk '{print $1}' > ssh_key_hv; echo -n $VAST_CONTAINERLABEL | md5sum | awk '{print $1}' > instance_id_hv; head -c -1 -q ssh_key_hv instance_id_hv > ~/.vast_api_key;
apt-get install -y wget; wget https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast; chmod +x vast;
# ./vast destroy instance ${VAST_CONTAINERLABEL:2}

sleep infinity
wait