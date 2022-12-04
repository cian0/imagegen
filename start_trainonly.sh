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
export MODEL_NAME=$_PRETRAINED_MODEL_PATH
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

curl -X POST \
     -H "Content-Type: application/json" \
     -d "{\"chat_id\": \"$TG_CHANNEL_ID\", \"text\": \"Starting processing for $MODEL_ID $MODEL_KEY $MODEL_CLASS\", \"disable_notification\": true}" \
     https://api.telegram.org/$TG_API_KEY/sendMessage

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

cd /workspace
# install dlib first

aws s3 cp s3://$MODEL_BUCKET/dlibwheels/ ./ --recursive
pip install /workspace/ubuntu1804/dlib-19.24.99-cp39-cp39-linux_x86_64.whl


if python -c "import dlib" &> /dev/null; then
    echo 'all good'
else
    echo 'uh oh'

    curl -X POST \
        -H "Content-Type: application/json" \
        -d "{\"chat_id\": \"$TG_CHANNEL_ID\", \"text\": \"DLIB has to be compiled for $MODEL_ID $MODEL_KEY $MODEL_CLASS\", \"disable_notification\": true}" \
        https://api.telegram.org/$TG_API_KEY/sendMessage
    pip uninstall -y dlib
    apt-get -y install build-essential cmake pkg-config libx11-dev libatlas-base-dev libgtk-3-dev libboost-python-dev
    git clone https://github.com/davisking/dlib
    cd dlib
    python3 setup.py install
fi
# aws s3 cp /workspace/dlib/dist/dlib-19.24.99-cp39-cp39-linux_x86_64.whl s3://aimodels-cyian/dlibwheels/ubuntu1804/

# python setup.py bdist_wheel # for building the wheel file

#install face recognition
pip3 install face_recognition

cd /workspace
mkdir -p ~/.huggingface
echo -n "$HUGGINGFACE_TOKEN" > ~/.huggingface/token


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

mv ./training_samples ./uncropped

python3 /workspace/imagegen/face_cropper.py /content/data/$MODEL_ID/uncropped /content/data/$MODEL_ID/training_samples

# TRAINING_FILE_COUNT=`ls /etc | wc -l`*100
STEPS_BASED_ON_FILES=$((`ls /content/data/$MODEL_ID/training_samples | wc -l`*110))

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
pip install -U --pre triton==2.0.0.dev20221105 \
    && pip install accelerate==0.12.0 transformers==4.24.0 ftfy==6.1.1 bitsandbytes gradio natsort==8.2.0

cd /workspace/sdw/
git checkout 219e279b0376d60382fce6a993641f806710ac44

# export MODEL_NAME="runwayml/stable-diffusion-v1-5" 
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
     -d "{\"chat_id\": \"$TG_CHANNEL_ID\", \"text\": \"Starting training for $MODEL_ID $MODEL_KEY $MODEL_CLASS with $STEPS_BASED_ON_FILES steps\", \"disable_notification\": true}" \
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
        --max_train_steps=$STEPS_BASED_ON_FILES \
        --save_interval=$STEPS_BASED_ON_FILES \
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
        --max_train_steps=$STEPS_BASED_ON_FILES \
        --save_interval=$STEPS_BASED_ON_FILES \
        --save_sample_prompt="photo of $MODEL_ID person" \
        --concepts_list="concepts_list.json"
fi

curl -X POST \
     -H "Content-Type: application/json" \
     -d "{\"chat_id\": \"$TG_CHANNEL_ID\", \"text\": \"Finished training for $MODEL_ID $MODEL_KEY $MODEL_CLASS\", \"disable_notification\": true}" \
     https://api.telegram.org/$TG_API_KEY/sendMessage
     
mv /workspace/imagegen/convert_diffusers_to_original_stable_diffusion.py ./convert_diffusers_to_original_stable_diffusion.py
python convert_diffusers_to_original_stable_diffusion.py --model_path "$OUTPUT_DIR/2200"  --checkpoint_path "$OUTPUT_DIR/2200/model.ckpt" --half

aws s3 cp "$OUTPUT_DIR/2200/model.ckpt" s3://$MODEL_BUCKET/$MODEL_PATH/$MODEL_ID/$MODEL_ID.ckpt
aws s3 cp "/workspace/logs" s3://$MODEL_BUCKET/$MODEL_PATH/$MODEL_ID/logs.txt

curl -X POST \
     -H "Content-Type: application/json" \
     -d "{\"chat_id\": \"$TG_CHANNEL_ID\", \"text\": \"Done uploading model for $MODEL_ID $MODEL_KEY $MODEL_CLASS\", \"disable_notification\": true}" \
     https://api.telegram.org/$TG_API_KEY/sendMessage

# end instance
cd ~
cat ~/.ssh/authorized_keys | md5sum | awk '{print $1}' > ssh_key_hv; echo -n $VAST_CONTAINERLABEL | md5sum | awk '{print $1}' > instance_id_hv; head -c -1 -q ssh_key_hv instance_id_hv > ~/.vast_api_key;
apt-get install -y wget; wget https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast; chmod +x vast;

FILE=$OUTPUT_DIR/2200/model.ckpt
if test -f "$FILE"; then
    echo "$FILE exists."
    ./vast destroy instance ${VAST_CONTAINERLABEL:2}
else 
    curl -X POST \
        -H "Content-Type: application/json" \
        -d "{\"chat_id\": \"$TG_CHANNEL_ID\", \"text\": \"Model cannot be uploaded for $MODEL_ID $MODEL_KEY $MODEL_CLASS, not shutting down\", \"disable_notification\": true}" \
        https://api.telegram.org/$TG_API_KEY/sendMessage
fi

sleep infinity
wait